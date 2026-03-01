"""
╔══════════════════════════════════════════════════════════════╗
║  Anomaly Detector — Real-time anomaly detection using       ║
║  Isolation Forest + statistical methods for power data      ║
╚══════════════════════════════════════════════════════════════╝

Methods:
    1. Isolation Forest — Unsupervised anomaly detection
    2. Statistical Z-Score — Deviation-based detection
    3. Moving Average Deviation — Trend-based detection
    4. Ensemble — Combined scoring for robustness
"""

import numpy as np
import pandas as pd
import joblib
import os
from typing import Tuple, Dict, List, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from collections import deque

from .config import (
    ANOMALY_CONTAMINATION, ANOMALY_WINDOW_SIZE, ANOMALY_SENSITIVITY,
    VOLTAGE_NOMINAL, VOLTAGE_HIGH, VOLTAGE_LOW,
    CURRENT_MAX, POWER_MAX, MODELS_DIR
)
from .feature_engineer import FeatureEngineer


class AnomalyDetector:
    """
    Multi-method anomaly detector for power monitoring data.
    
    Usage:
        detector = AnomalyDetector()
        detector.train(voltage_arr, current_arr, power_arr)
        
        # Real-time detection
        is_anomaly, score, details = detector.detect(v, i, p)
    """

    def __init__(self, contamination: float = ANOMALY_CONTAMINATION,
                 window_size: int = ANOMALY_WINDOW_SIZE,
                 sensitivity: float = ANOMALY_SENSITIVITY):
        
        self.contamination = contamination
        self.window_size = window_size
        self.sensitivity = sensitivity
        
        # Models
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            n_estimators=200,
            max_samples='auto',
            random_state=42,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.feature_engineer = FeatureEngineer(window_size=window_size)
        
        # Buffers for real-time detection
        self.voltage_buffer = deque(maxlen=window_size)
        self.current_buffer = deque(maxlen=window_size)
        self.power_buffer = deque(maxlen=window_size)
        
        # Training statistics (for Z-score method)
        self.train_stats = {}
        
        # State
        self.is_trained = False
        self.anomaly_history = deque(maxlen=1000)

    def train(self, voltage: np.ndarray, current: np.ndarray, power: np.ndarray,
              verbose: bool = True) -> Dict:
        """
        Train the anomaly detector on normal/mixed data.
        
        Args:
            voltage, current, power: Arrays of sensor data
            verbose: Print training progress
            
        Returns:
            Training metrics dict
        """
        if verbose:
            print("\n  🧠 Training Anomaly Detector...")
            print(f"     Input samples: {len(voltage):,}")
        
        # Store training statistics
        self.train_stats = {
            'voltage_mean': np.mean(voltage), 'voltage_std': np.std(voltage),
            'current_mean': np.mean(current), 'current_std': np.std(current),
            'power_mean': np.mean(power), 'power_std': np.std(power),
        }
        
        # Extract features from sliding windows
        df = pd.DataFrame({
            'voltage': voltage,
            'current': current,
            'power': power,
        })
        
        features_df = self.feature_engineer.extract_features_from_dataframe(df)
        
        if len(features_df) == 0:
            raise ValueError("Not enough data to extract features. Need at least "
                           f"{self.window_size} samples.")
        
        if verbose:
            print(f"     Feature vectors: {len(features_df):,}")
            print(f"     Features per vector: {features_df.shape[1]}")
        
        # Scale features
        X = features_df.values
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Isolation Forest
        self.isolation_forest.fit(X_scaled)
        
        # Evaluate on training data
        scores = self.isolation_forest.decision_function(X_scaled)
        predictions = self.isolation_forest.predict(X_scaled)
        n_anomalies = np.sum(predictions == -1)
        
        self.is_trained = True
        
        metrics = {
            'total_samples': len(X_scaled),
            'anomalies_found': int(n_anomalies),
            'anomaly_rate': float(n_anomalies / len(X_scaled)),
            'score_mean': float(np.mean(scores)),
            'score_std': float(np.std(scores)),
        }
        
        if verbose:
            print(f"     ✅ Training complete!")
            print(f"     Anomalies in training data: {n_anomalies} "
                  f"({metrics['anomaly_rate']:.2%})")
        
        return metrics

    def train_from_dataframe(self, df: pd.DataFrame,
                              voltage_col: str = 'voltage',
                              current_col: str = 'current',
                              power_col: str = 'power',
                              verbose: bool = True) -> Dict:
        """Train from a pandas DataFrame."""
        return self.train(
            df[voltage_col].values,
            df[current_col].values,
            df[power_col].values,
            verbose=verbose,
        )

    def detect(self, voltage: float, current: float, power: float) -> Tuple[bool, float, Dict]:
        """
        Detect anomaly for a single new data point (real-time).
        
        Args:
            voltage, current, power: Current sensor readings
            
        Returns:
            (is_anomaly, anomaly_score, details_dict)
        """
        # Add to buffers
        self.voltage_buffer.append(voltage)
        self.current_buffer.append(current)
        self.power_buffer.append(power)
        
        details = {
            'methods': {},
            'alerts': [],
        }
        
        # ── Method 1: Threshold-based detection ──
        threshold_score = self._threshold_check(voltage, current, power)
        details['methods']['threshold'] = threshold_score
        
        # ── Method 2: Z-Score detection ──
        zscore_result = self._zscore_check(voltage, current, power)
        details['methods']['zscore'] = zscore_result['score']
        if zscore_result['alerts']:
            details['alerts'].extend(zscore_result['alerts'])
        
        # ── Method 3: Isolation Forest (if trained and buffer full) ──
        if_score = 0.0
        if self.is_trained and len(self.voltage_buffer) >= self.window_size:
            v_arr = np.array(self.voltage_buffer)
            i_arr = np.array(self.current_buffer)
            p_arr = np.array(self.power_buffer)
            
            features = self.feature_engineer.extract_all_features(v_arr, i_arr, p_arr)
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            if_raw_score = self.isolation_forest.decision_function(features_scaled)[0]
            # Normalize: negative = anomaly, positive = normal
            # Convert to 0-1 where 1 = most anomalous
            if_score = max(0, -if_raw_score)
            details['methods']['isolation_forest'] = if_score
        
        # ── Method 4: Moving Average Deviation ──
        ma_score = self._moving_average_check()
        details['methods']['moving_avg'] = ma_score
        
        # ── Ensemble Score ──
        weights = {
            'threshold': 0.30,
            'zscore': 0.25,
            'isolation_forest': 0.30,
            'moving_avg': 0.15,
        }
        
        ensemble_score = (
            weights['threshold'] * threshold_score +
            weights['zscore'] * zscore_result['score'] +
            weights['isolation_forest'] * if_score +
            weights['moving_avg'] * ma_score
        )
        
        # Determine if anomaly based on sensitivity
        is_anomaly = ensemble_score > (1.0 - self.sensitivity)
        
        # Add to history
        self.anomaly_history.append({
            'is_anomaly': is_anomaly,
            'score': ensemble_score,
            'voltage': voltage,
            'current': current,
            'power': power,
        })
        
        return is_anomaly, float(ensemble_score), details

    def _threshold_check(self, v: float, i: float, p: float) -> float:
        """Simple threshold-based anomaly scoring."""
        score = 0.0
        
        if v > VOLTAGE_HIGH:
            score += min((v - VOLTAGE_HIGH) / (VOLTAGE_HIGH * 0.1), 1.0) * 0.4
        elif v < VOLTAGE_LOW:
            score += min((VOLTAGE_LOW - v) / (VOLTAGE_LOW * 0.1), 1.0) * 0.4
        
        if i > CURRENT_MAX:
            score += min((i - CURRENT_MAX) / (CURRENT_MAX * 0.2), 1.0) * 0.3
        
        if p > POWER_MAX:
            score += min((p - POWER_MAX) / (POWER_MAX * 0.2), 1.0) * 0.3
        
        return min(score, 1.0)

    def _zscore_check(self, v: float, i: float, p: float) -> Dict:
        """Z-score based anomaly detection."""
        alerts = []
        scores = []
        
        if self.train_stats:
            for name, value, key_prefix in [
                ('Voltage', v, 'voltage'),
                ('Current', i, 'current'),
                ('Power', p, 'power'),
            ]:
                mean = self.train_stats[f'{key_prefix}_mean']
                std = self.train_stats[f'{key_prefix}_std']
                
                if std > 0:
                    z = abs(value - mean) / std
                    scores.append(min(z / 4.0, 1.0))  # Normalize to 0-1
                    
                    if z > 3:
                        alerts.append(f"⚠️ {name} Z-score: {z:.2f} (extreme)")
                    elif z > 2:
                        alerts.append(f"⚡ {name} Z-score: {z:.2f} (high)")
        
        return {
            'score': float(np.mean(scores)) if scores else 0.0,
            'alerts': alerts,
        }

    def _moving_average_check(self) -> float:
        """Moving average deviation check."""
        if len(self.voltage_buffer) < 10:
            return 0.0
        
        v_arr = np.array(self.voltage_buffer)
        i_arr = np.array(self.current_buffer)
        p_arr = np.array(self.power_buffer)
        
        scores = []
        for arr, nominal in [(v_arr, VOLTAGE_NOMINAL), (i_arr, None), (p_arr, None)]:
            ma = np.mean(arr[:-1])
            latest = arr[-1]
            deviation = abs(latest - ma) / (ma if ma > 0 else 1)
            scores.append(min(deviation * 5, 1.0))  # Scale: 20% deviation = 1.0
        
        return float(np.mean(scores))

    def get_anomaly_rate(self, last_n: int = 100) -> float:
        """Get recent anomaly rate."""
        if not self.anomaly_history:
            return 0.0
        recent = list(self.anomaly_history)[-last_n:]
        return sum(1 for x in recent if x['is_anomaly']) / len(recent)

    def save(self, filename: str = "anomaly_detector"):
        """Save trained model to disk."""
        if not self.is_trained:
            print("  ⚠️ Model not trained yet!")
            return
        
        path = os.path.join(MODELS_DIR, f"{filename}.joblib")
        joblib.dump({
            'isolation_forest': self.isolation_forest,
            'scaler': self.scaler,
            'train_stats': self.train_stats,
            'contamination': self.contamination,
            'window_size': self.window_size,
            'sensitivity': self.sensitivity,
        }, path)
        print(f"  💾 Anomaly detector saved: {path}")

    def load(self, filename: str = "anomaly_detector"):
        """Load trained model from disk."""
        path = os.path.join(MODELS_DIR, f"{filename}.joblib")
        if not os.path.exists(path):
            print(f"  ⚠️ Model file not found: {path}")
            return False
        
        data = joblib.load(path)
        self.isolation_forest = data['isolation_forest']
        self.scaler = data['scaler']
        self.train_stats = data['train_stats']
        self.contamination = data['contamination']
        self.window_size = data['window_size']
        self.sensitivity = data['sensitivity']
        self.is_trained = True
        print(f"  📂 Anomaly detector loaded: {path}")
        return True
