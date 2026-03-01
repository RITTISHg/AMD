"""
╔══════════════════════════════════════════════════════════════╗
║  Fault Classifier — Multi-class fault classification        ║
║  using Random Forest + XGBoost ensemble for power faults    ║
╚══════════════════════════════════════════════════════════════╝

Classifies 10 fault types:
    0: Normal
    1: Overvoltage      5: Voltage Sag
    2: Undervoltage     6: Voltage Swell
    3: Overcurrent      7: Power Factor Issue
    4: Overload         8: Harmonic Distortion
                        9: Phase Imbalance
"""

import numpy as np
import pandas as pd
import joblib
import os
from typing import Tuple, Dict, List, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)
from collections import deque

from .config import (
    FAULT_CLASSES, FAULT_SEVERITY,
    CLASSIFIER_N_ESTIMATORS, CLASSIFIER_MAX_DEPTH,
    CLASSIFIER_TEST_SPLIT, MODELS_DIR, ANOMALY_WINDOW_SIZE
)
from .feature_engineer import FeatureEngineer


class FaultClassifier:
    """
    Multi-class fault classifier for power system events.
    
    Uses an ensemble of:
        1. Random Forest Classifier
        2. Gradient Boosting (XGBoost-style) Classifier
    
    Usage:
        classifier = FaultClassifier()
        metrics = classifier.train(X_features, y_labels)
        
        # Real-time classification
        fault_id, confidence, top3 = classifier.predict(features)
        fault_name = classifier.get_fault_name(fault_id)
    """

    def __init__(self, n_estimators: int = CLASSIFIER_N_ESTIMATORS,
                 max_depth: int = CLASSIFIER_MAX_DEPTH,
                 window_size: int = ANOMALY_WINDOW_SIZE):
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.window_size = window_size
        
        # Models
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )
        
        self.gb_model = GradientBoostingClassifier(
            n_estimators=min(n_estimators, 150),
            max_depth=min(max_depth, 8),
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        
        self.scaler = StandardScaler()
        self.feature_engineer = FeatureEngineer(window_size=window_size)
        
        # State
        self.is_trained = False
        self.training_metrics = {}
        self.feature_importances = {}
        
        # Real-time buffers
        self.voltage_buffer = deque(maxlen=window_size)
        self.current_buffer = deque(maxlen=window_size)
        self.power_buffer = deque(maxlen=window_size)
        
        # Classification history
        self.classification_history = deque(maxlen=500)

    def train(self, X: np.ndarray, y: np.ndarray,
              test_split: float = CLASSIFIER_TEST_SPLIT,
              verbose: bool = True) -> Dict:
        """
        Train the fault classifier.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Fault labels (n_samples,)
            test_split: Proportion for test set
            verbose: Print training progress
            
        Returns:
            Training metrics dictionary
        """
        if verbose:
            print("\n  🧠 Training Fault Classifier...")
            print(f"     Samples: {len(X):,}")
            print(f"     Features: {X.shape[1]}")
            print(f"     Classes: {len(np.unique(y))}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_split, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        if verbose:
            print("\n     📌 Training Random Forest...")
        self.rf_model.fit(X_train_scaled, y_train)
        rf_pred = self.rf_model.predict(X_test_scaled)
        rf_acc = accuracy_score(y_test, rf_pred)
        
        # Train Gradient Boosting
        if verbose:
            print("     📌 Training Gradient Boosting...")
        self.gb_model.fit(X_train_scaled, y_train)
        gb_pred = self.gb_model.predict(X_test_scaled)
        gb_acc = accuracy_score(y_test, gb_pred)
        
        # Ensemble predictions (soft voting)
        rf_proba = self.rf_model.predict_proba(X_test_scaled)
        gb_proba = self.gb_model.predict_proba(X_test_scaled)
        ensemble_proba = (rf_proba * 0.55 + gb_proba * 0.45)
        ensemble_pred = np.argmax(ensemble_proba, axis=1)
        ensemble_acc = accuracy_score(y_test, ensemble_pred)
        
        # Feature importances (from Random Forest)
        feature_names = self.feature_engineer.get_feature_names()
        if len(feature_names) == X.shape[1]:
            importances = self.rf_model.feature_importances_
            self.feature_importances = dict(
                sorted(zip(feature_names, importances),
                       key=lambda x: x[1], reverse=True)
            )
        
        # Cross-validation score
        cv_scores = cross_val_score(self.rf_model, X_train_scaled, y_train, cv=5, scoring='f1_weighted')
        
        # Compile metrics
        self.training_metrics = {
            'rf_accuracy': float(rf_acc),
            'gb_accuracy': float(gb_acc),
            'ensemble_accuracy': float(ensemble_acc),
            'f1_weighted': float(f1_score(y_test, ensemble_pred, average='weighted')),
            'precision_weighted': float(precision_score(y_test, ensemble_pred, average='weighted', zero_division=0)),
            'recall_weighted': float(recall_score(y_test, ensemble_pred, average='weighted', zero_division=0)),
            'cv_mean': float(np.mean(cv_scores)),
            'cv_std': float(np.std(cv_scores)),
            'confusion_matrix': confusion_matrix(y_test, ensemble_pred).tolist(),
            'classification_report': classification_report(
                y_test, ensemble_pred,
                target_names=[FAULT_CLASSES.get(i, f"Class_{i}") for i in sorted(np.unique(y))],
                output_dict=True,
                zero_division=0,
            ),
        }
        
        self.is_trained = True
        
        if verbose:
            print(f"\n     ✅ Training complete!")
            print(f"     Random Forest Accuracy:    {rf_acc:.4f}")
            print(f"     Gradient Boost Accuracy:    {gb_acc:.4f}")
            print(f"     Ensemble Accuracy:          {ensemble_acc:.4f}")
            print(f"     F1 Score (weighted):        {self.training_metrics['f1_weighted']:.4f}")
            print(f"     Cross-Val (5-fold):         {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
            print(f"\n     Top 10 Most Important Features:")
            for feat, imp in list(self.feature_importances.items())[:10]:
                bar = "█" * int(imp * 200)
                print(f"       {feat:30s} {imp:.4f} {bar}")
        
        return self.training_metrics

    def train_from_dataframe(self, df: pd.DataFrame,
                              voltage_col: str = 'voltage',
                              current_col: str = 'current',
                              power_col: str = 'power',
                              label_col: str = 'fault_label',
                              verbose: bool = True) -> Dict:
        """
        Train from a labeled DataFrame.
        Automatically extracts features using sliding windows.
        """
        if verbose:
            print("\n  📊 Extracting features from DataFrame...")
        
        features_df = self.feature_engineer.extract_features_from_dataframe(
            df, voltage_col, current_col, power_col
        )
        
        # Align labels with feature windows
        labels = df[label_col].values[self.window_size - 1:]
        labels = labels[:len(features_df)]
        
        X = features_df.values
        y = labels
        
        return self.train(X, y, verbose=verbose)

    def predict(self, features: np.ndarray) -> Tuple[int, float, List[Tuple[int, str, float]]]:
        """
        Predict fault class for a feature vector.
        
        Args:
            features: 1D feature array
            
        Returns:
            (fault_id, confidence, top3_predictions)
            top3: List of (fault_id, fault_name, probability)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained! Call train() first.")
        
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Ensemble prediction
        rf_proba = self.rf_model.predict_proba(features_scaled)[0]
        gb_proba = self.gb_model.predict_proba(features_scaled)[0]
        ensemble_proba = rf_proba * 0.55 + gb_proba * 0.45
        
        fault_id = int(np.argmax(ensemble_proba))
        confidence = float(ensemble_proba[fault_id])
        
        # Top 3 predictions
        top3_idx = np.argsort(ensemble_proba)[::-1][:3]
        top3 = [
            (int(idx), FAULT_CLASSES.get(int(idx), f"Unknown_{idx}"),
             float(ensemble_proba[idx]))
            for idx in top3_idx
        ]
        
        # Record in history
        self.classification_history.append({
            'fault_id': fault_id,
            'fault_name': FAULT_CLASSES.get(fault_id, "Unknown"),
            'confidence': confidence,
            'severity': FAULT_SEVERITY.get(FAULT_CLASSES.get(fault_id, ""), 0),
        })
        
        return fault_id, confidence, top3

    def predict_realtime(self, voltage: float, current: float, power: float
                         ) -> Optional[Tuple[int, float, List]]:
        """
        Real-time prediction from raw sensor values.
        Buffers data internally and predicts when window is full.
        
        Returns None if buffer isn't full yet, otherwise returns prediction.
        """
        self.voltage_buffer.append(voltage)
        self.current_buffer.append(current)
        self.power_buffer.append(power)
        
        if len(self.voltage_buffer) < self.window_size:
            return None
        
        v = np.array(self.voltage_buffer)
        i = np.array(self.current_buffer)
        p = np.array(self.power_buffer)
        
        features = self.feature_engineer.extract_all_features(v, i, p)
        return self.predict(features)

    @staticmethod
    def get_fault_name(fault_id: int) -> str:
        """Get human-readable fault name."""
        return FAULT_CLASSES.get(fault_id, f"Unknown Fault ({fault_id})")

    @staticmethod
    def get_fault_severity(fault_id: int) -> int:
        """Get fault severity level (0-5)."""
        name = FAULT_CLASSES.get(fault_id, "")
        return FAULT_SEVERITY.get(name, 0)

    def get_recent_fault_distribution(self, last_n: int = 100) -> Dict[str, int]:
        """Get distribution of recent fault predictions."""
        recent = list(self.classification_history)[-last_n:]
        dist = {}
        for record in recent:
            name = record['fault_name']
            dist[name] = dist.get(name, 0) + 1
        return dict(sorted(dist.items(), key=lambda x: x[1], reverse=True))

    def save(self, filename: str = "fault_classifier"):
        """Save trained model."""
        if not self.is_trained:
            print("  ⚠️ Model not trained!")
            return
        
        path = os.path.join(MODELS_DIR, f"{filename}.joblib")
        joblib.dump({
            'rf_model': self.rf_model,
            'gb_model': self.gb_model,
            'scaler': self.scaler,
            'training_metrics': self.training_metrics,
            'feature_importances': self.feature_importances,
        }, path)
        print(f"  💾 Fault classifier saved: {path}")

    def load(self, filename: str = "fault_classifier"):
        """Load trained model."""
        path = os.path.join(MODELS_DIR, f"{filename}.joblib")
        if not os.path.exists(path):
            print(f"  ⚠️ Model not found: {path}")
            return False
        
        data = joblib.load(path)
        self.rf_model = data['rf_model']
        self.gb_model = data['gb_model']
        self.scaler = data['scaler']
        self.training_metrics = data.get('training_metrics', {})
        self.feature_importances = data.get('feature_importances', {})
        self.is_trained = True
        print(f"  📂 Fault classifier loaded: {path}")
        return True
