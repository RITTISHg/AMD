"""
╔══════════════════════════════════════════════════════════════╗
║  Feature Engineering — Extract ML features from raw sensor  ║
║  data (Voltage, Current, Power) for all downstream models   ║
╚══════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from typing import Optional, Tuple, List, Dict
from .config import (
    VOLTAGE_NOMINAL, VOLTAGE_HIGH, VOLTAGE_LOW,
    CURRENT_MAX, POWER_MAX, ANOMALY_WINDOW_SIZE
)


class FeatureEngineer:
    """
    Extracts rich feature vectors from raw ESP32 sensor readings.
    
    Features include:
        - Statistical features (mean, std, min, max, skew, kurtosis)
        - Rolling window features (moving averages, trends)
        - Domain-specific features (power factor, voltage deviation, load %)
        - Temporal features (rate of change, gradients)
        - Cross-signal features (V×I correlation, apparent vs real power)
    """

    def __init__(self, window_size: int = ANOMALY_WINDOW_SIZE):
        self.window_size = window_size
        self.feature_names = []
        self._build_feature_names()

    def _build_feature_names(self):
        """Build the complete list of feature names."""
        base_signals = ['voltage', 'current', 'power']
        stat_suffixes = ['mean', 'std', 'min', 'max', 'range', 'skew', 'kurtosis',
                         'median', 'q25', 'q75', 'iqr', 'rms']
        
        self.feature_names = []
        
        # Per-signal statistical features
        for signal in base_signals:
            for suffix in stat_suffixes:
                self.feature_names.append(f"{signal}_{suffix}")
        
        # Trend features
        for signal in base_signals:
            self.feature_names.extend([
                f"{signal}_trend_slope",
                f"{signal}_trend_intercept",
                f"{signal}_rate_of_change",
                f"{signal}_acceleration",
            ])
        
        # Domain-specific features
        self.feature_names.extend([
            'voltage_deviation_pct',
            'voltage_thd_estimate',
            'current_load_pct',
            'power_load_pct',
            'power_factor',
            'apparent_power',
            'reactive_power',
            'voltage_crest_factor',
            'current_crest_factor',
            'energy_rate_wh_per_min',
        ])
        
        # Cross-signal features
        self.feature_names.extend([
            'vi_correlation',
            'vp_correlation',
            'ip_correlation',
            'power_voltage_ratio',
            'power_current_ratio',
        ])
        
        # Stability indicators
        self.feature_names.extend([
            'voltage_stability_idx',
            'current_stability_idx',
            'power_stability_idx',
            'is_voltage_in_range',
            'is_current_safe',
            'is_power_safe',
        ])

    def get_feature_names(self) -> List[str]:
        """Return list of all feature names."""
        return self.feature_names.copy()

    def get_num_features(self) -> int:
        """Return total number of features."""
        return len(self.feature_names)

    @staticmethod
    def _safe_stat(arr: np.ndarray, func, default=0.0):
        """Safely compute a statistic, returning default if array is too small."""
        try:
            if len(arr) < 2:
                return default
            result = func(arr)
            return result if np.isfinite(result) else default
        except Exception:
            return default

    def extract_statistical_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract comprehensive statistical features from a signal window."""
        features = {}
        
        if len(signal) == 0:
            return {k: 0.0 for k in ['mean', 'std', 'min', 'max', 'range',
                                       'skew', 'kurtosis', 'median', 'q25', 'q75',
                                       'iqr', 'rms']}
        
        features['mean'] = float(np.mean(signal))
        features['std'] = float(np.std(signal))
        features['min'] = float(np.min(signal))
        features['max'] = float(np.max(signal))
        features['range'] = features['max'] - features['min']
        features['skew'] = self._safe_stat(signal, scipy_stats.skew)
        features['kurtosis'] = self._safe_stat(signal, scipy_stats.kurtosis)
        features['median'] = float(np.median(signal))
        features['q25'] = float(np.percentile(signal, 25))
        features['q75'] = float(np.percentile(signal, 75))
        features['iqr'] = features['q75'] - features['q25']
        features['rms'] = float(np.sqrt(np.mean(signal ** 2)))
        
        return features

    def extract_trend_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract trend and rate-of-change features."""
        features = {}
        
        if len(signal) < 3:
            return {'trend_slope': 0.0, 'trend_intercept': 0.0,
                    'rate_of_change': 0.0, 'acceleration': 0.0}
        
        # Linear trend
        x = np.arange(len(signal))
        slope, intercept, _, _, _ = scipy_stats.linregress(x, signal)
        features['trend_slope'] = float(slope) if np.isfinite(slope) else 0.0
        features['trend_intercept'] = float(intercept) if np.isfinite(intercept) else 0.0
        
        # Rate of change (last value vs mean)
        features['rate_of_change'] = float(signal[-1] - np.mean(signal))
        
        # Acceleration (change in rate of change)
        if len(signal) >= 4:
            mid = len(signal) // 2
            roc_first = np.mean(np.diff(signal[:mid]))
            roc_second = np.mean(np.diff(signal[mid:]))
            features['acceleration'] = float(roc_second - roc_first)
        else:
            features['acceleration'] = 0.0
        
        return features

    def extract_domain_features(self, voltage: np.ndarray, current: np.ndarray,
                                 power: np.ndarray) -> Dict[str, float]:
        """Extract power-system domain-specific features."""
        features = {}
        
        v_mean = np.mean(voltage) if len(voltage) > 0 else VOLTAGE_NOMINAL
        i_mean = np.mean(current) if len(current) > 0 else 0.0
        p_mean = np.mean(power) if len(power) > 0 else 0.0
        
        # Voltage deviation from nominal
        features['voltage_deviation_pct'] = abs(v_mean - VOLTAGE_NOMINAL) / VOLTAGE_NOMINAL * 100
        
        # THD estimate (approximation from std/mean)
        v_std = np.std(voltage) if len(voltage) > 1 else 0.0
        features['voltage_thd_estimate'] = (v_std / v_mean * 100) if v_mean > 0 else 0.0
        
        # Load percentages
        features['current_load_pct'] = (i_mean / CURRENT_MAX * 100) if CURRENT_MAX > 0 else 0.0
        features['power_load_pct'] = (p_mean / POWER_MAX * 100) if POWER_MAX > 0 else 0.0
        
        # Power factor
        apparent = v_mean * i_mean
        features['apparent_power'] = apparent
        features['power_factor'] = (p_mean / apparent) if apparent > 0 else 0.0
        features['reactive_power'] = np.sqrt(max(apparent**2 - p_mean**2, 0))
        
        # Crest factors
        v_rms = np.sqrt(np.mean(voltage ** 2)) if len(voltage) > 0 else 1.0
        i_rms = np.sqrt(np.mean(current ** 2)) if len(current) > 0 else 1.0
        features['voltage_crest_factor'] = (np.max(voltage) / v_rms) if v_rms > 0 and len(voltage) > 0 else 1.0
        features['current_crest_factor'] = (np.max(current) / i_rms) if i_rms > 0 and len(current) > 0 else 1.0
        
        # Energy rate (Wh per minute estimate)
        features['energy_rate_wh_per_min'] = p_mean / 60.0
        
        return features

    def extract_cross_signal_features(self, voltage: np.ndarray, current: np.ndarray,
                                       power: np.ndarray) -> Dict[str, float]:
        """Extract cross-signal correlation features."""
        features = {}
        
        min_len = min(len(voltage), len(current), len(power))
        
        if min_len < 3:
            return {
                'vi_correlation': 0.0, 'vp_correlation': 0.0, 'ip_correlation': 0.0,
                'power_voltage_ratio': 0.0, 'power_current_ratio': 0.0,
            }
        
        v, i, p = voltage[:min_len], current[:min_len], power[:min_len]
        
        # Correlations
        features['vi_correlation'] = self._safe_stat(
            np.column_stack([v, i]), lambda x: np.corrcoef(x[:, 0], x[:, 1])[0, 1])
        features['vp_correlation'] = self._safe_stat(
            np.column_stack([v, p]), lambda x: np.corrcoef(x[:, 0], x[:, 1])[0, 1])
        features['ip_correlation'] = self._safe_stat(
            np.column_stack([i, p]), lambda x: np.corrcoef(x[:, 0], x[:, 1])[0, 1])
        
        # Ratios
        v_mean = np.mean(v)
        i_mean = np.mean(i)
        features['power_voltage_ratio'] = (np.mean(p) / v_mean) if v_mean > 0 else 0.0
        features['power_current_ratio'] = (np.mean(p) / i_mean) if i_mean > 0 else 0.0
        
        return features

    def extract_stability_features(self, voltage: np.ndarray, current: np.ndarray,
                                    power: np.ndarray) -> Dict[str, float]:
        """Extract stability indicator features."""
        features = {}
        
        # Coefficient of Variation as stability index (lower = more stable)
        v_mean = np.mean(voltage) if len(voltage) > 0 else 1.0
        i_mean = np.mean(current) if len(current) > 0 else 1.0
        p_mean = np.mean(power) if len(power) > 0 else 1.0
        
        features['voltage_stability_idx'] = 1.0 - min(np.std(voltage) / v_mean if v_mean > 0 else 0, 1.0) if len(voltage) > 1 else 1.0
        features['current_stability_idx'] = 1.0 - min(np.std(current) / i_mean if i_mean > 0 else 0, 1.0) if len(current) > 1 else 1.0
        features['power_stability_idx'] = 1.0 - min(np.std(power) / p_mean if p_mean > 0 else 0, 1.0) if len(power) > 1 else 1.0
        
        # Safety flags
        features['is_voltage_in_range'] = 1.0 if VOLTAGE_LOW <= v_mean <= VOLTAGE_HIGH else 0.0
        features['is_current_safe'] = 1.0 if i_mean <= CURRENT_MAX else 0.0
        features['is_power_safe'] = 1.0 if p_mean <= POWER_MAX else 0.0
        
        return features

    def extract_all_features(self, voltage: np.ndarray, current: np.ndarray,
                              power: np.ndarray) -> np.ndarray:
        """
        Extract ALL features from a window of sensor data.
        
        Args:
            voltage: Array of voltage readings (window)
            current: Array of current readings (window)
            power: Array of power readings (window)
            
        Returns:
            1D numpy array of all features
        """
        all_features = []
        
        # Statistical features for each signal
        for signal in [voltage, current, power]:
            stat_feats = self.extract_statistical_features(signal)
            all_features.extend(stat_feats.values())
        
        # Trend features for each signal
        for signal in [voltage, current, power]:
            trend_feats = self.extract_trend_features(signal)
            all_features.extend(trend_feats.values())
        
        # Domain features
        domain_feats = self.extract_domain_features(voltage, current, power)
        all_features.extend(domain_feats.values())
        
        # Cross-signal features
        cross_feats = self.extract_cross_signal_features(voltage, current, power)
        all_features.extend(cross_feats.values())
        
        # Stability features
        stability_feats = self.extract_stability_features(voltage, current, power)
        all_features.extend(stability_feats.values())
        
        feature_arr = np.array(all_features, dtype=np.float64)
        # Replace NaN/Inf with 0
        feature_arr = np.nan_to_num(feature_arr, nan=0.0, posinf=0.0, neginf=0.0)
        
        return feature_arr

    def extract_features_from_dataframe(self, df: pd.DataFrame,
                                         voltage_col: str = 'voltage',
                                         current_col: str = 'current',
                                         power_col: str = 'power') -> pd.DataFrame:
        """
        Extract features from a DataFrame using sliding windows.
        
        Args:
            df: Input DataFrame with sensor columns
            voltage_col, current_col, power_col: Column names
            
        Returns:
            DataFrame with extracted features (one row per valid window)
        """
        all_rows = []
        n = len(df)
        
        for i in range(self.window_size, n + 1):
            window = df.iloc[i - self.window_size:i]
            v = window[voltage_col].values
            c = window[current_col].values
            p = window[power_col].values
            
            features = self.extract_all_features(v, c, p)
            all_rows.append(features)
        
        if not all_rows:
            return pd.DataFrame(columns=self.feature_names)
        
        return pd.DataFrame(all_rows, columns=self.feature_names)

    def extract_single_point_features(self, voltage: float, current: float,
                                       power: float) -> Dict[str, float]:
        """
        Extract simplified features from a single data point.
        Useful for real-time inference when window data isn't available.
        """
        features = {
            'voltage': voltage,
            'current': current,
            'power': power,
            'voltage_deviation_pct': abs(voltage - VOLTAGE_NOMINAL) / VOLTAGE_NOMINAL * 100,
            'current_load_pct': (current / CURRENT_MAX) * 100,
            'power_load_pct': (power / POWER_MAX) * 100,
            'apparent_power': voltage * current,
            'power_factor': power / (voltage * current) if (voltage * current) > 0 else 0,
            'reactive_power': np.sqrt(max((voltage * current) ** 2 - power ** 2, 0)),
            'is_voltage_in_range': 1.0 if VOLTAGE_LOW <= voltage <= VOLTAGE_HIGH else 0.0,
            'is_current_safe': 1.0 if current <= CURRENT_MAX else 0.0,
            'is_power_safe': 1.0 if power <= POWER_MAX else 0.0,
        }
        return features
