"""
╔══════════════════════════════════════════════════════════════╗
║  Synthetic Data Generator — Generates realistic ESP32       ║
║  power monitoring data with labeled faults for training     ║
╚══════════════════════════════════════════════════════════════╝

Generates data for:
    - Normal operation patterns (daily load cycles)
    - Overvoltage / Undervoltage events
    - Overcurrent / Overload conditions
    - Voltage sags and swells
    - Power factor degradation
    - Harmonic distortion events
    - Phase imbalance scenarios
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict
from .config import (
    VOLTAGE_NOMINAL, VOLTAGE_HIGH, VOLTAGE_LOW,
    CURRENT_MAX, POWER_MAX, FAULT_CLASSES,
    SYNTHETIC_SAMPLES, SYNTHETIC_FAULT_RATIO, NOISE_LEVEL,
    DATA_DIR
)
import os


class SyntheticDataGenerator:
    """
    Generates realistic synthetic power monitoring data with labeled faults.
    
    The generator simulates:
        - Realistic daily load profiles (morning/evening peaks)
        - Gradual and sudden fault conditions
        - Sensor noise and measurement uncertainties
        - Seasonal and temporal patterns
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.fault_generators = {
            0: self._gen_normal,
            1: self._gen_overvoltage,
            2: self._gen_undervoltage,
            3: self._gen_overcurrent,
            4: self._gen_overload,
            5: self._gen_voltage_sag,
            6: self._gen_voltage_swell,
            7: self._gen_power_factor_issue,
            8: self._gen_harmonic_distortion,
            9: self._gen_phase_imbalance,
        }

    def _add_noise(self, value: float, noise_pct: float = NOISE_LEVEL) -> float:
        """Add Gaussian noise to a value."""
        return value * (1 + self.rng.normal(0, noise_pct))

    def _daily_load_profile(self, hour: float) -> float:
        """
        Simulate a realistic daily load profile.
        Returns a load multiplier (0.2 - 1.0) based on time of day.
        
        Pattern:
            - 00-05: Low load (0.2 - 0.3)
            - 06-09: Morning ramp-up (0.4 - 0.7)
            - 09-17: Daytime load (0.5 - 0.8)
            - 17-21: Evening peak (0.7 - 1.0)
            - 21-24: Wind-down (0.4 - 0.6)
        """
        # Use multiple sine waves to simulate realistic profile
        base = 0.5
        morning_peak = 0.15 * np.sin(np.pi * (hour - 7) / 4) if 5 <= hour <= 11 else 0
        evening_peak = 0.25 * np.sin(np.pi * (hour - 17) / 5) if 15 <= hour <= 22 else 0
        night_dip = -0.2 * np.cos(np.pi * hour / 12) if hour <= 6 or hour >= 22 else 0
        
        load = max(0.15, min(1.0, base + morning_peak + evening_peak + night_dip))
        return self._add_noise(load, 0.05)

    def _gen_normal(self, n: int, hour: float = 12.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate normal operating condition data."""
        load_factor = self._daily_load_profile(hour)
        
        voltage = np.array([self._add_noise(VOLTAGE_NOMINAL, 0.015) for _ in range(n)])
        current = np.array([self._add_noise(CURRENT_MAX * load_factor * 0.4, 0.03) for _ in range(n)])
        
        # Realistic power with slight power factor variation
        pf = self._add_noise(0.92, 0.02)
        power = voltage * current * pf
        
        return voltage, current, power

    def _gen_overvoltage(self, n: int, hour: float = 12.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate overvoltage event data (V > 250V)."""
        voltage_base = self.rng.uniform(252, 275)
        voltage = np.array([self._add_noise(voltage_base, 0.01) for _ in range(n)])
        
        load_factor = self._daily_load_profile(hour)
        current = np.array([self._add_noise(CURRENT_MAX * load_factor * 0.35, 0.03) for _ in range(n)])
        power = voltage * current * self._add_noise(0.90, 0.02)
        
        return voltage, current, power

    def _gen_undervoltage(self, n: int, hour: float = 12.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate undervoltage event data (V < 200V)."""
        voltage_base = self.rng.uniform(170, 198)
        voltage = np.array([self._add_noise(voltage_base, 0.01) for _ in range(n)])
        
        load_factor = self._daily_load_profile(hour)
        current = np.array([self._add_noise(CURRENT_MAX * load_factor * 0.5, 0.04) for _ in range(n)])
        power = voltage * current * self._add_noise(0.88, 0.03)
        
        return voltage, current, power

    def _gen_overcurrent(self, n: int, hour: float = 12.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate overcurrent event data (I > 15A)."""
        voltage = np.array([self._add_noise(VOLTAGE_NOMINAL, 0.02) for _ in range(n)])
        
        current_base = self.rng.uniform(15.5, 22)
        current = np.array([self._add_noise(current_base, 0.02) for _ in range(n)])
        power = voltage * current * self._add_noise(0.85, 0.03)
        
        return voltage, current, power

    def _gen_overload(self, n: int, hour: float = 12.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate overload event data (P > 3000W)."""
        voltage = np.array([self._add_noise(VOLTAGE_NOMINAL, 0.015) for _ in range(n)])
        
        # High current causing overload
        current_base = self.rng.uniform(13.5, 20)
        current = np.array([self._add_noise(current_base, 0.02) for _ in range(n)])
        power = np.array([self._add_noise(self.rng.uniform(3100, 4500), 0.02) for _ in range(n)])
        
        return voltage, current, power

    def _gen_voltage_sag(self, n: int, hour: float = 12.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate voltage sag (brief dip to 80-90% of nominal)."""
        sag_depth = self.rng.uniform(0.80, 0.92)
        
        voltage = np.array([self._add_noise(VOLTAGE_NOMINAL * sag_depth, 0.01) for _ in range(n)])
        
        load_factor = self._daily_load_profile(hour)
        current = np.array([self._add_noise(CURRENT_MAX * load_factor * 0.5, 0.03) for _ in range(n)])
        power = voltage * current * self._add_noise(0.88, 0.02)
        
        return voltage, current, power

    def _gen_voltage_swell(self, n: int, hour: float = 12.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate voltage swell (brief rise to 105-115% of nominal)."""
        swell_factor = self.rng.uniform(1.05, 1.15)
        
        voltage = np.array([self._add_noise(VOLTAGE_NOMINAL * swell_factor, 0.01) for _ in range(n)])
        
        load_factor = self._daily_load_profile(hour)
        current = np.array([self._add_noise(CURRENT_MAX * load_factor * 0.3, 0.03) for _ in range(n)])
        power = voltage * current * self._add_noise(0.91, 0.02)
        
        return voltage, current, power

    def _gen_power_factor_issue(self, n: int, hour: float = 12.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate data with poor power factor (PF < 0.7)."""
        voltage = np.array([self._add_noise(VOLTAGE_NOMINAL, 0.015) for _ in range(n)])
        
        load_factor = self._daily_load_profile(hour)
        current = np.array([self._add_noise(CURRENT_MAX * load_factor * 0.6, 0.03) for _ in range(n)])
        
        # Poor power factor — real power much less than apparent
        poor_pf = self.rng.uniform(0.45, 0.68)
        power = voltage * current * poor_pf
        
        return voltage, current, power

    def _gen_harmonic_distortion(self, n: int, hour: float = 12.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate data with harmonic distortion (high voltage fluctuation)."""
        # Base voltage with added harmonic ripple
        base_v = VOLTAGE_NOMINAL
        harmonic_amplitude = self.rng.uniform(5, 15)
        
        t = np.linspace(0, 2 * np.pi * n / 50, n)
        harmonic_3rd = harmonic_amplitude * np.sin(3 * t)
        harmonic_5th = harmonic_amplitude * 0.5 * np.sin(5 * t)
        
        voltage = base_v + harmonic_3rd + harmonic_5th
        voltage = np.array([self._add_noise(v, 0.01) for v in voltage])
        
        load_factor = self._daily_load_profile(hour)
        current = np.array([self._add_noise(CURRENT_MAX * load_factor * 0.45, 0.04) for _ in range(n)])
        power = voltage * current * self._add_noise(0.82, 0.03)
        
        return voltage, current, power

    def _gen_phase_imbalance(self, n: int, hour: float = 12.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate data simulating phase imbalance (asymmetric loading)."""
        # Voltage slightly off-nominal with periodic fluctuation
        v_offset = self.rng.uniform(-15, 15)
        fluctuation = self.rng.uniform(3, 10)
        
        t = np.linspace(0, 2 * np.pi * n / 30, n)
        voltage = VOLTAGE_NOMINAL + v_offset + fluctuation * np.sin(t)
        voltage = np.array([self._add_noise(v, 0.012) for v in voltage])
        
        # Unbalanced current
        load_factor = self._daily_load_profile(hour)
        current_base = CURRENT_MAX * load_factor * 0.5
        current_imbalance = self.rng.uniform(0.3, 0.7)
        current = np.array([self._add_noise(current_base * current_imbalance, 0.05) for _ in range(n)])
        
        power = voltage * current * self._add_noise(0.80, 0.04)
        
        return voltage, current, power

    def generate_dataset(self, total_samples: int = SYNTHETIC_SAMPLES,
                          fault_ratio: float = SYNTHETIC_FAULT_RATIO,
                          window_size: int = 30,
                          include_timestamps: bool = True) -> pd.DataFrame:
        """
        Generate a complete labeled dataset for training.
        
        Args:
            total_samples: Total number of data points
            fault_ratio: Proportion of fault samples (0-1)
            window_size: Points per event window
            include_timestamps: Whether to include timestamp column
            
        Returns:
            DataFrame with columns: [timestamp, voltage, current, power, fault_label, fault_name]
        """
        normal_count = int(total_samples * (1 - fault_ratio))
        fault_count = total_samples - normal_count
        
        all_data = []
        start_time = datetime(2025, 1, 1, 0, 0, 0)
        current_time = start_time
        sample_interval = timedelta(seconds=1)
        
        # Generate normal samples
        n_windows = normal_count // window_size + 1
        generated = 0
        for w in range(n_windows):
            if generated >= normal_count:
                break
            n = min(window_size, normal_count - generated)
            hour = (current_time.hour + current_time.minute / 60.0)
            v, i, p = self._gen_normal(n, hour)
            
            for j in range(n):
                ts = current_time + sample_interval * j
                all_data.append({
                    'timestamp': ts,
                    'voltage': round(v[j], 2),
                    'current': round(i[j], 4),
                    'power': round(p[j], 2),
                    'fault_label': 0,
                    'fault_name': 'Normal',
                })
            generated += n
            current_time += sample_interval * n
        
        # Generate fault samples (evenly distributed across fault types)
        fault_types = list(range(1, len(FAULT_CLASSES)))  # Exclude 0 (Normal)
        faults_per_type = fault_count // len(fault_types)
        
        for fault_id in fault_types:
            n = faults_per_type
            hour = self.rng.uniform(0, 24)
            v, i, p = self.fault_generators[fault_id](n, hour)
            
            for j in range(n):
                ts = current_time + sample_interval * j
                all_data.append({
                    'timestamp': ts,
                    'voltage': round(v[j], 2),
                    'current': round(i[j], 4),
                    'power': round(p[j], 2),
                    'fault_label': fault_id,
                    'fault_name': FAULT_CLASSES[fault_id],
                })
            current_time += sample_interval * n
        
        df = pd.DataFrame(all_data)
        
        # Shuffle dataset
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        if not include_timestamps:
            df = df.drop(columns=['timestamp'])
        
        return df

    def generate_time_series(self, hours: int = 24,
                              samples_per_hour: int = 3600,
                              inject_faults: bool = True,
                              fault_probability: float = 0.03) -> pd.DataFrame:
        """
        Generate continuous time-series data with optional random fault injection.
        Ideal for training the forecasting model.
        
        Args:
            hours: Duration in hours
            samples_per_hour: Samples per hour (3600 = 1 per second)
            inject_faults: Whether to inject random fault events
            fault_probability: Probability of fault at each window
            
        Returns:
            DataFrame with continuous time-series data
        """
        total = hours * samples_per_hour
        start_time = datetime(2025, 6, 15, 0, 0, 0)
        
        records = []
        window_size = 30
        idx = 0
        
        while idx < total:
            n = min(window_size, total - idx)
            hour = ((idx / samples_per_hour) % 24)
            
            # Decide if this window is a fault
            if inject_faults and self.rng.random() < fault_probability:
                fault_id = self.rng.integers(1, len(FAULT_CLASSES))
                v, i, p = self.fault_generators[fault_id](n, hour)
                fault_label = fault_id
                fault_name = FAULT_CLASSES[fault_id]
            else:
                v, i, p = self._gen_normal(n, hour)
                fault_label = 0
                fault_name = "Normal"
            
            for j in range(n):
                ts = start_time + timedelta(seconds=idx + j)
                records.append({
                    'timestamp': ts,
                    'voltage': round(v[j], 2),
                    'current': round(i[j], 4),
                    'power': round(p[j], 2),
                    'fault_label': fault_label,
                    'fault_name': fault_name,
                })
            
            idx += n
        
        return pd.DataFrame(records)

    def save_dataset(self, df: pd.DataFrame, filename: str = "training_data.csv"):
        """Save generated dataset to CSV."""
        filepath = os.path.join(DATA_DIR, filename)
        df.to_csv(filepath, index=False)
        print(f"  💾 Dataset saved: {filepath}")
        print(f"     Samples: {len(df):,}")
        print(f"     Fault distribution:")
        for label, name in FAULT_CLASSES.items():
            count = len(df[df['fault_label'] == label])
            pct = count / len(df) * 100
            print(f"       [{label}] {name}: {count:,} ({pct:.1f}%)")
        return filepath

    def load_dataset(self, filename: str = "training_data.csv") -> Optional[pd.DataFrame]:
        """Load a previously saved dataset."""
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, parse_dates=['timestamp'] if 'timestamp' in pd.read_csv(filepath, nrows=1).columns else None)
            print(f"  📂 Dataset loaded: {filepath} ({len(df):,} samples)")
            return df
        else:
            print(f"  ⚠️ File not found: {filepath}")
            return None
