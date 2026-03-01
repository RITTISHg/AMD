"""
╔══════════════════════════════════════════════════════════════╗
║  Model Manager — High-level interface to train,            ║
║  evaluate, and load all ML models in the AMD Power Suite    ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import pandas as pd
from typing import Dict, Optional

from .data_generator import SyntheticDataGenerator
from .anomaly_detector import AnomalyDetector
from .fault_classifier import FaultClassifier
from .power_forecaster import PowerForecaster


class ModelManager:
    """
    Centralized manager to orchestrate training, saving, and loading
    all ML models in the intelligence suite.
    """

    def __init__(self):
        self.anomaly_detector = AnomalyDetector()
        self.fault_classifier = FaultClassifier()
        self.power_forecaster = PowerForecaster()
        self.data_generator = SyntheticDataGenerator()

    def train_all_models(self, samples: int = 10000,
                          fault_ratio: float = 0.25,
                          force_retrain: bool = False):
        """
        Train all models organically using synthetic data.
        
        Args:
            samples: Number of synthetic samples to generate
            fault_ratio: Ratio of faults in classification data
            force_retrain: If False, will try to load existing models first
        """
        print("\n" + "═" * 60)
        print("  🚀 Initializing AMD Power ML Training Pipeline")
        print("═" * 60)

        # ── 1. Load or Retrain Anomaly Detector ──
        if not force_retrain and self.anomaly_detector.load():
            print("  ✅ Anomaly Detector loaded from disk.")
        else:
            print("\n  [1/4] Generating Normal Data for Anomaly Detector...")
            df_normal = self.data_generator.generate_dataset(
                total_samples=samples, fault_ratio=0.0
            )
            print("  [2/4] Training Anomaly Detector...")
            self.anomaly_detector.train_from_dataframe(df_normal)
            self.anomaly_detector.save()

        # ── 2. Load or Retrain Fault Classifier ──
        if not force_retrain and self.fault_classifier.load():
            print("  ✅ Fault Classifier loaded from disk.")
        else:
            print("\n  [3/4] Generating Labeled Fault Data...")
            df_faults = self.data_generator.generate_dataset(
                total_samples=samples, fault_ratio=fault_ratio
            )
            print("  [4/4] Training Fault Classifier...")
            self.fault_classifier.train_from_dataframe(df_faults)
            self.fault_classifier.save()

        # ── 3. Load or Retrain Power Forecaster ──
        if not force_retrain and self.power_forecaster.load():
            print("  ✅ Power Forecaster loaded from disk.")
        else:
            print("\n  [5/5] Generating Time-Series Data for Forecaster...")
            df_ts = self.data_generator.generate_time_series(hours=48, samples_per_hour=3600)
            print("  [6/6] Training Power Forecaster...")
            self.power_forecaster.train(
                power=df_ts['power'].values,
                voltage=df_ts['voltage'].values,
                current=df_ts['current'].values,
                epochs=50,  # lower for synthetic testing
            )
            self.power_forecaster.save()

        print("\n" + "═" * 60)
        print("  🎉 All Models Ready!")
        print("═" * 60 + "\n")

    def load_all_models(self) -> bool:
        """Attempt to load all models from disk. Returns True if all successful."""
        print("\n  📂 Loading ML Models...")
        a = self.anomaly_detector.load()
        f = self.fault_classifier.load()
        p = self.power_forecaster.load()
        return a and f and p


if __name__ == "__main__":
    manager = ModelManager()
    manager.train_all_models(samples=5000, force_retrain=True)
