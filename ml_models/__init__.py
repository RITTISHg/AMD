"""
╔══════════════════════════════════════════════════════════════╗
║     🧠 AMD Power Monitor — ML Intelligence Suite            ║
║     AI-Powered Insights, Fault Detection & Forecasting      ║
║     Optimized for AMD Ryzen™ High-Performance Processors    ║
╚══════════════════════════════════════════════════════════════╝

Modules:
    - anomaly_detector   : Real-time anomaly detection (Isolation Forest + Autoencoders)
    - fault_classifier   : Multi-class fault classification (Random Forest + XGBoost)
    - power_forecaster   : Time-series power consumption forecasting (LSTM + Prophet)
    - insights_engine    : AI-driven insights, health scoring & recommendations
    - data_generator     : Synthetic training data generator for all models
    - feature_engineer   : Feature extraction & preprocessing pipeline
    - model_manager      : Model training, saving, loading & evaluation utilities
"""

__version__ = "1.0.0"
__author__ = "AMD Power Intelligence"

from .feature_engineer import FeatureEngineer
from .anomaly_detector import AnomalyDetector
from .fault_classifier import FaultClassifier
from .power_forecaster import PowerForecaster
from .insights_engine import InsightsEngine
from .data_generator import SyntheticDataGenerator
from .model_manager import ModelManager
from .onnx_converter import ONNXModelConverter, ONNXPerformanceMonitor
