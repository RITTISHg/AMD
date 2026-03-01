"""
╔══════════════════════════════════════════════════════════════╗
║  ML Configuration — Centralized settings for all models     ║
╚══════════════════════════════════════════════════════════════╝
"""

import os

# ══════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Create directories if they don't exist
for d in [MODELS_DIR, DATA_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# SENSOR THRESHOLDS (must match dashboard config)
# ══════════════════════════════════════════════════════════════
VOLTAGE_NOMINAL = 230.0     # V — expected nominal voltage
VOLTAGE_HIGH = 250.0        # V — overvoltage threshold
VOLTAGE_LOW = 200.0         # V — undervoltage threshold
CURRENT_MAX = 15.0          # A — overcurrent threshold
POWER_MAX = 3000.0          # W — overload threshold
FREQUENCY_NOMINAL = 50.0    # Hz — nominal grid frequency

# ══════════════════════════════════════════════════════════════
# FAULT CLASSES
# ══════════════════════════════════════════════════════════════
FAULT_CLASSES = {
    0: "Normal",
    1: "Overvoltage",
    2: "Undervoltage",
    3: "Overcurrent",
    4: "Overload",
    5: "Voltage Sag",
    6: "Voltage Swell",
    7: "Power Factor Issue",
    8: "Harmonic Distortion",
    9: "Phase Imbalance",
}

FAULT_SEVERITY = {
    "Normal": 0,
    "Voltage Sag": 1,
    "Power Factor Issue": 1,
    "Voltage Swell": 2,
    "Harmonic Distortion": 2,
    "Phase Imbalance": 3,
    "Overvoltage": 3,
    "Undervoltage": 3,
    "Overcurrent": 4,
    "Overload": 5,
}

# ══════════════════════════════════════════════════════════════
# ANOMALY DETECTION SETTINGS
# ══════════════════════════════════════════════════════════════
ANOMALY_CONTAMINATION = 0.05        # Expected % of anomalies (5%)
ANOMALY_WINDOW_SIZE = 30            # Sliding window for feature extraction
ANOMALY_SENSITIVITY = 0.7           # Sensitivity threshold (0-1)

# ══════════════════════════════════════════════════════════════
# FORECASTING SETTINGS
# ══════════════════════════════════════════════════════════════
FORECAST_LOOKBACK = 60              # Number of past samples to use
FORECAST_HORIZON = 15               # Number of future steps to predict
FORECAST_EPOCHS = 100               # LSTM training epochs
FORECAST_BATCH_SIZE = 32            # Training batch size
FORECAST_LEARNING_RATE = 0.001      # Adam optimizer learning rate

# ══════════════════════════════════════════════════════════════
# FAULT CLASSIFIER SETTINGS
# ══════════════════════════════════════════════════════════════
CLASSIFIER_N_ESTIMATORS = 200       # Random Forest trees
CLASSIFIER_MAX_DEPTH = 15           # Max tree depth
CLASSIFIER_TEST_SPLIT = 0.2        # Train/test split ratio
CLASSIFIER_XGBOOST_ROUNDS = 150    # XGBoost boosting rounds

# ══════════════════════════════════════════════════════════════
# INSIGHTS ENGINE SETTINGS
# ══════════════════════════════════════════════════════════════
HEALTH_SCORE_WEIGHTS = {
    "voltage_stability": 0.25,
    "current_safety": 0.20,
    "power_efficiency": 0.20,
    "anomaly_rate": 0.15,
    "power_factor": 0.10,
    "energy_trend": 0.10,
}

# Cost settings
COST_PER_KWH = 6.50                # ₹ per kWh
CURRENCY = "₹"

# ══════════════════════════════════════════════════════════════
# DATA GENERATOR SETTINGS
# ══════════════════════════════════════════════════════════════
SYNTHETIC_SAMPLES = 50000           # Total synthetic samples to generate
SYNTHETIC_FAULT_RATIO = 0.25        # 25% fault samples
NOISE_LEVEL = 0.02                  # 2% Gaussian noise
