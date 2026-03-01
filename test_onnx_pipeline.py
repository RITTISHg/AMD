"""Quick test for the ONNX inference pipeline."""
import sys
import numpy as np

sys.path.insert(0, '.')
from ml_models.model_manager import ModelManager
from ml_models.onnx_converter import ONNXModelConverter
from ml_models.feature_engineer import FeatureEngineer
from ml_models.config import ANOMALY_WINDOW_SIZE
import os

print('=== Testing ML + ONNX Pipeline ===')

mm = ModelManager()
loaded = mm.load_all_models()
print(f'Models loaded: {loaded}')

onnx_conv = ONNXModelConverter()
fe = FeatureEngineer(window_size=ANOMALY_WINDOW_SIZE)
n_features = fe.get_num_features()
print(f'Feature count: {n_features}')

# Convert models to ONNX
if mm.anomaly_detector.is_trained:
    ok = onnx_conv.convert_isolation_forest(
        mm.anomaly_detector.isolation_forest,
        mm.anomaly_detector.scaler,
        n_features, 'anomaly_detector'
    )
    print(f'Anomaly converter: {ok}')

if mm.fault_classifier.is_trained:
    ok = onnx_conv.convert_classifier(
        mm.fault_classifier.rf_model,
        mm.fault_classifier.gb_model,
        mm.fault_classifier.scaler,
        n_features, 'fault'
    )
    print(f'Fault converter: {ok}')

# Load sessions
loaded_count = onnx_conv.load_all_sessions()
print(f'ONNX sessions loaded: {loaded_count}')

# Test inference
v = np.full(30, 230.0) + np.random.normal(0, 1, 30)
i = np.full(30, 2.0) + np.random.normal(0, 0.1, 30)
p = v * i * 0.92
features = fe.extract_all_features(v, i, p)
print(f'Features shape: {features.shape}')

is_anom, score = onnx_conv.infer_anomaly(features)
print(f'Anomaly: {is_anom}, Score: {score:.4f}')

fault_id, conf, top3 = onnx_conv.infer_fault(features)
print(f'Fault: {fault_id}, Confidence: {conf:.4f}')
print(f'Top3: {top3}')

perf = onnx_conv.monitor.get_stats()
print(f'Latency avg: {perf["avg_latency_ms"]:.3f}ms')
print(f'Total inferences: {perf["total_inferences"]}')
print('=== All tests passed! ===')
