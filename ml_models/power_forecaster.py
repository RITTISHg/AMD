"""
╔══════════════════════════════════════════════════════════════╗
║  Power Forecaster — Time-series power consumption           ║
║  prediction using LSTM neural network + trend analysis      ║
╚══════════════════════════════════════════════════════════════╝

Models:
    1. LSTM (Long Short-Term Memory) — Deep learning forecaster
    2. Linear Trend — Statistical baseline
    3. Moving Average — Smoothed prediction baseline
"""

import numpy as np
import pandas as pd
import os
import joblib
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import MinMaxScaler
from collections import deque

from .config import (
    FORECAST_LOOKBACK, FORECAST_HORIZON,
    FORECAST_EPOCHS, FORECAST_BATCH_SIZE,
    FORECAST_LEARNING_RATE, MODELS_DIR, POWER_MAX
)

# Conditional TensorFlow/Keras import
_HAS_TENSORFLOW = False
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    _HAS_TENSORFLOW = True
except ImportError:
    pass


class PowerForecaster:
    """
    Time-series power consumption forecaster.
    
    Supports:
        - LSTM-based deep learning forecasting (requires TensorFlow)
        - Statistical fallback (Linear Trend + Moving Average)
        - Multi-step ahead prediction
        - Confidence intervals
    
    Usage:
        forecaster = PowerForecaster()
        forecaster.train(power_series)
        predictions, confidence = forecaster.forecast(steps=15)
    """

    def __init__(self, lookback: int = FORECAST_LOOKBACK,
                 horizon: int = FORECAST_HORIZON,
                 use_lstm: bool = True):
        
        self.lookback = lookback
        self.horizon = horizon
        self.use_lstm = use_lstm and _HAS_TENSORFLOW
        
        # Scalers
        self.power_scaler = MinMaxScaler(feature_range=(0, 1))
        self.voltage_scaler = MinMaxScaler(feature_range=(0, 1))
        self.current_scaler = MinMaxScaler(feature_range=(0, 1))
        
        # LSTM model
        self.lstm_model = None
        
        # Statistical models
        self.trend_coeff = None
        self.seasonal_pattern = None
        
        # Buffers for real-time forecasting
        self.power_buffer = deque(maxlen=lookback * 2)
        self.voltage_buffer = deque(maxlen=lookback * 2)
        self.current_buffer = deque(maxlen=lookback * 2)
        
        # State
        self.is_trained = False
        self.training_history = {}

    def _build_lstm_model(self, n_features: int = 3) -> 'Sequential':
        """Build the LSTM forecasting model."""
        model = Sequential([
            Bidirectional(
                LSTM(64, return_sequences=True, input_shape=(self.lookback, n_features)),
                name='bilstm_1'
            ),
            Dropout(0.2),
            Bidirectional(
                LSTM(32, return_sequences=False),
                name='bilstm_2'
            ),
            Dropout(0.2),
            Dense(32, activation='relu', name='dense_1'),
            Dense(self.horizon, activation='linear', name='output'),
        ])
        model.compile(
            optimizer=Adam(learning_rate=FORECAST_LEARNING_RATE),
            loss='mse',
            metrics=['mae'],
        )
        return model

    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create input-output sequences for LSTM training."""
        X, y = [], []
        for i in range(len(data) - self.lookback - self.horizon + 1):
            X.append(data[i:i + self.lookback])
            y.append(data[i + self.lookback:i + self.lookback + self.horizon, 0])  # Predict power only
        return np.array(X), np.array(y)

    def train(self, power: np.ndarray,
              voltage: Optional[np.ndarray] = None,
              current: Optional[np.ndarray] = None,
              epochs: int = FORECAST_EPOCHS,
              verbose: bool = True) -> Dict:
        """
        Train the forecaster on historical data.
        
        Args:
            power: Historical power consumption array
            voltage: Optional voltage array (improves accuracy)
            current: Optional current array (improves accuracy)
            epochs: Training epochs for LSTM
            verbose: Print progress
            
        Returns:
            Training metrics
        """
        if verbose:
            print("\n  🧠 Training Power Forecaster...")
            print(f"     Data points: {len(power):,}")
            print(f"     Lookback: {self.lookback}")
            print(f"     Horizon: {self.horizon}")
            print(f"     LSTM available: {_HAS_TENSORFLOW}")
        
        # Fit scalers
        power_2d = power.reshape(-1, 1)
        power_scaled = self.power_scaler.fit_transform(power_2d).flatten()
        
        if voltage is not None and current is not None:
            voltage_scaled = self.voltage_scaler.fit_transform(voltage.reshape(-1, 1)).flatten()
            current_scaled = self.current_scaler.fit_transform(current.reshape(-1, 1)).flatten()
            multivariate = np.column_stack([power_scaled, voltage_scaled, current_scaled])
            n_features = 3
        else:
            multivariate = power_scaled.reshape(-1, 1)
            n_features = 1
        
        # ── Train Statistical Models ──
        self._train_statistical(power)
        
        metrics = {
            'statistical_trained': True,
            'lstm_trained': False,
        }
        
        # ── Train LSTM (if available) ──
        if self.use_lstm:
            if verbose:
                print("\n     📌 Training LSTM Network...")
            
            X, y = self._create_sequences(multivariate)
            
            if len(X) < 10:
                if verbose:
                    print("     ⚠️ Not enough data for LSTM. Need at least "
                          f"{self.lookback + self.horizon + 10} points.")
            else:
                # Train/val split (time-series aware — no shuffle)
                split = int(len(X) * 0.85)
                X_train, X_val = X[:split], X[split:]
                y_train, y_val = y[:split], y[split:]
                
                self.lstm_model = self._build_lstm_model(n_features)
                
                callbacks = [
                    EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
                    ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-6),
                ]
                
                history = self.lstm_model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=FORECAST_BATCH_SIZE,
                    callbacks=callbacks,
                    verbose=1 if verbose else 0,
                )
                
                self.training_history = {
                    'loss': [float(x) for x in history.history['loss']],
                    'val_loss': [float(x) for x in history.history['val_loss']],
                    'mae': [float(x) for x in history.history['mae']],
                    'val_mae': [float(x) for x in history.history['val_mae']],
                }
                
                # Evaluate
                val_loss, val_mae = self.lstm_model.evaluate(X_val, y_val, verbose=0)
                
                metrics['lstm_trained'] = True
                metrics['lstm_val_loss'] = float(val_loss)
                metrics['lstm_val_mae'] = float(val_mae)
                metrics['lstm_epochs_run'] = len(history.history['loss'])
                
                if verbose:
                    print(f"\n     ✅ LSTM Training complete!")
                    print(f"     Val Loss (MSE):  {val_loss:.6f}")
                    print(f"     Val MAE:         {val_mae:.6f}")
                    print(f"     Epochs run:      {len(history.history['loss'])}")
        
        self.is_trained = True
        
        if verbose:
            print(f"\n     ✅ Forecaster ready!")
        
        return metrics

    def _train_statistical(self, power: np.ndarray):
        """Train statistical baseline models."""
        # Linear trend
        x = np.arange(len(power))
        coeffs = np.polyfit(x, power, deg=1)
        self.trend_coeff = coeffs
        
        # Simple seasonal pattern (if enough data)
        if len(power) > 100:
            period = min(60, len(power) // 3)
            n_periods = len(power) // period
            if n_periods >= 2:
                reshaped = power[:n_periods * period].reshape(n_periods, period)
                self.seasonal_pattern = np.mean(reshaped, axis=0)

    def forecast(self, steps: int = None,
                 return_confidence: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Forecast future power consumption.
        
        Args:
            steps: Number of steps to forecast (default: self.horizon)
            return_confidence: Whether to return confidence intervals
            
        Returns:
            (predictions, confidence_bounds) or (predictions, None)
            confidence_bounds: (n_steps, 2) array of [lower, upper] bounds
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained! Call train() first.")
        
        steps = steps or self.horizon
        
        predictions = np.zeros(steps)
        confidence = np.zeros((steps, 2)) if return_confidence else None
        
        if self.use_lstm and self.lstm_model is not None and len(self.power_buffer) >= self.lookback:
            # LSTM prediction
            predictions = self._lstm_forecast(steps)
        else:
            # Statistical fallback
            predictions = self._statistical_forecast(steps)
        
        # Calculate confidence intervals
        if return_confidence and len(self.power_buffer) > 10:
            recent = np.array(list(self.power_buffer)[-50:])
            std = np.std(recent)
            expanding_std = std * np.sqrt(np.arange(1, steps + 1) / steps)
            confidence[:, 0] = predictions - 1.96 * expanding_std
            confidence[:, 1] = predictions + 1.96 * expanding_std
            confidence[:, 0] = np.maximum(confidence[:, 0], 0)  # Power can't be negative
        
        return predictions, confidence

    def _lstm_forecast(self, steps: int) -> np.ndarray:
        """Generate LSTM-based forecast."""
        power_arr = np.array(list(self.power_buffer)[-self.lookback:])
        power_scaled = self.power_scaler.transform(power_arr.reshape(-1, 1)).flatten()
        
        if hasattr(self, 'voltage_buffer') and len(self.voltage_buffer) >= self.lookback:
            v_arr = np.array(list(self.voltage_buffer)[-self.lookback:])
            i_arr = np.array(list(self.current_buffer)[-self.lookback:])
            v_scaled = self.voltage_scaler.transform(v_arr.reshape(-1, 1)).flatten()
            i_scaled = self.current_scaler.transform(i_arr.reshape(-1, 1)).flatten()
            input_seq = np.column_stack([power_scaled, v_scaled, i_scaled])
        else:
            input_seq = power_scaled.reshape(-1, 1)
        
        input_seq = input_seq.reshape(1, self.lookback, -1)
        pred_scaled = self.lstm_model.predict(input_seq, verbose=0)[0]
        
        # Inverse transform
        pred = self.power_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        
        # If need more steps than horizon, extend with trend
        if steps > self.horizon:
            extra = steps - self.horizon
            last_val = pred[-1]
            if self.trend_coeff is not None:
                slope = self.trend_coeff[0]
                extension = last_val + slope * np.arange(1, extra + 1)
            else:
                extension = np.full(extra, last_val)
            pred = np.concatenate([pred, extension])
        
        return pred[:steps]

    def _statistical_forecast(self, steps: int) -> np.ndarray:
        """Statistical baseline forecast."""
        if len(self.power_buffer) < 3:
            return np.zeros(steps)
        
        recent = np.array(list(self.power_buffer))
        
        # Combine trend + seasonal
        n = len(recent)
        predictions = np.zeros(steps)
        
        for s in range(steps):
            # Trend component
            if self.trend_coeff is not None:
                trend = np.polyval(self.trend_coeff, n + s)
            else:
                trend = np.mean(recent[-10:])
            
            # Seasonal component
            if self.seasonal_pattern is not None:
                period = len(self.seasonal_pattern)
                seasonal = self.seasonal_pattern[(n + s) % period]
                predictions[s] = 0.6 * trend + 0.4 * seasonal
            else:
                predictions[s] = trend
        
        return np.maximum(predictions, 0)

    def add_point(self, power: float, voltage: float = None, current: float = None):
        """Add a new data point to the buffer (for real-time forecasting)."""
        self.power_buffer.append(power)
        if voltage is not None:
            self.voltage_buffer.append(voltage)
        if current is not None:
            self.current_buffer.append(current)

    def get_trend(self) -> Dict:
        """Analyze current power consumption trend."""
        if len(self.power_buffer) < 10:
            return {'direction': 'unknown', 'slope': 0.0, 'description': 'Insufficient data'}
        
        recent = np.array(list(self.power_buffer))
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        
        if abs(slope) < 0.5:
            direction = 'stable'
            desc = 'Power consumption is stable'
        elif slope > 0:
            direction = 'increasing'
            desc = f'Power consumption increasing at ~{slope:.1f} W/sample'
        else:
            direction = 'decreasing'
            desc = f'Power consumption decreasing at ~{abs(slope):.1f} W/sample'
        
        return {
            'direction': direction,
            'slope': float(slope),
            'current_avg': float(np.mean(recent[-10:])),
            'description': desc,
        }

    def save(self, filename: str = "power_forecaster"):
        """Save trained model."""
        if not self.is_trained:
            print("  ⚠️ Model not trained!")
            return
        
        path = os.path.join(MODELS_DIR, f"{filename}.joblib")
        state = {
            'power_scaler': self.power_scaler,
            'voltage_scaler': self.voltage_scaler,
            'current_scaler': self.current_scaler,
            'trend_coeff': self.trend_coeff,
            'seasonal_pattern': self.seasonal_pattern,
            'lookback': self.lookback,
            'horizon': self.horizon,
            'training_history': self.training_history,
        }
        joblib.dump(state, path)
        
        # Save LSTM separately (Keras format)
        if self.lstm_model is not None:
            lstm_path = os.path.join(MODELS_DIR, f"{filename}_lstm.keras")
            self.lstm_model.save(lstm_path)
            print(f"  💾 LSTM model saved: {lstm_path}")
        
        print(f"  💾 Forecaster state saved: {path}")

    def load(self, filename: str = "power_forecaster"):
        """Load trained model."""
        path = os.path.join(MODELS_DIR, f"{filename}.joblib")
        if not os.path.exists(path):
            print(f"  ⚠️ Model not found: {path}")
            return False
        
        state = joblib.load(path)
        self.power_scaler = state['power_scaler']
        self.voltage_scaler = state['voltage_scaler']
        self.current_scaler = state['current_scaler']
        self.trend_coeff = state['trend_coeff']
        self.seasonal_pattern = state['seasonal_pattern']
        self.lookback = state['lookback']
        self.horizon = state['horizon']
        self.training_history = state.get('training_history', {})
        
        # Load LSTM
        if _HAS_TENSORFLOW:
            lstm_path = os.path.join(MODELS_DIR, f"{filename}_lstm.keras")
            if os.path.exists(lstm_path):
                self.lstm_model = keras.models.load_model(lstm_path)
                self.use_lstm = True
        
        self.is_trained = True
        print(f"  📂 Forecaster loaded: {path}")
        return True
