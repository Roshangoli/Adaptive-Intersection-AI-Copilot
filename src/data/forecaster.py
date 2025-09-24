#!/usr/bin/env python3
"""
Demand Forecasting for Adaptive Intersection AI Copilot.
This module provides short-term demand prediction using Prophet and LSTM models.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DemandForecaster:
    """Demand forecasting using Prophet and LSTM models."""
    
    def __init__(self, model_type: str = "prophet"):
        """
        Initialize forecaster.
        
        Args:
            model_type: Type of model to use ("prophet" or "lstm")
        """
        self.model_type = model_type
        self.prophet_model = None
        self.lstm_model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        
        # Model parameters
        self.forecast_horizon = 15  # 15 minutes ahead
        self.lookback_window = 60    # 60 minutes of history
        
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for forecasting.
        
        Args:
            data: Raw traffic data
            
        Returns:
            Prepared dataframe
        """
        # Ensure we have required columns
        required_cols = ['timestamp', 'pedestrians', 'vehicles']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Convert timestamp to datetime
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Sort by timestamp
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        # Add time features
        data['hour'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
        
        # Add rolling averages
        data['pedestrians_ma'] = data['pedestrians'].rolling(window=5, min_periods=1).mean()
        data['vehicles_ma'] = data['vehicles'].rolling(window=5, min_periods=1).mean()
        
        return data
    
    def train_prophet_model(self, data: pd.DataFrame) -> None:
        """
        Train Prophet model for demand forecasting.
        
        Args:
            data: Prepared traffic data
        """
        try:
            # Prepare data for Prophet
            prophet_data = data[['timestamp', 'pedestrians']].copy()
            prophet_data.columns = ['ds', 'y']
            
            # Initialize and train Prophet model
            self.prophet_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True,
                seasonality_mode='multiplicative'
            )
            
            # Add additional regressors
            self.prophet_model.add_regressor('vehicles')
            self.prophet_model.add_regressor('hour')
            self.prophet_model.add_regressor('is_weekend')
            
            # Prepare data with regressors
            prophet_data['vehicles'] = data['vehicles'].values
            prophet_data['hour'] = data['hour'].values
            prophet_data['is_weekend'] = data['is_weekend'].values
            
            # Train model
            self.prophet_model.fit(prophet_data)
            
            logger.info("Prophet model trained successfully")
            
        except Exception as e:
            logger.error(f"Prophet training failed: {e}")
            raise
    
    def train_lstm_model(self, data: pd.DataFrame) -> None:
        """
        Train LSTM model for demand forecasting.
        
        Args:
            data: Prepared traffic data
        """
        try:
            # Prepare features
            features = ['pedestrians', 'vehicles', 'hour', 'day_of_week', 'is_weekend']
            X = data[features].values
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Create sequences
            X_seq, y_seq = self._create_sequences(X_scaled, self.lookback_window)
            
            # Split data
            split_idx = int(0.8 * len(X_seq))
            X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
            y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
            
            # Build LSTM model
            self.lstm_model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(self.lookback_window, len(features))),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            # Compile model
            self.lstm_model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            # Train model
            history = self.lstm_model.fit(
                X_train, y_train,
                batch_size=32,
                epochs=50,
                validation_data=(X_test, y_test),
                verbose=0
            )
            
            logger.info("LSTM model trained successfully")
            
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
            raise
    
    def _create_sequences(self, data: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            data: Scaled feature data
            lookback: Number of time steps to look back
            
        Returns:
            X sequences and y targets
        """
        X, y = [], []
        
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i, 0])  # Predict pedestrians (first feature)
        
        return np.array(X), np.array(y)
    
    def train(self, data: pd.DataFrame) -> None:
        """
        Train the forecasting model.
        
        Args:
            data: Traffic data for training
        """
        logger.info(f"Training {self.model_type} model...")
        
        # Prepare data
        prepared_data = self.prepare_data(data)
        
        # Train model based on type
        if self.model_type == "prophet":
            self.train_prophet_model(prepared_data)
        elif self.model_type == "lstm":
            self.train_lstm_model(prepared_data)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.is_trained = True
        logger.info("Model training completed")
    
    def forecast(self, current_data: pd.DataFrame, 
                forecast_minutes: int = 15) -> Dict[str, List]:
        """
        Generate demand forecast.
        
        Args:
            current_data: Current traffic data
            forecast_minutes: Minutes to forecast ahead
            
        Returns:
            Forecast results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before forecasting")
        
        if self.model_type == "prophet":
            return self._forecast_prophet(current_data, forecast_minutes)
        elif self.model_type == "lstm":
            return self._forecast_lstm(current_data, forecast_minutes)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _forecast_prophet(self, current_data: pd.DataFrame, 
                         forecast_minutes: int) -> Dict[str, List]:
        """Generate Prophet forecast."""
        try:
            # Prepare future dataframe
            last_timestamp = current_data['timestamp'].iloc[-1]
            future_times = pd.date_range(
                start=last_timestamp + timedelta(minutes=1),
                periods=forecast_minutes,
                freq='1min'
            )
            
            future_df = pd.DataFrame({
                'ds': future_times,
                'vehicles': current_data['vehicles'].iloc[-1],  # Use last known value
                'hour': future_times.hour,
                'is_weekend': future_times.dayofweek.isin([5, 6]).astype(int)
            })
            
            # Generate forecast
            forecast = self.prophet_model.predict(future_df)
            
            return {
                'timestamps': future_times.tolist(),
                'pedestrians': forecast['yhat'].tolist(),
                'pedestrians_lower': forecast['yhat_lower'].tolist(),
                'pedestrians_upper': forecast['yhat_upper'].tolist(),
                'vehicles': [current_data['vehicles'].iloc[-1]] * forecast_minutes
            }
            
        except Exception as e:
            logger.error(f"Prophet forecast failed: {e}")
            return self._fallback_forecast(current_data, forecast_minutes)
    
    def _forecast_lstm(self, current_data: pd.DataFrame, 
                      forecast_minutes: int) -> Dict[str, List]:
        """Generate LSTM forecast."""
        try:
            # Prepare features
            features = ['pedestrians', 'vehicles', 'hour', 'day_of_week', 'is_weekend']
            X = current_data[features].values
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Create sequence for prediction
            X_seq = X_scaled[-self.lookback_window:].reshape(1, self.lookback_window, len(features))
            
            # Generate forecast
            forecasts = []
            current_seq = X_seq.copy()
            
            for _ in range(forecast_minutes):
                pred = self.lstm_model.predict(current_seq, verbose=0)[0, 0]
                forecasts.append(pred)
                
                # Update sequence (simple approach)
                new_row = current_seq[0, -1, :].copy()
                new_row[0] = pred  # Update pedestrians prediction
                current_seq = np.roll(current_seq, -1, axis=1)
                current_seq[0, -1, :] = new_row
            
            # Generate timestamps
            last_timestamp = current_data['timestamp'].iloc[-1]
            future_times = pd.date_range(
                start=last_timestamp + timedelta(minutes=1),
                periods=forecast_minutes,
                freq='1min'
            )
            
            return {
                'timestamps': future_times.tolist(),
                'pedestrians': forecasts,
                'pedestrians_lower': [max(0, p - 2) for p in forecasts],
                'pedestrians_upper': [p + 2 for p in forecasts],
                'vehicles': [current_data['vehicles'].iloc[-1]] * forecast_minutes
            }
            
        except Exception as e:
            logger.error(f"LSTM forecast failed: {e}")
            return self._fallback_forecast(current_data, forecast_minutes)
    
    def _fallback_forecast(self, current_data: pd.DataFrame, 
                          forecast_minutes: int) -> Dict[str, List]:
        """Fallback forecast using simple moving average."""
        logger.warning("Using fallback forecast")
        
        # Simple moving average
        avg_pedestrians = current_data['pedestrians'].tail(5).mean()
        avg_vehicles = current_data['vehicles'].tail(5).mean()
        
        # Generate timestamps
        last_timestamp = current_data['timestamp'].iloc[-1]
        future_times = pd.date_range(
            start=last_timestamp + timedelta(minutes=1),
            periods=forecast_minutes,
            freq='1min'
        )
        
        return {
            'timestamps': future_times.tolist(),
            'pedestrians': [avg_pedestrians] * forecast_minutes,
            'pedestrians_lower': [max(0, avg_pedestrians - 1)] * forecast_minutes,
            'pedestrians_upper': [avg_pedestrians + 1] * forecast_minutes,
            'vehicles': [avg_vehicles] * forecast_minutes
        }
    
    def evaluate_model(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            test_data: Test data for evaluation
            
        Returns:
            Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Generate forecast
        forecast = self.forecast(test_data, len(test_data))
        
        # Calculate metrics
        actual = test_data['pedestrians'].values
        predicted = forecast['pedestrians']
        
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': np.mean(np.abs((actual - predicted) / actual)) * 100
        }
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained model to file.
        
        Args:
            filepath: Path to save model
        """
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model_type': self.model_type,
            'forecast_horizon': self.forecast_horizon,
            'lookback_window': self.lookback_window
        }
        
        if self.model_type == "prophet" and self.prophet_model:
            # Save Prophet model
            self.prophet_model.save(f"{filepath}_prophet.json")
            
        elif self.model_type == "lstm" and self.lstm_model:
            # Save LSTM model
            self.lstm_model.save(f"{filepath}_lstm.h5")
            
            # Save scaler
            import joblib
            joblib.dump(self.scaler, f"{filepath}_scaler.pkl")
        
        # Save metadata
        with open(f"{filepath}_metadata.json", 'w') as f:
            json.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load trained model from file.
        
        Args:
            filepath: Path to load model from
        """
        # Load metadata
        with open(f"{filepath}_metadata.json", 'r') as f:
            model_data = json.load(f)
        
        self.model_type = model_data['model_type']
        self.forecast_horizon = model_data['forecast_horizon']
        self.lookback_window = model_data['lookback_window']
        
        if self.model_type == "prophet":
            # Load Prophet model
            self.prophet_model = Prophet()
            self.prophet_model.load(f"{filepath}_prophet.json")
            
        elif self.model_type == "lstm":
            # Load LSTM model
            self.lstm_model = tf.keras.models.load_model(f"{filepath}_lstm.h5")
            
            # Load scaler
            import joblib
            self.scaler = joblib.load(f"{filepath}_scaler.pkl")
        
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")

# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='1min')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'pedestrians': np.random.poisson(5, 1000),
        'vehicles': np.random.poisson(3, 1000)
    })
    
    # Initialize forecaster
    forecaster = DemandForecaster(model_type="prophet")
    
    # Train model
    forecaster.train(sample_data)
    
    # Generate forecast
    forecast = forecaster.forecast(sample_data.tail(100), forecast_minutes=15)
    
    print("DemandForecaster initialized successfully!")
    print(f"Forecast for next 15 minutes: {forecast['pedestrians'][:5]} pedestrians")