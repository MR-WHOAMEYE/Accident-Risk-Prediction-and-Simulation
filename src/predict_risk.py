"""
Real-time Risk Prediction Module
Predicts accident risk during simulation or for historical data
"""

import pandas as pd
import numpy as np
import joblib
import yaml
import os
from tensorflow import keras

from feature_engineering import FeatureEngineer


class RiskPredictor:
    """Predict accident risk using trained models"""
    
    def __init__(self, model_path, scaler_path, features_path, config_path='config.yaml'):
        """Initialize predictor with trained model"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load model
        if model_path.endswith('.pkl'):
            self.model = joblib.load(model_path)
            self.model_type = 'sklearn'
        elif model_path.endswith('.h5'):
            self.model = keras.models.load_model(model_path)
            self.model_type = 'keras'
        else:
            raise ValueError("Unsupported model format")
        
        # Load scaler and features
        self.scaler = joblib.load(scaler_path)
        self.feature_columns = joblib.load(features_path)
        
        # Feature engineer
        self.feature_engineer = FeatureEngineer(config_path)
        
        print(f"Loaded model: {model_path}")
        print(f"Model type: {self.model_type}")
        print(f"Features: {len(self.feature_columns)}")
    
    def predict_batch(self, data):
        """Predict risk for batch data from CSV or DataFrame"""
        # Handle both file path and DataFrame inputs
        if isinstance(data, str):
            print(f"Loading data from {data}...")
            df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError("data must be a file path (str) or DataFrame")
        
        # Apply feature engineering
        df = self.feature_engineer.process_data(df)
        
        # Prepare features (handle missing columns)
        available_features = [col for col in self.feature_columns if col in df.columns]
        missing_features = [col for col in self.feature_columns if col not in df.columns]
        
        if missing_features:
            # Add missing features with default values
            for col in missing_features:
                df[col] = 0
        
        X = df[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Convert scaled data back to DataFrame with feature names to avoid sklearn warning
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_columns, index=X.index)
        
        # Predict
        if self.model_type == 'sklearn':
            predictions = self.model.predict(X_scaled_df)
            probabilities = self.model.predict_proba(X_scaled_df)[:, 1]
        else:  # keras
            probabilities = self.model.predict(X_scaled).flatten()
            predictions = (probabilities > 0.5).astype(int)
        
        # Add predictions to dataframe
        df['predicted_risk'] = predictions
        df['risk_probability'] = probabilities
        
        if isinstance(data, str):
            print(f"Predictions complete!")
            print(f"High risk predictions: {predictions.sum()} ({predictions.sum()/len(predictions)*100:.1f}%)")
        
        return df
    
    def predict_single(self, features_dict):
        """Predict risk for a single data point"""
        # Convert to DataFrame
        df = pd.DataFrame([features_dict])
        
        # Prepare features
        X = df[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        if self.model_type == 'sklearn':
            prediction = self.model.predict(X_scaled)[0]
            probability = self.model.predict_proba(X_scaled)[0, 1]
        else:  # keras
            probability = self.model.predict(X_scaled)[0, 0]
            prediction = int(probability > 0.5)
        
        return {
            'risk_level': 'HIGH' if prediction == 1 else 'LOW',
            'risk_probability': float(probability)
        }
    
    def save_predictions(self, df, output_path):
        """Save predictions to CSV"""
        df.to_csv(output_path, index=False)
        print(f"Predictions saved: {output_path}")


def load_latest_model(models_dir='models'):
    """Load the most recently trained model"""
    # Find latest model files
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl') or f.endswith('.h5')]
    
    if not model_files:
        raise FileNotFoundError("No trained models found")
    
    # Sort by timestamp (assuming filename format: modelname_timestamp.ext)
    model_files.sort(reverse=True)
    
    # Get latest model (prefer gradient_boosting)
    gb_models = [f for f in model_files if 'gradient_boosting' in f and f.endswith('.pkl')]
    if gb_models:
        model_file = gb_models[0]
    else:
        # Try random forest
        rf_models = [f for f in model_files if 'random_forest' in f and f.endswith('.pkl')]
        if rf_models:
            model_file = rf_models[0]
        else:
            model_file = model_files[0]
    
    model_path = os.path.join(models_dir, model_file)
    
    # Extract timestamp - handle filenames like "gradient_boosting_20251202_164825.pkl"
    # Split by underscore and get the last two parts before extension
    parts = model_file.replace('.pkl', '').replace('.h5', '').split('_')
    # Timestamp is the last two parts joined (e.g., "20251202_164825")
    timestamp = '_'.join(parts[-2:])
    
    # Find corresponding scaler and features
    scaler_path = os.path.join(models_dir, f'scaler_{timestamp}.pkl')
    features_path = os.path.join(models_dir, f'feature_columns_{timestamp}.pkl')
    
    return model_path, scaler_path, features_path


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python predict_risk.py <data_file.csv> [output_file.csv]")
        print("Example: python predict_risk.py data/vehicle_data_20231201_120000.csv predictions.csv")
        sys.exit(1)
    
    data_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'predictions.csv'
    
    # Load latest model
    model_path, scaler_path, features_path = load_latest_model()
    
    # Create predictor
    predictor = RiskPredictor(model_path, scaler_path, features_path)
    
    # Predict
    df_with_predictions = predictor.predict_batch(data_path)
    
    # Save
    predictor.save_predictions(df_with_predictions, output_path)
