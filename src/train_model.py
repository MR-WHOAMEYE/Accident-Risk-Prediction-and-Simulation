"""
Machine Learning Model Training Module
Trains multiple models for accident risk prediction
"""

import pandas as pd
import numpy as np
import yaml
import joblib
import os
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import seaborn as sns

from feature_engineering import FeatureEngineer


class AccidentRiskModel:
    """Train and evaluate accident risk prediction models"""
    
    def __init__(self, config_path='config.yaml'):
        """Initialize model trainer"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_engineer = FeatureEngineer(config_path)
        self.feature_columns = None
        self.results = {}
    
    def load_and_prepare_data(self, data_path):
        """Load data and prepare for training"""
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        
        print(f"Loaded {len(df)} records")
        
        # Apply feature engineering
        df = self.feature_engineer.process_data(df)
        
        # Get feature columns
        self.feature_columns = self.feature_engineer.get_feature_columns()
        
        # Filter to only include available columns
        self.feature_columns = [col for col in self.feature_columns if col in df.columns]
        
        print(f"Using {len(self.feature_columns)} features")
        
        return df
    
    def prepare_train_test_split(self, df):
        """Split data into train and test sets"""
        # Features and target
        X = df[self.feature_columns].fillna(0)
        y = df['high_risk']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['ml']['test_size'],
            random_state=self.config['ml']['random_state'],
            stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Positive class ratio: {y_train.mean():.3f}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest classifier"""
        print("\n" + "="*50)
        print("Training Random Forest Classifier")
        print("="*50)
        
        rf_params = self.config['ml']['models']['random_forest']
        
        rf_model = RandomForestClassifier(
            n_estimators=rf_params['n_estimators'],
            max_depth=rf_params['max_depth'],
            min_samples_split=rf_params['min_samples_split'],
            random_state=self.config['ml']['random_state'],
            n_jobs=-1,
            class_weight='balanced'
        )
        
        rf_model.fit(X_train, y_train)
        
        self.models['random_forest'] = rf_model
        print("Random Forest training complete!")
        
        return rf_model
    
    def train_gradient_boosting(self, X_train, y_train):
        """Train Gradient Boosting classifier"""
        print("\n" + "="*50)
        print("Training Gradient Boosting Classifier")
        print("="*50)
        
        gb_params = self.config['ml']['models']['xgboost']
        
        gb_model = GradientBoostingClassifier(
            n_estimators=gb_params['n_estimators'],
            max_depth=gb_params['max_depth'],
            learning_rate=gb_params['learning_rate'],
            random_state=self.config['ml']['random_state'],
            subsample=0.8,
            verbose=0
        )
        
        gb_model.fit(X_train, y_train)
        
        self.models['gradient_boosting'] = gb_model
        print("Gradient Boosting training complete!")
        
        return gb_model
    
    def train_neural_network(self, X_train, y_train, X_test, y_test):
        """Train Neural Network classifier"""
        print("\n" + "="*50)
        print("Training Neural Network")
        print("="*50)
        
        nn_params = self.config['ml']['models']['neural_network']
        
        # Build model
        model = keras.Sequential()
        model.add(layers.Input(shape=(X_train.shape[1],)))
        
        for units in nn_params['hidden_layers']:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(0.3))
        
        model.add(layers.Dense(1, activation='sigmoid'))
        
        # Compile
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        # Calculate class weights
        class_weight = {
            0: 1.0,
            1: (y_train == 0).sum() / (y_train == 1).sum()
        }
        
        # Train
        history = model.fit(
            X_train, y_train,
            epochs=nn_params['epochs'],
            batch_size=nn_params['batch_size'],
            validation_data=(X_test, y_test),
            class_weight=class_weight,
            verbose=1
        )
        
        self.models['neural_network'] = model
        self.models['nn_history'] = history
        print("Neural Network training complete!")
        
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model performance"""
        print(f"\n{'='*50}")
        print(f"Evaluating {model_name}")
        print('='*50)
        
        # Predictions
        if model_name == 'neural_network':
            y_pred_proba = model.predict(X_test).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"ROC AUC:   {roc_auc:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Store results
        self.results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        return self.results[model_name]
    
    def plot_feature_importance(self, model_name, output_dir='outputs/visualizations'):
        """Plot feature importance for tree-based models"""
        os.makedirs(output_dir, exist_ok=True)
        
        if model_name not in ['random_forest', 'gradient_boosting']:
            print(f"Feature importance not available for {model_name}")
            return
        
        model = self.models[model_name]
        
        importances = model.feature_importances_
        
        # Create DataFrame
        feature_imp = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Plot top 20 features
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_imp.head(20), x='importance', y='feature')
        plt.title(f'Top 20 Feature Importances - {model_name}')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        filename = os.path.join(output_dir, f'feature_importance_{model_name}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature importance plot saved: {filename}")
        
        return feature_imp
    
    def save_models(self, output_dir='models'):
        """Save trained models"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save sklearn models
        for model_name in ['random_forest', 'gradient_boosting']:
            if model_name in self.models:
                filename = os.path.join(output_dir, f'{model_name}_{timestamp}.pkl')
                joblib.dump(self.models[model_name], filename)
                print(f"Saved {model_name}: {filename}")
        
        # Save neural network
        if 'neural_network' in self.models:
            filename = os.path.join(output_dir, f'neural_network_{timestamp}.h5')
            self.models['neural_network'].save(filename)
            print(f"Saved neural_network: {filename}")
        
        # Save scaler
        scaler_file = os.path.join(output_dir, f'scaler_{timestamp}.pkl')
        joblib.dump(self.scaler, scaler_file)
        print(f"Saved scaler: {scaler_file}")
        
        # Save feature columns
        feature_file = os.path.join(output_dir, f'feature_columns_{timestamp}.pkl')
        joblib.dump(self.feature_columns, feature_file)
        print(f"Saved feature columns: {feature_file}")
    
    def train_all_models(self, data_path):
        """Complete training pipeline"""
        print("="*70)
        print("ACCIDENT RISK PREDICTION - MODEL TRAINING")
        print("="*70)
        
        # Load and prepare data
        df = self.load_and_prepare_data(data_path)
        
        # Train-test split
        X_train, X_test, y_train, y_test, X_train_orig, X_test_orig = \
            self.prepare_train_test_split(df)
        
        # Train Random Forest
        self.train_random_forest(X_train_orig, y_train)
        self.evaluate_model(self.models['random_forest'], X_test_orig, y_test, 'random_forest')
        self.plot_feature_importance('random_forest')
        
        # Train Gradient Boosting
        self.train_gradient_boosting(X_train_orig, y_train)
        self.evaluate_model(self.models['gradient_boosting'], X_test_orig, y_test, 'gradient_boosting')
        self.plot_feature_importance('gradient_boosting')
        
        # Train Neural Network
        self.train_neural_network(X_train, y_train, X_test, y_test)
        self.evaluate_model(self.models['neural_network'], X_test, y_test, 'neural_network')
        
        # Save models
        self.save_models()
        
        # Print summary
        print("\n" + "="*70)
        print("TRAINING SUMMARY")
        print("="*70)
        
        results_df = pd.DataFrame(self.results).T
        print(results_df[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']])
        
        print("\nAll models trained and saved successfully!")
        
        return self.results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python train_model.py <data_file.csv>")
        print("Example: python train_model.py data/vehicle_data_20231201_120000.csv")
        sys.exit(1)
    
    data_path = sys.argv[1]
    
    trainer = AccidentRiskModel()
    results = trainer.train_all_models(data_path)
