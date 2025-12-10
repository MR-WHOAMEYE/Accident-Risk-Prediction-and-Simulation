"""
Feature Engineering Module
Creates advanced features and generates risk labels for ML training
"""

import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import LabelEncoder


class FeatureEngineer:
    """Engineer features and generate risk labels for accident prediction"""
    
    def __init__(self, config_path='config.yaml'):
        """Initialize feature engineer with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.label_encoders = {}
    
    def create_rolling_features(self, df, window=10):
        """Create rolling window statistics for time-series features"""
        print("Creating rolling features...")
        
        # Sort by vehicle and timestamp
        df = df.sort_values(['vehicle_id', 'timestamp'])
        
        # Rolling features for each vehicle
        rolling_features = []
        
        for veh_id in df['vehicle_id'].unique():
            veh_data = df[df['vehicle_id'] == veh_id].copy()
            
            # Rolling mean and std for speed
            veh_data['speed_rolling_mean'] = veh_data['speed'].rolling(window=window, min_periods=1).mean()
            veh_data['speed_rolling_std'] = veh_data['speed'].rolling(window=window, min_periods=1).std().fillna(0)
            
            # Rolling mean for acceleration
            veh_data['accel_rolling_mean'] = veh_data['acceleration'].rolling(window=window, min_periods=1).mean()
            veh_data['accel_rolling_std'] = veh_data['acceleration'].rolling(window=window, min_periods=1).std().fillna(0)
            
            # Speed change rate
            veh_data['speed_change_rate'] = veh_data['speed'].diff().fillna(0)
            
            rolling_features.append(veh_data)
        
        df_enhanced = pd.concat(rolling_features, ignore_index=True)
        
        return df_enhanced
    
    def create_interaction_features(self, df):
        """Create interaction and derived features"""
        print("Creating interaction features...")
        
        # Congestion index (combination of density and waiting time)
        df['congestion_index'] = (df['traffic_density'] * 0.5 + 
                                  df['waiting_time'] / 60 * 0.5)
        
        # Normalize congestion index to 0-1
        max_congestion = df['congestion_index'].max()
        if max_congestion > 0:
            df['congestion_index'] = df['congestion_index'] / max_congestion
        
        # Risk of collision (inverse of TTC, capped)
        df['collision_risk'] = np.where(df['ttc'] < 999, 1 / (df['ttc'] + 0.1), 0)
        
        # Pedestrian conflict potential
        df['pedestrian_conflict'] = df['pedestrian_proximity'] * (df['speed'] / 13.89)  # Normalized by typical speed
        
        # Sudden braking indicator
        df['sudden_braking'] = (df['acceleration'] < -3.0).astype(int)
        
        # Aggressive driving score
        df['aggressive_driving'] = (
            (df['speed_variance'] > 10).astype(int) +
            (df['lane_changes'] > 2).astype(int) +
            (df['acceleration'] > 3.0).astype(int)
        ) / 3.0
        
        # Distance to intersection (approximation)
        df['near_intersection'] = (
            (np.abs(df['x']) < 50) | (np.abs(df['y']) < 50) |
            (np.abs(df['x'] - 200) < 50) | (np.abs(df['y'] - 200) < 50) |
            (np.abs(df['x'] + 200) < 50) | (np.abs(df['y'] + 200) < 50)
        ).astype(int)
        
        # Traffic light conflict (approaching red light at high speed)
        df['tls_conflict'] = 0
        df.loc[(df['tls_state'].isin(['r', 'y'])) & 
               (df['tls_distance'] < 30) & 
               (df['speed'] > 8), 'tls_conflict'] = 1
        
        return df
    
    def encode_categorical_features(self, df):
        """Encode categorical features"""
        print("Encoding categorical features...")
        
        categorical_cols = ['vehicle_type', 'tls_state', 'road_id', 'lane_id']
        
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    # Handle unseen categories
                    df[f'{col}_encoded'] = df[col].apply(
                        lambda x: self.label_encoders[col].transform([str(x)])[0] 
                        if str(x) in self.label_encoders[col].classes_ 
                        else -1
                    )
        
        return df
    
    def generate_risk_labels(self, df):
        """Generate risk labels based on multi-factor thresholds"""
        print("Generating risk labels...")
        
        thresholds = self.config['risk_thresholds']
        
        # Initialize risk level
        df['risk_level'] = 0  # 0 = low, 1 = medium, 2 = high
        
        # High risk conditions
        high_risk_conditions = [
            df['speed_variance'] > thresholds['high_risk']['conditions']['speed_variance'],
            df['ttc'] < thresholds['high_risk']['conditions']['ttc_threshold'],
            df['pedestrian_proximity'] > 0,
            df['congestion_index'] > thresholds['high_risk']['conditions']['congestion_level'],
            df['acceleration'] < thresholds['high_risk']['conditions']['sudden_deceleration'],
            df['lane_changes'] >= thresholds['high_risk']['conditions']['lane_change_frequency']
        ]
        
        high_risk_count = sum(high_risk_conditions)
        df.loc[high_risk_count >= thresholds['high_risk']['min_conditions'], 'risk_level'] = 2
        
        # Medium risk conditions
        medium_risk_conditions = [
            df['speed_variance'] > thresholds['medium_risk']['conditions']['speed_variance'],
            df['ttc'] < thresholds['medium_risk']['conditions']['ttc_threshold'],
            df['pedestrian_proximity'] > 0,
            df['congestion_index'] > thresholds['medium_risk']['conditions']['congestion_level'],
            df['acceleration'] < thresholds['medium_risk']['conditions']['sudden_deceleration'],
            df['lane_changes'] >= thresholds['medium_risk']['conditions']['lane_change_frequency']
        ]
        
        medium_risk_count = sum(medium_risk_conditions)
        df.loc[(df['risk_level'] == 0) & 
               (medium_risk_count >= thresholds['medium_risk']['min_conditions']), 'risk_level'] = 1
        
        # Binary risk label (for classification)
        df['high_risk'] = (df['risk_level'] >= 2).astype(int)
        
        # Risk score (continuous 0-1)
        df['risk_score'] = (
            df['collision_risk'] * 0.3 +
            df['congestion_index'] * 0.2 +
            df['pedestrian_conflict'] * 0.2 +
            df['aggressive_driving'] * 0.15 +
            df['tls_conflict'] * 0.15
        )
        
        # Normalize risk score
        max_score = df['risk_score'].max()
        if max_score > 0:
            df['risk_score'] = df['risk_score'] / max_score
        
        print(f"Risk distribution:")
        print(f"  Low risk: {(df['risk_level'] == 0).sum()} ({(df['risk_level'] == 0).sum()/len(df)*100:.1f}%)")
        print(f"  Medium risk: {(df['risk_level'] == 1).sum()} ({(df['risk_level'] == 1).sum()/len(df)*100:.1f}%)")
        print(f"  High risk: {(df['risk_level'] == 2).sum()} ({(df['risk_level'] == 2).sum()/len(df)*100:.1f}%)")
        
        return df
    
    def process_data(self, df):
        """Complete feature engineering pipeline"""
        print("Starting feature engineering pipeline...")
        print(f"Input data shape: {df.shape}")
        
        # Create rolling features
        df = self.create_rolling_features(df)
        
        # Create interaction features
        df = self.create_interaction_features(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Generate risk labels
        df = self.generate_risk_labels(df)
        
        print(f"Output data shape: {df.shape}")
        print("Feature engineering complete!")
        
        return df
    
    def get_feature_columns(self):
        """Return list of feature columns for ML training"""
        feature_cols = [
            # Kinematic features
            'speed', 'acceleration', 'speed_variance',
            'speed_rolling_mean', 'speed_rolling_std',
            'accel_rolling_mean', 'accel_rolling_std',
            'speed_change_rate',
            
            # Spatial features
            'lane_position', 'lane_index',
            
            # Traffic features
            'leader_distance', 'ttc', 'traffic_density',
            'waiting_time', 'accumulated_waiting',
            'lane_changes',
            
            # Pedestrian features
            'pedestrian_proximity', 'pedestrian_conflict',
            
            # Derived features
            'congestion_index', 'collision_risk',
            'sudden_braking', 'aggressive_driving',
            'near_intersection', 'tls_conflict',
            
            # Encoded categorical
            'vehicle_type_encoded', 'tls_state_encoded'
        ]
        
        return feature_cols


if __name__ == "__main__":
    print("Feature Engineering Module - Use with collected data")
