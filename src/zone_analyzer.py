"""
High-Risk Zone Analysis Module
Identifies and analyzes high-risk zones using spatial aggregation
"""

import pandas as pd
import numpy as np
import yaml
import os
from collections import defaultdict


class ZoneAnalyzer:
    """Analyze and identify high-risk zones"""
    
    def __init__(self, config_path='config.yaml'):
        """Initialize zone analyzer"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.grid_size = self.config['zone_analysis']['grid_size']
        self.zone_stats = {}
    
    def create_spatial_grid(self, df):
        """Create spatial grid for zone analysis"""
        print("Creating spatial grid...")
        
        # Calculate grid indices
        df['grid_x'] = (df['x'] // self.grid_size).astype(int)
        df['grid_y'] = (df['y'] // self.grid_size).astype(int)
        df['zone_id'] = df['grid_x'].astype(str) + '_' + df['grid_y'].astype(str)
        
        return df
    
    def calculate_zone_risk(self, df):
        """Calculate risk metrics for each zone"""
        print("Calculating zone risk metrics...")
        
        # Ensure we have predictions
        if 'risk_probability' not in df.columns:
            print("Warning: No risk predictions found. Using risk_score instead.")
            if 'risk_score' in df.columns:
                df['risk_probability'] = df['risk_score']
            else:
                raise ValueError("No risk data available")
        
        # Group by zone
        zone_groups = df.groupby('zone_id')
        
        zone_stats = []
        
        for zone_id, zone_data in zone_groups:
            # Calculate statistics
            stats = {
                'zone_id': zone_id,
                'grid_x': zone_data['grid_x'].iloc[0],
                'grid_y': zone_data['grid_y'].iloc[0],
                'center_x': zone_data['x'].mean(),
                'center_y': zone_data['y'].mean(),
                'sample_count': len(zone_data),
                'avg_risk_probability': zone_data['risk_probability'].mean(),
                'max_risk_probability': zone_data['risk_probability'].max(),
                'high_risk_count': (zone_data['risk_probability'] > 0.7).sum(),
                'high_risk_ratio': (zone_data['risk_probability'] > 0.7).sum() / len(zone_data),
                'avg_speed': zone_data['speed'].mean(),
                'avg_congestion': zone_data['congestion_index'].mean() if 'congestion_index' in zone_data.columns else 0,
                'avg_ttc': zone_data['ttc'].mean() if 'ttc' in zone_data.columns else 999,
                'total_lane_changes': zone_data['lane_changes'].sum() if 'lane_changes' in zone_data.columns else 0,
                'pedestrian_incidents': (zone_data['pedestrian_proximity'] > 0).sum() if 'pedestrian_proximity' in zone_data.columns else 0
            }
            
            zone_stats.append(stats)
        
        zone_df = pd.DataFrame(zone_stats)
        
        # Filter zones with minimum samples
        min_samples = self.config['zone_analysis']['min_samples_per_zone']
        zone_df = zone_df[zone_df['sample_count'] >= min_samples]
        
        # Sort by risk
        zone_df = zone_df.sort_values('avg_risk_probability', ascending=False)
        
        self.zone_stats = zone_df
        
        print(f"Analyzed {len(zone_df)} zones")
        
        return zone_df
    
    def identify_high_risk_zones(self, top_n=None):
        """Identify top high-risk zones"""
        if top_n is None:
            top_n = self.config['zone_analysis']['top_n_zones']
        
        high_risk_zones = self.zone_stats.head(top_n)
        
        print(f"\nTop {top_n} High-Risk Zones:")
        print("="*80)
        
        for idx, zone in high_risk_zones.iterrows():
            print(f"\nZone {zone['zone_id']} (Center: {zone['center_x']:.1f}, {zone['center_y']:.1f})")
            print(f"  Average Risk Probability: {zone['avg_risk_probability']:.3f}")
            print(f"  High Risk Ratio: {zone['high_risk_ratio']:.2%}")
            print(f"  Sample Count: {zone['sample_count']}")
            print(f"  Average Speed: {zone['avg_speed']:.2f} m/s")
            print(f"  Average Congestion: {zone['avg_congestion']:.3f}")
            print(f"  Pedestrian Incidents: {zone['pedestrian_incidents']}")
        
        return high_risk_zones
    
    def analyze_intersection_risk(self, df):
        """Analyze risk at specific intersections"""
        print("\nAnalyzing intersection risk...")
        
        # Define intersection locations
        intersections = {
            'center': (0, 0),
            'north': (0, 200),
            'south': (0, -200),
            'east': (200, 0),
            'west': (-200, 0),
            'northeast': (200, 200)
        }
        
        intersection_stats = []
        
        for name, (ix, iy) in intersections.items():
            # Find data points near intersection (within 50m)
            near_intersection = df[
                (np.abs(df['x'] - ix) < 50) & 
                (np.abs(df['y'] - iy) < 50)
            ]
            
            if len(near_intersection) > 0:
                stats = {
                    'intersection': name,
                    'x': ix,
                    'y': iy,
                    'sample_count': len(near_intersection),
                    'avg_risk': near_intersection['risk_probability'].mean() if 'risk_probability' in near_intersection.columns else near_intersection['risk_score'].mean(),
                    'high_risk_count': (near_intersection['risk_probability'] > 0.7).sum() if 'risk_probability' in near_intersection.columns else 0,
                    'avg_speed': near_intersection['speed'].mean(),
                    'avg_waiting_time': near_intersection['waiting_time'].mean() if 'waiting_time' in near_intersection.columns else 0
                }
                
                intersection_stats.append(stats)
        
        
        intersection_df = pd.DataFrame(intersection_stats)
        
        # Only sort if we have data
        if len(intersection_df) > 0 and 'avg_risk' in intersection_df.columns:
            intersection_df = intersection_df.sort_values('avg_risk', ascending=False)
            
            print("\nIntersection Risk Ranking:")
            print("="*80)
            for idx, inter in intersection_df.iterrows():
                print(f"{inter['intersection']:12s}: Risk={inter['avg_risk']:.3f}, "
                      f"High-Risk Count={inter['high_risk_count']:4d}, "
                      f"Avg Speed={inter['avg_speed']:.2f} m/s")
        else:
            print("No intersection data available for risk analysis")
        
        return intersection_df
    
    def temporal_risk_analysis(self, df):
        """Analyze risk patterns over time"""
        print("\nAnalyzing temporal risk patterns...")
        
        # Create time bins (e.g., every 5 minutes)
        df['time_bin'] = (df['timestamp'] // 300).astype(int)  # 5-minute bins
        
        temporal_stats = df.groupby('time_bin').agg({
            'risk_probability': ['mean', 'max', 'std'],
            'vehicle_id': 'count',
            'speed': 'mean',
            'congestion_index': 'mean'
        }).reset_index()
        
        temporal_stats.columns = ['time_bin', 'avg_risk', 'max_risk', 'risk_std', 
                                  'vehicle_count', 'avg_speed', 'avg_congestion']
        
        temporal_stats['time_minutes'] = temporal_stats['time_bin'] * 5
        
        return temporal_stats
    
    def generate_report(self, output_dir='outputs/reports'):
        """Generate comprehensive risk analysis report"""
        os.makedirs(output_dir, exist_ok=True)
        
        report_path = os.path.join(output_dir, 'risk_zone_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("HIGH-RISK ZONE ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total Zones Analyzed: {len(self.zone_stats)}\n")
            f.write(f"Grid Size: {self.grid_size} meters\n\n")
            
            f.write("TOP 10 HIGH-RISK ZONES\n")
            f.write("-"*80 + "\n")
            
            for idx, zone in self.zone_stats.head(10).iterrows():
                f.write(f"\nZone {zone['zone_id']}\n")
                f.write(f"  Location: ({zone['center_x']:.1f}, {zone['center_y']:.1f})\n")
                f.write(f"  Average Risk: {zone['avg_risk_probability']:.3f}\n")
                f.write(f"  High Risk Ratio: {zone['high_risk_ratio']:.2%}\n")
                f.write(f"  Sample Count: {zone['sample_count']}\n")
                f.write(f"  Average Speed: {zone['avg_speed']:.2f} m/s\n")
                f.write(f"  Average Congestion: {zone['avg_congestion']:.3f}\n")
                f.write(f"  Pedestrian Incidents: {zone['pedestrian_incidents']}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("="*80 + "\n")
            
            # Generate recommendations based on top zones
            top_zone = self.zone_stats.iloc[0]
            f.write(f"\n1. Highest Risk Zone: {top_zone['zone_id']}\n")
            f.write(f"   - Consider additional traffic calming measures\n")
            f.write(f"   - Increase police presence during peak hours\n")
            
            if top_zone['avg_congestion'] > 0.5:
                f.write(f"   - High congestion detected - optimize signal timing\n")
            
            if top_zone['pedestrian_incidents'] > 10:
                f.write(f"   - High pedestrian activity - improve crosswalk visibility\n")
            
            f.write("\n2. General Recommendations:\n")
            f.write("   - Install speed cameras in top 5 high-risk zones\n")
            f.write("   - Conduct safety audits at identified intersections\n")
            f.write("   - Implement real-time warning systems\n")
        
        print(f"\nReport saved: {report_path}")
        
        return report_path
    
    def save_zone_data(self, output_dir='outputs/reports'):
        """Save zone statistics to CSV"""
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, 'zone_statistics.csv')
        self.zone_stats.to_csv(output_path, index=False)
        
        print(f"Zone statistics saved: {output_path}")
        
        return output_path


if __name__ == "__main__":
    print("Zone Analyzer Module - Use with prediction data")
