"""
Complete Analysis Pipeline
Runs the entire workflow from data to visualizations
"""

import os
import sys
import yaml
import glob
import pandas as pd
from datetime import datetime

from feature_engineering import FeatureEngineer
from train_model import AccidentRiskModel
from predict_risk import RiskPredictor, load_latest_model
from zone_analyzer import ZoneAnalyzer
from visualizer import RiskVisualizer


class AnalysisPipeline:
    """Complete end-to-end analysis pipeline"""
    
    def __init__(self, config_path='config.yaml'):
        """Initialize pipeline"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_file = None
        self.predictions_file = None
        self.results = {}
    
    def run_complete_analysis(self, data_file):
        """Run complete analysis pipeline"""
        print("\n" + "="*80)
        print("COMPLETE ACCIDENT RISK ANALYSIS PIPELINE")
        print("="*80)
        
        self.data_file = data_file
        
        # Step 1: Feature Engineering and Model Training
        print("\n" + "="*80)
        print("STEP 1: FEATURE ENGINEERING & MODEL TRAINING")
        print("="*80)
        
        trainer = AccidentRiskModel()
        self.results['training'] = trainer.train_all_models(data_file)
        
        # Step 2: Make Predictions
        print("\n" + "="*80)
        print("STEP 2: RISK PREDICTION")
        print("="*80)
        
        model_path, scaler_path, features_path = load_latest_model()
        predictor = RiskPredictor(model_path, scaler_path, features_path)
        
        df_predictions = predictor.predict_batch(data_file)
        
        # Save predictions
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.predictions_file = f'data/predictions_{timestamp}.csv'
        predictor.save_predictions(df_predictions, self.predictions_file)
        
        # Step 3: Zone Analysis
        print("\n" + "="*80)
        print("STEP 3: HIGH-RISK ZONE ANALYSIS")
        print("="*80)
        
        analyzer = ZoneAnalyzer()
        df_with_zones = analyzer.create_spatial_grid(df_predictions)
        zone_df = analyzer.calculate_zone_risk(df_with_zones)
        
        # Identify high-risk zones
        high_risk_zones = analyzer.identify_high_risk_zones()
        
        # Intersection analysis
        intersection_df = analyzer.analyze_intersection_risk(df_with_zones)
        
        # Temporal analysis
        temporal_df = analyzer.temporal_risk_analysis(df_with_zones)
        
        # Generate report
        report_path = analyzer.generate_report()
        analyzer.save_zone_data()
        
        self.results['zones'] = {
            'zone_df': zone_df,
            'high_risk_zones': high_risk_zones,
            'intersection_df': intersection_df,
            'temporal_df': temporal_df
        }
        
        # Step 4: Visualizations
        print("\n" + "="*80)
        print("STEP 4: GENERATING VISUALIZATIONS")
        print("="*80)
        
        visualizer = RiskVisualizer()
        
        # Risk heatmap
        visualizer.plot_risk_heatmap(zone_df)
        visualizer.plot_interactive_heatmap(zone_df)
        
        # Temporal analysis
        visualizer.plot_temporal_risk(temporal_df)
        
        # Intersection comparison
        visualizer.plot_intersection_comparison(intersection_df)
        
        # Model comparison
        visualizer.plot_model_comparison(self.results['training'])
        visualizer.plot_confusion_matrices(self.results['training'])
        
        # Dashboard summary
        visualizer.create_dashboard_summary(zone_df, intersection_df, temporal_df)
        
        # Final Summary
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        
        print(f"\n📊 Results Summary:")
        print(f"  - Input Data: {data_file}")
        print(f"  - Predictions: {self.predictions_file}")
        print(f"  - Total Records: {len(df_predictions):,}")
        print(f"  - High-Risk Events: {(df_predictions['risk_probability'] > 0.7).sum():,}")
        print(f"  - High-Risk Zones: {len(high_risk_zones)}")
        print(f"  - Report: {report_path}")
        
        print(f"\n📈 Model Performance:")
        for model_name, metrics in self.results['training'].items():
            print(f"  {model_name}:")
            print(f"    - Accuracy: {metrics['accuracy']:.4f}")
            print(f"    - F1 Score: {metrics['f1']:.4f}")
            print(f"    - ROC-AUC: {metrics['roc_auc']:.4f}")
        
        print(f"\n🗺️ Top 3 High-Risk Zones:")
        for idx, zone in high_risk_zones.head(3).iterrows():
            print(f"  {idx+1}. Zone {zone['zone_id']}: Risk={zone['avg_risk_probability']:.3f}")
        
        print(f"\n🚦 Highest Risk Intersection:")
        top_intersection = intersection_df.iloc[0]
        print(f"  {top_intersection['intersection']}: Risk={top_intersection['avg_risk']:.3f}")
        
        print(f"\n📁 Outputs:")
        print(f"  - Models: models/")
        print(f"  - Visualizations: outputs/visualizations/")
        print(f"  - Reports: outputs/reports/")
        
        print(f"\n🚀 Next Steps:")
        print(f"  1. View interactive dashboard: streamlit run src/dashboard.py")
        print(f"  2. Review visualizations in outputs/visualizations/")
        print(f"  3. Read detailed report in outputs/reports/risk_zone_report.txt")
        
        return self.results


def find_latest_data_file(data_dir='data'):
    """Find the most recent data file"""
    data_files = glob.glob(os.path.join(data_dir, 'vehicle_data_*.csv'))
    
    if not data_files:
        # Try merged data
        merged_files = glob.glob(os.path.join(data_dir, 'merged_data.csv'))
        if merged_files:
            return merged_files[0]
        else:
            raise FileNotFoundError("No data files found. Run simulation first!")
    
    # Sort by timestamp and return latest
    data_files.sort(reverse=True)
    return data_files[0]


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run complete analysis pipeline')
    parser.add_argument('--data', type=str, default=None, 
                       help='Path to data file (default: latest in data/)')
    
    args = parser.parse_args()
    
    # Find data file
    if args.data:
        data_file = args.data
    else:
        try:
            data_file = find_latest_data_file()
            print(f"Using latest data file: {data_file}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("\nPlease run simulation first:")
            print("  python src/run_simulation.py --test")
            sys.exit(1)
    
    # Check if file exists
    if not os.path.exists(data_file):
        print(f"Error: Data file not found: {data_file}")
        sys.exit(1)
    
    # Run pipeline
    pipeline = AnalysisPipeline()
    results = pipeline.run_complete_analysis(data_file)
