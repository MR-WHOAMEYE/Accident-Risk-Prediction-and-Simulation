"""
Visualization Module
Creates comprehensive visualizations for risk analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml
import os


class RiskVisualizer:
    """Create visualizations for accident risk analysis"""
    
    def __init__(self, config_path='config.yaml'):
        """Initialize visualizer"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = self.config['visualization']['figure_size']
    
    def plot_risk_heatmap(self, zone_df, output_dir='outputs/visualizations'):
        """Create spatial heatmap of risk zones"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("Creating risk heatmap...")
        
        # Prepare data for heatmap
        pivot_data = zone_df.pivot_table(
            values='avg_risk_probability',
            index='grid_y',
            columns='grid_x',
            fill_value=0
        )
        
        # Create heatmap
        plt.figure(figsize=(14, 10))
        sns.heatmap(
            pivot_data,
            cmap=self.config['visualization']['colormap'],
            annot=False,
            cbar_kws={'label': 'Average Risk Probability'},
            vmin=0,
            vmax=1
        )
        
        plt.title('Spatial Risk Heatmap - Downtown Network', fontsize=16, fontweight='bold')
        plt.xlabel('Grid X', fontsize=12)
        plt.ylabel('Grid Y', fontsize=12)
        plt.tight_layout()
        
        filename = os.path.join(output_dir, 'risk_heatmap.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Heatmap saved: {filename}")
        
        return filename
    
    def plot_interactive_heatmap(self, zone_df, output_dir='outputs/visualizations'):
        """Create interactive heatmap using Plotly"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("Creating interactive heatmap...")
        
        # Create scatter plot with size based on risk
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=zone_df['center_x'],
            y=zone_df['center_y'],
            mode='markers',
            marker=dict(
                size=zone_df['avg_risk_probability'] * 50,
                color=zone_df['avg_risk_probability'],
                colorscale='YlOrRd',
                showscale=True,
                colorbar=dict(title="Risk Probability"),
                line=dict(width=1, color='black')
            ),
            text=[f"Zone: {z}<br>Risk: {r:.3f}<br>Samples: {s}" 
                  for z, r, s in zip(zone_df['zone_id'], 
                                    zone_df['avg_risk_probability'],
                                    zone_df['sample_count'])],
            hoverinfo='text'
        ))
        
        # Add intersection markers
        intersections = {
            'Center': (0, 0),
            'North': (0, 200),
            'South': (0, -200),
            'East': (200, 0),
            'West': (-200, 0),
            'Northeast': (200, 200)
        }
        
        for name, (x, y) in intersections.items():
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode='markers+text',
                marker=dict(size=15, color='blue', symbol='diamond'),
                text=[name],
                textposition='top center',
                name=name,
                showlegend=False
            ))
        
        fig.update_layout(
            title='Interactive Risk Heatmap - Downtown Network',
            xaxis_title='X Position (m)',
            yaxis_title='Y Position (m)',
            hovermode='closest',
            width=1000,
            height=800
        )
        
        filename = os.path.join(output_dir, 'interactive_heatmap.html')
        fig.write_html(filename)
        
        print(f"Interactive heatmap saved: {filename}")
        
        return filename
    
    def plot_temporal_risk(self, temporal_df, output_dir='outputs/visualizations'):
        """Plot risk trends over time"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("Creating temporal risk plot...")
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Risk over time
        axes[0].plot(temporal_df['time_minutes'], temporal_df['avg_risk'], 
                    label='Average Risk', linewidth=2, color='red')
        axes[0].fill_between(temporal_df['time_minutes'], 
                            temporal_df['avg_risk'] - temporal_df['risk_std'],
                            temporal_df['avg_risk'] + temporal_df['risk_std'],
                            alpha=0.3, color='red')
        axes[0].set_xlabel('Time (minutes)', fontsize=12)
        axes[0].set_ylabel('Risk Probability', fontsize=12)
        axes[0].set_title('Risk Probability Over Time', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Vehicle count and congestion
        ax2 = axes[1].twinx()
        axes[1].bar(temporal_df['time_minutes'], temporal_df['vehicle_count'], 
                   alpha=0.6, label='Vehicle Count', color='blue')
        ax2.plot(temporal_df['time_minutes'], temporal_df['avg_congestion'], 
                label='Avg Congestion', linewidth=2, color='orange')
        
        axes[1].set_xlabel('Time (minutes)', fontsize=12)
        axes[1].set_ylabel('Vehicle Count', fontsize=12, color='blue')
        ax2.set_ylabel('Congestion Index', fontsize=12, color='orange')
        axes[1].set_title('Traffic Volume and Congestion Over Time', fontsize=14, fontweight='bold')
        axes[1].legend(loc='upper left')
        ax2.legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = os.path.join(output_dir, 'temporal_risk.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Temporal risk plot saved: {filename}")
        
        return filename
    
    def plot_intersection_comparison(self, intersection_df, output_dir='outputs/visualizations'):
        """Compare risk across intersections"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("Creating intersection comparison...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Risk by intersection
        sns.barplot(data=intersection_df, x='intersection', y='avg_risk', ax=axes[0], palette='YlOrRd')
        axes[0].set_title('Average Risk by Intersection', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Intersection', fontsize=12)
        axes[0].set_ylabel('Average Risk Probability', fontsize=12)
        axes[0].tick_params(axis='x', rotation=45)
        
        # High-risk count by intersection
        sns.barplot(data=intersection_df, x='intersection', y='high_risk_count', ax=axes[1], palette='Reds')
        axes[1].set_title('High-Risk Incidents by Intersection', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Intersection', fontsize=12)
        axes[1].set_ylabel('High-Risk Count', fontsize=12)
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        filename = os.path.join(output_dir, 'intersection_comparison.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Intersection comparison saved: {filename}")
        
        return filename
    
    def plot_model_comparison(self, results_dict, output_dir='outputs/visualizations'):
        """Compare performance of different models"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("Creating model comparison...")
        
        # Prepare data
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        models = list(results_dict.keys())
        
        data = []
        for model in models:
            for metric in metrics:
                data.append({
                    'Model': model,
                    'Metric': metric.upper().replace('_', ' '),
                    'Score': results_dict[model][metric]
                })
        
        df = pd.DataFrame(data)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='Metric', y='Score', hue='Model', palette='Set2')
        plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.ylabel('Score', fontsize=12)
        plt.xlabel('Metric', fontsize=12)
        plt.ylim(0, 1.1)
        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        filename = os.path.join(output_dir, 'model_comparison.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Model comparison saved: {filename}")
        
        return filename
    
    def plot_confusion_matrices(self, results_dict, output_dir='outputs/visualizations'):
        """Plot confusion matrices for all models"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("Creating confusion matrices...")
        
        n_models = len(results_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, results) in enumerate(results_dict.items()):
            cm = results['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['Low Risk', 'High Risk'],
                       yticklabels=['Low Risk', 'High Risk'])
            axes[idx].set_title(f'{model_name}\nConfusion Matrix', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True Label', fontsize=10)
            axes[idx].set_xlabel('Predicted Label', fontsize=10)
        
        plt.tight_layout()
        
        filename = os.path.join(output_dir, 'confusion_matrices.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrices saved: {filename}")
        
        return filename
    
    def create_dashboard_summary(self, zone_df, intersection_df, temporal_df, 
                                 output_dir='outputs/visualizations'):
        """Create comprehensive dashboard summary"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("Creating dashboard summary...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Risk Heatmap', 'Intersection Risk', 
                          'Temporal Risk Trend', 'Top High-Risk Zones'),
            specs=[[{'type': 'scatter'}, {'type': 'bar'}],
                   [{'type': 'scatter'}, {'type': 'bar'}]]
        )
        
        # 1. Risk heatmap (scatter)
        fig.add_trace(
            go.Scatter(
                x=zone_df['center_x'],
                y=zone_df['center_y'],
                mode='markers',
                marker=dict(
                    size=zone_df['avg_risk_probability'] * 30,
                    color=zone_df['avg_risk_probability'],
                    colorscale='YlOrRd',
                    showscale=True
                ),
                name='Zones'
            ),
            row=1, col=1
        )
        
        # 2. Intersection risk
        fig.add_trace(
            go.Bar(
                x=intersection_df['intersection'],
                y=intersection_df['avg_risk'],
                marker_color='indianred',
                name='Risk'
            ),
            row=1, col=2
        )
        
        # 3. Temporal trend
        fig.add_trace(
            go.Scatter(
                x=temporal_df['time_minutes'],
                y=temporal_df['avg_risk'],
                mode='lines',
                line=dict(color='red', width=2),
                name='Avg Risk'
            ),
            row=2, col=1
        )
        
        # 4. Top zones
        top_zones = zone_df.head(10)
        fig.add_trace(
            go.Bar(
                x=top_zones['zone_id'],
                y=top_zones['avg_risk_probability'],
                marker_color='crimson',
                name='Risk'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=900,
            showlegend=False,
            title_text="Accident Risk Analysis Dashboard",
            title_font_size=20
        )
        
        filename = os.path.join(output_dir, 'dashboard_summary.html')
        fig.write_html(filename)
        
        print(f"Dashboard summary saved: {filename}")
        
        return filename


if __name__ == "__main__":
    print("Visualization Module - Use with analyzed data")
