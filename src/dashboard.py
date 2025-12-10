"""
Real-time Accident Risk Monitoring Dashboard
Interactive Streamlit dashboard for visualization and monitoring
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import glob

from zone_analyzer import ZoneAnalyzer
from visualizer import RiskVisualizer
from predict_risk import load_latest_model, RiskPredictor


# Page configuration
st.set_page_config(
    page_title="Accident Risk Monitoring Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF4B4B;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(file_path):
    """Load data with caching"""
    return pd.read_csv(file_path)


@st.cache_resource
def load_predictor():
    """Load trained model with caching"""
    try:
        model_path, scaler_path, features_path = load_latest_model()
        predictor = RiskPredictor(model_path, scaler_path, features_path)
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def main():
    # Header
    st.markdown('<div class="main-header">🚦 Accident Risk Monitoring Dashboard</div>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data file selection
        data_dir = "data"
        if os.path.exists(data_dir):
            # Get all CSV files
            all_files = glob.glob(os.path.join(data_dir, "*.csv"))
            
            # Only show vehicle_data files (predictions are auto-generated)
            vehicle_files = [f for f in all_files if 'vehicle_data' in f.lower()]
            
            if vehicle_files:
                selected_file = st.selectbox(
                    "Select Data File",
                    vehicle_files,
                    format_func=lambda x: os.path.basename(x)
                )
            else:
                st.warning("No data files found. Run simulation first!")
                st.info("Run: `python src/run_simulation.py --test`")
                st.stop()
        else:
            st.error("Data directory not found!")
            st.stop()
        
        st.markdown("---")
        
        # Analysis options
        st.header("📊 Analysis Options")
        show_predictions = st.checkbox("Show Risk Predictions", value=True)
        show_zones = st.checkbox("Show High-Risk Zones", value=True)
        show_temporal = st.checkbox("Show Temporal Analysis", value=True)
        show_intersections = st.checkbox("Show Intersection Analysis", value=True)
        
        st.markdown("---")
        
        # Risk threshold
        risk_threshold = st.slider(
            "High Risk Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05
        )
        
        st.markdown("---")
        st.info("💡 **Tip**: Use the filters above to customize your analysis")
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data(selected_file)
    
    # Auto-predict if vehicle_data file is selected (not predictions file)
    if 'vehicle_data' in selected_file.lower() and 'risk_probability' not in df.columns:
        st.info("📊 Detected raw vehicle data - running ML predictions automatically...")
        
        with st.spinner("🧠 Running ML predictions... This may take a moment."):
            try:
                # Load predictor
                predictor = load_predictor()
                
                if predictor:
                    # Run predictions
                    df = predictor.predict_batch(df)
                    st.success("✅ ML predictions complete! Risk analysis is now available.")
                else:
                    st.error("❌ Could not load ML model. Using raw data without predictions.")
            except Exception as e:
                st.error(f"❌ Prediction error: {e}")
                st.info("Displaying raw data without risk predictions.")
    
    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", 
        "🗺️ Risk Zones", 
        "⏱️ Temporal Analysis", 
        "🔍 Detailed Analysis",
        "🛡️ Accident Prevention"
    ])
    
    # TAB 1: Overview
    with tab1:
        st.header("Overview Statistics")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Records",
                value=f"{len(df):,}",
                delta=None
            )
        
        with col2:
            unique_vehicles = df['vehicle_id'].nunique() if 'vehicle_id' in df.columns else len(df)
            st.metric(
                label="Unique Vehicles",
                value=f"{unique_vehicles:,}",
                delta=None
            )
        
        with col3:
            if 'risk_probability' in df.columns:
                avg_risk = df['risk_probability'].mean()
                st.metric(
                    label="Average Risk",
                    value=f"{avg_risk:.3f}",
                    delta=None
                )
            else:
                st.metric(label="Average Risk", value="N/A")
        
        with col4:
            if 'risk_probability' in df.columns:
                high_risk_count = (df['risk_probability'] > risk_threshold).sum()
                high_risk_pct = high_risk_count / len(df) * 100
                st.metric(
                    label="High Risk Events",
                    value=f"{high_risk_count:,}",
                    delta=f"{high_risk_pct:.1f}%"
                )
            else:
                st.metric(label="High Risk Events", value="N/A")
        
        st.markdown("---")
        
        # Data preview
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(100), use_container_width=True)
        
        # Basic statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Speed Distribution")
            if 'speed' in df.columns:
                fig_speed = px.histogram(
                    df, 
                    x='speed', 
                    nbins=50,
                    title='Vehicle Speed Distribution',
                    labels={'speed': 'Speed (m/s)', 'count': 'Frequency'}
                )
                fig_speed.update_traces(marker_color='lightblue')
                st.plotly_chart(fig_speed, use_container_width=True)
            else:
                st.info("Speed data not available in this file")
        
        with col2:
            st.subheader("🚗 Vehicle Type Distribution")
            if 'vehicle_type' in df.columns:
                vehicle_counts = df['vehicle_type'].value_counts()
                fig_veh = px.pie(
                    values=vehicle_counts.values,
                    names=vehicle_counts.index,
                    title='Vehicle Type Distribution'
                )
                st.plotly_chart(fig_veh, use_container_width=True)
    
    # TAB 2: Risk Zones
    with tab2:
        if show_zones and 'risk_probability' in df.columns:
            st.header("🗺️ High-Risk Zone Analysis")
            
            # Analyze zones
            with st.spinner("Analyzing risk zones..."):
                analyzer = ZoneAnalyzer()
                df_with_zones = analyzer.create_spatial_grid(df)
                zone_df = analyzer.calculate_zone_risk(df_with_zones)
            
            # Display top zones
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Risk Heatmap")
                
                # Interactive heatmap
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
                        colorbar=dict(title="Risk"),
                        line=dict(width=1, color='black')
                    ),
                    text=[f"Zone: {z}<br>Risk: {r:.3f}<br>Samples: {s}" 
                          for z, r, s in zip(zone_df['zone_id'], 
                                            zone_df['avg_risk_probability'],
                                            zone_df['sample_count'])],
                    hoverinfo='text'
                ))
                
                # Add intersections
                intersections = {
                    'Center': (0, 0), 'North': (0, 200), 'South': (0, -200),
                    'East': (200, 0), 'West': (-200, 0), 'Northeast': (200, 200)
                }
                
                for name, (x, y) in intersections.items():
                    fig.add_trace(go.Scatter(
                        x=[x], y=[y],
                        mode='markers+text',
                        marker=dict(size=15, color='blue', symbol='diamond'),
                        text=[name],
                        textposition='top center',
                        showlegend=False
                    ))
                
                fig.update_layout(
                    xaxis_title='X Position (m)',
                    yaxis_title='Y Position (m)',
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Top 10 High-Risk Zones")
                
                top_zones = zone_df.head(10)
                
                for idx, zone in top_zones.iterrows():
                    with st.expander(f"Zone {zone['zone_id']}", expanded=(idx == 0)):
                        st.metric("Risk Probability", f"{zone['avg_risk_probability']:.3f}")
                        st.metric("Sample Count", f"{zone['sample_count']:,}")
                        st.metric("Avg Speed", f"{zone['avg_speed']:.2f} m/s")
                        st.metric("Congestion", f"{zone['avg_congestion']:.3f}")
        else:
            st.info("Run predictions first to see risk zones")
    
    # TAB 3: Temporal Analysis
    with tab3:
        if show_temporal:
            st.header("⏱️ Temporal Risk Analysis")
            
            # Create time bins
            df['time_bin'] = (df['timestamp'] // 300).astype(int)
            
            temporal_stats = df.groupby('time_bin').agg({
                'vehicle_id': 'count',
                'speed': 'mean',
            }).reset_index()
            
            temporal_stats.columns = ['time_bin', 'vehicle_count', 'avg_speed']
            temporal_stats['time_minutes'] = temporal_stats['time_bin'] * 5
            
            if 'risk_probability' in df.columns:
                risk_temporal = df.groupby('time_bin')['risk_probability'].agg(['mean', 'max', 'std']).reset_index()
                temporal_stats = temporal_stats.merge(risk_temporal, on='time_bin')
            
            # Plot
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Risk Over Time', 'Traffic Volume Over Time'),
                vertical_spacing=0.15
            )
            
            if 'mean' in temporal_stats.columns:
                fig.add_trace(
                    go.Scatter(
                        x=temporal_stats['time_minutes'],
                        y=temporal_stats['mean'],
                        mode='lines',
                        name='Avg Risk',
                        line=dict(color='red', width=2)
                    ),
                    row=1, col=1
                )
            
            fig.add_trace(
                go.Bar(
                    x=temporal_stats['time_minutes'],
                    y=temporal_stats['vehicle_count'],
                    name='Vehicle Count',
                    marker_color='lightblue'
                ),
                row=2, col=1
            )
            
            fig.update_xaxes(title_text="Time (minutes)", row=2, col=1)
            fig.update_yaxes(title_text="Risk Probability", row=1, col=1)
            fig.update_yaxes(title_text="Vehicle Count", row=2, col=1)
            fig.update_layout(height=700, showlegend=True)
            
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 4: Detailed Analysis
    with tab4:
        st.header("🔍 Detailed Analysis")
        
        if show_intersections and 'risk_probability' in df.columns:
            st.subheader("Intersection Risk Analysis")
            
            analyzer = ZoneAnalyzer()
            intersection_df = analyzer.analyze_intersection_risk(df)
            
            # Only plot if we have intersection data
            if len(intersection_df) > 0 and 'avg_risk' in intersection_df.columns:
                # Plot
                fig = px.bar(
                    intersection_df,
                    x='intersection',
                    y='avg_risk',
                    color='avg_risk',
                    color_continuous_scale='YlOrRd',
                    title='Average Risk by Intersection'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Table
                st.dataframe(intersection_df, use_container_width=True)
            else:
                st.info("No intersection data available in the selected time range. Try selecting a different data file with more spatial coverage.")
        
        # Feature correlations
        st.subheader("Feature Correlations")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            corr_features = st.multiselect(
                "Select features for correlation analysis",
                numeric_cols,
                default=numeric_cols[:5]
            )
            
            if len(corr_features) > 1:
                corr_matrix = df[corr_features].corr()
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    aspect='auto',
                    color_continuous_scale='RdBu_r',
                    title='Feature Correlation Matrix'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # TAB 5: Accident Prevention
    with tab5:
        st.header("🛡️ Accident Prevention System")
        
        st.markdown("""
        This module demonstrates the effectiveness of ML-based accident prevention by comparing:
        - **BEFORE**: Simulation with aggressive accident scenarios (no prevention)
        - **AFTER**: Same scenarios but with ML-based intervention and prevention
        """)
        
        st.markdown("---")
        
        # Check for existing reports
        report_path = "outputs/reports/accident_prevention_report.txt"
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🎮 Simulation Controls")
            
            sim_duration = st.slider(
                "Simulation Duration (seconds)",
                min_value=100,
                max_value=2000,
                value=300,
                step=50
            )
            
            accident_freq = st.slider(
                "Accident Scenario Frequency (steps)",
                min_value=10,
                max_value=100,
                value=40,
                step=5,
                help="Lower = more accident scenarios"
            )
            
            skip_training = st.checkbox(
                "Skip Model Training (use existing model)",
                value=False,
                help="Uncheck to train a fresh model on BEFORE scenario data"
            )
            
            st.markdown("---")
            
            if st.button("🚀 Run Full Simulation", type="primary", use_container_width=True):
                st.info("Starting simulation... This will open SUMO GUI windows.")
                st.warning("⚠️ Please close this and run from command line for best experience:")
                
                cmd = f"python src/accident_prevention_sim.py --mode both --duration {sim_duration} --accident-freq {accident_freq}"
                if skip_training:
                    cmd += " --skip-training"
                
                st.code(cmd, language="bash")
                
                st.markdown("### Simulation Workflow:")
                st.markdown("""
                1. **STEP 1**: Run BEFORE scenario (accidents created, no prevention)
                2. **STEP 2**: Train ML model on collected data
                3. **STEP 3**: Run AFTER scenario (same triggers, ML prevention active)
                4. **STEP 4**: Generate comparison report
                """)
        
        with col2:
            st.subheader("📊 Latest Results")
            
            if os.path.exists(report_path):
                with open(report_path, 'r') as f:
                    report_content = f.read()
                
                # Parse results for visualization
                try:
                    lines = report_content.split('\n')
                    before_accidents = 0
                    after_accidents = 0
                    prevented = 0
                    interventions = 0
                    steps = 0
                    prevention_rate = 0
                    reduction_pct = 0
                    
                    # Track which section we're in
                    in_before_section = False
                    in_after_section = False
                    
                    for line in lines:
                        if 'SCENARIO 1' in line or 'WITHOUT' in line:
                            in_before_section = True
                            in_after_section = False
                        elif 'SCENARIO 2' in line or 'WITH ML' in line:
                            in_before_section = False
                            in_after_section = True
                        elif 'IMPACT ANALYSIS' in line:
                            in_before_section = False
                            in_after_section = False
                        
                        if 'Total Accidents:' in line:
                            val = int(line.split(':')[1].strip())
                            if in_before_section:
                                before_accidents = val
                            elif in_after_section:
                                after_accidents = val
                        elif 'Prevented Accidents:' in line:
                            prevented = int(line.split(':')[1].strip())
                        elif 'Total Interventions:' in line:
                            interventions = int(line.split(':')[1].strip())
                        elif 'Simulation Steps:' in line:
                            steps = int(line.split(':')[1].strip())
                        elif 'Prevention Rate:' in line:
                            prevention_rate = float(line.split(':')[1].strip().replace('%', ''))
                        elif 'Accident Reduction:' in line and '(' in line:
                            reduction_pct = float(line.split('(')[1].split('%')[0])
                    
                    # Stylish Card Layout
                    st.markdown("""
                    <style>
                    .report-card {
                        padding: 1.5rem;
                        border-radius: 12px;
                        margin-bottom: 1rem;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    }
                    .before-card {
                        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%);
                        color: white;
                    }
                    .after-card {
                        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
                        color: white;
                    }
                    .stats-card {
                        background: linear-gradient(135deg, #339af0 0%, #228be6 100%);
                        color: white;
                    }
                    .big-number {
                        font-size: 3rem;
                        font-weight: bold;
                        margin: 0;
                    }
                    .card-label {
                        font-size: 1rem;
                        opacity: 0.9;
                        margin-bottom: 0.5rem;
                    }
                    .improvement-badge {
                        background: #ffd43b;
                        color: #212529;
                        padding: 0.5rem 1rem;
                        border-radius: 20px;
                        font-weight: bold;
                        display: inline-block;
                        margin-top: 0.5rem;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # BEFORE Card
                    st.markdown(f"""
                    <div class="report-card before-card">
                        <div class="card-label">🚨 BEFORE (No Prevention)</div>
                        <p class="big-number">{before_accidents}</p>
                        <div>Total Accidents</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # AFTER Card
                    st.markdown(f"""
                    <div class="report-card after-card">
                        <div class="card-label">🛡️ AFTER (ML Prevention)</div>
                        <p class="big-number">{after_accidents}</p>
                        <div>Total Accidents</div>
                        <span class="improvement-badge">⬇️ {reduction_pct:.1f}% Reduction</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Stats Card
                    st.markdown(f"""
                    <div class="report-card stats-card">
                        <div class="card-label">📊 Prevention Statistics</div>
                        <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
                            <div style="text-align: center;">
                                <div style="font-size: 1.8rem; font-weight: bold;">{prevented}</div>
                                <div style="font-size: 0.85rem;">Prevented</div>
                            </div>
                            <div style="text-align: center;">
                                <div style="font-size: 1.8rem; font-weight: bold;">{interventions}</div>
                                <div style="font-size: 0.85rem;">Interventions</div>
                            </div>
                            <div style="text-align: center;">
                                <div style="font-size: 1.8rem; font-weight: bold;">{prevention_rate:.1f}%</div>
                                <div style="font-size: 0.85rem;">Success Rate</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Expandable raw report
                    with st.expander("📄 View Raw Report"):
                        st.code(report_content, language="text")
                    
                except Exception as e:
                    st.warning(f"Could not parse report: {e}")
                    st.text_area("Prevention Report", report_content, height=300)
            else:
                st.info("No simulation results yet. Run the simulation to see results.")
                st.markdown("### Quick Start:")
                st.code("python src/accident_prevention_sim.py --mode both --duration 300 --accident-freq 40", language="bash")
        
        st.markdown("---")
        
        # Prevention comparison visualization
        st.subheader("📈 Prevention Effectiveness Comparison")
        
        if os.path.exists(report_path):
            try:
                with open(report_path, 'r') as f:
                    report_content = f.read()
                
                # Parse for visualization
                lines = report_content.split('\n')
                before_accidents = 0
                after_accidents = 0
                prevented = 0
                interventions = 0
                
                # Track which section we're in
                in_before_section = False
                in_after_section = False
                
                for line in lines:
                    if 'SCENARIO 1' in line or 'WITHOUT' in line:
                        in_before_section = True
                        in_after_section = False
                    elif 'SCENARIO 2' in line or 'WITH ML' in line:
                        in_before_section = False
                        in_after_section = True
                    elif 'IMPACT ANALYSIS' in line:
                        in_before_section = False
                        in_after_section = False
                    
                    if 'Total Accidents:' in line:
                        val = int(line.split(':')[1].strip())
                        if in_before_section:
                            before_accidents = val
                        elif in_after_section:
                            after_accidents = val
                    elif 'Prevented Accidents:' in line:
                        prevented = int(line.split(':')[1].strip())
                    elif 'Total Interventions:' in line:
                        interventions = int(line.split(':')[1].strip())
                
                # Create comparison chart
                col1, col2 = st.columns(2)
                
                with col1:
                    # Bar chart comparison
                    fig_compare = go.Figure()
                    
                    fig_compare.add_trace(go.Bar(
                        name='Accidents',
                        x=['BEFORE (No Prevention)', 'AFTER (ML Prevention)'],
                        y=[before_accidents, after_accidents],
                        marker_color=['#FF4B4B', '#00CC96']
                    ))
                    
                    fig_compare.add_trace(go.Bar(
                        name='Prevented',
                        x=['BEFORE (No Prevention)', 'AFTER (ML Prevention)'],
                        y=[0, prevented],
                        marker_color=['#636EFA', '#636EFA']
                    ))
                    
                    fig_compare.update_layout(
                        title='Accident Comparison: Before vs After',
                        barmode='group',
                        yaxis_title='Count',
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_compare, use_container_width=True)
                
                with col2:
                    # Pie chart for prevention rate
                    if after_accidents + prevented > 0:
                        fig_pie = go.Figure(data=[go.Pie(
                            labels=['Accidents (Failed Prevention)', 'Successfully Prevented'],
                            values=[after_accidents, prevented],
                            marker_colors=['#FF4B4B', '#00CC96'],
                            hole=0.4
                        )])
                        
                        fig_pie.update_layout(
                            title='Prevention Success Rate',
                            annotations=[dict(text=f'{prevented/(after_accidents+prevented)*100:.0f}%', 
                                            x=0.5, y=0.5, font_size=20, showarrow=False)]
                        )
                        
                        st.plotly_chart(fig_pie, use_container_width=True)
                    else:
                        st.info("No prevention data available yet")
                
                # Improvement metrics
                st.subheader("📊 Key Metrics")
                
                m1, m2, m3, m4 = st.columns(4)
                
                with m1:
                    if before_accidents > 0:
                        reduction_pct = ((before_accidents - after_accidents) / before_accidents) * 100
                        st.metric("Accident Reduction", f"{reduction_pct:.1f}%")
                    else:
                        st.metric("Accident Reduction", "N/A")
                
                with m2:
                    if after_accidents + prevented > 0:
                        prevention_rate = (prevented / (after_accidents + prevented)) * 100
                        st.metric("Prevention Rate", f"{prevention_rate:.1f}%")
                    else:
                        st.metric("Prevention Rate", "N/A")
                
                with m3:
                    st.metric("Total Interventions", f"{interventions}")
                
                with m4:
                    if interventions > 0:
                        effectiveness = (prevented / interventions) * 100 if interventions > 0 else 0
                        st.metric("Intervention Effectiveness", f"{effectiveness:.1f}%")
                    else:
                        st.metric("Intervention Effectiveness", "N/A")
                
            except Exception as e:
                st.error(f"Error parsing report: {e}")
        else:
            # Show placeholder
            st.info("Run the accident prevention simulation to see comparison charts")
            
            # Show sample visualization with placeholder data
            fig_sample = go.Figure()
            fig_sample.add_trace(go.Bar(
                name='Expected Results (Sample)',
                x=['BEFORE', 'AFTER'],
                y=[10, 2],
                marker_color=['#FF4B4B', '#00CC96']
            ))
            fig_sample.update_layout(
                title='Sample: Expected Accident Reduction',
                yaxis_title='Accidents',
                annotations=[dict(text='Run simulation to see actual results', 
                                 xref='paper', yref='paper', x=0.5, y=0.5,
                                 showarrow=False, font_size=14)]
            )
            st.plotly_chart(fig_sample, use_container_width=True)
        
        st.markdown("---")
        
        # Model training section
        st.subheader("🧠 Model Training")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Training Workflow:**
            1. BEFORE scenario collects vehicle/accident data
            2. Data saved to `data/vehicle_data_YYYYMMDD_HHMMSS.csv`
            3. ML models trained: Random Forest, Gradient Boosting, Neural Network
            4. Models saved to `models/` directory
            5. AFTER scenario loads and uses the trained model
            """)
        
        with col2:
            # Check for trained models
            models_dir = "models"
            if os.path.exists(models_dir):
                # Look for .pkl (sklearn) and .h5 (keras) model files
                pkl_files = glob.glob(os.path.join(models_dir, "*.pkl"))
                h5_files = glob.glob(os.path.join(models_dir, "*.h5"))
                # Filter to only actual model files (not scalers or feature columns)
                model_files = [f for f in pkl_files if 'gradient_boosting' in f or 'random_forest' in f]
                model_files.extend(h5_files)
                
                if model_files:
                    st.success(f"✅ {len(model_files)} trained model(s) found")
                    for mf in model_files[:6]:
                        st.text(f"  • {os.path.basename(mf)}")
                else:
                    st.warning("No trained models found. Run simulation with training enabled.")
            else:
                st.warning("Models directory not found. Train models first.")


if __name__ == "__main__":
    main()
