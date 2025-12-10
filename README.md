# Machine Learning-Based Road Accident Risk Prediction & Prevention System

A comprehensive system that uses **SUMO traffic simulation** to generate realistic traffic data, applies **machine learning** to predict accident risks, identifies high-risk zones, and implements intelligent accident prevention strategies.

---

## Module 1: Machine Learning-Based Accident Prediction

### Project Description
This module develops an ML-based accident risk prediction system using SUMO traffic simulation and TraCI for real-time data collection. It trains Random Forest, XGBoost, and Neural Network models to classify accident risk based on features like speed, acceleration, TTC, lane changes, and congestion.

### My Contribution
I implemented the ML pipeline including TraCI-based data collection, feature engineering (rolling statistics, interaction features), multi-model training framework, and real-time risk prediction module.

### Model Description
The system uses a modular architecture consisting of a SUMO Simulation Layer that provides the traffic network with roads, intersections, and mixed vehicles, a TraCI Data Collector for real-time vehicle feature extraction (speed, TTC, acceleration), a Feature Engineering Pipeline that creates rolling statistics, interaction terms, and risk labeling, an ML Model Trainer that trains Random Forest, XGBoost, and Neural Network classifiers, and a Risk Predictor that performs batch and real-time risk probability scoring.

---

## Module 2: Intelligent Accident Prevention & High-Risk Zone Identification

### Project Description
This module develops an intelligent prevention system that identifies high-risk zones using spatial grid analysis and applies ML-based interventions to prevent accidents. It includes before/after simulation comparisons and an interactive Streamlit dashboard with risk heatmaps.

### My Contribution
I implemented the accident prevention simulation with safe intervention logic, zone analysis module for spatial risk profiling, and the interactive dashboard for real-time monitoring and visualization.

### Model Description
The system uses a modular architecture consisting of an ML-Based Risk Monitoring Layer that performs continuous real-time prediction using trained models, a High-Risk Zone Identification Engine that uses spatial grid aggregation and intersection-level risk analysis, an Intelligent Intervention System that applies safe, graduated interventions such as speed reduction and lane advisories based on risk thresholds, and an Interactive Visualization Dashboard that displays risk heatmaps, temporal trends, before/after comparison reports, and zone statistics.

---

## 🎯 Features

- **SUMO Traffic Simulation**: Realistic downtown network with 8 roads, 5 signalized intersections, and mixed traffic
- **Data Collection**: Real-time feature extraction via TraCI (speed, acceleration, TTC, lane changes, congestion)
- **Machine Learning Models**: Random Forest, XGBoost, Neural Network
- **Risk Prediction**: Real-time accident risk scoring (0-1 probability)
- **High-Risk Zone Identification**: Spatial heatmaps and intersection-level risk rankings
- **Accident Prevention**: ML-based interventions with before/after comparison
- **Interactive Dashboard**: Streamlit-based real-time monitoring

## 📋 Requirements

### Software Dependencies
- **SUMO** (Simulation of Urban MObility) - [Download](https://sumo.dlr.de/docs/Downloads.php)
- **Python 3.8+**

### Python Packages
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Generate SUMO Network
```bash
cd sumo_network
generate_network.bat
```

### 2. Run Test Simulation
```bash
python src/run_simulation.py --test
```

### 3. Train ML Models
```bash
python src/train_model.py data/vehicle_data_TIMESTAMP.csv
```

### 4. Make Risk Predictions
```bash
python src/predict_risk.py data/vehicle_data_TIMESTAMP.csv predictions.csv
```

### 5. Run Accident Prevention Simulation
```bash
python src/accident_prevention_sim.py
```

### 6. Launch Dashboard
```bash
streamlit run src/dashboard.py
```

## 📁 Project Structure

```
FDS/
├── sumo_network/              # SUMO network files
├── src/                       # Source code
│   ├── data_collector.py      # TraCI-based data collection
│   ├── feature_engineering.py # Feature creation and risk labeling
│   ├── train_model.py         # ML model training
│   ├── predict_risk.py        # Risk prediction
│   ├── zone_analyzer.py       # High-risk zone identification
│   ├── accident_prevention_sim.py # Prevention simulation
│   └── dashboard.py           # Streamlit dashboard
├── data/                      # Collected simulation data
├── models/                    # Trained ML models
├── outputs/                   # Analysis outputs
├── config.yaml                # System configuration
└── requirements.txt           # Python dependencies
```

## 📈 Model Performance

- **Random Forest**: Excellent interpretability with feature importance
- **XGBoost**: High accuracy with gradient boosting
- **Neural Network**: Captures complex non-linear patterns

Evaluation metrics: Accuracy, Precision, Recall, F1 Score, ROC-AUC

---

**Built with**: SUMO, Python, scikit-learn, XGBoost, TensorFlow, Streamlit, Plotly
