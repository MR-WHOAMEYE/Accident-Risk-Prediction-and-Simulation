# 🚗 Machine Learning-Based Road Accident Risk Prediction & Prevention System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![SUMO](https://img.shields.io/badge/SUMO-Traffic%20Simulation-green.svg)](https://sumo.dlr.de)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-Educational-yellow.svg)]()

A comprehensive AI-powered system that uses **SUMO traffic simulation** to generate realistic traffic data, applies **machine learning** to predict accident risks, identifies high-risk zones, and implements intelligent accident prevention strategies.

---

## 📊 Module 1: Machine Learning-Based Accident Prediction

### Project Description
This module develops an ML-based accident risk prediction system using SUMO traffic simulation and TraCI for real-time data collection. It trains Random Forest, XGBoost, and Neural Network models to classify accident risk based on features like speed, acceleration, Time-to-Collision (TTC), lane changes, and congestion levels.

### Contributor
**Akshaykumar-B** - [GitHub Profile](https://github.com/Akshaykumar-B)

### Contribution
Implemented the core ML pipeline including TraCI-based data collection, feature engineering (rolling statistics, interaction features), multi-model training framework, and real-time risk prediction module.

### Model Architecture
The system uses a modular architecture consisting of a SUMO Simulation Layer that provides the traffic network with roads, intersections, and mixed vehicles, a TraCI Data Collector for real-time vehicle feature extraction (speed, TTC, acceleration), a Feature Engineering Pipeline that creates rolling statistics, interaction terms, and risk labeling, an ML Model Trainer that trains Random Forest, XGBoost, and Neural Network classifiers, and a Risk Predictor that performs batch and real-time risk probability scoring.

**Key Components:**
| Component | Description |
|-----------|-------------|
| `data_collector.py` | TraCI-based real-time feature extraction |
| `feature_engineering.py` | Rolling statistics, interaction features, risk labeling |
| `train_model.py` | Multi-model training (RF, XGBoost, NN) |
| `predict_risk.py` | Batch and real-time risk scoring |
| `dashboard.py` | Streamlit visualization dashboard |
| `visualizer.py` | Risk analysis visualization tools |

---

## 🛡️ Module 2: Intelligent Accident Prevention & High-Risk Zone Identification

### Project Description
This module develops an intelligent prevention system that identifies high-risk zones using spatial grid analysis and applies ML-based interventions to prevent accidents. It includes before/after simulation comparisons and an interactive Streamlit dashboard with risk heatmaps, temporal trends, and comparison reports.

### Contributor
**MR-WHOAMEYE** - [GitHub Profile](https://github.com/MR-WHOAMEYE)

### Contribution
Implemented the accident prevention simulation with safe intervention logic, zone analysis module for spatial risk profiling, and the interactive dashboard for real-time monitoring and visualization.

### Model Architecture
The system uses a modular architecture consisting of an ML-Based Risk Monitoring Layer that performs continuous real-time prediction using trained models, a High-Risk Zone Identification Engine that uses spatial grid aggregation and intersection-level risk analysis, an Intelligent Intervention System that applies safe, graduated interventions such as speed reduction and lane advisories based on risk thresholds, and an Interactive Visualization Dashboard that displays risk heatmaps, temporal trends, before/after comparison reports, and zone statistics.

**Key Components:**
| Component | Description |
|-----------|-------------|
| `zone_analyzer.py` | Spatial grid analysis, intersection risk profiling |
| `accident_prevention_sim.py` | Before/after scenario simulation with interventions |
| `dashboard.py` | Interactive risk monitoring dashboard |

---

## 🎯 Features

- 🚦 **SUMO Traffic Simulation** - Realistic downtown network with 8 roads, 5 signalized intersections, and mixed traffic
- 📡 **Real-time Data Collection** - Feature extraction via TraCI (speed, acceleration, TTC, lane changes)
- 🤖 **Machine Learning Models** - Random Forest, XGBoost, Neural Network ensemble
- 📈 **Risk Prediction** - Real-time accident risk scoring (0-1 probability)
- 🗺️ **High-Risk Zone Identification** - Spatial heatmaps and intersection-level rankings
- 🛑 **Accident Prevention** - ML-based interventions with before/after comparison
- 📊 **Interactive Dashboard** - Streamlit-based real-time monitoring

---

## 📋 Requirements

### Software
- **SUMO** (Simulation of Urban MObility) - [Download](https://sumo.dlr.de/docs/Downloads.php)
- **Python 3.8+**

### Installation
```bash
pip install -r requirements.txt
```

---

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

---

## 📁 Project Structure

```
FDS/
├── 📂 sumo_network/           # SUMO network configuration
│   ├── downtown.nod.xml       # Node definitions
│   ├── downtown.edg.xml       # Edge definitions
│   ├── downtown.rou.xml       # Routes and vehicle flows
│   └── downtown.sumocfg       # SUMO configuration
│
├── 📂 src/                    # Source code
│   ├── data_collector.py      # TraCI data collection
│   ├── feature_engineering.py # Feature creation
│   ├── train_model.py         # ML model training
│   ├── predict_risk.py        # Risk prediction
│   ├── zone_analyzer.py       # Zone analysis
│   ├── accident_prevention_sim.py # Prevention simulation
│   ├── dashboard.py           # Streamlit dashboard
│   └── visualizer.py          # Visualization tools
│
├── 📂 data/                   # Simulation data
├── 📂 models/                 # Trained ML models
├── 📂 outputs/                # Analysis outputs
├── config.yaml                # Configuration
└── requirements.txt           # Dependencies
```

---

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | High | High | Medium | High |
| XGBoost | High | High | High | High |
| Neural Network | High | Medium | High | High |

---

## 🔍 Risk Factors Analyzed

- 🚗 **Speed Variance** - Sudden acceleration/deceleration
- ⏱️ **Time to Collision (TTC)** - Distance to leading vehicle
- 🔄 **Lane Changes** - Frequent lane switching
- 🚶 **Pedestrian Proximity** - Nearby pedestrian activity
- 🚦 **Traffic Signal Conflicts** - Red light approach at high speed
- 📍 **Intersection Proximity** - Distance to intersections
- 🚌 **Vehicle Type** - Different risk profiles for cars, buses, bikes

---

## 👥 TEAM MEMBERS

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/Akshaykumar-B">
        <img src="https://github.com/Akshaykumar-B.png" width="100px;" alt="Akshaykumar-B"/>
        <br /><sub><b>Akshaykumar-B</b></sub>
      </a>
      <br />Module 1: ML Prediction
    </td>
    <td align="center">
      <a href="https://github.com/MR-WHOAMEYE">
        <img src="https://github.com/MR-WHOAMEYE.png" width="100px;" alt="MR-WHOAMEYE"/>
        <br /><sub><b>MR-WHOAMEYE</b></sub>
      </a>
      <br />Module 2: Prevention & Zones
    </td>
  </tr>
</table>

---

## 📧 Support

For questions or issues:
- SUMO Documentation: https://sumo.dlr.de/docs/
- Check `config.yaml` for settings
- Review console output for errors

---

**Built with:** SUMO, Python, scikit-learn, XGBoost, TensorFlow, Streamlit, and Plotly
