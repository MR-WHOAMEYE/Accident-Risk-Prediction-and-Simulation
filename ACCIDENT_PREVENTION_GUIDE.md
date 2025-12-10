# Accident Prevention Simulation - Quick Start Guide

## Overview

This enhanced system demonstrates **before and after** accident prevention using ML predictions:

1. **BEFORE**: Simulates traffic with actual collision scenarios
2. **AFTER**: Uses ML to detect high-risk situations and apply interventions to prevent accidents

## Quick Start

### Run Both Scenarios (Recommended)

```bash
python src/accident_prevention_sim.py --mode both --duration 600
```

This will:
1. Run 10-minute simulation WITHOUT prevention (shows accidents)
2. Run 10-minute simulation WITH ML-based prevention
3. Generate comparison report

### Run Individual Scenarios

**Before Only** (No Prevention):
```bash
python src/accident_prevention_sim.py --mode before --duration 300
```

**After Only** (With Prevention):
```bash
python src/accident_prevention_sim.py --mode after --duration 300
```

## How It Works

### Accident Scenarios

The system introduces realistic accident scenarios:
- **Rear-end collisions**: Sudden braking
- **Intersection accidents**: Running red lights
- **Lane-change accidents**: Aggressive lane switching

### ML-Based Interventions

When high risk is detected (risk > 0.5):
- **Critical Risk (>0.9)**: Emergency braking (70% speed reduction)
- **High Risk (>0.7)**: Moderate braking (40% speed reduction)
- **Medium Risk (>0.5)**: Visual warning (orange color)

### Risk Calculation

Real-time risk based on:
- Vehicle speed
- Sudden deceleration
- Following distance (Time to Collision)
- Waiting time (congestion)

## Expected Results

**BEFORE Scenario**:
- Multiple accidents occur
- No interventions
- High accident rate

**AFTER Scenario**:
- Accidents prevented through interventions
- Real-time risk monitoring
- Significant accident reduction (50-80%)

## Output Files

- `outputs/reports/accident_prevention_report.txt` - Detailed comparison
- `data/vehicle_data_*.csv` - Simulation data
- Console output shows real-time accidents and interventions

## Advanced Options

```bash
# Longer simulation with more frequent accidents
python src/accident_prevention_sim.py --mode both --duration 1200 --accident-freq 30

# Quick test (5 minutes)
python src/accident_prevention_sim.py --mode both --duration 300 --accident-freq 40
```

## Visualization

The simulation runs with **SUMO GUI** so you can:
- Watch accidents happen in real-time
- See intervention warnings (orange vehicles)
- Observe traffic flow changes

## Integration with Dashboard

After running the simulation:
```bash
streamlit run src/dashboard.py
```

Select the generated data file to analyze:
- Risk patterns before accidents
- Intervention effectiveness
- Spatial distribution of prevented accidents

---

**Note**: Make sure you have trained ML models before running the AFTER scenario:
```bash
python src/train_model.py data/vehicle_data_*.csv
```
