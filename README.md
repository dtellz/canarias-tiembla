# Canary Islands Earthquake Monitor & Eruption Prediction

A real-time earthquake monitoring dashboard for the Canary Islands with an experimental machine learning system for volcanic eruption prediction.

**Live Demo**: https://dtellz.github.io/canarias-tiembla/

## Features

### Real-Time Monitoring Dashboard
- **Live Data**: Fetches real-time earthquake data from IGN Spain every 2 minutes
- **Interactive Map**: Dark-themed map focused on the Canary Islands with animated earthquake markers
- **Magnitude Visualization**: Color-coded markers and size scaling based on earthquake magnitude
- **Statistics Dashboard**: Total events, max magnitude, average depth, and recent activity
- **Time Filtering**: View earthquakes from the last 24 hours, 7 days, or 30 days
- **Earthquake List**: Scrollable list of recent earthquakes with quick details
- **Magnitude Distribution Chart**: Visual breakdown of earthquake magnitudes
- **Responsive Design**: Works on desktop and mobile devices

### Eruption Prediction Model (Experimental)
- **Temporal Convolutional Network (TCN)**: Deep learning model for time-series analysis
- **Multi-modal Data**: Combines seismicity, ground deformation, and gas emissions
- **Multiple Prediction Windows**: 7, 30, and 90-day eruption forecasts
- **Apple Silicon Optimized**: Runs on M-series Macs using PyTorch MPS backend
- **XGBoost Baseline**: Gradient boosting model for comparison

## Project Structure

```
├── index.html              # Real-time monitoring dashboard
├── app.js                  # Dashboard JavaScript
├── styles.css              # Dashboard styling
├── scripts/                # Data pipeline scripts
│   ├── download_seismic.py    # Fetch IGN earthquake data
│   ├── download_eruptions.py  # Historical eruption records
│   ├── process_seismic.py     # Feature engineering
│   ├── build_dataset.py       # Create ML training data
│   └── predict.py             # Run inference
├── models/                 # ML model implementations
│   ├── tcn_model.py           # TCN, LSTM, Attention models
│   └── xgboost_baseline.py    # XGBoost baseline
├── training/               # Training scripts
│   ├── train_model.py         # Model training
│   └── evaluate_model.py      # Evaluation & plots
└── data/                   # Data directories
    ├── raw/                   # Raw downloaded data
    ├── processed/             # Processed datasets
    └── features/              # Engineered features
```

## Quick Start

### Dashboard Only
```bash
# Open index.html in your browser, or serve locally:
python -m http.server 8000
```

### ML Pipeline (requires Python 3.12+)
```bash
# Setup environment
python3.12 -m venv volcano-env
source volcano-env/bin/activate
pip install -r requirements.txt

# Run pipeline
python scripts/download_eruptions.py
python scripts/download_seismic.py
python scripts/process_seismic.py
python scripts/build_dataset.py
python training/train_model.py
python scripts/predict.py
```

## Tech Stack

**Dashboard**
- HTML5, CSS3, Vanilla JavaScript
- Leaflet.js for interactive mapping

**ML Pipeline**
- PyTorch (MPS backend for Apple Silicon)
- XGBoost, scikit-learn
- pandas, numpy, geopandas

## Data Sources

- **Seismic Data**: [Instituto Geográfico Nacional (IGN)](https://www.ign.es/web/ign/portal/sis-catalogo-terremotos)
- **Historical Eruptions**: Smithsonian Global Volcanism Program
- **Gas Emissions**: INVOLCAN (requires data access request)

## ⚠️ Disclaimer

The eruption prediction model is **experimental** and for research purposes only. Outputs must NOT be used for official eruption forecasting. Always consult professional volcanologists and official sources (IGN, INVOLCAN) for volcanic hazard assessments.

## License

MIT License
