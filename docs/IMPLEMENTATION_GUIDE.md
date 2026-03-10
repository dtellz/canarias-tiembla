# Tenerife Volcanic Eruption Prediction Model
## Step-by-Step Implementation Guide

---

# Phase 0: Prerequisites & Environment Setup

## Step 0.1: Verify System Requirements

```bash
# Check macOS version and chip
system_profiler SPHardwareDataType | grep -E "Chip|Memory"

# Expected output should show M4 Pro and 48GB memory
```

## Step 0.2: Install Homebrew Dependencies

```bash
# Install Homebrew if not present
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install system dependencies
brew install python@3.12 git cmake gdal proj
```

## Step 0.3: Create Python Virtual Environment

```bash
cd /Users/diego/Desktop/projects/tener-moto

# Create virtual environment
python3.12 -m venv volcano-env

# Activate environment
source volcano-env/bin/activate

# Upgrade pip
pip install --upgrade pip
```

## Step 0.4: Install Python Dependencies

Create `requirements.txt` first (see Phase 1), then:

```bash
pip install -r requirements.txt
```

## Step 0.5: Verify PyTorch MPS Backend

```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

# Test MPS
if torch.backends.mps.is_available():
    x = torch.randn(3, 3, device="mps")
    print(f"Tensor on MPS: {x.device}")
```

---

# Phase 1: Project Structure Setup

## Step 1.1: Create Directory Structure

```bash
cd /Users/diego/Desktop/projects/tener-moto

mkdir -p data/{raw,processed,features}
mkdir -p data/raw/{seismic,gas,insar,eruptions}
mkdir -p scripts
mkdir -p models
mkdir -p training
mkdir -p notebooks
mkdir -p configs
```

## Step 1.2: Create requirements.txt

```
# Core ML
torch>=2.1.0
torchvision>=0.16.0
torchaudio>=2.1.0

# Data processing
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.11.0

# ML utilities
scikit-learn>=1.3.0
xgboost>=2.0.0
imbalanced-learn>=0.11.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0

# Geospatial
geopandas>=0.14.0
rasterio>=1.3.0
xarray>=2023.6.0
netcdf4>=1.6.0
pyproj>=3.6.0
shapely>=2.0.0

# Data access
requests>=2.31.0
beautifulsoup4>=4.12.0
pystac-client>=0.7.0

# Deep learning utilities
pytorch-lightning>=2.1.0
tensorboard>=2.14.0

# Explainability
shap>=0.42.0
captum>=0.6.0

# Hyperparameter tuning
optuna>=3.3.0

# Utilities
tqdm>=4.66.0
python-dateutil>=2.8.0
pyyaml>=6.0.0
```

---

# Phase 2: Data Acquisition

## Step 2.1: Seismic Data from IGN

**Source**: Instituto Geográfico Nacional (IGN)  
**URL**: https://www.ign.es/web/ign/portal/sis-catalogo-terremotos

### Download Strategy

The IGN provides earthquake catalogs via web interface. We'll scrape/download the Canary Islands region.

**Bounding box for Canary Islands**:
- Latitude: 27.5°N to 29.5°N
- Longitude: 18.5°W to 13.5°W

**Script**: `scripts/download_seismic.py`

```python
"""
Download seismic data from IGN for Canary Islands region.
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import time

# IGN earthquake catalog endpoint
IGN_URL = "https://www.ign.es/web/ign/portal/sis-catalogo-terremotos"

# Canary Islands bounding box
CANARY_BBOX = {
    "lat_min": 27.5,
    "lat_max": 29.5,
    "lon_min": -18.5,
    "lon_max": -13.5
}

# Teide volcano coordinates
TEIDE_COORDS = {
    "lat": 28.2723,
    "lon": -16.6421
}

def download_ign_catalog(start_year: int, end_year: int, output_dir: Path):
    """
    Download IGN earthquake catalog.
    
    Note: IGN may require manual download or API access.
    Alternative: Use FDSN web services if available.
    """
    # FDSN alternative endpoint (if IGN supports it)
    fdsn_url = "https://www.ign.es/fdsnws/event/1/query"
    
    all_events = []
    
    for year in range(start_year, end_year + 1):
        params = {
            "starttime": f"{year}-01-01",
            "endtime": f"{year}-12-31",
            "minlatitude": CANARY_BBOX["lat_min"],
            "maxlatitude": CANARY_BBOX["lat_max"],
            "minlongitude": CANARY_BBOX["lon_min"],
            "maxlongitude": CANARY_BBOX["lon_max"],
            "format": "text",
            "orderby": "time"
        }
        
        try:
            response = requests.get(fdsn_url, params=params, timeout=60)
            if response.status_code == 200:
                # Parse FDSN text format
                lines = response.text.strip().split('\n')
                if len(lines) > 1:  # Has header + data
                    for line in lines[1:]:
                        all_events.append(line)
                print(f"Downloaded {year}: {len(lines)-1} events")
            else:
                print(f"Failed {year}: HTTP {response.status_code}")
        except Exception as e:
            print(f"Error {year}: {e}")
        
        time.sleep(1)  # Rate limiting
    
    # Save raw data
    output_file = output_dir / "ign_earthquakes_canarias.csv"
    with open(output_file, 'w') as f:
        f.write('\n'.join(all_events))
    
    return output_file


def parse_earthquake_catalog(input_file: Path) -> pd.DataFrame:
    """Parse IGN earthquake catalog into DataFrame."""
    
    # Expected columns from FDSN text format
    columns = [
        'event_id', 'timestamp', 'latitude', 'longitude', 
        'depth', 'magnitude', 'mag_type', 'location'
    ]
    
    df = pd.read_csv(input_file, sep='|', names=columns, skiprows=1)
    
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculate distance to Teide
    from math import radians, sin, cos, sqrt, atan2
    
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371  # Earth radius in km
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        return 2 * R * atan2(sqrt(a), sqrt(1-a))
    
    df['distance_to_teide_km'] = df.apply(
        lambda row: haversine(
            row['latitude'], row['longitude'],
            TEIDE_COORDS['lat'], TEIDE_COORDS['lon']
        ), axis=1
    )
    
    return df


if __name__ == "__main__":
    output_dir = Path("data/raw/seismic")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download 2000-2024
    catalog_file = download_ign_catalog(2000, 2024, output_dir)
    
    # Parse and save
    df = parse_earthquake_catalog(catalog_file)
    df.to_parquet(output_dir / "earthquakes_processed.parquet")
    print(f"Saved {len(df)} earthquakes")
```

### Alternative: Manual Download

If API fails, manually download from:
1. Go to https://www.ign.es/web/ign/portal/sis-catalogo-terremotos
2. Set region to Canary Islands
3. Set date range (2000-2024)
4. Export as CSV

---

## Step 2.2: Historical Eruption Data

**Source**: Smithsonian Global Volcanism Program  
**URL**: https://volcano.si.edu/

**Script**: `scripts/download_eruptions.py`

```python
"""
Download and parse historical eruption data for Canary Islands.
"""
import pandas as pd
from pathlib import Path

# Canary Islands eruption history (manually curated)
CANARY_ERUPTIONS = [
    # Teide-Pico Viejo complex
    {"volcano": "Teide", "start": "1909-11-18", "end": "1909-11-27", "vei": 2, "lat": 28.27, "lon": -16.64},
    {"volcano": "Teide", "start": "1798-06-09", "end": "1798-09-15", "vei": 3, "lat": 28.27, "lon": -16.64},
    {"volcano": "Teide", "start": "1706-05-05", "end": "1706-06-13", "vei": 3, "lat": 28.27, "lon": -16.64},
    {"volcano": "Teide", "start": "1704-12-31", "end": "1705-02-27", "vei": 3, "lat": 28.27, "lon": -16.64},
    
    # La Palma (Cumbre Vieja)
    {"volcano": "Cumbre Vieja", "start": "2021-09-19", "end": "2021-12-13", "vei": 3, "lat": 28.57, "lon": -17.84},
    {"volcano": "Cumbre Vieja", "start": "1971-10-26", "end": "1971-11-18", "vei": 2, "lat": 28.57, "lon": -17.84},
    {"volcano": "Cumbre Vieja", "start": "1949-06-24", "end": "1949-07-30", "vei": 2, "lat": 28.57, "lon": -17.84},
    
    # El Hierro
    {"volcano": "El Hierro", "start": "2011-10-10", "end": "2012-03-05", "vei": 2, "lat": 27.73, "lon": -18.03},
    
    # Lanzarote
    {"volcano": "Timanfaya", "start": "1824-07-31", "end": "1824-10-24", "vei": 2, "lat": 29.03, "lon": -13.73},
    {"volcano": "Timanfaya", "start": "1730-09-01", "end": "1736-04-16", "vei": 3, "lat": 29.03, "lon": -13.73},
]

def create_eruption_dataset():
    """Create eruption dataset for Canary Islands."""
    df = pd.DataFrame(CANARY_ERUPTIONS)
    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])
    df['duration_days'] = (df['end'] - df['start']).dt.days
    
    return df


def create_eruption_labels(eruption_df: pd.DataFrame, 
                           date_range: pd.DatetimeIndex,
                           prediction_window_days: int = 30) -> pd.Series:
    """
    Create binary labels for eruption prediction.
    
    Label = 1 if eruption starts within `prediction_window_days` of date.
    """
    labels = pd.Series(0, index=date_range)
    
    for _, eruption in eruption_df.iterrows():
        # Mark days before eruption start
        window_start = eruption['start'] - pd.Timedelta(days=prediction_window_days)
        window_end = eruption['start']
        
        mask = (labels.index >= window_start) & (labels.index < window_end)
        labels.loc[mask] = 1
    
    return labels


if __name__ == "__main__":
    output_dir = Path("data/raw/eruptions")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = create_eruption_dataset()
    df.to_csv(output_dir / "canary_eruptions.csv", index=False)
    print(f"Saved {len(df)} eruption records")
```

---

## Step 2.3: Ground Deformation Data (InSAR)

**Recommended Source**: LiCSAR (pre-processed InSAR)  
**URL**: https://comet.nerc.ac.uk/comet-lics-portal/

LiCSAR provides pre-processed Sentinel-1 InSAR time series, avoiding complex raw processing.

**Script**: `scripts/download_insar.py`

```python
"""
Download pre-processed InSAR deformation data from LiCSAR.
"""
import requests
from pathlib import Path
import xarray as xr

# LiCSAR frame covering Tenerife
# Frame ID needs to be identified from LiCSAR portal
TENERIFE_FRAME = "106D_05291_131313"  # Example - verify on portal

LICSAR_BASE_URL = "https://gws-access.jasmin.ac.uk/public/nceo_geohazards/LiCSAR_products"


def download_licsar_timeseries(frame_id: str, output_dir: Path):
    """
    Download LiCSAR time series for a frame.
    
    Note: LiCSAR data may require registration or direct download from portal.
    """
    # Construct URL for cumulative displacement
    url = f"{LICSAR_BASE_URL}/{frame_id[:3]}/{frame_id}/metadata/metadata.txt"
    
    print(f"Checking frame: {frame_id}")
    print(f"URL: {url}")
    print("\nManual download instructions:")
    print("1. Go to https://comet.nerc.ac.uk/comet-lics-portal/")
    print("2. Search for Tenerife region")
    print("3. Download cumulative displacement GeoTIFFs")
    print("4. Save to data/raw/insar/")
    
    return None


def process_insar_timeseries(insar_dir: Path) -> pd.DataFrame:
    """
    Process InSAR GeoTIFFs into time series at Teide location.
    """
    import rasterio
    import glob
    
    teide_lat, teide_lon = 28.2723, -16.6421
    
    results = []
    
    for tif_file in sorted(glob.glob(str(insar_dir / "*.tif"))):
        # Extract date from filename
        filename = Path(tif_file).stem
        # Assuming format: YYYYMMDD_displacement.tif
        date_str = filename.split('_')[0]
        
        with rasterio.open(tif_file) as src:
            # Get pixel value at Teide location
            row, col = src.index(teide_lon, teide_lat)
            value = src.read(1)[row, col]
            
            results.append({
                'date': pd.to_datetime(date_str),
                'displacement_mm': value
            })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    output_dir = Path("data/raw/insar")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    download_licsar_timeseries(TENERIFE_FRAME, output_dir)
```

---

## Step 2.4: Gas Emissions Data

**Source**: INVOLCAN publications and reports  
**Challenge**: Not publicly available in bulk format

**Workaround Options**:
1. Contact INVOLCAN directly for data access
2. Extract data from published papers
3. Use synthetic/proxy data for initial model development

**Script**: `scripts/create_synthetic_gas.py`

```python
"""
Create synthetic gas emission data for model development.
Real data should be obtained from INVOLCAN.
"""
import pandas as pd
import numpy as np
from pathlib import Path

def create_synthetic_gas_data(start_date: str, end_date: str, 
                              eruption_dates: list) -> pd.DataFrame:
    """
    Create synthetic gas emission data with eruption precursor patterns.
    
    WARNING: This is for model development only. 
    Real INVOLCAN data required for production.
    """
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Baseline values (typical non-eruptive)
    np.random.seed(42)
    
    co2_baseline = 500  # t/d
    so2_baseline = 50   # t/d
    
    co2_flux = np.random.normal(co2_baseline, 100, len(dates))
    so2_flux = np.random.normal(so2_baseline, 20, len(dates))
    
    # Add eruption precursor signals
    for eruption_date in eruption_dates:
        eruption_dt = pd.to_datetime(eruption_date)
        
        # Precursor window: 60 days before eruption
        for days_before in range(60, 0, -1):
            precursor_date = eruption_dt - pd.Timedelta(days=days_before)
            if precursor_date in dates:
                idx = dates.get_loc(precursor_date)
                
                # Exponential increase approaching eruption
                multiplier = 1 + (60 - days_before) / 20
                co2_flux[idx] *= multiplier
                so2_flux[idx] *= multiplier
    
    df = pd.DataFrame({
        'date': dates,
        'co2_flux_td': co2_flux,
        'so2_flux_td': so2_flux,
        'is_synthetic': True
    })
    
    return df


if __name__ == "__main__":
    output_dir = Path("data/raw/gas")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use known eruption dates
    eruption_dates = ["2021-09-19", "2011-10-10"]  # La Palma, El Hierro
    
    df = create_synthetic_gas_data("2000-01-01", "2024-12-31", eruption_dates)
    df.to_csv(output_dir / "synthetic_gas_emissions.csv", index=False)
    print(f"Created synthetic gas data: {len(df)} days")
    print("WARNING: Replace with real INVOLCAN data for production use")
```

---

# Phase 3: Data Processing & Feature Engineering

## Step 3.1: Process Seismic Data

**Script**: `scripts/process_seismic.py`

```python
"""
Process seismic data and extract features.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats


def calculate_b_value(magnitudes: np.ndarray, mc: float = 1.5) -> float:
    """
    Calculate Gutenberg-Richter b-value.
    
    b = log10(e) / (mean_magnitude - mc)
    """
    magnitudes = magnitudes[magnitudes >= mc]
    if len(magnitudes) < 10:
        return np.nan
    
    mean_mag = np.mean(magnitudes)
    b_value = np.log10(np.e) / (mean_mag - mc + 0.05)
    return b_value


def detect_swarms(df: pd.DataFrame, 
                  time_window_hours: int = 24,
                  min_events: int = 10,
                  max_distance_km: float = 10) -> pd.DataFrame:
    """
    Detect earthquake swarms based on spatiotemporal clustering.
    """
    df = df.sort_values('timestamp').copy()
    df['swarm_id'] = 0
    
    current_swarm = 0
    
    for i in range(len(df)):
        if df.iloc[i]['swarm_id'] > 0:
            continue
            
        # Find events within time window
        time_mask = (
            (df['timestamp'] >= df.iloc[i]['timestamp']) &
            (df['timestamp'] <= df.iloc[i]['timestamp'] + pd.Timedelta(hours=time_window_hours))
        )
        
        window_events = df[time_mask]
        
        if len(window_events) >= min_events:
            current_swarm += 1
            df.loc[time_mask, 'swarm_id'] = current_swarm
    
    return df


def extract_daily_features(df: pd.DataFrame, 
                           teide_distance_threshold: float = 50) -> pd.DataFrame:
    """
    Extract daily seismic features.
    """
    # Filter to events near Teide
    df_teide = df[df['distance_to_teide_km'] <= teide_distance_threshold].copy()
    df_teide['date'] = df_teide['timestamp'].dt.date
    
    daily_features = []
    
    for date, group in df_teide.groupby('date'):
        features = {
            'date': pd.to_datetime(date),
            'earthquake_count': len(group),
            'mean_magnitude': group['magnitude'].mean(),
            'max_magnitude': group['magnitude'].max(),
            'depth_mean': group['depth'].mean(),
            'depth_std': group['depth'].std(),
            'depth_min': group['depth'].min(),
            'b_value': calculate_b_value(group['magnitude'].values),
            'energy_release': np.sum(10 ** (1.5 * group['magnitude'] + 4.8)),
            'swarm_events': (group['swarm_id'] > 0).sum(),
        }
        daily_features.append(features)
    
    return pd.DataFrame(daily_features)


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling window features."""
    df = df.sort_values('date').copy()
    
    # Rolling windows
    for window in [7, 14, 30]:
        df[f'eq_count_{window}d'] = df['earthquake_count'].rolling(window, min_periods=1).sum()
        df[f'max_mag_{window}d'] = df['max_magnitude'].rolling(window, min_periods=1).max()
        df[f'energy_{window}d'] = df['energy_release'].rolling(window, min_periods=1).sum()
    
    # Depth migration (trend in depth over time)
    df['depth_migration_7d'] = df['depth_mean'].rolling(7, min_periods=3).apply(
        lambda x: stats.linregress(range(len(x)), x)[0] if len(x) >= 3 else 0
    )
    
    # Rate of change
    df['eq_rate_change'] = df['eq_count_7d'].pct_change(periods=7)
    
    return df


if __name__ == "__main__":
    input_file = Path("data/raw/seismic/earthquakes_processed.parquet")
    output_dir = Path("data/features")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = pd.read_parquet(input_file)
    
    # Detect swarms
    df = detect_swarms(df)
    
    # Extract daily features
    daily_df = extract_daily_features(df)
    
    # Add rolling features
    daily_df = add_rolling_features(daily_df)
    
    # Save
    daily_df.to_parquet(output_dir / "seismic_features.parquet")
    print(f"Saved seismic features: {len(daily_df)} days")
```

---

## Step 3.2: Build Unified Dataset

**Script**: `scripts/build_dataset.py`

```python
"""
Build unified training dataset combining all features.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle


def load_and_merge_features(feature_dir: Path, 
                            eruption_file: Path) -> pd.DataFrame:
    """Load all feature files and merge on date."""
    
    # Load seismic features
    seismic_df = pd.read_parquet(feature_dir / "seismic_features.parquet")
    seismic_df['date'] = pd.to_datetime(seismic_df['date'])
    
    # Load gas features (if available)
    gas_file = Path("data/raw/gas/synthetic_gas_emissions.csv")
    if gas_file.exists():
        gas_df = pd.read_csv(gas_file)
        gas_df['date'] = pd.to_datetime(gas_df['date'])
        seismic_df = seismic_df.merge(gas_df, on='date', how='left')
    
    # Load eruption labels
    eruption_df = pd.read_csv(eruption_file)
    eruption_df['start'] = pd.to_datetime(eruption_df['start'])
    
    return seismic_df, eruption_df


def create_labels(df: pd.DataFrame, 
                  eruption_df: pd.DataFrame,
                  windows: list = [7, 30, 90]) -> pd.DataFrame:
    """Create eruption labels for multiple prediction windows."""
    
    for window in windows:
        col_name = f'eruption_{window}d'
        df[col_name] = 0
        
        for _, eruption in eruption_df.iterrows():
            window_start = eruption['start'] - pd.Timedelta(days=window)
            window_end = eruption['start']
            
            mask = (df['date'] >= window_start) & (df['date'] < window_end)
            df.loc[mask, col_name] = 1
    
    return df


def create_sequences(df: pd.DataFrame, 
                     sequence_length: int = 60,
                     target_col: str = 'eruption_30d') -> tuple:
    """
    Create sequences for temporal model training.
    
    Returns:
        X: (n_samples, sequence_length, n_features)
        y: (n_samples,)
    """
    feature_cols = [c for c in df.columns if c not in 
                    ['date', 'eruption_7d', 'eruption_30d', 'eruption_90d', 'is_synthetic']]
    
    df = df.sort_values('date').reset_index(drop=True)
    
    X, y, dates = [], [], []
    
    for i in range(sequence_length, len(df)):
        X.append(df.loc[i-sequence_length:i-1, feature_cols].values)
        y.append(df.loc[i, target_col])
        dates.append(df.loc[i, 'date'])
    
    return np.array(X), np.array(y), dates


def train_test_split_temporal(X: np.ndarray, 
                              y: np.ndarray,
                              dates: list,
                              test_start_date: str = "2018-01-01",
                              val_start_date: str = "2015-01-01") -> dict:
    """
    Split data temporally (no data leakage).
    """
    dates = pd.to_datetime(dates)
    test_start = pd.to_datetime(test_start_date)
    val_start = pd.to_datetime(val_start_date)
    
    train_mask = dates < val_start
    val_mask = (dates >= val_start) & (dates < test_start)
    test_mask = dates >= test_start
    
    return {
        'X_train': X[train_mask],
        'y_train': y[train_mask],
        'X_val': X[val_mask],
        'y_val': y[val_mask],
        'X_test': X[test_mask],
        'y_test': y[test_mask],
        'dates_test': dates[test_mask]
    }


if __name__ == "__main__":
    feature_dir = Path("data/features")
    eruption_file = Path("data/raw/eruptions/canary_eruptions.csv")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and merge
    df, eruption_df = load_and_merge_features(feature_dir, eruption_file)
    
    # Create labels
    df = create_labels(df, eruption_df)
    
    # Fill missing values
    df = df.fillna(method='ffill').fillna(0)
    
    # Save full dataset
    df.to_parquet(output_dir / "full_dataset.parquet")
    
    # Create sequences
    X, y, dates = create_sequences(df, sequence_length=60, target_col='eruption_30d')
    
    # Split
    splits = train_test_split_temporal(X, y, dates)
    
    # Normalize features
    scaler = StandardScaler()
    n_samples, seq_len, n_features = splits['X_train'].shape
    
    # Fit on training data
    X_train_flat = splits['X_train'].reshape(-1, n_features)
    scaler.fit(X_train_flat)
    
    # Transform all splits
    for key in ['X_train', 'X_val', 'X_test']:
        X_flat = splits[key].reshape(-1, n_features)
        X_scaled = scaler.transform(X_flat)
        splits[key] = X_scaled.reshape(-1, seq_len, n_features)
    
    # Save
    np.savez(output_dir / "sequences.npz", **splits)
    with open(output_dir / "scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Training samples: {len(splits['X_train'])}")
    print(f"Validation samples: {len(splits['X_val'])}")
    print(f"Test samples: {len(splits['X_test'])}")
    print(f"Positive rate (train): {splits['y_train'].mean():.4f}")
```

---

# Phase 4: Model Implementation

## Step 4.1: Temporal Convolutional Network (TCN)

**Script**: `models/tcn_model.py`

```python
"""
Temporal Convolutional Network for eruption prediction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """Causal convolution with proper padding."""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
    
    def forward(self, x):
        x = self.conv(x)
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x


class TCNBlock(nn.Module):
    """Residual TCN block."""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        residual = self.residual(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        return self.relu(x + residual)


class TCN(nn.Module):
    """
    Temporal Convolutional Network for binary classification.
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 64,
                 num_layers: int = 4,
                 kernel_size: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        
        layers = []
        in_channels = input_size
        
        for i in range(num_layers):
            dilation = 2 ** i
            out_channels = hidden_size
            layers.append(TCNBlock(in_channels, out_channels, kernel_size, dilation, dropout))
            in_channels = out_channels
        
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        x = self.tcn(x)
        x = x[:, :, -1]  # Take last timestep
        x = self.fc(x)
        return x.squeeze(-1)


class EruptionPredictor(nn.Module):
    """
    Full eruption prediction model with attention.
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 64,
                 num_layers: int = 4,
                 dropout: float = 0.2):
        super().__init__()
        
        self.tcn = TCN(input_size, hidden_size, num_layers, dropout=dropout)
        
        # Attention for interpretability
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        return self.tcn(x)


if __name__ == "__main__":
    # Test model
    batch_size = 16
    seq_len = 60
    n_features = 15
    
    model = TCN(input_size=n_features)
    x = torch.randn(batch_size, seq_len, n_features)
    
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

---

## Step 4.2: XGBoost Baseline

**Script**: `models/xgboost_baseline.py`

```python
"""
XGBoost baseline model for comparison.
"""
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, classification_report
import pickle
from pathlib import Path


def flatten_sequences(X: np.ndarray) -> np.ndarray:
    """Flatten sequences for tree-based models."""
    n_samples, seq_len, n_features = X.shape
    
    # Use last N days + aggregations
    features = []
    
    for i in range(n_samples):
        seq = X[i]
        
        # Last day features
        last_day = seq[-1]
        
        # Aggregations over sequence
        seq_mean = seq.mean(axis=0)
        seq_std = seq.std(axis=0)
        seq_max = seq.max(axis=0)
        seq_min = seq.min(axis=0)
        
        # Trend (last 7 days vs first 7 days)
        trend = seq[-7:].mean(axis=0) - seq[:7].mean(axis=0)
        
        features.append(np.concatenate([
            last_day, seq_mean, seq_std, seq_max, seq_min, trend
        ]))
    
    return np.array(features)


def train_xgboost(X_train: np.ndarray, 
                  y_train: np.ndarray,
                  X_val: np.ndarray,
                  y_val: np.ndarray,
                  scale_pos_weight: float = None) -> xgb.XGBClassifier:
    """Train XGBoost classifier."""
    
    # Flatten sequences
    X_train_flat = flatten_sequences(X_train)
    X_val_flat = flatten_sequences(X_val)
    
    # Calculate class weight if not provided
    if scale_pos_weight is None:
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric='auc',
        early_stopping_rounds=20,
        random_state=42
    )
    
    model.fit(
        X_train_flat, y_train,
        eval_set=[(X_val_flat, y_val)],
        verbose=True
    )
    
    return model


if __name__ == "__main__":
    # Load data
    data = np.load("data/processed/sequences.npz")
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Train
    model = train_xgboost(X_train, y_train, X_val, y_val)
    
    # Evaluate
    X_test_flat = flatten_sequences(X_test)
    y_pred = model.predict_proba(X_test_flat)[:, 1]
    
    print(f"\nTest ROC-AUC: {roc_auc_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, (y_pred > 0.5).astype(int)))
    
    # Save model
    model.save_model("models/xgboost_baseline.json")
```

---

# Phase 5: Model Training

## Step 5.1: Training Script

**Script**: `training/train_model.py`

```python
"""
Train eruption prediction model.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.tcn_model import TCN
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from tqdm import tqdm


class FocalLoss(nn.Module):
    """Focal loss for imbalanced classification."""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    
    auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
    return total_loss / len(dataloader), auc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            
            total_loss += loss.item()
            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
    return total_loss / len(dataloader), auc, np.array(all_preds), np.array(all_labels)


def main():
    # Config
    BATCH_SIZE = 32
    EPOCHS = 100
    LR = 1e-4
    HIDDEN_SIZE = 64
    NUM_LAYERS = 4
    DROPOUT = 0.3
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Load data
    data = np.load("data/processed/sequences.npz")
    
    X_train = torch.FloatTensor(data['X_train'])
    y_train = torch.FloatTensor(data['y_train'])
    X_val = torch.FloatTensor(data['X_val'])
    y_val = torch.FloatTensor(data['y_val'])
    X_test = torch.FloatTensor(data['X_test'])
    y_test = torch.FloatTensor(data['y_test'])
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Input features: {X_train.shape[2]}")
    print(f"Sequence length: {X_train.shape[1]}")
    print(f"Positive rate: {y_train.mean():.4f}")
    
    # Dataloaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=BATCH_SIZE
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=BATCH_SIZE
    )
    
    # Model
    model = TCN(
        input_size=X_train.shape[2],
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )
    
    # Training loop
    best_val_auc = 0
    history = {'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': []}
    
    for epoch in range(EPOCHS):
        train_loss, train_auc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc, _, _ = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(val_auc)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), "models/tcn_best.pt")
            print(f"  -> New best model saved (AUC: {val_auc:.4f})")
    
    # Load best model and evaluate on test
    model.load_state_dict(torch.load("models/tcn_best.pt"))
    test_loss, test_auc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    
    print(f"\n{'='*50}")
    print(f"Test Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  ROC-AUC: {test_auc:.4f}")
    
    # Metrics at different thresholds
    for threshold in [0.3, 0.5, 0.7]:
        preds_binary = (test_preds > threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, preds_binary, average='binary', zero_division=0
        )
        print(f"  Threshold {threshold}: P={precision:.3f} R={recall:.3f} F1={f1:.3f}")
    
    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].set_title('Training Loss')
    
    axes[1].plot(history['train_auc'], label='Train')
    axes[1].plot(history['val_auc'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('ROC-AUC')
    axes[1].legend()
    axes[1].set_title('ROC-AUC')
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    print("\nTraining history saved to training_history.png")


if __name__ == "__main__":
    main()
```

---

# Phase 6: Evaluation & Inference

## Step 6.1: Evaluation Script

**Script**: `training/evaluate_model.py`

```python
"""
Evaluate trained model and generate reports.
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, classification_report
)
import seaborn as sns
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.tcn_model import TCN


def plot_roc_curve(y_true, y_pred, save_path):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_precision_recall(y_true, y_pred, save_path):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_predictions_timeline(dates, y_true, y_pred, save_path):
    plt.figure(figsize=(14, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(dates, y_pred, 'b-', alpha=0.7, label='Predicted probability')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold 0.5')
    plt.ylabel('Eruption Probability')
    plt.legend()
    plt.title('Eruption Probability Over Time')
    
    plt.subplot(2, 1, 2)
    plt.fill_between(dates, y_true, alpha=0.5, color='red', label='Actual eruption window')
    plt.ylabel('Actual Label')
    plt.xlabel('Date')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    # Load model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    data = np.load("data/processed/sequences.npz", allow_pickle=True)
    X_test = torch.FloatTensor(data['X_test'])
    y_test = data['y_test']
    dates_test = pd.to_datetime(data['dates_test'])
    
    model = TCN(input_size=X_test.shape[2])
    model.load_state_dict(torch.load("models/tcn_best.pt", map_location=device))
    model.to(device)
    model.eval()
    
    # Predict
    with torch.no_grad():
        outputs = model(X_test.to(device))
        y_pred = torch.sigmoid(outputs).cpu().numpy()
    
    # Generate plots
    output_dir = Path("evaluation")
    output_dir.mkdir(exist_ok=True)
    
    plot_roc_curve(y_test, y_pred, output_dir / "roc_curve.png")
    plot_precision_recall(y_test, y_pred, output_dir / "precision_recall.png")
    plot_predictions_timeline(dates_test, y_test, y_pred, output_dir / "timeline.png")
    
    # Classification report
    report = classification_report(y_test, (y_pred > 0.5).astype(int))
    print(report)
    
    with open(output_dir / "classification_report.txt", 'w') as f:
        f.write(report)
    
    print(f"\nEvaluation results saved to {output_dir}/")


if __name__ == "__main__":
    main()
```

---

# Phase 7: Inference API

## Step 7.1: Inference Script

**Script**: `scripts/predict.py`

```python
"""
Run inference on new data.
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.tcn_model import TCN


class EruptionPredictor:
    """Eruption prediction inference class."""
    
    def __init__(self, model_path: str, scaler_path: str, device: str = None):
        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load model (infer input size from scaler)
        n_features = self.scaler.n_features_in_
        self.model = TCN(input_size=n_features)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.sequence_length = 60
    
    def predict(self, features_df: pd.DataFrame) -> dict:
        """
        Predict eruption probability from feature DataFrame.
        
        Args:
            features_df: DataFrame with required features (last 60 days)
        
        Returns:
            dict with probability and alert level
        """
        # Ensure we have enough data
        if len(features_df) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} days of data")
        
        # Take last sequence_length days
        features = features_df.tail(self.sequence_length).values
        
        # Scale
        features_scaled = self.scaler.transform(features)
        
        # To tensor
        X = torch.FloatTensor(features_scaled).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(X)
            probability = torch.sigmoid(output).item()
        
        # Determine alert level
        if probability >= 0.8:
            alert_level = "HIGH"
        elif probability >= 0.6:
            alert_level = "ELEVATED"
        elif probability >= 0.4:
            alert_level = "MODERATE"
        else:
            alert_level = "LOW"
        
        return {
            "eruption_probability_30d": probability,
            "alert_level": alert_level,
            "timestamp": pd.Timestamp.now().isoformat()
        }


if __name__ == "__main__":
    # Example usage
    predictor = EruptionPredictor(
        model_path="models/tcn_best.pt",
        scaler_path="data/processed/scaler.pkl"
    )
    
    # Load recent features
    features_df = pd.read_parquet("data/features/seismic_features.parquet")
    
    result = predictor.predict(features_df)
    
    print("\n" + "="*50)
    print("ERUPTION PREDICTION RESULT")
    print("="*50)
    print(f"Probability (30-day window): {result['eruption_probability_30d']:.2%}")
    print(f"Alert Level: {result['alert_level']}")
    print(f"Timestamp: {result['timestamp']}")
    print("="*50)
    print("\nWARNING: This is an experimental model.")
    print("Do not use for official eruption forecasting.")
```

---

# Quick Start Commands

```bash
# 1. Setup environment
cd /Users/diego/Desktop/projects/tener-moto
python3.12 -m venv volcano-env
source volcano-env/bin/activate
pip install -r requirements.txt

# 2. Create directory structure
mkdir -p data/{raw,processed,features}/{seismic,gas,insar,eruptions}
mkdir -p scripts models training notebooks evaluation configs

# 3. Download data (run scripts)
python scripts/download_seismic.py
python scripts/download_eruptions.py
python scripts/create_synthetic_gas.py

# 4. Process data
python scripts/process_seismic.py
python scripts/build_dataset.py

# 5. Train models
python training/train_model.py
python models/xgboost_baseline.py

# 6. Evaluate
python training/evaluate_model.py

# 7. Run inference
python scripts/predict.py
```

---

# Important Notes

1. **Data Limitations**: Teide hasn't erupted since 1909. The model primarily learns from La Palma 2021 and El Hierro 2011 precursor patterns.

2. **Gas Data**: INVOLCAN data requires direct request. Synthetic data is provided for development only.

3. **InSAR Data**: Requires manual download from LiCSAR portal or significant processing from raw Sentinel-1.

4. **Model Validation**: This is an experimental research tool. Outputs must NOT be used for official eruption forecasting without validation by volcanologists.

5. **Class Imbalance**: Eruptions are rare. The model uses focal loss and class weighting to handle imbalance.

---

END
