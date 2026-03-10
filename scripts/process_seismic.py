"""
Process seismic data and extract features for eruption prediction.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats


def calculate_b_value(magnitudes: np.ndarray, mc: float = 1.5) -> float:
    """
    Calculate Gutenberg-Richter b-value.
    
    The b-value describes the frequency-magnitude distribution of earthquakes.
    Lower b-values may indicate increased stress and potential volcanic activity.
    
    Args:
        magnitudes: Array of earthquake magnitudes
        mc: Magnitude of completeness (minimum reliable magnitude)
    
    Returns:
        b-value or NaN if insufficient data
    """
    magnitudes = magnitudes[magnitudes >= mc]
    if len(magnitudes) < 10:
        return np.nan
    
    mean_mag = np.mean(magnitudes)
    # Aki-Utsu estimator
    b_value = np.log10(np.e) / (mean_mag - mc + 0.05)
    return b_value


def detect_swarms(df: pd.DataFrame, 
                  time_window_hours: int = 24,
                  min_events: int = 10,
                  max_distance_km: float = 10) -> pd.DataFrame:
    """
    Detect earthquake swarms based on spatiotemporal clustering.
    
    Swarms are clusters of earthquakes occurring close in time and space,
    often associated with volcanic activity.
    
    Args:
        df: DataFrame with earthquake data
        time_window_hours: Time window for clustering
        min_events: Minimum events to constitute a swarm
        max_distance_km: Maximum spatial extent of swarm
    
    Returns:
        DataFrame with swarm_id column added
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
    Extract daily seismic features from earthquake catalog.
    
    Args:
        df: DataFrame with earthquake data
        teide_distance_threshold: Maximum distance from Teide in km
    
    Returns:
        DataFrame with daily aggregated features
    """
    # Filter to events near Teide
    df_teide = df[df['distance_to_teide_km'] <= teide_distance_threshold].copy()
    
    if len(df_teide) == 0:
        print(f"Warning: No earthquakes within {teide_distance_threshold}km of Teide")
        return pd.DataFrame()
    
    df_teide['date'] = pd.to_datetime(df_teide['timestamp']).dt.date
    
    daily_features = []
    
    for date, group in df_teide.groupby('date'):
        mags = group['magnitude'].dropna().values
        depths = group['depth'].dropna().values
        
        features = {
            'date': pd.to_datetime(date),
            'earthquake_count': len(group),
            'mean_magnitude': np.mean(mags) if len(mags) > 0 else 0,
            'max_magnitude': np.max(mags) if len(mags) > 0 else 0,
            'min_magnitude': np.min(mags) if len(mags) > 0 else 0,
            'std_magnitude': np.std(mags) if len(mags) > 1 else 0,
            'depth_mean': np.mean(depths) if len(depths) > 0 else 0,
            'depth_std': np.std(depths) if len(depths) > 1 else 0,
            'depth_min': np.min(depths) if len(depths) > 0 else 0,
            'depth_max': np.max(depths) if len(depths) > 0 else 0,
            'b_value': calculate_b_value(mags),
            'energy_release': np.sum(10 ** (1.5 * mags + 4.8)) if len(mags) > 0 else 0,
            'swarm_events': (group['swarm_id'] > 0).sum() if 'swarm_id' in group.columns else 0,
            'mean_distance_km': group['distance_to_teide_km'].mean(),
        }
        daily_features.append(features)
    
    return pd.DataFrame(daily_features)


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling window features to capture temporal patterns.
    
    Args:
        df: DataFrame with daily features
    
    Returns:
        DataFrame with additional rolling features
    """
    df = df.sort_values('date').copy()
    
    # Rolling windows (7, 14, 30 days)
    for window in [7, 14, 30]:
        df[f'eq_count_{window}d'] = df['earthquake_count'].rolling(window, min_periods=1).sum()
        df[f'max_mag_{window}d'] = df['max_magnitude'].rolling(window, min_periods=1).max()
        df[f'mean_mag_{window}d'] = df['mean_magnitude'].rolling(window, min_periods=1).mean()
        df[f'energy_{window}d'] = df['energy_release'].rolling(window, min_periods=1).sum()
        df[f'depth_mean_{window}d'] = df['depth_mean'].rolling(window, min_periods=1).mean()
    
    # Depth migration (trend in depth over time - shallowing may indicate rising magma)
    def calc_slope(x):
        if len(x) < 3:
            return 0
        return stats.linregress(range(len(x)), x)[0]
    
    df['depth_migration_7d'] = df['depth_mean'].rolling(7, min_periods=3).apply(calc_slope)
    df['depth_migration_30d'] = df['depth_mean'].rolling(30, min_periods=7).apply(calc_slope)
    
    # Rate of change
    df['eq_rate_change_7d'] = df['eq_count_7d'].pct_change(periods=7).replace([np.inf, -np.inf], 0)
    df['eq_rate_change_30d'] = df['eq_count_30d'].pct_change(periods=30).replace([np.inf, -np.inf], 0)
    
    # Acceleration (second derivative)
    df['eq_acceleration'] = df['eq_rate_change_7d'].diff()
    
    # Days since last significant event
    significant_threshold = df['max_magnitude'].quantile(0.9)
    df['is_significant'] = df['max_magnitude'] >= significant_threshold
    df['days_since_significant'] = df.groupby(df['is_significant'].cumsum()).cumcount()
    
    # Cumulative seismic moment
    df['cumulative_energy'] = df['energy_release'].cumsum()
    
    return df


def fill_missing_dates(df: pd.DataFrame, 
                       start_date: str = None, 
                       end_date: str = None) -> pd.DataFrame:
    """
    Fill missing dates with zero earthquake activity.
    
    Args:
        df: DataFrame with daily features
        start_date: Optional start date
        end_date: Optional end date
    
    Returns:
        DataFrame with continuous date range
    """
    if len(df) == 0:
        return df
    
    if start_date is None:
        start_date = df['date'].min()
    if end_date is None:
        end_date = df['date'].max()
    
    # Create full date range
    full_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    full_df = pd.DataFrame({'date': full_dates})
    
    # Merge with existing data
    df['date'] = pd.to_datetime(df['date'])
    merged = full_df.merge(df, on='date', how='left')
    
    # Fill missing values
    # For count-based features, fill with 0
    count_cols = ['earthquake_count', 'swarm_events']
    for col in count_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)
    
    # For other features, forward fill then backward fill
    merged = merged.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return merged


if __name__ == "__main__":
    input_file = Path(__file__).parent.parent / "data" / "raw" / "seismic" / "earthquakes_processed.parquet"
    output_dir = Path(__file__).parent.parent / "data" / "features"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if input exists
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        print("Please run download_seismic.py first")
        exit(1)
    
    # Load data
    print(f"Loading data from {input_file}")
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df)} earthquakes")
    
    # Detect swarms
    print("Detecting earthquake swarms...")
    df = detect_swarms(df)
    n_swarms = df['swarm_id'].max()
    print(f"Detected {n_swarms} swarms")
    
    # Extract daily features
    print("Extracting daily features...")
    daily_df = extract_daily_features(df, teide_distance_threshold=50)
    
    if len(daily_df) == 0:
        print("No data within distance threshold. Trying with larger threshold...")
        daily_df = extract_daily_features(df, teide_distance_threshold=100)
    
    if len(daily_df) == 0:
        print("Error: Could not extract features. Check input data.")
        exit(1)
    
    print(f"Extracted features for {len(daily_df)} days")
    
    # Fill missing dates
    print("Filling missing dates...")
    daily_df = fill_missing_dates(daily_df)
    print(f"Total days after filling: {len(daily_df)}")
    
    # Add rolling features
    print("Adding rolling features...")
    daily_df = add_rolling_features(daily_df)
    
    # Save
    output_file = output_dir / "seismic_features.parquet"
    daily_df.to_parquet(output_file, index=False)
    
    print(f"\n{'='*50}")
    print(f"Saved seismic features to {output_file}")
    print(f"Date range: {daily_df['date'].min()} to {daily_df['date'].max()}")
    print(f"Total days: {len(daily_df)}")
    print(f"Features: {len(daily_df.columns)}")
    print(f"{'='*50}")
    
    # Summary
    print("\nFeature columns:")
    for col in daily_df.columns:
        print(f"  - {col}")
