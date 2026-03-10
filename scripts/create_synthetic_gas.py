"""
Create synthetic gas emission data for model development.

WARNING: This is for model development only.
Real data should be obtained from INVOLCAN (Instituto Volcanológico de Canarias).
Contact: https://www.involcan.org/
"""
import pandas as pd
import numpy as np
from pathlib import Path


def create_synthetic_gas_data(start_date: str, 
                               end_date: str, 
                               eruption_dates: list) -> pd.DataFrame:
    """
    Create synthetic gas emission data with eruption precursor patterns.
    
    This simulates the typical pattern where CO2 and SO2 emissions increase
    in the weeks/months before a volcanic eruption.
    
    Args:
        start_date: Start date for synthetic data
        end_date: End date for synthetic data
        eruption_dates: List of eruption start dates (YYYY-MM-DD format)
    
    Returns:
        DataFrame with synthetic gas emission data
    """
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Baseline values (typical non-eruptive Teide values)
    # Based on literature values for quiescent volcanoes
    co2_baseline = 500   # t/d (tonnes per day)
    so2_baseline = 50    # t/d
    
    # Generate baseline with seasonal variation and noise
    n_days = len(dates)
    
    # Seasonal component (slightly higher in summer due to temperature effects)
    seasonal = 1 + 0.1 * np.sin(2 * np.pi * np.arange(n_days) / 365)
    
    # Random walk component for natural variation
    random_walk = np.cumsum(np.random.normal(0, 5, n_days))
    random_walk = random_walk - random_walk.mean()  # Center around zero
    random_walk = np.clip(random_walk, -100, 100)  # Limit range
    
    # Base emissions
    co2_flux = co2_baseline * seasonal + random_walk + np.random.normal(0, 50, n_days)
    so2_flux = so2_baseline * seasonal + random_walk * 0.1 + np.random.normal(0, 10, n_days)
    
    # Ensure positive values
    co2_flux = np.maximum(co2_flux, 50)
    so2_flux = np.maximum(so2_flux, 5)
    
    # Add eruption precursor signals
    for eruption_date in eruption_dates:
        eruption_dt = pd.to_datetime(eruption_date)
        
        # Precursor window: gradual increase 90 days before eruption
        # with exponential acceleration in final 30 days
        for i, date in enumerate(dates):
            days_before = (eruption_dt - date).days
            
            if 0 < days_before <= 90:
                # Gradual increase phase (90-30 days before)
                if days_before > 30:
                    multiplier = 1 + (90 - days_before) / 120  # Up to 1.5x
                # Exponential increase phase (30-0 days before)
                else:
                    base_mult = 1.5
                    exp_mult = np.exp((30 - days_before) / 15) - 1
                    multiplier = base_mult + exp_mult  # Can reach 3-5x
                
                co2_flux[i] *= multiplier
                so2_flux[i] *= multiplier * 1.2  # SO2 increases more dramatically
    
    df = pd.DataFrame({
        'date': dates,
        'co2_flux_td': np.round(co2_flux, 1),
        'so2_flux_td': np.round(so2_flux, 1),
        'is_synthetic': True
    })
    
    return df


def add_gas_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived gas features.
    
    Args:
        df: DataFrame with gas emission data
    
    Returns:
        DataFrame with additional features
    """
    df = df.copy()
    
    # Rolling statistics
    for window in [7, 14, 30]:
        df[f'co2_mean_{window}d'] = df['co2_flux_td'].rolling(window, min_periods=1).mean()
        df[f'so2_mean_{window}d'] = df['so2_flux_td'].rolling(window, min_periods=1).mean()
    
    # Z-scores (anomaly detection)
    co2_mean = df['co2_flux_td'].rolling(90, min_periods=30).mean()
    co2_std = df['co2_flux_td'].rolling(90, min_periods=30).std()
    df['co2_zscore'] = (df['co2_flux_td'] - co2_mean) / (co2_std + 1e-6)
    
    so2_mean = df['so2_flux_td'].rolling(90, min_periods=30).mean()
    so2_std = df['so2_flux_td'].rolling(90, min_periods=30).std()
    df['so2_zscore'] = (df['so2_flux_td'] - so2_mean) / (so2_std + 1e-6)
    
    # Rate of change
    df['co2_rate_7d'] = df['co2_flux_td'].pct_change(periods=7)
    df['so2_rate_7d'] = df['so2_flux_td'].pct_change(periods=7)
    
    # CO2/SO2 ratio (can indicate magma depth/type)
    df['co2_so2_ratio'] = df['co2_flux_td'] / (df['so2_flux_td'] + 1e-6)
    
    return df


if __name__ == "__main__":
    output_dir = Path(__file__).parent.parent / "data" / "raw" / "gas"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Known Canary Islands eruption dates
    eruption_dates = [
        "2021-09-19",  # La Palma
        "2011-10-10",  # El Hierro
    ]
    
    # Create synthetic data
    print("Creating synthetic gas emission data...")
    df = create_synthetic_gas_data("2000-01-01", "2024-12-31", eruption_dates)
    
    # Add derived features
    df = add_gas_features(df)
    
    # Save
    output_file = output_dir / "synthetic_gas_emissions.csv"
    df.to_csv(output_file, index=False)
    
    print(f"Created synthetic gas data: {len(df)} days")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Saved to: {output_file}")
    
    # Summary statistics
    print("\nBaseline statistics (excluding precursor periods):")
    print(f"  CO2 flux: {df['co2_flux_td'].median():.1f} t/d (median)")
    print(f"  SO2 flux: {df['so2_flux_td'].median():.1f} t/d (median)")
    
    print("\n" + "=" * 60)
    print("WARNING: This is SYNTHETIC data for model development only!")
    print("For real volcanic monitoring, contact INVOLCAN:")
    print("  https://www.involcan.org/")
    print("=" * 60)
