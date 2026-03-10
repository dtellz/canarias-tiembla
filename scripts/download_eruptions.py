"""
Download and parse historical eruption data for Canary Islands.
"""
import pandas as pd
from pathlib import Path

# Canary Islands eruption history (manually curated from Smithsonian GVP)
CANARY_ERUPTIONS = [
    # Teide-Pico Viejo complex (Tenerife)
    {"volcano": "Teide", "island": "Tenerife", "start": "1909-11-18", "end": "1909-11-27", "vei": 2, "lat": 28.27, "lon": -16.64, "type": "Strombolian"},
    {"volcano": "Teide", "island": "Tenerife", "start": "1798-06-09", "end": "1798-09-15", "vei": 3, "lat": 28.27, "lon": -16.64, "type": "Strombolian"},
    {"volcano": "Teide", "island": "Tenerife", "start": "1706-05-05", "end": "1706-06-13", "vei": 3, "lat": 28.27, "lon": -16.64, "type": "Effusive"},
    {"volcano": "Teide", "island": "Tenerife", "start": "1704-12-31", "end": "1705-02-27", "vei": 3, "lat": 28.27, "lon": -16.64, "type": "Strombolian"},
    
    # La Palma (Cumbre Vieja)
    {"volcano": "Cumbre Vieja", "island": "La Palma", "start": "2021-09-19", "end": "2021-12-13", "vei": 3, "lat": 28.57, "lon": -17.84, "type": "Strombolian"},
    {"volcano": "Cumbre Vieja", "island": "La Palma", "start": "1971-10-26", "end": "1971-11-18", "vei": 2, "lat": 28.57, "lon": -17.84, "type": "Strombolian"},
    {"volcano": "Cumbre Vieja", "island": "La Palma", "start": "1949-06-24", "end": "1949-07-30", "vei": 2, "lat": 28.57, "lon": -17.84, "type": "Strombolian"},
    {"volcano": "Cumbre Vieja", "island": "La Palma", "start": "1712-10-09", "end": "1712-12-03", "vei": 2, "lat": 28.57, "lon": -17.84, "type": "Strombolian"},
    {"volcano": "Cumbre Vieja", "island": "La Palma", "start": "1677-11-17", "end": "1678-01-21", "vei": 3, "lat": 28.57, "lon": -17.84, "type": "Strombolian"},
    {"volcano": "Cumbre Vieja", "island": "La Palma", "start": "1646-10-02", "end": "1646-12-21", "vei": 3, "lat": 28.57, "lon": -17.84, "type": "Strombolian"},
    {"volcano": "Cumbre Vieja", "island": "La Palma", "start": "1585-05-19", "end": "1585-08-10", "vei": 3, "lat": 28.57, "lon": -17.84, "type": "Strombolian"},
    
    # El Hierro
    {"volcano": "El Hierro", "island": "El Hierro", "start": "2011-10-10", "end": "2012-03-05", "vei": 2, "lat": 27.73, "lon": -18.03, "type": "Submarine"},
    
    # Lanzarote (Timanfaya)
    {"volcano": "Timanfaya", "island": "Lanzarote", "start": "1824-07-31", "end": "1824-10-24", "vei": 2, "lat": 29.03, "lon": -13.73, "type": "Strombolian"},
    {"volcano": "Timanfaya", "island": "Lanzarote", "start": "1730-09-01", "end": "1736-04-16", "vei": 3, "lat": 29.03, "lon": -13.73, "type": "Effusive"},
]


def create_eruption_dataset() -> pd.DataFrame:
    """Create eruption dataset for Canary Islands."""
    df = pd.DataFrame(CANARY_ERUPTIONS)
    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])
    df['duration_days'] = (df['end'] - df['start']).dt.days
    
    return df


def create_eruption_labels(eruption_df: pd.DataFrame, 
                           date_range: pd.DatetimeIndex,
                           prediction_window_days: int = 30,
                           volcano_filter: str = None) -> pd.Series:
    """
    Create binary labels for eruption prediction.
    
    Label = 1 if eruption starts within `prediction_window_days` of date.
    
    Args:
        eruption_df: DataFrame with eruption records
        date_range: DatetimeIndex of dates to label
        prediction_window_days: Days before eruption to mark as positive
        volcano_filter: Optional volcano name to filter (e.g., "Teide")
    
    Returns:
        Series with binary labels indexed by date
    """
    labels = pd.Series(0, index=date_range)
    
    df = eruption_df.copy()
    if volcano_filter:
        df = df[df['volcano'] == volcano_filter]
    
    for _, eruption in df.iterrows():
        # Mark days before eruption start as positive
        window_start = eruption['start'] - pd.Timedelta(days=prediction_window_days)
        window_end = eruption['start']
        
        mask = (labels.index >= window_start) & (labels.index < window_end)
        labels.loc[mask] = 1
    
    return labels


def get_recent_eruptions(eruption_df: pd.DataFrame, 
                         min_year: int = 1900) -> pd.DataFrame:
    """Get eruptions from recent history with better data availability."""
    return eruption_df[eruption_df['start'].dt.year >= min_year]


if __name__ == "__main__":
    output_dir = Path(__file__).parent.parent / "data" / "raw" / "eruptions"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset
    df = create_eruption_dataset()
    
    # Save full dataset
    df.to_csv(output_dir / "canary_eruptions.csv", index=False)
    
    # Save recent eruptions (better for ML training)
    recent_df = get_recent_eruptions(df, min_year=1900)
    recent_df.to_csv(output_dir / "canary_eruptions_recent.csv", index=False)
    
    print(f"Saved {len(df)} total eruption records")
    print(f"Saved {len(recent_df)} recent eruption records (since 1900)")
    
    print("\nEruption Summary:")
    print(df.groupby('island').size().to_string())
    
    print("\nRecent eruptions (since 1900):")
    for _, row in recent_df.iterrows():
        print(f"  {row['start'].strftime('%Y-%m-%d')} - {row['volcano']} ({row['island']}) VEI {row['vei']}")
    
    print("\nKey eruptions for ML training:")
    print("  - La Palma 2021: High-resolution seismic precursor data available")
    print("  - El Hierro 2011: Submarine eruption with detailed monitoring")
    print("  - Teide 1909: Last Teide eruption (limited data)")
