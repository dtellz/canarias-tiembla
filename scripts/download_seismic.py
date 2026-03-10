"""
Download seismic data from IGN for Canary Islands region.
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import time
from math import radians, sin, cos, sqrt, atan2

# IGN FDSN web service endpoint
IGN_FDSN_URL = "https://www.ign.es/fdsnws/event/1/query"

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


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in km."""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1-a))


def download_ign_catalog(start_year: int, end_year: int, output_dir: Path) -> Path:
    """
    Download IGN earthquake catalog via FDSN web service.
    
    Args:
        start_year: Start year for data download
        end_year: End year for data download
        output_dir: Directory to save raw data
    
    Returns:
        Path to saved catalog file
    """
    all_events = []
    
    print(f"Downloading IGN earthquake catalog ({start_year}-{end_year})...")
    
    for year in range(start_year, end_year + 1):
        params = {
            "starttime": f"{year}-01-01T00:00:00",
            "endtime": f"{year}-12-31T23:59:59",
            "minlatitude": CANARY_BBOX["lat_min"],
            "maxlatitude": CANARY_BBOX["lat_max"],
            "minlongitude": CANARY_BBOX["lon_min"],
            "maxlongitude": CANARY_BBOX["lon_max"],
            "format": "text",
            "orderby": "time"
        }
        
        try:
            response = requests.get(IGN_FDSN_URL, params=params, timeout=120)
            
            if response.status_code == 200:
                lines = response.text.strip().split('\n')
                # Skip header line
                data_lines = lines[1:] if len(lines) > 1 else []
                all_events.extend(data_lines)
                print(f"  {year}: {len(data_lines)} events")
            elif response.status_code == 204:
                print(f"  {year}: No events found")
            else:
                print(f"  {year}: HTTP {response.status_code}")
                
        except requests.exceptions.Timeout:
            print(f"  {year}: Timeout - retrying...")
            time.sleep(5)
            try:
                response = requests.get(IGN_FDSN_URL, params=params, timeout=180)
                if response.status_code == 200:
                    lines = response.text.strip().split('\n')
                    data_lines = lines[1:] if len(lines) > 1 else []
                    all_events.extend(data_lines)
                    print(f"  {year}: {len(data_lines)} events (retry)")
            except Exception as e:
                print(f"  {year}: Failed on retry - {e}")
                
        except Exception as e:
            print(f"  {year}: Error - {e}")
        
        time.sleep(1)  # Rate limiting
    
    # Save raw data
    output_file = output_dir / "ign_earthquakes_raw.txt"
    with open(output_file, 'w') as f:
        # Write header
        f.write("EventID|Time|Latitude|Longitude|Depth|Author|Catalog|Contributor|ContributorID|MagType|Magnitude|MagAuthor|EventLocationName\n")
        f.write('\n'.join(all_events))
    
    print(f"\nSaved {len(all_events)} total events to {output_file}")
    return output_file


def parse_earthquake_catalog(input_file: Path) -> pd.DataFrame:
    """
    Parse IGN earthquake catalog into DataFrame.
    
    Args:
        input_file: Path to raw catalog file
    
    Returns:
        Processed DataFrame with earthquake data
    """
    # Read the file
    df = pd.read_csv(input_file, sep='|', on_bad_lines='skip')
    
    # Rename columns to standard names
    column_mapping = {
        'EventID': 'event_id',
        'Time': 'timestamp',
        'Latitude': 'latitude',
        'Longitude': 'longitude',
        'Depth': 'depth',
        'Magnitude': 'magnitude',
        'MagType': 'mag_type',
        'EventLocationName': 'location'
    }
    
    df = df.rename(columns=column_mapping)
    
    # Keep only needed columns
    keep_cols = ['event_id', 'timestamp', 'latitude', 'longitude', 'depth', 'magnitude', 'mag_type', 'location']
    df = df[[c for c in keep_cols if c in df.columns]]
    
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Convert numeric columns
    for col in ['latitude', 'longitude', 'depth', 'magnitude']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with missing critical data
    df = df.dropna(subset=['timestamp', 'latitude', 'longitude'])
    
    # Calculate distance to Teide
    df['distance_to_teide_km'] = df.apply(
        lambda row: haversine(
            row['latitude'], row['longitude'],
            TEIDE_COORDS['lat'], TEIDE_COORDS['lon']
        ), axis=1
    )
    
    # Add date column
    df['date'] = df['timestamp'].dt.date
    
    return df


def create_fallback_data(output_dir: Path) -> pd.DataFrame:
    """
    Create sample data structure if API fails.
    This shows the expected format for manual data entry.
    """
    print("\nCreating sample data structure for manual entry...")
    
    sample_data = {
        'event_id': ['sample_001', 'sample_002'],
        'timestamp': ['2024-01-01 12:00:00', '2024-01-02 14:30:00'],
        'latitude': [28.27, 28.30],
        'longitude': [-16.64, -16.60],
        'depth': [10.0, 15.0],
        'magnitude': [2.1, 1.8],
        'mag_type': ['ML', 'ML'],
        'location': ['Tenerife', 'Tenerife'],
        'distance_to_teide_km': [0.5, 5.2],
        'date': ['2024-01-01', '2024-01-02']
    }
    
    df = pd.DataFrame(sample_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    return df


if __name__ == "__main__":
    output_dir = Path(__file__).parent.parent / "data" / "raw" / "seismic"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to download from IGN
    try:
        catalog_file = download_ign_catalog(2000, 2024, output_dir)
        df = parse_earthquake_catalog(catalog_file)
    except Exception as e:
        print(f"\nFailed to download from IGN: {e}")
        print("Creating fallback sample data...")
        df = create_fallback_data(output_dir)
    
    # Save processed data
    output_file = output_dir / "earthquakes_processed.parquet"
    df.to_parquet(output_file, index=False)
    
    print(f"\n{'='*50}")
    print(f"Processed {len(df)} earthquakes")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Saved to: {output_file}")
    print(f"{'='*50}")
    
    # Summary statistics
    if len(df) > 0:
        print(f"\nSummary:")
        print(f"  Events within 50km of Teide: {(df['distance_to_teide_km'] <= 50).sum()}")
        print(f"  Magnitude range: {df['magnitude'].min():.1f} - {df['magnitude'].max():.1f}")
        print(f"  Depth range: {df['depth'].min():.1f} - {df['depth'].max():.1f} km")
