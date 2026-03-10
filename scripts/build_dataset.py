"""
Build unified training dataset combining all features and labels.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle


def load_and_merge_features(feature_dir: Path, eruption_file: Path) -> tuple:
    """
    Load all feature files and eruption data.
    
    Args:
        feature_dir: Directory containing feature parquet files
        eruption_file: Path to eruption CSV file
    
    Returns:
        Tuple of (features DataFrame, eruptions DataFrame)
    """
    # Load seismic features
    seismic_file = feature_dir / "seismic_features.parquet"
    if not seismic_file.exists():
        raise FileNotFoundError(f"Seismic features not found: {seismic_file}")
    
    seismic_df = pd.read_parquet(seismic_file)
    seismic_df['date'] = pd.to_datetime(seismic_df['date'])
    print(f"Loaded seismic features: {len(seismic_df)} days, {len(seismic_df.columns)} columns")
    
    # Load gas features (if available)
    gas_file = Path(__file__).parent.parent / "data" / "raw" / "gas" / "synthetic_gas_emissions.csv"
    if gas_file.exists():
        gas_df = pd.read_csv(gas_file)
        gas_df['date'] = pd.to_datetime(gas_df['date'])
        seismic_df = seismic_df.merge(gas_df[['date', 'co2_flux_td', 'so2_flux_td']], 
                                       on='date', how='left')
        print(f"Merged gas features")
    
    # Load eruption data
    eruption_df = pd.read_csv(eruption_file)
    eruption_df['start'] = pd.to_datetime(eruption_df['start'])
    eruption_df['end'] = pd.to_datetime(eruption_df['end'])
    print(f"Loaded {len(eruption_df)} eruption records")
    
    return seismic_df, eruption_df


def create_labels(df: pd.DataFrame, 
                  eruption_df: pd.DataFrame,
                  windows: list = [7, 30, 90]) -> pd.DataFrame:
    """
    Create eruption labels for multiple prediction windows.
    
    Args:
        df: Features DataFrame with date column
        eruption_df: Eruptions DataFrame
        windows: List of prediction window sizes in days
    
    Returns:
        DataFrame with label columns added
    """
    df = df.copy()
    
    for window in windows:
        col_name = f'eruption_{window}d'
        df[col_name] = 0
        
        for _, eruption in eruption_df.iterrows():
            window_start = eruption['start'] - pd.Timedelta(days=window)
            window_end = eruption['start']
            
            mask = (df['date'] >= window_start) & (df['date'] < window_end)
            df.loc[mask, col_name] = 1
        
        n_positive = df[col_name].sum()
        print(f"  {col_name}: {n_positive} positive days ({100*n_positive/len(df):.2f}%)")
    
    return df


def create_sequences(df: pd.DataFrame, 
                     sequence_length: int = 60,
                     target_col: str = 'eruption_30d',
                     feature_cols: list = None) -> tuple:
    """
    Create sequences for temporal model training.
    
    Args:
        df: DataFrame with features and labels
        sequence_length: Number of days in each sequence
        target_col: Target column name
        feature_cols: List of feature columns (auto-detected if None)
    
    Returns:
        Tuple of (X, y, dates) where:
            X: (n_samples, sequence_length, n_features)
            y: (n_samples,)
            dates: list of prediction dates
    """
    # Auto-detect feature columns
    if feature_cols is None:
        exclude_cols = ['date', 'eruption_7d', 'eruption_30d', 'eruption_90d', 
                        'is_synthetic', 'is_significant']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    print(f"Using {len(feature_cols)} features")
    
    df = df.sort_values('date').reset_index(drop=True)
    
    X, y, dates = [], [], []
    
    for i in range(sequence_length, len(df)):
        # Get sequence of features
        seq = df.loc[i-sequence_length:i-1, feature_cols].values
        
        # Skip if any NaN in sequence
        if np.isnan(seq).any():
            continue
        
        X.append(seq)
        y.append(df.loc[i, target_col])
        dates.append(df.loc[i, 'date'])
    
    return np.array(X), np.array(y), dates


def train_test_split_temporal(X: np.ndarray, 
                              y: np.ndarray,
                              dates: list,
                              test_start_date: str = "2018-01-01",
                              val_start_date: str = "2015-01-01") -> dict:
    """
    Split data temporally to prevent data leakage.
    
    Args:
        X: Feature sequences
        y: Labels
        dates: Dates for each sample
        test_start_date: Start of test period
        val_start_date: Start of validation period
    
    Returns:
        Dictionary with train/val/test splits
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
        'dates_test': np.array(dates[test_mask])
    }


def normalize_features(splits: dict, output_dir: Path) -> dict:
    """
    Normalize features using StandardScaler fit on training data.
    
    Args:
        splits: Dictionary with train/val/test splits
        output_dir: Directory to save scaler
    
    Returns:
        Dictionary with normalized splits
    """
    scaler = StandardScaler()
    
    n_samples, seq_len, n_features = splits['X_train'].shape
    
    # Fit on training data (flattened)
    X_train_flat = splits['X_train'].reshape(-1, n_features)
    scaler.fit(X_train_flat)
    
    # Transform all splits
    normalized = {}
    for key in ['X_train', 'X_val', 'X_test']:
        X = splits[key]
        n = X.shape[0]
        X_flat = X.reshape(-1, n_features)
        X_scaled = scaler.transform(X_flat)
        normalized[key] = X_scaled.reshape(n, seq_len, n_features)
    
    # Copy labels and dates
    for key in ['y_train', 'y_val', 'y_test', 'dates_test']:
        normalized[key] = splits[key]
    
    # Save scaler
    scaler_path = output_dir / "scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to {scaler_path}")
    
    return normalized


if __name__ == "__main__":
    feature_dir = Path(__file__).parent.parent / "data" / "features"
    eruption_file = Path(__file__).parent.parent / "data" / "raw" / "eruptions" / "canary_eruptions.csv"
    output_dir = Path(__file__).parent.parent / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check inputs
    if not eruption_file.exists():
        print(f"Error: Eruption file not found: {eruption_file}")
        print("Please run download_eruptions.py first")
        exit(1)
    
    # Load and merge features
    print("Loading features...")
    df, eruption_df = load_and_merge_features(feature_dir, eruption_file)
    
    # Create labels
    print("\nCreating labels...")
    df = create_labels(df, eruption_df, windows=[7, 30, 90])
    
    # Fill missing values
    print("\nFilling missing values...")
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # Save full dataset
    full_dataset_path = output_dir / "full_dataset.parquet"
    df.to_parquet(full_dataset_path, index=False)
    print(f"Saved full dataset to {full_dataset_path}")
    
    # Create sequences
    print("\nCreating sequences...")
    sequence_length = 60
    X, y, dates = create_sequences(df, sequence_length=sequence_length, target_col='eruption_30d')
    print(f"Created {len(X)} sequences of length {sequence_length}")
    
    # Split temporally
    print("\nSplitting data temporally...")
    splits = train_test_split_temporal(X, y, dates)
    
    print(f"  Training samples: {len(splits['X_train'])}")
    print(f"  Validation samples: {len(splits['X_val'])}")
    print(f"  Test samples: {len(splits['X_test'])}")
    
    # Check class balance
    for split_name in ['train', 'val', 'test']:
        y_split = splits[f'y_{split_name}']
        pos_rate = y_split.mean() if len(y_split) > 0 else 0
        print(f"  {split_name} positive rate: {pos_rate:.4f}")
    
    # Normalize
    print("\nNormalizing features...")
    normalized = normalize_features(splits, output_dir)
    
    # Save sequences
    sequences_path = output_dir / "sequences.npz"
    np.savez(sequences_path, **normalized)
    print(f"Saved sequences to {sequences_path}")
    
    print(f"\n{'='*50}")
    print("Dataset build complete!")
    print(f"{'='*50}")
    print(f"\nNext steps:")
    print("  1. Train model: python training/train_model.py")
    print("  2. Train baseline: python models/xgboost_baseline.py")
