"""
Run inference on new data for eruption prediction.
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
    """
    Eruption prediction inference class.
    
    Loads a trained model and scaler to make predictions on new data.
    """
    
    def __init__(self, model_path: str, scaler_path: str, device: str = None):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model weights
            scaler_path: Path to fitted scaler
            device: Device to use ('mps', 'cuda', 'cpu', or None for auto)
        """
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load model
        n_features = self.scaler.n_features_in_
        self.model = TCN(input_size=n_features)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.sequence_length = 60
        self.n_features = n_features
    
    def predict(self, features_df: pd.DataFrame) -> dict:
        """
        Predict eruption probability from feature DataFrame.
        
        Args:
            features_df: DataFrame with required features (last 60 days minimum)
        
        Returns:
            Dictionary with probability and alert level
        """
        # Ensure we have enough data
        if len(features_df) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} days of data, got {len(features_df)}")
        
        # Get feature columns (exclude non-feature columns)
        exclude_cols = ['date', 'eruption_7d', 'eruption_30d', 'eruption_90d', 
                        'is_synthetic', 'is_significant']
        feature_cols = [c for c in features_df.columns if c not in exclude_cols]
        
        # Take last sequence_length days
        features = features_df[feature_cols].tail(self.sequence_length).values
        
        # Handle NaN values
        features = np.nan_to_num(features, nan=0.0)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Convert to tensor
        X = torch.FloatTensor(features_scaled).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(X)
            probability = torch.sigmoid(output).item()
        
        # Determine alert level
        if probability >= 0.8:
            alert_level = "HIGH"
            alert_description = "Elevated eruption risk - immediate attention required"
        elif probability >= 0.6:
            alert_level = "ELEVATED"
            alert_description = "Increased volcanic activity detected"
        elif probability >= 0.4:
            alert_level = "MODERATE"
            alert_description = "Some anomalous signals detected"
        else:
            alert_level = "LOW"
            alert_description = "Normal background activity"
        
        return {
            "eruption_probability_30d": round(probability, 4),
            "alert_level": alert_level,
            "alert_description": alert_description,
            "timestamp": pd.Timestamp.now().isoformat(),
            "model_version": "tcn_v1",
            "sequence_days": self.sequence_length
        }
    
    def predict_batch(self, features_df: pd.DataFrame, 
                      step: int = 1) -> pd.DataFrame:
        """
        Generate predictions for multiple dates.
        
        Args:
            features_df: DataFrame with features
            step: Step size between predictions
        
        Returns:
            DataFrame with dates and predictions
        """
        results = []
        
        for i in range(self.sequence_length, len(features_df), step):
            window = features_df.iloc[i-self.sequence_length:i]
            
            try:
                pred = self.predict(window)
                results.append({
                    'date': features_df.iloc[i-1]['date'] if 'date' in features_df.columns else i,
                    'probability': pred['eruption_probability_30d'],
                    'alert_level': pred['alert_level']
                })
            except Exception as e:
                print(f"Warning: Prediction failed at index {i}: {e}")
                continue
        
        return pd.DataFrame(results)


def format_prediction_output(result: dict) -> str:
    """Format prediction result for display."""
    lines = [
        "",
        "=" * 60,
        "TEIDE VOLCANIC ERUPTION PREDICTION",
        "=" * 60,
        "",
        f"Prediction Window: 30 days",
        f"Eruption Probability: {result['eruption_probability_30d']:.1%}",
        "",
        f"Alert Level: {result['alert_level']}",
        f"Description: {result['alert_description']}",
        "",
        f"Timestamp: {result['timestamp']}",
        f"Model: {result['model_version']}",
        f"Based on: {result['sequence_days']} days of data",
        "",
        "=" * 60,
        "",
        "WARNING: This is an EXPERIMENTAL model.",
        "Do NOT use for official eruption forecasting.",
        "Consult volcanologists for official assessments.",
        "",
        "=" * 60,
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    # Paths
    base_dir = Path(__file__).parent.parent
    model_path = base_dir / "models" / "tcn_best.pt"
    scaler_path = base_dir / "data" / "processed" / "scaler.pkl"
    features_path = base_dir / "data" / "features" / "seismic_features.parquet"
    
    # Check files exist
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please run training/train_model.py first")
        exit(1)
    
    if not scaler_path.exists():
        print(f"Error: Scaler not found at {scaler_path}")
        print("Please run scripts/build_dataset.py first")
        exit(1)
    
    if not features_path.exists():
        print(f"Error: Features not found at {features_path}")
        print("Please run scripts/process_seismic.py first")
        exit(1)
    
    # Initialize predictor
    print("Loading model...")
    predictor = EruptionPredictor(
        model_path=str(model_path),
        scaler_path=str(scaler_path)
    )
    print(f"Model loaded on device: {predictor.device}")
    
    # Load recent features
    print("Loading features...")
    features_df = pd.read_parquet(features_path)
    print(f"Loaded {len(features_df)} days of data")
    
    # Make prediction
    print("Running prediction...")
    result = predictor.predict(features_df)
    
    # Display result
    print(format_prediction_output(result))
