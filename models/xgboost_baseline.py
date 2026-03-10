"""
XGBoost baseline model for eruption prediction.
"""
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve
from pathlib import Path
import json


def flatten_sequences(X: np.ndarray) -> np.ndarray:
    """
    Flatten sequences for tree-based models.
    
    Creates features from:
    - Last day values
    - Sequence aggregations (mean, std, max, min)
    - Trend (last 7 days vs first 7 days)
    
    Args:
        X: Sequences of shape (n_samples, seq_len, n_features)
    
    Returns:
        Flattened features of shape (n_samples, n_flat_features)
    """
    n_samples, seq_len, n_features = X.shape
    
    features = []
    
    for i in range(n_samples):
        seq = X[i]
        
        # Last day features
        last_day = seq[-1]
        
        # Aggregations over full sequence
        seq_mean = seq.mean(axis=0)
        seq_std = seq.std(axis=0)
        seq_max = seq.max(axis=0)
        seq_min = seq.min(axis=0)
        
        # Trend (last 7 days vs first 7 days)
        if seq_len >= 14:
            trend = seq[-7:].mean(axis=0) - seq[:7].mean(axis=0)
        else:
            trend = seq[-seq_len//2:].mean(axis=0) - seq[:seq_len//2].mean(axis=0)
        
        # Recent volatility (std of last 7 days)
        recent_std = seq[-7:].std(axis=0) if seq_len >= 7 else seq.std(axis=0)
        
        features.append(np.concatenate([
            last_day, seq_mean, seq_std, seq_max, seq_min, trend, recent_std
        ]))
    
    return np.array(features)


def train_xgboost(X_train: np.ndarray, 
                  y_train: np.ndarray,
                  X_val: np.ndarray,
                  y_val: np.ndarray,
                  scale_pos_weight: float = None) -> xgb.XGBClassifier:
    """
    Train XGBoost classifier.
    
    Args:
        X_train: Training sequences
        y_train: Training labels
        X_val: Validation sequences
        y_val: Validation labels
        scale_pos_weight: Class weight for imbalance (auto-calculated if None)
    
    Returns:
        Trained XGBClassifier
    """
    # Flatten sequences
    X_train_flat = flatten_sequences(X_train)
    X_val_flat = flatten_sequences(X_val)
    
    print(f"Flattened features: {X_train_flat.shape[1]}")
    
    # Calculate class weight if not provided
    if scale_pos_weight is None:
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos_weight = n_neg / max(n_pos, 1)
        print(f"Auto scale_pos_weight: {scale_pos_weight:.2f}")
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric='auc',
        early_stopping_rounds=20,
        random_state=42,
        tree_method='hist',  # Fast histogram-based method
        device='cpu'  # XGBoost on Apple Silicon works best with CPU
    )
    
    model.fit(
        X_train_flat, y_train,
        eval_set=[(X_val_flat, y_val)],
        verbose=True
    )
    
    return model


def evaluate_model(model: xgb.XGBClassifier, 
                   X: np.ndarray, 
                   y: np.ndarray,
                   threshold: float = 0.5) -> dict:
    """
    Evaluate XGBoost model.
    
    Args:
        model: Trained model
        X: Test sequences
        y: Test labels
        threshold: Classification threshold
    
    Returns:
        Dictionary of metrics
    """
    X_flat = flatten_sequences(X)
    y_pred_proba = model.predict_proba(X_flat)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # ROC-AUC
    try:
        auc = roc_auc_score(y, y_pred_proba)
    except ValueError:
        auc = 0.5
    
    # Find optimal threshold
    precision, recall, thresholds = precision_recall_curve(y, y_pred_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    return {
        'roc_auc': auc,
        'best_threshold': best_threshold,
        'best_f1': f1_scores[best_idx],
        'predictions': y_pred_proba
    }


def get_feature_importance(model: xgb.XGBClassifier, 
                           n_original_features: int) -> dict:
    """
    Get feature importance grouped by original features.
    
    Args:
        model: Trained model
        n_original_features: Number of original features before flattening
    
    Returns:
        Dictionary mapping feature groups to importance
    """
    importance = model.feature_importances_
    
    # Group by feature type (7 groups: last, mean, std, max, min, trend, recent_std)
    n_groups = 7
    group_names = ['last_day', 'seq_mean', 'seq_std', 'seq_max', 'seq_min', 'trend', 'recent_std']
    
    grouped_importance = {}
    for i, name in enumerate(group_names):
        start_idx = i * n_original_features
        end_idx = (i + 1) * n_original_features
        if end_idx <= len(importance):
            grouped_importance[name] = float(importance[start_idx:end_idx].sum())
    
    return grouped_importance


if __name__ == "__main__":
    data_path = Path(__file__).parent.parent / "data" / "processed" / "sequences.npz"
    
    if not data_path.exists():
        print(f"Error: Data not found at {data_path}")
        print("Please run scripts/build_dataset.py first")
        exit(1)
    
    # Load data
    print("Loading data...")
    data = np.load(data_path, allow_pickle=True)
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Sequence shape: {X_train.shape[1:]} (length, features)")
    print(f"Positive rate (train): {y_train.mean():.4f}")
    
    # Train
    print("\nTraining XGBoost...")
    model = train_xgboost(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    metrics = evaluate_model(model, X_test, y_test)
    
    print(f"\nTest Results:")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"  Best threshold: {metrics['best_threshold']:.3f}")
    print(f"  Best F1: {metrics['best_f1']:.4f}")
    
    # Classification report at different thresholds
    X_test_flat = flatten_sequences(X_test)
    y_pred_proba = model.predict_proba(X_test_flat)[:, 1]
    
    print("\nClassification reports at different thresholds:")
    for threshold in [0.3, 0.5, 0.7]:
        print(f"\n--- Threshold: {threshold} ---")
        y_pred = (y_pred_proba >= threshold).astype(int)
        print(classification_report(y_test, y_pred, zero_division=0))
    
    # Feature importance
    print("\nFeature importance by group:")
    importance = get_feature_importance(model, X_train.shape[2])
    for name, imp in sorted(importance.items(), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.4f}")
    
    # Save model
    model_path = Path(__file__).parent / "xgboost_baseline.json"
    model.save_model(str(model_path))
    print(f"\nModel saved to {model_path}")
    
    # Save metrics
    metrics_path = Path(__file__).parent / "xgboost_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({
            'roc_auc': metrics['roc_auc'],
            'best_threshold': metrics['best_threshold'],
            'best_f1': metrics['best_f1'],
            'feature_importance': importance
        }, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
