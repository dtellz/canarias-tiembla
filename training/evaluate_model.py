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


def plot_roc_curve(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path):
    """Plot and save ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return roc_auc


def plot_precision_recall(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path):
    """Plot and save precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return pr_auc


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                          threshold: float, save_path: Path):
    """Plot and save confusion matrix."""
    y_pred_binary = (y_pred >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_binary)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Eruption', 'Eruption'],
                yticklabels=['No Eruption', 'Eruption'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix (threshold={threshold})')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return cm


def plot_predictions_timeline(dates: np.ndarray, y_true: np.ndarray, 
                              y_pred: np.ndarray, save_path: Path):
    """Plot predictions over time."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Prediction probability
    axes[0].plot(dates, y_pred, 'b-', alpha=0.7, linewidth=1, label='Predicted probability')
    axes[0].axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Threshold 0.5')
    axes[0].axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='Threshold 0.7')
    axes[0].fill_between(dates, 0, y_pred, alpha=0.3)
    axes[0].set_ylabel('Eruption Probability')
    axes[0].set_ylim(0, 1)
    axes[0].legend(loc='upper right')
    axes[0].set_title('Eruption Probability Over Time')
    axes[0].grid(True, alpha=0.3)
    
    # Actual labels
    axes[1].fill_between(dates, y_true, alpha=0.7, color='red', label='Actual eruption window')
    axes[1].set_ylabel('Actual Label')
    axes[1].set_xlabel('Date')
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_threshold_analysis(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path):
    """Analyze metrics across different thresholds."""
    thresholds = np.arange(0.1, 0.95, 0.05)
    
    precisions, recalls, f1s = [], [], []
    
    for thresh in thresholds:
        y_pred_binary = (y_pred >= thresh).astype(int)
        
        tp = ((y_pred_binary == 1) & (y_true == 1)).sum()
        fp = ((y_pred_binary == 1) & (y_true == 0)).sum()
        fn = ((y_pred_binary == 0) & (y_true == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, 'b-', label='Precision', linewidth=2)
    plt.plot(thresholds, recalls, 'g-', label='Recall', linewidth=2)
    plt.plot(thresholds, f1s, 'r-', label='F1 Score', linewidth=2)
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs Classification Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Find optimal threshold
    best_idx = np.argmax(f1s)
    return thresholds[best_idx], f1s[best_idx]


def main():
    # Paths
    models_dir = Path(__file__).parent.parent / "models"
    data_path = Path(__file__).parent.parent / "data" / "processed" / "sequences.npz"
    output_dir = Path(__file__).parent.parent / "evaluation"
    output_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "tcn_best.pt"
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please run training/train_model.py first")
        return
    
    if not data_path.exists():
        print(f"Error: Data not found at {data_path}")
        print("Please run scripts/build_dataset.py first")
        return
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    data = np.load(data_path, allow_pickle=True)
    X_test = torch.FloatTensor(data['X_test'])
    y_test = data['y_test']
    dates_test = pd.to_datetime(data['dates_test'])
    
    print(f"Test samples: {len(X_test)}")
    print(f"Positive samples: {y_test.sum()}")
    
    # Load model
    print("Loading model...")
    model = TCN(input_size=X_test.shape[2])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Predict
    print("Running inference...")
    with torch.no_grad():
        outputs = model(X_test.to(device))
        y_pred = torch.sigmoid(outputs).cpu().numpy()
    
    # Generate evaluation plots
    print("\nGenerating evaluation plots...")
    
    roc_auc = plot_roc_curve(y_test, y_pred, output_dir / "roc_curve.png")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    
    pr_auc = plot_precision_recall(y_test, y_pred, output_dir / "precision_recall.png")
    print(f"  PR-AUC: {pr_auc:.4f}")
    
    plot_confusion_matrix(y_test, y_pred, 0.5, output_dir / "confusion_matrix.png")
    
    if len(dates_test) > 0:
        plot_predictions_timeline(dates_test, y_test, y_pred, output_dir / "timeline.png")
    
    best_threshold, best_f1 = plot_threshold_analysis(y_test, y_pred, 
                                                       output_dir / "threshold_analysis.png")
    print(f"  Best threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")
    
    # Classification report
    print("\nClassification Report (threshold=0.5):")
    report = classification_report(y_test, (y_pred >= 0.5).astype(int), 
                                   target_names=['No Eruption', 'Eruption'],
                                   zero_division=0)
    print(report)
    
    # Save report
    with open(output_dir / "classification_report.txt", 'w') as f:
        f.write("Eruption Prediction Model Evaluation\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test samples: {len(y_test)}\n")
        f.write(f"Positive samples: {y_test.sum()}\n")
        f.write(f"Positive rate: {y_test.mean():.4f}\n\n")
        f.write(f"ROC-AUC: {roc_auc:.4f}\n")
        f.write(f"PR-AUC: {pr_auc:.4f}\n")
        f.write(f"Best threshold: {best_threshold:.2f}\n")
        f.write(f"Best F1: {best_f1:.4f}\n\n")
        f.write("Classification Report (threshold=0.5):\n")
        f.write(report)
    
    print(f"\nEvaluation results saved to {output_dir}/")
    print("Generated files:")
    for f in output_dir.glob("*"):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
