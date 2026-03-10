"""
Train eruption prediction model using PyTorch with Apple Silicon MPS support.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import sys
import json

sys.path.append(str(Path(__file__).parent.parent))

from models.tcn_model import TCN, LSTMModel, EruptionPredictor
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from tqdm import tqdm


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.
    
    Reduces the loss contribution from easy examples and focuses
    on hard examples (rare eruption precursors).
    
    Args:
        alpha: Weighting factor for positive class
        gamma: Focusing parameter (higher = more focus on hard examples)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def get_device() -> torch.device:
    """Get the best available device (MPS for Apple Silicon, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def train_epoch(model: nn.Module, 
                dataloader: DataLoader, 
                optimizer: torch.optim.Optimizer, 
                criterion: nn.Module, 
                device: torch.device) -> tuple:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    
    # Calculate AUC
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc = 0.5
    
    return total_loss / len(dataloader), auc


def evaluate(model: nn.Module, 
             dataloader: DataLoader, 
             criterion: nn.Module, 
             device: torch.device) -> tuple:
    """Evaluate model on a dataset."""
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
    
    # Calculate AUC
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc = 0.5
    
    return total_loss / len(dataloader), auc, np.array(all_preds), np.array(all_labels)


def plot_training_history(history: dict, save_path: Path):
    """Plot and save training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['train_auc'], label='Train')
    axes[1].plot(history['val_auc'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('ROC-AUC')
    axes[1].legend()
    axes[1].set_title('ROC-AUC')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    # Configuration
    config = {
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 1e-4,
        'hidden_size': 64,
        'num_layers': 4,
        'dropout': 0.3,
        'patience': 15,
        'model_type': 'tcn'  # 'tcn', 'lstm', or 'attention'
    }
    
    # Paths
    data_path = Path(__file__).parent.parent / "data" / "processed" / "sequences.npz"
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    if not data_path.exists():
        print(f"Error: Data not found at {data_path}")
        print("Please run scripts/build_dataset.py first")
        return
    
    # Device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    data = np.load(data_path, allow_pickle=True)
    
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
    print(f"Positive rate (train): {y_train.mean():.4f}")
    
    # Handle edge case of no validation data
    if len(X_val) == 0:
        print("Warning: No validation data. Using 20% of training data.")
        split_idx = int(0.8 * len(X_train))
        X_val = X_train[split_idx:]
        y_val = y_train[split_idx:]
        X_train = X_train[:split_idx]
        y_train = y_train[:split_idx]
    
    # Dataloaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=config['batch_size'],
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=config['batch_size']
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=config['batch_size']
    )
    
    # Model selection
    input_size = X_train.shape[2]
    
    if config['model_type'] == 'tcn':
        model = TCN(
            input_size=input_size,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        )
    elif config['model_type'] == 'lstm':
        model = LSTMModel(
            input_size=input_size,
            hidden_size=config['hidden_size'],
            num_layers=2,
            dropout=config['dropout']
        )
    else:
        model = EruptionPredictor(
            input_size=input_size,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        )
    
    model = model.to(device)
    print(f"Model: {config['model_type'].upper()}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    # Use higher alpha for focal loss since eruptions are rare
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    alpha = min(0.9, 1 - 1/pos_weight) if pos_weight > 1 else 0.5
    criterion = FocalLoss(alpha=alpha, gamma=2.0)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=1e-4
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )
    
    # Training loop
    best_val_auc = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': []}
    
    print(f"\nStarting training for {config['epochs']} epochs...")
    print("-" * 70)
    
    for epoch in range(config['epochs']):
        train_loss, train_auc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc, _, _ = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(val_auc)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)
        
        # Progress output
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d}/{config['epochs']} | "
              f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | LR: {lr:.2e}")
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), models_dir / "tcn_best.pt")
            print(f"  -> New best model saved (AUC: {val_auc:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Load best model and evaluate on test
    print("\n" + "=" * 70)
    print("Final Evaluation on Test Set")
    print("=" * 70)
    
    model.load_state_dict(torch.load(models_dir / "tcn_best.pt", map_location=device))
    test_loss, test_auc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test ROC-AUC: {test_auc:.4f}")
    
    # Metrics at different thresholds
    print("\nMetrics at different thresholds:")
    for threshold in [0.3, 0.5, 0.7]:
        preds_binary = (test_preds > threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, preds_binary, average='binary', zero_division=0
        )
        print(f"  Threshold {threshold}: Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f}")
    
    # Save training history plot
    plot_training_history(history, Path(__file__).parent.parent / "training_history.png")
    print("\nTraining history saved to training_history.png")
    
    # Save config and results
    results = {
        'config': config,
        'best_val_auc': best_val_auc,
        'test_auc': test_auc,
        'test_loss': test_loss,
        'epochs_trained': len(history['train_loss'])
    }
    
    with open(models_dir / "training_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"Best model saved to: {models_dir / 'tcn_best.pt'}")


if __name__ == "__main__":
    main()
