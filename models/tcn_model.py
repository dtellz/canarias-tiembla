"""
Temporal Convolutional Network for eruption prediction.
"""
import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    """Causal convolution with proper padding to prevent future leakage."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x


class TCNBlock(nn.Module):
    """Residual TCN block with batch normalization and dropout."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        # Residual connection (1x1 conv if channel mismatch)
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    
    Uses dilated causal convolutions to capture long-range temporal dependencies
    while maintaining computational efficiency.
    
    Args:
        input_size: Number of input features per timestep
        hidden_size: Number of channels in TCN blocks
        num_layers: Number of TCN blocks (receptive field = 2^num_layers)
        kernel_size: Convolution kernel size
        dropout: Dropout probability
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 64,
                 num_layers: int = 4,
                 kernel_size: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        layers = []
        in_channels = input_size
        
        for i in range(num_layers):
            dilation = 2 ** i
            out_channels = hidden_size
            layers.append(TCNBlock(in_channels, out_channels, kernel_size, dilation, dropout))
            in_channels = out_channels
        
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, 1)
        
        # Calculate receptive field
        self.receptive_field = sum([2 ** i * (kernel_size - 1) for i in range(num_layers)]) + 1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
        
        Returns:
            Output tensor of shape (batch,) with logits
        """
        # Transpose to (batch, features, seq_len) for Conv1d
        x = x.transpose(1, 2)
        x = self.tcn(x)
        # Take last timestep
        x = x[:, :, -1]
        x = self.fc(x)
        return x.squeeze(-1)


class LSTMModel(nn.Module):
    """
    LSTM baseline model for comparison.
    
    Args:
        input_size: Number of input features per timestep
        hidden_size: LSTM hidden state size
        num_layers: Number of LSTM layers
        dropout: Dropout probability
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
        
        Returns:
            Output tensor of shape (batch,) with logits
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = h_n[-1]
        
        out = self.dropout(last_hidden)
        out = self.fc(out)
        return out.squeeze(-1)


class EruptionPredictor(nn.Module):
    """
    Full eruption prediction model with attention mechanism.
    
    Combines TCN for temporal feature extraction with attention
    for interpretability.
    
    Args:
        input_size: Number of input features per timestep
        hidden_size: Hidden layer size
        num_layers: Number of TCN layers
        dropout: Dropout probability
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 64,
                 num_layers: int = 4,
                 dropout: float = 0.2):
        super().__init__()
        
        self.tcn = nn.Sequential()
        in_channels = input_size
        
        for i in range(num_layers):
            dilation = 2 ** i
            self.tcn.add_module(
                f'block_{i}',
                TCNBlock(in_channels, hidden_size, kernel_size=3, dilation=dilation, dropout=dropout)
            )
            in_channels = hidden_size
        
        # Temporal attention
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention."""
        # x: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        x = self.tcn(x)  # (batch, hidden, seq_len)
        x = x.transpose(1, 2)  # (batch, seq_len, hidden)
        
        # Attention weights
        attn_weights = self.attention(x)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Weighted sum
        context = torch.sum(x * attn_weights, dim=1)  # (batch, hidden)
        
        out = self.fc(context)
        return out.squeeze(-1)
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get attention weights for interpretability."""
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = x.transpose(1, 2)
        
        attn_weights = self.attention(x)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        return attn_weights.squeeze(-1)


if __name__ == "__main__":
    # Test models
    batch_size = 16
    seq_len = 60
    n_features = 15
    
    print("Testing TCN model...")
    model = TCN(input_size=n_features)
    x = torch.randn(batch_size, seq_len, n_features)
    out = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Receptive field: {model.receptive_field} timesteps")
    
    print("\nTesting LSTM model...")
    lstm_model = LSTMModel(input_size=n_features)
    out = lstm_model(x)
    print(f"  Output shape: {out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")
    
    print("\nTesting EruptionPredictor with attention...")
    attn_model = EruptionPredictor(input_size=n_features)
    out = attn_model(x)
    attn_weights = attn_model.get_attention_weights(x)
    print(f"  Output shape: {out.shape}")
    print(f"  Attention weights shape: {attn_weights.shape}")
    print(f"  Parameters: {sum(p.numel() for p in attn_model.parameters()):,}")
    
    # Test MPS availability
    print("\nDevice availability:")
    print(f"  MPS available: {torch.backends.mps.is_available()}")
    print(f"  MPS built: {torch.backends.mps.is_built()}")
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
        x = x.to(device)
        out = model(x)
        print(f"  MPS test passed: output on {out.device}")
