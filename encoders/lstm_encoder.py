# encoders/lstm_encoder.py
import torch
import torch.nn as nn
from .base_encoder import BaseEncoder

class LSTMEncoder(BaseEncoder):
    """
    LSTM encoder for text feature extraction.
    Wraps the existing LSTM implementation from the original model.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 50, hidden_size: int = 128, 
                 num_layers: int = 1, output_dim: int = 128):
        super().__init__(name="LSTM")
        self.output_dim = output_dim
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        
        # Projection layer to match desired output dimension
        if hidden_size != output_dim:
            self.projection = nn.Linear(hidden_size, output_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM encoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len) containing token indices
            
        Returns:
            Encoded features of shape (batch_size, output_dim)
        """
        # Embed tokens
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # Pass through LSTM
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        
        # Take the hidden state from the last LSTM layer
        features = h_n[-1]  # (batch_size, hidden_size)
        
        # Project to desired output dimension
        features = self.projection(features)
        
        return features 