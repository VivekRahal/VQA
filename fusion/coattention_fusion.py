# fusion/coattention_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_fusion import BaseFusion

class CoAttentionFusion(BaseFusion):
    """
    Co-attention fusion strategy that implements cross-modal attention.
    This allows the model to attend to relevant parts of the image based on the question
    and vice versa.
    """
    
    def __init__(self, image_dim: int, text_dim: int, hidden_dim: int = 512, 
                 num_heads: int = 8, output_dim: int = 512, dropout: float = 0.1):
        super().__init__(name="CoAttention", image_dim=image_dim, text_dim=text_dim)
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Project image and text features to common hidden dimension
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Co-attention mechanism
        self.coattention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward networks
        self.ff1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.ff2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Final projection to output dimension
        self.final_proj = nn.Linear(hidden_dim * 2, output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through co-attention fusion.
        
        Args:
            image_features: Image features of shape (batch_size, image_dim)
            text_features: Text features of shape (batch_size, text_dim)
            
        Returns:
            Fused features of shape (batch_size, output_dim)
        """
        batch_size = image_features.size(0)
        
        # Project features to common hidden dimension
        image_hidden = self.image_proj(image_features).unsqueeze(1)  # (batch_size, 1, hidden_dim)
        text_hidden = self.text_proj(text_features).unsqueeze(1)     # (batch_size, 1, hidden_dim)
        
        # Combine features for co-attention
        combined_features = torch.cat([image_hidden, text_hidden], dim=1)  # (batch_size, 2, hidden_dim)
        
        # Self-attention on combined features
        attended_features, _ = self.coattention(
            combined_features, combined_features, combined_features
        )
        attended_features = self.norm1(combined_features + self.dropout(attended_features))
        
        # Feed-forward on attended features
        ff_output = self.ff1(attended_features)
        attended_features = self.norm2(attended_features + self.dropout(ff_output))
        
        # Cross-modal attention: image attends to text and vice versa
        image_attended, _ = self.cross_attention(
            image_hidden, text_hidden, text_hidden
        )
        text_attended, _ = self.cross_attention(
            text_hidden, image_hidden, image_hidden
        )
        
        # Combine attended features
        image_final = attended_features[:, 0]  # Image representation
        text_final = attended_features[:, 1]   # Text representation
        
        # Concatenate cross-attended features
        fused_features = torch.cat([image_final, text_final], dim=1)
        
        # Final projection
        fused_features = self.final_proj(fused_features)
        
        return fused_features 