# fusion/bilinear_fusion.py
import torch
import torch.nn as nn
from .base_fusion import BaseFusion

class BilinearFusion(BaseFusion):
    """
    Bilinear fusion strategy that implements multiplicative interaction between modalities.
    This captures complex interactions between image and text features.
    """
    
    def __init__(self, image_dim: int, text_dim: int, hidden_dim: int = 512, 
                 output_dim: int = 512, dropout: float = 0.1):
        super().__init__(name="Bilinear", image_dim=image_dim, text_dim=text_dim)
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # Project image and text features to common hidden dimension
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Bilinear layer for multiplicative interaction
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim)
        
        # Additional linear layers for different fusion strategies
        self.additive_fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.multiplicative_fusion = nn.Linear(hidden_dim, hidden_dim)
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Sigmoid()
        )
        
        # Final projection
        self.final_proj = nn.Linear(hidden_dim, output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through bilinear fusion.
        
        Args:
            image_features: Image features of shape (batch_size, image_dim)
            text_features: Text features of shape (batch_size, text_dim)
            
        Returns:
            Fused features of shape (batch_size, output_dim)
        """
        # Project features to common hidden dimension
        image_hidden = self.image_proj(image_features)
        text_hidden = self.text_proj(text_features)
        
        # Bilinear interaction
        bilinear_features = self.bilinear(image_hidden, text_hidden)
        
        # Additive fusion
        additive_features = self.additive_fusion(torch.cat([image_hidden, text_hidden], dim=1))
        
        # Multiplicative fusion (element-wise multiplication)
        multiplicative_features = self.multiplicative_fusion(image_hidden * text_hidden)
        
        # Combine all fusion strategies
        combined_features = torch.cat([
            bilinear_features,
            additive_features, 
            multiplicative_features
        ], dim=1)
        
        # Gating mechanism
        gate_weights = self.gate(combined_features)
        
        # Apply gating
        gated_features = gate_weights * bilinear_features + (1 - gate_weights) * additive_features
        
        # Final projection
        fused_features = self.final_proj(self.dropout(gated_features))
        
        return fused_features 