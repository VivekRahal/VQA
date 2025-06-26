# fusion/concatenation_fusion.py
import torch
import torch.nn as nn
from .base_fusion import BaseFusion

class ConcatenationFusion(BaseFusion):
    """
    Simple concatenation fusion strategy.
    This is the same as the original implementation in the VQA model.
    """
    
    def __init__(self, image_dim: int, text_dim: int, output_dim: int = None):
        super().__init__(name="Concatenation", image_dim=image_dim, text_dim=text_dim)
        
        # If output_dim is not specified, use concatenated dimension
        if output_dim is None:
            output_dim = image_dim + text_dim
        
        self.output_dim = output_dim
        
        # Projection layer to match desired output dimension
        if (image_dim + text_dim) != output_dim:
            self.projection = nn.Linear(image_dim + text_dim, output_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through concatenation fusion.
        
        Args:
            image_features: Image features of shape (batch_size, image_dim)
            text_features: Text features of shape (batch_size, text_dim)
            
        Returns:
            Fused features of shape (batch_size, output_dim)
        """
        # Concatenate features along the feature dimension
        fused_features = torch.cat([image_features, text_features], dim=1)
        
        # Project to desired output dimension
        fused_features = self.projection(fused_features)
        
        return fused_features 