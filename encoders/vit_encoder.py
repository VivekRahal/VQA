# encoders/vit_encoder.py
"""
Vision Transformer (ViT) encoder module for VQA system.

This module provides a ViT encoder that uses pretrained Vision Transformer models from Hugging Face.
It wraps the transformers.ViTModel for easy integration into the VQA system.
"""

from typing import Optional
import torch
import torch.nn as nn
from transformers import ViTModel, ViTImageProcessor
from .base_encoder import BaseEncoder


class ViTEncoder(BaseEncoder):
    """
    Vision Transformer (ViT) encoder using pretrained ViT model from Hugging Face.
    
    This encoder loads a pretrained ViT model and provides a simple interface
    for image encoding in the VQA system. It automatically handles image preprocessing
    and provides the [CLS] token representation as output.
    
    Attributes:
        model_name (str): Name of the pretrained ViT model
        vit (ViTModel): The pretrained ViT model
        vit_embed_dim (int): Embedding dimension of the ViT model
        projection (nn.Module): Optional projection layer to match output dimension
    """
    
    def __init__(self, 
                 img_size: int = 224, 
                 patch_size: int = 16, 
                 in_channels: int = 3, 
                 embed_dim: int = 768, 
                 num_heads: int = 12, 
                 num_layers: int = 12, 
                 mlp_ratio: float = 4.0, 
                 dropout: float = 0.1, 
                 output_dim: int = 768, 
                 model_name: str = "google/vit-base-patch16-224") -> None:
        """
        Initialize the ViT encoder.
        
        Args:
            img_size: Input image size (not used for pretrained models)
            patch_size: Patch size (not used for pretrained models)
            in_channels: Number of input channels (not used for pretrained models)
            embed_dim: Embedding dimension (not used for pretrained models)
            num_heads: Number of attention heads (not used for pretrained models)
            num_layers: Number of layers (not used for pretrained models)
            mlp_ratio: MLP ratio (not used for pretrained models)
            dropout: Dropout rate (not used for pretrained models)
            output_dim: Desired output dimension
            model_name: Name of the pretrained ViT model to load
            
        Raises:
            ValueError: If model_name is invalid or model cannot be loaded
        """
        super().__init__(name="ViT")
        
        if not isinstance(model_name, str) or not model_name:
            raise ValueError(f"model_name must be a non-empty string, got {model_name}")
        if not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError(f"output_dim must be a positive integer, got {output_dim}")
        
        self.model_name: str = model_name
        self.output_dim = output_dim
        
        # Load pretrained ViT model
        try:
            print(f"🔄 Loading pretrained ViT model: {model_name}")
            self.vit = ViTModel.from_pretrained(model_name)
        except Exception as e:
            raise ValueError(f"Failed to load ViT model '{model_name}': {e}")
        
        # Get the actual embedding dimension from ViT
        self.vit_embed_dim: int = self.vit.config.hidden_size
        
        # Projection layer to match desired output dimension
        if self.vit_embed_dim != output_dim:
            self.projection = nn.Linear(self.vit_embed_dim, output_dim)
        else:
            self.projection = nn.Identity()
        
        print(f"✅ ViT encoder initialized with {self.vit_embed_dim} -> {output_dim} projection")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through pretrained ViT encoder.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Encoded features of shape (batch_size, output_dim)
            
        Raises:
            ValueError: If input tensor has incorrect shape
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor (batch, channels, height, width), got shape {x.shape}")
        
        if x.size(1) != 3:
            raise ValueError(f"Expected 3 channels, got {x.size(1)} channels")
        
        # Pass through pretrained ViT
        vit_outputs = self.vit(pixel_values=x)
        
        # Use the [CLS] token representation (first token)
        pooled_output = vit_outputs.pooler_output
        
        # Project to desired output dimension
        output = self.projection(pooled_output)
        
        return output
    
    def get_processor(self) -> ViTImageProcessor:
        """
        Get the ViT image processor for preprocessing images.
        
        Returns:
            ViT image processor instance
            
        Raises:
            ValueError: If processor cannot be loaded
        """
        try:
            return ViTImageProcessor.from_pretrained(self.model_name)
        except Exception as e:
            raise ValueError(f"Failed to load ViT processor for '{self.model_name}': {e}")
    
    def get_config(self) -> dict:
        """
        Get encoder configuration as a dictionary.
        
        Returns:
            Dictionary containing encoder configuration
        """
        config = super().get_config()
        config.update({
            'model_name': self.model_name,
            'vit_embed_dim': self.vit_embed_dim,
            'has_projection': not isinstance(self.projection, nn.Identity)
        })
        return config 