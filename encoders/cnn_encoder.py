# encoders/cnn_encoder.py
"""
CNN encoder module for VQA system.

This module provides a CNN encoder that uses pretrained CNN models from torchvision.
It wraps various pretrained CNN architectures for easy integration into the VQA system.
"""

from typing import Dict, Any
import torch
import torch.nn as nn
import torchvision.models as models
from .base_encoder import BaseEncoder


class CNNEncoder(BaseEncoder):
    """
    CNN encoder using pretrained CNN models from torchvision.
    
    This encoder loads a pretrained CNN model and provides a simple interface
    for image encoding in the VQA system. It supports various ResNet and EfficientNet
    architectures and automatically handles the final classification layer removal.
    
    Attributes:
        model_name (str): Name of the pretrained CNN model
        cnn (nn.Module): The pretrained CNN model
        cnn_embed_dim (int): Embedding dimension of the CNN model
        projection (nn.Module): Optional projection layer to match output dimension
    """
    
    # Supported model configurations
    SUPPORTED_MODELS: Dict[str, int] = {
        "resnet18": 512,
        "resnet34": 512,
        "resnet50": 2048,
        "resnet101": 2048,
        "resnet152": 2048,
        "efficientnet_b0": 1280,
        "efficientnet_b1": 1280,
        "efficientnet_b2": 1408
    }
    
    def __init__(self, 
                 input_channels: int = 3, 
                 output_dim: int = 512, 
                 model_name: str = "resnet50", 
                 pretrained: bool = True) -> None:
        """
        Initialize the CNN encoder.
        
        Args:
            input_channels: Number of input channels (must be 3 for pretrained models)
            output_dim: Desired output dimension
            model_name: Name of the pretrained CNN model to load
            pretrained: Whether to load pretrained weights
            
        Raises:
            ValueError: If model_name is unsupported or parameters are invalid
        """
        super().__init__(name="CNN")
        
        if not isinstance(model_name, str) or model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_name}. Supported models: {list(self.SUPPORTED_MODELS.keys())}")
        if not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError(f"output_dim must be a positive integer, got {output_dim}")
        if not isinstance(pretrained, bool):
            raise ValueError(f"pretrained must be a boolean, got {pretrained}")
        
        self.model_name: str = model_name
        self.output_dim = output_dim
        
        # Load pretrained CNN model
        try:
            print(f"🔄 Loading pretrained CNN model: {model_name}")
            self.cnn = self._load_cnn_model(model_name, pretrained)
        except Exception as e:
            raise ValueError(f"Failed to load CNN model '{model_name}': {e}")
        
        # Get the actual embedding dimension from CNN
        self.cnn_embed_dim: int = self.SUPPORTED_MODELS[model_name]
        
        # Remove the final classification layer
        self._remove_classification_layer()
        
        # Projection layer to match desired output dimension
        if self.cnn_embed_dim != output_dim:
            self.projection = nn.Linear(self.cnn_embed_dim, output_dim)
        else:
            self.projection = nn.Identity()
        
        print(f"✅ CNN encoder initialized with {self.cnn_embed_dim} -> {output_dim} projection")
    
    def _load_cnn_model(self, model_name: str, pretrained: bool) -> nn.Module:
        """
        Load the specified CNN model.
        
        Args:
            model_name: Name of the model to load
            pretrained: Whether to load pretrained weights
            
        Returns:
            Loaded CNN model
            
        Raises:
            ValueError: If model_name is unsupported
        """
        if model_name == "resnet18":
            return models.resnet18(pretrained=pretrained)
        elif model_name == "resnet34":
            return models.resnet34(pretrained=pretrained)
        elif model_name == "resnet50":
            return models.resnet50(pretrained=pretrained)
        elif model_name == "resnet101":
            return models.resnet101(pretrained=pretrained)
        elif model_name == "resnet152":
            return models.resnet152(pretrained=pretrained)
        elif model_name == "efficientnet_b0":
            return models.efficientnet_b0(pretrained=pretrained)
        elif model_name == "efficientnet_b1":
            return models.efficientnet_b1(pretrained=pretrained)
        elif model_name == "efficientnet_b2":
            return models.efficientnet_b2(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def _remove_classification_layer(self) -> None:
        """Remove the final classification layer from the CNN model."""
        if hasattr(self.cnn, 'classifier'):
            # For EfficientNet
            self.cnn.classifier = nn.Identity()
        else:
            # For ResNet
            self.cnn.fc = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through pretrained CNN encoder.
        
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
        
        # Pass through pretrained CNN
        features = self.cnn(x)
        
        # Project to desired output dimension
        output = self.projection(features)
        
        return output
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get encoder configuration as a dictionary.
        
        Returns:
            Dictionary containing encoder configuration
        """
        config = super().get_config()
        config.update({
            'model_name': self.model_name,
            'cnn_embed_dim': self.cnn_embed_dim,
            'has_projection': not isinstance(self.projection, nn.Identity)
        })
        return config 