# fusion/base_fusion.py
from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BaseFusion(ABC, nn.Module):
    """
    Abstract base class for all fusion strategies.
    All fusion methods must implement the forward method and provide output dimension.
    """
    
    def __init__(self, name: str, image_dim: int, text_dim: int):
        super().__init__()
        self.name = name
        self.image_dim = image_dim
        self.text_dim = text_dim
        self._output_dim = None
    
    @property
    def output_dim(self) -> int:
        """Return the output dimension of the fusion."""
        if self._output_dim is None:
            raise ValueError(f"Output dimension not set for {self.name}")
        return self._output_dim
    
    @output_dim.setter
    def output_dim(self, value: int):
        """Set the output dimension of the fusion."""
        self._output_dim = value
    
    @abstractmethod
    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the fusion strategy.
        
        Args:
            image_features: Image features of shape (batch_size, image_dim)
            text_features: Text features of shape (batch_size, text_dim)
            
        Returns:
            Fused features of shape (batch_size, output_dim)
        """
        pass
    
    def get_config(self) -> dict:
        """Return fusion configuration as a dictionary."""
        return {
            'name': self.name,
            'image_dim': self.image_dim,
            'text_dim': self.text_dim,
            'output_dim': self.output_dim,
            'type': self.__class__.__name__
        } 