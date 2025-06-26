# fusion/base_fusion.py
from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
import torch.nn as nn

"""
Base fusion module for VQA system.

This module provides the abstract base class for all fusion strategies in the VQA system.
All fusion methods must inherit from BaseFusion and implement the required methods.
"""

class BaseFusion(ABC, nn.Module):
    """
    Abstract base class for all fusion strategies in the VQA system.
    
    This class defines the interface that all fusion strategies must implement.
    It provides common functionality like output dimension management
    and configuration retrieval.
    
    Attributes:
        name (str): Name identifier for the fusion strategy
        image_dim (int): Dimension of image features
        text_dim (int): Dimension of text features
        _output_dim (Optional[int]): Output dimension of the fusion
    """
    
    def __init__(self, name: str, image_dim: int, text_dim: int) -> None:
        """
        Initialize the base fusion strategy.
        
        Args:
            name: Name identifier for the fusion strategy
            image_dim: Dimension of image features
            text_dim: Dimension of text features
            
        Raises:
            ValueError: If dimensions are not positive integers
        """
        super().__init__()
        
        if not isinstance(image_dim, int) or image_dim <= 0:
            raise ValueError(f"image_dim must be a positive integer, got {image_dim}")
        if not isinstance(text_dim, int) or text_dim <= 0:
            raise ValueError(f"text_dim must be a positive integer, got {text_dim}")
        
        self.name: str = name
        self.image_dim: int = image_dim
        self.text_dim: int = text_dim
        self._output_dim: int = None
    
    @property
    def output_dim(self) -> int:
        """
        Get the output dimension of the fusion strategy.
        
        Returns:
            Output dimension as integer
            
        Raises:
            ValueError: If output dimension is not set
        """
        if self._output_dim is None:
            raise ValueError(f"Output dimension not set for {self.name}")
        return self._output_dim
    
    @output_dim.setter
    def output_dim(self, value: int) -> None:
        """
        Set the output dimension of the fusion strategy.
        
        Args:
            value: Output dimension to set
            
        Raises:
            ValueError: If value is not a positive integer
        """
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"Output dimension must be a positive integer, got {value}")
        self._output_dim = value
    
    @abstractmethod
    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the fusion strategy.
        
        This method must be implemented by all subclasses.
        
        Args:
            image_features: Image features of shape (batch_size, image_dim)
            text_features: Text features of shape (batch_size, text_dim)
            
        Returns:
            Fused features of shape (batch_size, output_dim)
            
        Raises:
            ValueError: If input tensor shapes are incorrect
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get fusion strategy configuration as a dictionary.
        
        Returns:
            Dictionary containing fusion strategy configuration
        """
        return {
            'name': self.name,
            'image_dim': self.image_dim,
            'text_dim': self.text_dim,
            'output_dim': self.output_dim,
            'type': self.__class__.__name__
        }
    
    def __str__(self) -> str:
        """String representation of the fusion strategy."""
        return f"{self.__class__.__name__}(name='{self.name}', image_dim={self.image_dim}, text_dim={self.text_dim}, output_dim={self.output_dim})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the fusion strategy."""
        return self.__str__() 