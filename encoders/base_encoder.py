# encoders/base_encoder.py
"""
Base encoder module for VQA system.

This module provides the abstract base class for all encoders in the VQA system.
All encoders must inherit from BaseEncoder and implement the required methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch
import torch.nn as nn


class BaseEncoder(ABC, nn.Module):
    """
    Abstract base class for all encoders in the VQA system.
    
    This class defines the interface that all encoders must implement.
    It provides common functionality like output dimension management
    and configuration retrieval.
    
    Attributes:
        name (str): Name identifier for the encoder
        _output_dim (Optional[int]): Output dimension of the encoder
    """
    
    def __init__(self, name: str) -> None:
        """
        Initialize the base encoder.
        
        Args:
            name: Name identifier for the encoder
        """
        super().__init__()
        self.name: str = name
        self._output_dim: Optional[int] = None
    
    @property
    def output_dim(self) -> int:
        """
        Get the output dimension of the encoder.
        
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
        Set the output dimension of the encoder.
        
        Args:
            value: Output dimension to set
        """
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"Output dimension must be a positive integer, got {value}")
        self._output_dim = value
    
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Forward pass of the encoder.
        
        This method must be implemented by all subclasses.
        
        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
            
        Returns:
            Encoded features as torch.Tensor
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get encoder configuration as a dictionary.
        
        Returns:
            Dictionary containing encoder configuration
        """
        return {
            'name': self.name,
            'output_dim': self.output_dim,
            'type': self.__class__.__name__
        }
    
    def __str__(self) -> str:
        """String representation of the encoder."""
        return f"{self.__class__.__name__}(name='{self.name}', output_dim={self.output_dim})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the encoder."""
        return self.__str__() 