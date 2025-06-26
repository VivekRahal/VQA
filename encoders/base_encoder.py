# encoders/base_encoder.py
from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BaseEncoder(ABC, nn.Module):
    """
    Abstract base class for all encoders.
    All encoders must implement the forward method and provide output dimension.
    """
    
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self._output_dim = None
    
    @property
    def output_dim(self) -> int:
        """Return the output dimension of the encoder."""
        if self._output_dim is None:
            raise ValueError(f"Output dimension not set for {self.name}")
        return self._output_dim
    
    @output_dim.setter
    def output_dim(self, value: int):
        """Set the output dimension of the encoder."""
        self._output_dim = value
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass of the encoder. Must be implemented by subclasses."""
        pass
    
    def get_config(self) -> dict:
        """Return encoder configuration as a dictionary."""
        return {
            'name': self.name,
            'output_dim': self.output_dim,
            'type': self.__class__.__name__
        } 