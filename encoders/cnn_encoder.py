# encoders/cnn_encoder.py
import torch
import torch.nn as nn
from .base_encoder import BaseEncoder

class CNNEncoder(BaseEncoder):
    """
    CNN encoder for image feature extraction.
    Wraps the existing CNN implementation from the original model.
    """
    
    def __init__(self, input_channels: int = 3, output_dim: int = 512):
        super().__init__(name="CNN")
        self.output_dim = output_dim
        
        # Define CNN layers (same as original implementation)
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.MaxPool2d(2),                                       # 32x32 -> 16x16
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 16x16 -> 8x8
            nn.ReLU(),
            nn.MaxPool2d(2),                                       # 8x8 -> 4x4
        )
        
        # Calculate actual output dimension
        self._calculate_output_dim()
        
        # Projection layer to match desired output dimension
        if self._actual_output_dim != output_dim:
            self.projection = nn.Linear(self._actual_output_dim, output_dim)
        else:
            self.projection = nn.Identity()
    
    def _calculate_output_dim(self):
        """Calculate the actual output dimension of the CNN."""
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 64, 64)
            output = self.cnn(dummy_input)
            self._actual_output_dim = output.view(1, -1).size(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN encoder.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Encoded features of shape (batch_size, output_dim)
        """
        batch_size = x.size(0)
        x = self.cnn(x)
        x = x.view(batch_size, -1)  # Flatten
        x = self.projection(x)
        return x 