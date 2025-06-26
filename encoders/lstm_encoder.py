# encoders/lstm_encoder.py
"""
LSTM encoder module for VQA system.

This module provides an LSTM encoder for text feature extraction.
It can use either custom embeddings or pretrained word embeddings from Hugging Face.
"""

from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from .base_encoder import BaseEncoder


class LSTMEncoder(BaseEncoder):
    """
    LSTM encoder for text feature extraction.
    
    This encoder processes text sequences using LSTM networks. It can use either
    custom embeddings or pretrained word embeddings from Hugging Face.
    
    Attributes:
        use_pretrained_embeddings (bool): Whether to use pretrained embeddings
        embedding_model (str): Name of the pretrained embedding model
        embedding_dim (int): Dimension of embeddings
        lstm (nn.LSTM): LSTM layer for sequence processing
        projection (nn.Module): Optional projection layer to match output dimension
    """
    
    def __init__(self, 
                 vocab_size: int, 
                 embedding_dim: int = 50, 
                 hidden_size: int = 128, 
                 num_layers: int = 1, 
                 output_dim: int = 128, 
                 use_pretrained_embeddings: bool = False, 
                 embedding_model: str = "glove-wiki-gigaword-50") -> None:
        """
        Initialize the LSTM encoder.
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of embeddings (for custom embeddings)
            hidden_size: Hidden size of LSTM layers
            num_layers: Number of LSTM layers
            output_dim: Desired output dimension
            use_pretrained_embeddings: Whether to use pretrained embeddings
            embedding_model: Name of the pretrained embedding model
            
        Raises:
            ValueError: If parameters are invalid
        """
        super().__init__(name="LSTM")
        
        if not isinstance(vocab_size, int) or vocab_size <= 0:
            raise ValueError(f"vocab_size must be a positive integer, got {vocab_size}")
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError(f"hidden_size must be a positive integer, got {hidden_size}")
        if not isinstance(num_layers, int) or num_layers <= 0:
            raise ValueError(f"num_layers must be a positive integer, got {num_layers}")
        if not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError(f"output_dim must be a positive integer, got {output_dim}")
        if not isinstance(use_pretrained_embeddings, bool):
            raise ValueError(f"use_pretrained_embeddings must be a boolean, got {use_pretrained_embeddings}")
        
        self.use_pretrained_embeddings: bool = use_pretrained_embeddings
        self.output_dim = output_dim
        
        if use_pretrained_embeddings:
            # Use pretrained embeddings from Hugging Face
            if not isinstance(embedding_model, str) or not embedding_model:
                raise ValueError(f"embedding_model must be a non-empty string, got {embedding_model}")
            
            try:
                print(f"🔄 Loading pretrained embeddings: {embedding_model}")
                self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
                self.embedding_model = AutoModel.from_pretrained(embedding_model)
                self.embedding_dim = self.embedding_model.config.hidden_size
            except Exception as e:
                raise ValueError(f"Failed to load pretrained embeddings '{embedding_model}': {e}")
        else:
            # Use custom embeddings
            if not isinstance(embedding_dim, int) or embedding_dim <= 0:
                raise ValueError(f"embedding_dim must be a positive integer, got {embedding_dim}")
            
            self.embedding_dim = embedding_dim
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(self.embedding_dim, hidden_size, num_layers, batch_first=True)
        
        # Projection layer to match desired output dimension
        if hidden_size != output_dim:
            self.projection = nn.Linear(hidden_size, output_dim)
        else:
            self.projection = nn.Identity()
        
        print(f"✅ LSTM encoder initialized with {hidden_size} -> {output_dim} projection")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM encoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len) containing token indices
            
        Returns:
            Encoded features of shape (batch_size, output_dim)
            
        Raises:
            ValueError: If input tensor has incorrect shape
        """
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input tensor (batch, seq_len), got shape {x.shape}")
        
        if self.use_pretrained_embeddings:
            # Use pretrained embeddings
            with torch.no_grad():
                embedded = self.embedding_model(x).last_hidden_state
        else:
            # Use custom embeddings
            embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # Pass through LSTM
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        
        # Take the hidden state from the last LSTM layer
        features = h_n[-1]  # (batch_size, hidden_size)
        
        # Project to desired output dimension
        features = self.projection(features)
        
        return features
    
    def get_config(self) -> dict:
        """
        Get encoder configuration as a dictionary.
        
        Returns:
            Dictionary containing encoder configuration
        """
        config = super().get_config()
        config.update({
            'use_pretrained_embeddings': self.use_pretrained_embeddings,
            'embedding_dim': self.embedding_dim,
            'has_projection': not isinstance(self.projection, nn.Identity)
        })
        
        if self.use_pretrained_embeddings:
            config['embedding_model'] = getattr(self, 'embedding_model', 'unknown')
        
        return config 