"""
BERT encoder module for VQA system.

This module provides a BERT encoder that uses pretrained BERT models from Hugging Face.
It wraps the transformers.BertModel for easy integration into the VQA system.
"""

from typing import Optional
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from .base_encoder import BaseEncoder


class BERTEncoder(BaseEncoder):
    """
    BERT encoder using pretrained BERT model from Hugging Face.
    
    This encoder loads a pretrained BERT model and provides a simple interface
    for text encoding in the VQA system. It automatically handles tokenization
    and provides the [CLS] token representation as output.
    
    Attributes:
        model_name (str): Name of the pretrained BERT model
        bert (BertModel): The pretrained BERT model
        bert_embed_dim (int): Embedding dimension of the BERT model
        projection (nn.Module): Optional projection layer to match output dimension
    """
    
    def __init__(self, 
                 vocab_size: Optional[int] = None, 
                 embed_dim: int = 768, 
                 num_heads: int = 12, 
                 num_layers: int = 6, 
                 ff_dim: int = 3072, 
                 max_seq_len: int = 512, 
                 dropout: float = 0.1, 
                 output_dim: int = 768, 
                 model_name: str = "bert-base-uncased") -> None:
        """
        Initialize the BERT encoder.
        
        Args:
            vocab_size: Vocabulary size (not used for pretrained models)
            embed_dim: Embedding dimension (not used for pretrained models)
            num_heads: Number of attention heads (not used for pretrained models)
            num_layers: Number of layers (not used for pretrained models)
            ff_dim: Feed-forward dimension (not used for pretrained models)
            max_seq_len: Maximum sequence length (not used for pretrained models)
            dropout: Dropout rate (not used for pretrained models)
            output_dim: Desired output dimension
            model_name: Name of the pretrained BERT model to load
            
        Raises:
            ValueError: If model_name is invalid or model cannot be loaded
        """
        super().__init__(name="BERT")
        
        if not isinstance(model_name, str) or not model_name:
            raise ValueError(f"model_name must be a non-empty string, got {model_name}")
        if not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError(f"output_dim must be a positive integer, got {output_dim}")
        
        self.model_name: str = model_name
        self.output_dim = output_dim
        
        # Load pretrained BERT model
        try:
            print(f"🔄 Loading pretrained BERT model: {model_name}")
            self.bert = BertModel.from_pretrained(model_name)
        except Exception as e:
            raise ValueError(f"Failed to load BERT model '{model_name}': {e}")
        
        # Get the actual embedding dimension from BERT
        self.bert_embed_dim: int = self.bert.config.hidden_size
        
        # Projection layer to match desired output dimension
        if self.bert_embed_dim != output_dim:
            self.projection = nn.Linear(self.bert_embed_dim, output_dim)
        else:
            self.projection = nn.Identity()
        
        print(f"✅ BERT encoder initialized with {self.bert_embed_dim} -> {output_dim} projection")
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through pretrained BERT encoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len) containing token indices
            attention_mask: Attention mask of shape (batch_size, seq_len), 
                           where 1 indicates tokens to attend to and 0 indicates padding
            
        Returns:
            Encoded features of shape (batch_size, output_dim)
            
        Raises:
            ValueError: If input tensor has incorrect shape
        """
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input tensor, got shape {x.shape}")
        
        if attention_mask is not None and attention_mask.shape != x.shape:
            raise ValueError(f"Attention mask shape {attention_mask.shape} must match input shape {x.shape}")
        
        # Pass through pretrained BERT
        bert_outputs = self.bert(input_ids=x, attention_mask=attention_mask)
        
        # Use the [CLS] token representation (first token)
        pooled_output = bert_outputs.pooler_output
        
        # Project to desired output dimension
        output = self.projection(pooled_output)
        
        return output
    
    def get_tokenizer(self) -> BertTokenizer:
        """
        Get the BERT tokenizer for preprocessing text.
        
        Returns:
            BERT tokenizer instance
            
        Raises:
            ValueError: If tokenizer cannot be loaded
        """
        try:
            return BertTokenizer.from_pretrained(self.model_name)
        except Exception as e:
            raise ValueError(f"Failed to load BERT tokenizer for '{self.model_name}': {e}")
    
    def get_config(self) -> dict:
        """
        Get encoder configuration as a dictionary.
        
        Returns:
            Dictionary containing encoder configuration
        """
        config = super().get_config()
        config.update({
            'model_name': self.model_name,
            'bert_embed_dim': self.bert_embed_dim,
            'has_projection': not isinstance(self.projection, nn.Identity)
        })
        return config 