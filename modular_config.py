"""
Modular configuration module for VQA system.

This module provides a comprehensive configuration class for the modular VQA system.
It supports different encoders, fusion strategies, and pretrained models with preset configurations.
"""

from typing import Dict, Any, Optional
import torch
import os


class ModularConfig:
    """
    Configuration class for modular VQA model.
    
    This class manages all configuration parameters for the VQA system including
    encoders, fusion strategies, training parameters, and pretrained model settings.
    
    Attributes:
        batch_size (int): Training batch size
        num_epochs (int): Number of training epochs
        learning_rate (float): Learning rate for optimization
        device (torch.device): Device for training (CPU/GPU)
        num_classes (int): Number of output classes
        use_pretrained_models (bool): Whether to use pretrained models
        image_encoder_type (str): Type of image encoder
        text_encoder_type (str): Type of text encoder
        fusion_type (str): Type of fusion strategy
    """
    
    def __init__(self) -> None:
        """Initialize the configuration with default values."""
        # Basic training parameters
        self.batch_size: int = 2
        self.num_epochs: int = 5
        self.learning_rate: float = 0.001
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes: int = 3
        
        # DataLoader parameters
        self.num_workers: int = 0  # 0 for Windows, 2-4 for Linux/Mac
        self.pin_memory: bool = True if torch.cuda.is_available() else False
        self.persistent_workers: bool = False
        
        # Encoder types
        self.image_encoder_type: str = "cnn"
        self.text_encoder_type: str = "lstm"
        
        # Fusion type
        self.fusion_type: str = "concatenation"
        
        # Dimensions
        self.image_encoder_dim: int = 512
        self.text_encoder_dim: int = 128
        self.fusion_dim: int = 640
        self.image_size: int = 224  # Standard size for pretrained models
        
        # Vocabulary and classes (set dynamically)
        self.vocab_size: int = 1000
        self.num_classes: int = 3
        
        # Pretrained model configurations
        self.use_pretrained_models: bool = True
        
        # CNN encoder parameters
        self.cnn_model_name: str = "resnet50"
        self.cnn_pretrained: bool = True
        
        # LSTM encoder parameters
        self.lstm_embed_dim: int = 50
        self.lstm_hidden_size: int = 128
        self.lstm_num_layers: int = 1
        self.lstm_use_pretrained_embeddings: bool = False
        self.lstm_embedding_model: str = "glove-wiki-gigaword-50"
        
        # ViT encoder parameters
        self.vit_model_name: str = "google/vit-base-patch16-224"
        self.vit_pretrained: bool = True
        
        # BERT encoder parameters
        self.bert_model_name: str = "bert-base-uncased"
        self.bert_pretrained: bool = True
        
        # Fusion parameters
        self.coattention_hidden_dim: int = 256
        self.coattention_num_heads: int = 8
        self.bilinear_hidden_dim: int = 256
    
    def set_vocab_size(self, vocab_size: int) -> None:
        """
        Set vocabulary size dynamically.
        
        Args:
            vocab_size: Size of the vocabulary
            
        Raises:
            ValueError: If vocab_size is not a positive integer
        """
        if not isinstance(vocab_size, int) or vocab_size <= 0:
            raise ValueError(f"vocab_size must be a positive integer, got {vocab_size}")
        self.vocab_size = vocab_size
    
    def set_num_classes(self, num_classes: int) -> None:
        """
        Set number of classes dynamically.
        
        Args:
            num_classes: Number of output classes
            
        Raises:
            ValueError: If num_classes is not a positive integer
        """
        if not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError(f"num_classes must be a positive integer, got {num_classes}")
        self.num_classes = num_classes
    
    def get_preset_config(self, preset_name: str) -> 'ModularConfig':
        """
        Get a preset configuration for common combinations.
        
        Args:
            preset_name: Name of the preset configuration
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If preset_name is unknown
        """
        presets = self._get_available_presets()
        
        if preset_name not in presets:
            available = list(presets.keys())
            raise ValueError(f"Unknown preset: {preset_name}. Available presets: {available}")
        
        # Apply preset configuration
        for key, value in presets[preset_name].items():
            setattr(self, key, value)
        
        print(f"Applied preset configuration: {preset_name}")
        return self
    
    def _get_available_presets(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all available preset configurations.
        
        Returns:
            Dictionary of preset configurations
        """
        return {
            "original": {
                "image_encoder_type": "cnn",
                "text_encoder_type": "lstm", 
                "fusion_type": "concatenation",
                "image_encoder_dim": 512,
                "text_encoder_dim": 128,
                "fusion_dim": 640,
                "use_pretrained_models": False
            },
            "pretrained_cnn_lstm": {
                "image_encoder_type": "cnn",
                "text_encoder_type": "lstm",
                "fusion_type": "concatenation",
                "image_encoder_dim": 2048,
                "text_encoder_dim": 128,
                "fusion_dim": 2176,
                "use_pretrained_models": True,
                "cnn_model_name": "resnet50",
                "cnn_pretrained": True
            },
            "pretrained_vit_bert": {
                "image_encoder_type": "vit",
                "text_encoder_type": "bert",
                "fusion_type": "concatenation",
                "image_encoder_dim": 768,
                "text_encoder_dim": 768,
                "fusion_dim": 1536,
                "use_pretrained_models": True,
                "vit_model_name": "google/vit-base-patch16-224",
                "bert_model_name": "bert-base-uncased"
            },
            "pretrained_vit_bert_coattention": {
                "image_encoder_type": "vit",
                "text_encoder_type": "bert",
                "fusion_type": "coattention",
                "image_encoder_dim": 768,
                "text_encoder_dim": 768,
                "fusion_dim": 1536,
                "use_pretrained_models": True,
                "vit_model_name": "google/vit-base-patch16-224",
                "bert_model_name": "bert-base-uncased",
                "coattention_hidden_dim": 512
            },
            "pretrained_vit_bert_bilinear": {
                "image_encoder_type": "vit",
                "text_encoder_type": "bert",
                "fusion_type": "bilinear",
                "image_encoder_dim": 768,
                "text_encoder_dim": 768,
                "fusion_dim": 1536,
                "use_pretrained_models": True,
                "vit_model_name": "google/vit-base-patch16-224",
                "bert_model_name": "bert-base-uncased",
                "bilinear_hidden_dim": 512
            },
            "pretrained_cnn_bert": {
                "image_encoder_type": "cnn",
                "text_encoder_type": "bert",
                "fusion_type": "concatenation",
                "image_encoder_dim": 2048,
                "text_encoder_dim": 768,
                "fusion_dim": 2816,
                "use_pretrained_models": True,
                "cnn_model_name": "resnet50",
                "bert_model_name": "bert-base-uncased"
            },
            "pretrained_cnn_bert_coattention": {
                "image_encoder_type": "cnn",
                "text_encoder_type": "bert",
                "fusion_type": "coattention",
                "image_encoder_dim": 2048,
                "text_encoder_dim": 768,
                "fusion_dim": 2816,
                "use_pretrained_models": True,
                "cnn_model_name": "resnet50",
                "bert_model_name": "bert-base-uncased",
                "coattention_hidden_dim": 512
            },
            "pretrained_cnn_bert_bilinear": {
                "image_encoder_type": "cnn",
                "text_encoder_type": "bert",
                "fusion_type": "bilinear",
                "image_encoder_dim": 2048,
                "text_encoder_dim": 768,
                "fusion_dim": 2816,
                "use_pretrained_models": True,
                "cnn_model_name": "resnet50",
                "bert_model_name": "bert-base-uncased",
                "bilinear_hidden_dim": 512
            },
            "efficient_vit_bert": {
                "image_encoder_type": "cnn",
                "text_encoder_type": "bert",
                "fusion_type": "concatenation",
                "image_encoder_dim": 1280,
                "text_encoder_dim": 768,
                "fusion_dim": 2048,
                "use_pretrained_models": True,
                "cnn_model_name": "efficientnet_b0",
                "bert_model_name": "bert-base-uncased"
            }
        }
    
    def print_config(self) -> None:
        """Print current configuration in a formatted way."""
        print("\n=== Modular VQA Configuration ===")
        print(f"Image Encoder: {self.image_encoder_type} ({self.image_encoder_dim}D)")
        print(f"Text Encoder: {self.text_encoder_type} ({self.text_encoder_dim}D)")
        print(f"Fusion Strategy: {self.fusion_type} ({self.fusion_dim}D)")
        print(f"Use Pretrained Models: {self.use_pretrained_models}")
        
        if self.use_pretrained_models:
            if self.image_encoder_type == "cnn":
                print(f"CNN Model: {self.cnn_model_name} (pretrained: {self.cnn_pretrained})")
            elif self.image_encoder_type == "vit":
                print(f"ViT Model: {self.vit_model_name} (pretrained: {self.vit_pretrained})")
            
            if self.text_encoder_type == "bert":
                print(f"BERT Model: {self.bert_model_name} (pretrained: {self.bert_pretrained})")
            elif self.text_encoder_type == "lstm":
                print(f"LSTM Pretrained Embeddings: {self.lstm_use_pretrained_embeddings}")
        
        print(f"Batch Size: {self.batch_size}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Device: {self.device}")
        print(f"DataLoader Workers: {self.num_workers}")
        print(f"Pin Memory: {self.pin_memory}")
        print(f"Persistent Workers: {self.persistent_workers}")
        print("=" * 35)
    
    def get_config_dict(self) -> Dict[str, Any]:
        """
        Get configuration as a dictionary.
        
        Returns:
            Dictionary containing all configuration parameters
        """
        return {
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'learning_rate': self.learning_rate,
            'device': str(self.device),
            'num_classes': self.num_classes,
            'use_pretrained_models': self.use_pretrained_models,
            'image_encoder_type': self.image_encoder_type,
            'text_encoder_type': self.text_encoder_type,
            'fusion_type': self.fusion_type,
            'image_encoder_dim': self.image_encoder_dim,
            'text_encoder_dim': self.text_encoder_dim,
            'fusion_dim': self.fusion_dim,
            'vocab_size': self.vocab_size,
            'cnn_model_name': self.cnn_model_name,
            'bert_model_name': self.bert_model_name,
            'vit_model_name': self.vit_model_name
        } 