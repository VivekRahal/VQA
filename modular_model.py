"""
Modular VQA model module.

This module provides the main VQA model class that combines different encoders
and fusion strategies in a modular, configurable way.
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from encoders import CNNEncoder, LSTMEncoder, ViTEncoder, BERTEncoder
from fusion import ConcatenationFusion, CoAttentionFusion, BilinearFusion
from modular_config import ModularConfig


class ModularVQAModel(nn.Module):
    """
    Modular VQA model that supports different encoders and fusion strategies.
    
    This model allows easy experimentation with different combinations of
    image encoders, text encoders, and fusion strategies. It automatically
    handles the creation and configuration of components based on the provided config.
    
    Attributes:
        config (ModularConfig): Configuration object
        image_encoder (BaseEncoder): Image encoder instance
        text_encoder (BaseEncoder): Text encoder instance
        fusion (BaseFusion): Fusion strategy instance
        classifier (nn.Linear): Final classification layer
    """
    
    def __init__(self, config: ModularConfig) -> None:
        """
        Initialize the modular VQA model.
        
        Args:
            config: Configuration object containing all model parameters
            
        Raises:
            ValueError: If configuration is invalid or components cannot be created
        """
        super().__init__()
        
        if not isinstance(config, ModularConfig):
            raise ValueError(f"config must be a ModularConfig instance, got {type(config)}")
        
        self.config = config
        
        # Initialize components
        self.image_encoder = self._create_image_encoder()
        self.text_encoder = self._create_text_encoder()
        self.fusion = self._create_fusion_strategy()
        
        # Final classification layer
        self.classifier = nn.Linear(self.fusion.output_dim, config.num_classes)
        
        # Print model configuration
        self._print_config()
    
    def _create_image_encoder(self) -> nn.Module:
        """
        Create image encoder based on configuration.
        
        Returns:
            Image encoder instance
            
        Raises:
            ValueError: If image encoder type is unsupported
        """
        encoder_type = self.config.image_encoder_type.lower()
        
        if encoder_type == "cnn":
            return CNNEncoder(
                input_channels=3,
                output_dim=self.config.image_encoder_dim,
                model_name=self.config.cnn_model_name,
                pretrained=self.config.cnn_pretrained
            )
        elif encoder_type == "vit":
            return ViTEncoder(
                img_size=self.config.image_size,
                patch_size=16,  # Standard for pretrained ViT
                in_channels=3,
                embed_dim=768,  # Standard for pretrained ViT
                num_heads=12,   # Standard for pretrained ViT
                num_layers=12,  # Standard for pretrained ViT
                output_dim=self.config.image_encoder_dim,
                model_name=self.config.vit_model_name
            )
        else:
            raise ValueError(f"Unsupported image encoder type: {encoder_type}")
    
    def _create_text_encoder(self) -> nn.Module:
        """
        Create text encoder based on configuration.
        
        Returns:
            Text encoder instance
            
        Raises:
            ValueError: If text encoder type is unsupported
        """
        encoder_type = self.config.text_encoder_type.lower()
        
        if encoder_type == "lstm":
            return LSTMEncoder(
                vocab_size=self.config.vocab_size,
                embedding_dim=self.config.lstm_embed_dim,
                hidden_size=self.config.lstm_hidden_size,
                num_layers=self.config.lstm_num_layers,
                output_dim=self.config.text_encoder_dim,
                use_pretrained_embeddings=self.config.lstm_use_pretrained_embeddings,
                embedding_model=self.config.lstm_embedding_model
            )
        elif encoder_type == "bert":
            return BERTEncoder(
                vocab_size=self.config.vocab_size,
                embed_dim=768,  # Standard for pretrained BERT
                num_heads=12,   # Standard for pretrained BERT
                num_layers=12,  # Standard for pretrained BERT
                ff_dim=3072,    # Standard for pretrained BERT
                max_seq_len=512, # Standard for pretrained BERT
                output_dim=self.config.text_encoder_dim,
                model_name=self.config.bert_model_name
            )
        else:
            raise ValueError(f"Unsupported text encoder type: {encoder_type}")
    
    def _create_fusion_strategy(self) -> nn.Module:
        """
        Create fusion strategy based on configuration.
        
        Returns:
            Fusion strategy instance
            
        Raises:
            ValueError: If fusion type is unsupported
        """
        fusion_type = self.config.fusion_type.lower()
        
        if fusion_type == "concatenation":
            return ConcatenationFusion(
                image_dim=self.config.image_encoder_dim,
                text_dim=self.config.text_encoder_dim,
                output_dim=self.config.fusion_dim
            )
        elif fusion_type == "coattention":
            return CoAttentionFusion(
                image_dim=self.config.image_encoder_dim,
                text_dim=self.config.text_encoder_dim,
                hidden_dim=self.config.coattention_hidden_dim,
                num_heads=self.config.coattention_num_heads,
                output_dim=self.config.fusion_dim
            )
        elif fusion_type == "bilinear":
            return BilinearFusion(
                image_dim=self.config.image_encoder_dim,
                text_dim=self.config.text_encoder_dim,
                hidden_dim=self.config.bilinear_hidden_dim,
                output_dim=self.config.fusion_dim
            )
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")
    
    def _print_config(self) -> None:
        """Print model configuration for debugging."""
        print(f"\n=== Modular VQA Model Configuration ===")
        print(f"Image Encoder: {self.image_encoder.name} ({self.config.image_encoder_dim}D)")
        print(f"Text Encoder: {self.text_encoder.name} ({self.config.text_encoder_dim}D)")
        print(f"Fusion Strategy: {self.fusion.name} ({self.fusion.output_dim}D)")
        print(f"Number of Classes: {self.config.num_classes}")
        print(f"Use Pretrained Models: {self.config.use_pretrained_models}")
        print(f"Total Parameters: {sum(p.numel() for p in self.parameters()):,}")
        print("=" * 40)
    
    def forward(self, 
                image: torch.Tensor, 
                text: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the modular VQA model.
        
        Args:
            image: Image tensor of shape (batch_size, channels, height, width)
            text: Text tensor of shape (batch_size, seq_len)
            attention_mask: Attention mask for BERT (optional)
            
        Returns:
            Logits of shape (batch_size, num_classes)
            
        Raises:
            ValueError: If input tensors have incorrect shapes
        """
        if image.dim() != 4:
            raise ValueError(f"Expected 4D image tensor, got shape {image.shape}")
        if text.dim() != 2:
            raise ValueError(f"Expected 2D text tensor, got shape {text.shape}")
        if image.size(0) != text.size(0):
            raise ValueError(f"Batch sizes must match: image {image.size(0)} vs text {text.size(0)}")
        
        # Encode image
        image_features = self.image_encoder(image)
        
        # Encode text
        if isinstance(self.text_encoder, BERTEncoder) and attention_mask is not None:
            text_features = self.text_encoder(text, attention_mask)
        else:
            text_features = self.text_encoder(text)
        
        # Fuse features
        fused_features = self.fusion(image_features, text_features)
        
        # Classify
        logits = self.classifier(fused_features)
        
        return logits
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the model architecture.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'image_encoder': self.image_encoder.get_config(),
            'text_encoder': self.text_encoder.get_config(),
            'fusion': self.fusion.get_config(),
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    
    def __str__(self) -> str:
        """String representation of the model."""
        return f"ModularVQAModel({self.config.image_encoder_type}+{self.config.text_encoder_type}+{self.config.fusion_type})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the model."""
        return self.__str__() 