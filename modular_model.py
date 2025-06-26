# modular_model.py
import torch
import torch.nn as nn
from encoders import CNNEncoder, LSTMEncoder, ViTEncoder, BERTEncoder
from fusion import ConcatenationFusion, CoAttentionFusion, BilinearFusion

class ModularVQAModel(nn.Module):
    """
    Modular VQA model that supports different encoders and fusion strategies.
    This allows easy experimentation with different combinations.
    """
    
    def __init__(self, config):
        super(ModularVQAModel, self).__init__()
        self.config = config
        
        # Initialize image encoder
        self.image_encoder = self._create_image_encoder()
        
        # Initialize text encoder
        self.text_encoder = self._create_text_encoder()
        
        # Initialize fusion strategy
        self.fusion = self._create_fusion_strategy()
        
        # Final classification layer
        self.classifier = nn.Linear(self.fusion.output_dim, config.num_classes)
        
        # Print model configuration
        self._print_config()
    
    def _create_image_encoder(self):
        """Create image encoder based on configuration."""
        encoder_type = self.config.image_encoder_type.lower()
        
        if encoder_type == "cnn":
            return CNNEncoder(
                input_channels=3,
                output_dim=self.config.image_encoder_dim
            )
        elif encoder_type == "vit":
            return ViTEncoder(
                img_size=self.config.image_size,
                patch_size=self.config.vit_patch_size,
                embed_dim=self.config.vit_embed_dim,
                num_heads=self.config.vit_num_heads,
                num_layers=self.config.vit_num_layers,
                output_dim=self.config.image_encoder_dim
            )
        else:
            raise ValueError(f"Unsupported image encoder type: {encoder_type}")
    
    def _create_text_encoder(self):
        """Create text encoder based on configuration."""
        encoder_type = self.config.text_encoder_type.lower()
        
        if encoder_type == "lstm":
            return LSTMEncoder(
                vocab_size=self.config.vocab_size,
                embedding_dim=self.config.lstm_embed_dim,
                hidden_size=self.config.lstm_hidden_size,
                num_layers=self.config.lstm_num_layers,
                output_dim=self.config.text_encoder_dim
            )
        elif encoder_type == "bert":
            return BERTEncoder(
                vocab_size=self.config.vocab_size,
                embed_dim=self.config.bert_embed_dim,
                num_heads=self.config.bert_num_heads,
                num_layers=self.config.bert_num_layers,
                ff_dim=self.config.bert_ff_dim,
                max_seq_len=self.config.max_seq_len,
                output_dim=self.config.text_encoder_dim
            )
        else:
            raise ValueError(f"Unsupported text encoder type: {encoder_type}")
    
    def _create_fusion_strategy(self):
        """Create fusion strategy based on configuration."""
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
    
    def _print_config(self):
        """Print model configuration for debugging."""
        print(f"\n=== Modular VQA Model Configuration ===")
        print(f"Image Encoder: {self.image_encoder.name} ({self.config.image_encoder_dim}D)")
        print(f"Text Encoder: {self.text_encoder.name} ({self.config.text_encoder_dim}D)")
        print(f"Fusion Strategy: {self.fusion.name} ({self.fusion.output_dim}D)")
        print(f"Number of Classes: {self.config.num_classes}")
        print(f"Total Parameters: {sum(p.numel() for p in self.parameters()):,}")
        print("=" * 40)
    
    def forward(self, image, text, attention_mask=None):
        """
        Forward pass through the modular VQA model.
        
        Args:
            image: Image tensor of shape (batch_size, channels, height, width)
            text: Text tensor of shape (batch_size, seq_len) or (batch_size, seq_len) for BERT
            attention_mask: Attention mask for BERT (optional)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
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
    
    def get_model_info(self):
        """Get detailed information about the model architecture."""
        return {
            'image_encoder': self.image_encoder.get_config(),
            'text_encoder': self.text_encoder.get_config(),
            'fusion': self.fusion.get_config(),
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        } 