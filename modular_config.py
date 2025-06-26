# modular_config.py
import torch
import os

class ModularConfig:
    """
    Extended configuration class for modular VQA model.
    Supports different encoders and fusion strategies.
    """
    
    def __init__(self):
        # Basic training parameters
        self.batch_size = 2
        self.num_epochs = 5
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = 3
        
        # DataLoader parameters
        self.num_workers = 0  # 0 for Windows, 2-4 for Linux/Mac, adjust based on CPU cores
        self.pin_memory = True if torch.cuda.is_available() else False
        self.persistent_workers = False  # Set to True if num_workers > 0 for better performance
        
        # Encoder types
        self.image_encoder_type = "cnn"  # Options: "cnn", "vit"
        self.text_encoder_type = "lstm"  # Options: "lstm", "bert"
        
        # Fusion type
        self.fusion_type = "concatenation"  # Options: "concatenation", "coattention", "bilinear"
        
        # Image encoder dimensions
        self.image_encoder_dim = 512
        self.image_size = 64  # For ViT
        
        # Text encoder dimensions
        self.text_encoder_dim = 128
        self.vocab_size = 1000  # Will be set dynamically
        
        # Fusion dimensions
        self.fusion_dim = 640  # Default: image_dim + text_dim
        
        # CNN encoder parameters (already set in CNNEncoder)
        
        # LSTM encoder parameters
        self.lstm_embed_dim = 50
        self.lstm_hidden_size = 128
        self.lstm_num_layers = 1
        
        # ViT encoder parameters
        self.vit_patch_size = 8  # Smaller patches for 64x64 images
        self.vit_embed_dim = 256  # Smaller for efficiency
        self.vit_num_heads = 8
        self.vit_num_layers = 6  # Fewer layers for efficiency
        
        # BERT encoder parameters
        self.bert_embed_dim = 256  # Smaller for efficiency
        self.bert_num_heads = 8
        self.bert_num_layers = 4  # Fewer layers for efficiency
        self.bert_ff_dim = 1024
        self.max_seq_len = 64  # Shorter sequences for VQA
        
        # Co-attention fusion parameters
        self.coattention_hidden_dim = 256
        self.coattention_num_heads = 8
        
        # Bilinear fusion parameters
        self.bilinear_hidden_dim = 256
    
    def set_vocab_size(self, vocab_size: int):
        """Set vocabulary size dynamically."""
        self.vocab_size = vocab_size
    
    def set_num_classes(self, num_classes: int):
        """Set number of classes dynamically."""
        self.num_classes = num_classes
    
    def get_preset_config(self, preset_name: str):
        """Get a preset configuration for common combinations."""
        presets = {
            "original": {
                "image_encoder_type": "cnn",
                "text_encoder_type": "lstm", 
                "fusion_type": "concatenation",
                "image_encoder_dim": 512,
                "text_encoder_dim": 128,
                "fusion_dim": 640
            },
            "vit_bert_coattention": {
                "image_encoder_type": "vit",
                "text_encoder_type": "bert",
                "fusion_type": "coattention",
                "image_encoder_dim": 256,
                "text_encoder_dim": 256,
                "fusion_dim": 512,
                "vit_embed_dim": 256,
                "vit_num_layers": 6,
                "bert_embed_dim": 256,
                "bert_num_layers": 4,
                "coattention_hidden_dim": 256
            },
            "vit_bert_bilinear": {
                "image_encoder_type": "vit",
                "text_encoder_type": "bert",
                "fusion_type": "bilinear",
                "image_encoder_dim": 256,
                "text_encoder_dim": 256,
                "fusion_dim": 512,
                "vit_embed_dim": 256,
                "vit_num_layers": 6,
                "bert_embed_dim": 256,
                "bert_num_layers": 4,
                "bilinear_hidden_dim": 256
            },
            "cnn_bert_coattention": {
                "image_encoder_type": "cnn",
                "text_encoder_type": "bert",
                "fusion_type": "coattention",
                "image_encoder_dim": 512,
                "text_encoder_dim": 256,
                "fusion_dim": 512,
                "bert_embed_dim": 256,
                "bert_num_layers": 4,
                "coattention_hidden_dim": 256
            },
            "cnn_bert_bilinear": {
                "image_encoder_type": "cnn",
                "text_encoder_type": "bert",
                "fusion_type": "bilinear",
                "image_encoder_dim": 512,
                "text_encoder_dim": 256,
                "fusion_dim": 512,
                "bert_embed_dim": 256,
                "bert_num_layers": 4,
                "bilinear_hidden_dim": 256
            }
        }
        
        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}. Available presets: {list(presets.keys())}")
        
        # Apply preset configuration
        for key, value in presets[preset_name].items():
            setattr(self, key, value)
        
        print(f"Applied preset configuration: {preset_name}")
        return self
    
    def print_config(self):
        """Print current configuration."""
        print("\n=== Modular VQA Configuration ===")
        print(f"Image Encoder: {self.image_encoder_type} ({self.image_encoder_dim}D)")
        print(f"Text Encoder: {self.text_encoder_type} ({self.text_encoder_dim}D)")
        print(f"Fusion Strategy: {self.fusion_type} ({self.fusion_dim}D)")
        print(f"Batch Size: {self.batch_size}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Device: {self.device}")
        print(f"DataLoader Workers: {self.num_workers}")
        print(f"Pin Memory: {self.pin_memory}")
        print(f"Persistent Workers: {self.persistent_workers}")
        print("=" * 35) 