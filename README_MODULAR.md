# Modular VQA System

This is a modular implementation of the Visual Question Answering (VQA) system that allows easy experimentation with different encoder and fusion combinations while maintaining clean, object-oriented code structure.

## 🏗️ Architecture Overview

The system is built with a modular design that separates concerns and allows easy swapping of components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Image Encoder │    │   Text Encoder  │    │  Fusion Strategy│
│                 │    │                 │    │                 │
│ • CNN           │    │ • LSTM          │    │ • Concatenation │
│ • ViT           │    │ • BERT          │    │ • Co-attention  │
│                 │    │                 │    │ • Bilinear      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Classifier    │
                    │                 │
                    │ Linear Layer    │
                    └─────────────────┘
```

## 📁 File Structure

```
VQA/
├── encoders/                    # Encoder implementations
│   ├── __init__.py
│   ├── base_encoder.py         # Abstract base class
│   ├── cnn_encoder.py          # CNN for images
│   ├── lstm_encoder.py         # LSTM for text
│   ├── vit_encoder.py          # Vision Transformer
│   └── bert_encoder.py         # BERT for text
├── fusion/                      # Fusion strategies
│   ├── __init__.py
│   ├── base_fusion.py          # Abstract base class
│   ├── concatenation_fusion.py # Simple concatenation
│   ├── coattention_fusion.py   # Co-attention mechanism
│   └── bilinear_fusion.py      # Bilinear interaction
├── modular_model.py            # Main modular model
├── modular_config.py           # Configuration management
├── modular_trainer.py          # Training logic
├── demo_modular_vqa.py         # Demo script
└── README_MODULAR.md           # This file
```

## 🚀 Quick Start

### 1. Run the Demo

```bash
python demo_modular_vqa.py
```

This will show you different encoder and fusion combinations and their parameter counts.

### 2. Use Preset Configurations

```python
from modular_config import ModularConfig
from modular_trainer import ModularTrainer

# Use ViT + BERT + Co-attention
config = ModularConfig().get_preset_config("vit_bert_coattention")

trainer = ModularTrainer(
    train_csv="data/data_train.csv",
    img_dir="data/images",
    answer_space_file="data/answer_space.txt",
    train_img_list="data/train_images_list.txt",
    config=config
)

# Train the model
trainer.train()
```

### 3. Create Custom Configuration

```python
from modular_config import ModularConfig

config = ModularConfig()
config.image_encoder_type = "vit"
config.text_encoder_type = "bert"
config.fusion_type = "bilinear"
config.image_encoder_dim = 256
config.text_encoder_dim = 256
config.fusion_dim = 512

# Use the custom config
trainer = ModularTrainer(..., config=config)
```

## 🔧 Available Components

### Image Encoders

1. **CNN Encoder** (`cnn`)
   - Traditional convolutional neural network
   - Good for spatial feature extraction
   - Efficient and well-understood

2. **Vision Transformer (ViT)** (`vit`)
   - Transformer-based image encoder
   - Captures global dependencies
   - State-of-the-art performance

### Text Encoders

1. **LSTM Encoder** (`lstm`)
   - Recurrent neural network
   - Good for sequential data
   - Efficient for short sequences

2. **BERT Encoder** (`bert`)
   - Transformer-based text encoder
   - Captures contextual relationships
   - Pre-trained language understanding

### Fusion Strategies

1. **Concatenation Fusion** (`concatenation`)
   - Simple concatenation of features
   - Baseline approach
   - Fast and memory efficient

2. **Co-attention Fusion** (`coattention`)
   - Cross-modal attention mechanism
   - Allows modalities to attend to each other
   - Captures complex interactions

3. **Bilinear Fusion** (`bilinear`)
   - Multiplicative interaction between modalities
   - Captures fine-grained relationships
   - More expressive than concatenation

## 📊 Preset Configurations

The system comes with several preset configurations:

| Preset Name | Image Encoder | Text Encoder | Fusion | Description |
|-------------|---------------|--------------|---------|-------------|
| `original` | CNN | LSTM | Concatenation | Original implementation |
| `vit_bert_coattention` | ViT | BERT | Co-attention | Modern transformer-based |
| `vit_bert_bilinear` | ViT | BERT | Bilinear | Transformer with bilinear fusion |
| `cnn_bert_coattention` | CNN | BERT | Co-attention | Hybrid approach |

## 🎯 Usage Examples

### Example 1: Train with ViT + BERT + Co-attention

```python
from modular_config import ModularConfig
from modular_trainer import ModularTrainer

# Load preset configuration
config = ModularConfig().get_preset_config("vit_bert_coattention")

# Create trainer
trainer = ModularTrainer(
    train_csv="data/data_train.csv",
    img_dir="data/images",
    answer_space_file="data/answer_space.txt",
    train_img_list="data/train_images_list.txt",
    config=config
)

# Train the model
trainer.train()

# Save the model
trainer.save_model("vit_bert_coattention_model.pth")
```

### Example 2: Experiment with Different Configurations

```python
from modular_config import ModularConfig

# Test different combinations
configs = [
    ("cnn_lstm", {"image_encoder_type": "cnn", "text_encoder_type": "lstm", "fusion_type": "concatenation"}),
    ("vit_bert", {"image_encoder_type": "vit", "text_encoder_type": "bert", "fusion_type": "coattention"}),
    ("cnn_bert", {"image_encoder_type": "cnn", "text_encoder_type": "bert", "fusion_type": "bilinear"}),
]

for name, config_dict in configs:
    config = ModularConfig()
    for key, value in config_dict.items():
        setattr(config, key, value)
    
    print(f"\nTraining {name} configuration...")
    trainer = ModularTrainer(..., config=config)
    trainer.train()
```

### Example 3: Custom Model Architecture

```python
from modular_config import ModularConfig

# Create custom configuration
config = ModularConfig()

# Image encoder settings
config.image_encoder_type = "vit"
config.image_encoder_dim = 384
config.vit_embed_dim = 384
config.vit_num_layers = 8
config.vit_num_heads = 12

# Text encoder settings
config.text_encoder_type = "bert"
config.text_encoder_dim = 384
config.bert_embed_dim = 384
config.bert_num_layers = 6
config.bert_num_heads = 12

# Fusion settings
config.fusion_type = "coattention"
config.fusion_dim = 768
config.coattention_hidden_dim = 384
config.coattention_num_heads = 12

# Training settings
config.batch_size = 4
config.learning_rate = 0.0001
config.num_epochs = 10

# Use the custom config
trainer = ModularTrainer(..., config=config)
```

## 🔍 Model Information

You can get detailed information about any model:

```python
# Get model architecture info
model_info = trainer.get_model_info()
print(f"Total Parameters: {model_info['total_parameters']:,}")
print(f"Trainable Parameters: {model_info['trainable_parameters']:,}")

# Get encoder configurations
print(f"Image Encoder: {model_info['image_encoder']}")
print(f"Text Encoder: {model_info['text_encoder']}")
print(f"Fusion: {model_info['fusion']}")
```

## 💾 Model Persistence

### Save Model

```python
# Save with all metadata
trainer.save_model("my_model.pth")

# This saves:
# - Model state dict
# - Configuration
# - Vocabulary
# - Answer mapping
```

### Load Model

```python
# Load saved model
trainer.load_model("my_model.pth")
```

## 🎨 Adding New Components

### Adding a New Encoder

1. Create a new encoder class in `encoders/`:

```python
# encoders/my_encoder.py
from .base_encoder import BaseEncoder

class MyEncoder(BaseEncoder):
    def __init__(self, **kwargs):
        super().__init__(name="MyEncoder")
        # Your implementation here
    
    def forward(self, x):
        # Your forward pass here
        return features
```

2. Add it to `encoders/__init__.py`:

```python
from .my_encoder import MyEncoder
__all__ = [..., 'MyEncoder']
```

3. Add support in `modular_model.py`:

```python
def _create_image_encoder(self):
    # ... existing code ...
    elif encoder_type == "my_encoder":
        return MyEncoder(...)
```

### Adding a New Fusion Strategy

1. Create a new fusion class in `fusion/`:

```python
# fusion/my_fusion.py
from .base_fusion import BaseFusion

class MyFusion(BaseFusion):
    def __init__(self, image_dim, text_dim, **kwargs):
        super().__init__(name="MyFusion", image_dim=image_dim, text_dim=text_dim)
        # Your implementation here
    
    def forward(self, image_features, text_features):
        # Your fusion logic here
        return fused_features
```

2. Add it to the modular model and configuration.

## 🚀 Performance Tips

1. **Memory Efficiency**: Use smaller embedding dimensions for faster training
2. **Batch Size**: Adjust based on your GPU memory
3. **Learning Rate**: Start with 0.001 and adjust based on convergence
4. **Model Size**: Use fewer layers for faster experimentation

## 🔧 Configuration Parameters

### Image Encoder Parameters

- `image_encoder_type`: "cnn" or "vit"
- `image_encoder_dim`: Output dimension of image encoder
- `image_size`: Input image size (for ViT)
- `vit_patch_size`: Patch size for ViT
- `vit_embed_dim`: Embedding dimension for ViT
- `vit_num_heads`: Number of attention heads for ViT
- `vit_num_layers`: Number of transformer layers for ViT

### Text Encoder Parameters

- `text_encoder_type`: "lstm" or "bert"
- `text_encoder_dim`: Output dimension of text encoder
- `vocab_size`: Vocabulary size (set automatically)
- `lstm_embed_dim`: Embedding dimension for LSTM
- `lstm_hidden_size`: Hidden size for LSTM
- `lstm_num_layers`: Number of LSTM layers
- `bert_embed_dim`: Embedding dimension for BERT
- `bert_num_heads`: Number of attention heads for BERT
- `bert_num_layers`: Number of transformer layers for BERT
- `bert_ff_dim`: Feed-forward dimension for BERT
- `max_seq_len`: Maximum sequence length

### Fusion Parameters

- `fusion_type`: "concatenation", "coattention", or "bilinear"
- `fusion_dim`: Output dimension of fusion
- `coattention_hidden_dim`: Hidden dimension for co-attention
- `coattention_num_heads`: Number of attention heads for co-attention
- `bilinear_hidden_dim`: Hidden dimension for bilinear fusion

### Training Parameters

- `batch_size`: Training batch size
- `num_epochs`: Number of training epochs
- `learning_rate`: Learning rate
- `device`: Training device (CPU/GPU)

## 🤝 Contributing

To add new components:

1. Follow the existing code structure
2. Inherit from base classes
3. Implement required methods
4. Add proper documentation
5. Update the demo script
6. Test with different configurations

## 📝 License

This implementation follows the same license as the original VQA project. 