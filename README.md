# Modular Visual Question Answering (VQA) System

A clean, modular, and extensible Visual Question Answering system built with PyTorch that supports multiple encoder and fusion strategies with pretrained models.

## 🚀 Features

- **Modular Architecture**: Easy to experiment with different encoder and fusion combinations
- **Pretrained Models**: Support for BERT, ViT, ResNet, and EfficientNet from Hugging Face
- **Multiple Fusion Strategies**: Concatenation, Co-attention, and Bilinear fusion
- **Clean Code**: Well-documented, type-hinted, and follows OOP principles
- **GPU Optimization**: Automatic GPU detection and memory management
- **Easy Configuration**: Preset configurations for common combinations

## 📁 Project Structure

```
VQA/
├── encoders/                 # Image and text encoders
│   ├── base_encoder.py      # Abstract base class
│   ├── cnn_encoder.py       # CNN encoder (ResNet, EfficientNet)
│   ├── vit_encoder.py       # Vision Transformer encoder
│   ├── bert_encoder.py      # BERT text encoder
│   └── lstm_encoder.py      # LSTM text encoder
├── fusion/                  # Fusion strategies
│   ├── base_fusion.py       # Abstract base class
│   ├── concatenation_fusion.py
│   ├── coattention_fusion.py
│   └── bilinear_fusion.py
├── utils/                   # Utility modules
│   ├── helper.py           # Helper functions
│   ├── gpu_utils.py        # GPU utilities
│   └── dataloader_utils.py # DataLoader optimization
├── data/                   # Dataset files
├── modular_config.py       # Configuration management
├── modular_model.py        # Main VQA model
├── modular_trainer.py      # Training logic
├── main_modular.py         # Main training script
├── prepare_data.py         # Data preparation
├── demo_pretrained_vqa.py  # Demo script
└── requirements.txt        # Dependencies
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd VQA
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install CUDA PyTorch** (for GPU support):
   ```bash
   # For CUDA 11.8 (adjust version as needed)
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## 📊 Available Presets

The system comes with several preset configurations:

| Preset | Image Encoder | Text Encoder | Fusion | Description |
|--------|---------------|--------------|---------|-------------|
| `original` | Custom CNN | LSTM | Concatenation | Original implementation |
| `pretrained_cnn_lstm` | ResNet50 | LSTM | Concatenation | Pretrained CNN + LSTM |
| `pretrained_vit_bert` | ViT | BERT | Concatenation | Modern transformer-based |
| `pretrained_vit_bert_coattention` | ViT | BERT | Co-attention | Advanced attention fusion |
| `pretrained_vit_bert_bilinear` | ViT | BERT | Bilinear | Bilinear fusion |
| `pretrained_cnn_bert` | ResNet50 | BERT | Concatenation | CNN + BERT combination |
| `pretrained_cnn_bert_coattention` | ResNet50 | BERT | Co-attention | CNN + BERT + Co-attention |
| `pretrained_cnn_bert_bilinear` | ResNet50 | BERT | Bilinear | CNN + BERT + Bilinear |
| `efficient_vit_bert` | EfficientNet | BERT | Concatenation | Efficient architecture |

## 🚀 Quick Start

### 1. Prepare Data

First, prepare your dataset:

```bash
python prepare_data.py
```

This will:
- Split your data into train/validation/test sets
- Create corresponding image list files
- Set up the data structure for training

### 2. Train a Model

Train with a preset configuration:

```bash
# Train with ViT + BERT + Co-attention (default)
python main_modular.py

# Train with a specific preset
python main_modular.py pretrained_cnn_bert_bilinear

# Train with custom preset
python main_modular.py pretrained_vit_bert
```

### 3. Demo

Run the demo to see the system in action:

```bash
python demo_pretrained_vqa.py
```

## 🔧 Configuration

### Using Presets

```python
from modular_config import ModularConfig

# Use a preset
config = ModularConfig()
config.get_preset_config("pretrained_vit_bert_coattention")

# Customize parameters
config.batch_size = 4
config.learning_rate = 0.0001
config.num_epochs = 10
```

### Custom Configuration

```python
from modular_config import ModularConfig

config = ModularConfig()

# Set encoders
config.image_encoder_type = "vit"
config.text_encoder_type = "bert"
config.fusion_type = "coattention"

# Set dimensions
config.image_encoder_dim = 768
config.text_encoder_dim = 768
config.fusion_dim = 1536

# Set pretrained models
config.vit_model_name = "google/vit-base-patch16-224"
config.bert_model_name = "bert-base-uncased"
```

## 🏗️ Architecture

### Base Classes

The system uses abstract base classes for extensibility:

- **BaseEncoder**: Abstract base for all encoders
- **BaseFusion**: Abstract base for all fusion strategies

### Adding New Components

#### New Encoder

```python
from encoders.base_encoder import BaseEncoder

class MyEncoder(BaseEncoder):
    def __init__(self, name: str, **kwargs):
        super().__init__(name)
        # Your implementation
        
    def forward(self, x):
        # Your forward pass
        return output
```

#### New Fusion Strategy

```python
from fusion.base_fusion import BaseFusion

class MyFusion(BaseFusion):
    def __init__(self, name: str, image_dim: int, text_dim: int):
        super().__init__(name, image_dim, text_dim)
        # Your implementation
        
    def forward(self, image_features, text_features):
        # Your fusion logic
        return fused_features
```

## 📈 Training

### Training Process

1. **Data Loading**: Efficient DataLoader with optimal settings
2. **Model Creation**: Automatic component initialization
3. **Training Loop**: With validation and early stopping
4. **Monitoring**: Loss and accuracy tracking
5. **Saving**: Best model checkpoint

### Training Output

```
🚀 Starting Modular VQA Training
==================================================
Applied preset configuration: pretrained_vit_bert_coattention

=== Modular VQA Configuration ===
Image Encoder: ViT (768D)
Text Encoder: BERT (768D)
Fusion Strategy: CoAttention (1536D)
Use Pretrained Models: True
ViT Model: google/vit-base-patch16-224 (pretrained: True)
BERT Model: bert-base-uncased (pretrained: True)
Batch Size: 2
Learning Rate: 0.001
Device: cuda
DataLoader Workers: 0
Pin Memory: True
Persistent Workers: False
===================================

✅ Data loaders created:
   Train: 4756 samples, 2378 batches
   Val: 1359 samples, 680 batches

=== Modular VQA Model Configuration ===
Image Encoder: ViT (768D)
Text Encoder: BERT (768D)
Fusion Strategy: CoAttention (1536D)
Number of Classes: 3
Use Pretrained Models: True
Total Parameters: 110,595,843
========================================

🎯 Training with preset: pretrained_vit_bert_coattention
Epoch 1/5: 100%|██████████| 2378/2378 [05:23<00:00, 7.37it/s]
Train Loss: 0.9876, Train Acc: 0.4567
Val Loss: 0.9234, Val Acc: 0.4789
```

## 🔍 Monitoring

### Training Metrics

- **Loss**: Training and validation loss
- **Accuracy**: Training and validation accuracy
- **Learning Curves**: Automatic plotting
- **Early Stopping**: Prevents overfitting

### GPU Monitoring

```python
from utils.gpu_utils import GPUMonitor

monitor = GPUMonitor()
monitor.print_gpu_info()
monitor.monitor_gpu_usage()
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size
   - Use smaller models
   - Enable gradient accumulation

2. **Slow Training**:
   - Increase num_workers
   - Enable pin_memory
   - Use persistent_workers

3. **Import Errors**:
   - Check virtual environment
   - Install missing dependencies
   - Verify Python path

### Performance Optimization

```python
# Optimize DataLoader
config.num_workers = 4  # Adjust based on CPU cores
config.pin_memory = True
config.persistent_workers = True

# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler
```

## 📝 Code Quality

### OOP Principles

- **Encapsulation**: Private methods and attributes
- **Inheritance**: Base classes for common functionality
- **Polymorphism**: Interface-based design
- **Abstraction**: Abstract base classes

### Clean Code Practices

- **Type Hints**: Full type annotation
- **Documentation**: Comprehensive docstrings
- **Error Handling**: Proper exception handling
- **Validation**: Input parameter validation
- **Separation of Concerns**: Modular design

### Code Style

- **PEP 8**: Python style guide compliance
- **Naming**: Clear and descriptive names
- **Comments**: Meaningful documentation
- **Structure**: Logical organization

## 🤝 Contributing

1. Follow the existing code style
2. Add type hints and documentation
3. Include error handling
4. Write tests for new features
5. Update documentation

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face for pretrained models
- PyTorch team for the framework
- VQA community for research insights 