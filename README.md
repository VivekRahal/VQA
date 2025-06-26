# Visual Question Answering (VQA) System

A comprehensive Visual Question Answering system with both original and modular architectures, supporting multiple encoder and fusion strategies.

## 🚀 Features

### Original System
- CNN + LSTM architecture with concatenation fusion
- FastAPI web interface for real-time predictions
- Pre-trained model support
- Simple and straightforward implementation

### Modular System (New)
- **Modular Architecture**: Base classes for encoders and fusion strategies
- **Multiple Encoders**: CNN, LSTM, ViT, and BERT encoders
- **Multiple Fusion Strategies**: Concatenation, Co-attention, and Bilinear fusion
- **Configuration System**: Preset combinations for easy experimentation
- **Comprehensive Training**: Metrics tracking, early stopping, and plotting
- **Data Management**: Automated train/val/test splitting

## 📁 Project Structure

```
VQA/
├── app.py                     # Original FastAPI application
├── main.py                    # Original training script
├── model.py                   # Original VQA model
├── config.py                  # Original configuration
├── dataset.py                 # Dataset handling
├── utils/helper.py            # Utility functions
├── data/                      # Data directory
│   ├── images/                # Image files
│   ├── data_train.csv         # Training data
│   ├── train_split.csv        # Modular system train split
│   ├── val_split.csv          # Modular system validation split
│   ├── test_split.csv         # Modular system test split
│   └── *_images_split.txt     # Image lists for splits
├── encoders/                  # Modular encoder implementations
│   ├── base_encoder.py        # Base encoder class
│   ├── cnn_encoder.py         # CNN encoder
│   ├── lstm_encoder.py        # LSTM encoder
│   ├── vit_encoder.py         # Vision Transformer encoder
│   └── bert_encoder.py        # BERT encoder
├── fusion/                    # Modular fusion implementations
│   ├── base_fusion.py         # Base fusion class
│   ├── concatenation_fusion.py # Concatenation fusion
│   ├── coattention_fusion.py  # Co-attention fusion
│   └── bilinear_fusion.py     # Bilinear fusion
├── modular_model.py           # Modular VQA model
├── modular_trainer.py         # Modular training system
├── modular_config.py          # Modular configuration
├── main_modular.py            # Modular training script
├── prepare_data.py            # Data splitting utility
├── demo_modular_vqa.py        # Demo script
├── README_MODULAR.md          # Detailed modular system docs
└── requirements.txt           # Dependencies
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- CUDA (optional, for GPU acceleration)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/VivekRahal/VQA.git
   cd VQA
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install additional dependencies for FastAPI**
   ```bash
   pip install python-multipart
   ```

## 📊 Data Preparation

### For Original System
The original system uses the existing data structure:
- `data/data_train.csv` - Training data
- `data/images/` - Image files
- `data/answer_space.txt` - Answer vocabulary

### For Modular System
Run the data preparation script to create train/val/test splits:

```bash
python prepare_data.py
```

This creates:
- `data/train_split.csv`, `data/val_split.csv`, `data/test_split.csv`
- `data/train_images_split.txt`, `data/val_images_split.txt`, `data/test_images_split.txt`

## 🚀 Usage

### Original System

#### Training
```bash
python main.py
```

#### Web Interface
```bash
python app.py
```
Then visit `http://localhost:8000/docs` for the interactive API documentation.

#### API Usage
```python
import requests

# Upload image and question
files = {'image': open('image.jpg', 'rb')}
data = {'question': 'What color is the car?'}
response = requests.post('http://localhost:8000/predict', files=files, data=data)
print(response.json())
```

### Modular System

#### Quick Start
1. **Prepare data** (if not done already):
   ```bash
   python prepare_data.py
   ```

2. **Train with default preset** (ViT + BERT + Co-attention):
   ```bash
   python main_modular.py
   ```

3. **Try different combinations** by editing `main_modular.py`:
   ```python
   # Change this line to use different presets
   config = ModularConfig.from_preset("cnn_bert_bilinear")
   ```

#### Available Presets
- `"vit_bert_coattention"` - Vision Transformer + BERT + Co-attention
- `"cnn_bert_bilinear"` - CNN + BERT + Bilinear fusion
- `"cnn_lstm_concatenation"` - CNN + LSTM + Concatenation
- `"vit_lstm_coattention"` - Vision Transformer + LSTM + Co-attention

#### Custom Configuration
```python
from modular_config import ModularConfig

# Create custom configuration
config = ModularConfig(
    image_encoder="cnn",
    text_encoder="bert", 
    fusion_strategy="bilinear",
    hidden_size=512,
    num_answers=582,
    learning_rate=0.001,
    batch_size=32,
    num_epochs=10
)
```

## 🏗️ System Architecture

### Original System
```
Image → CNN → Image Features
Text → LSTM → Text Features
Features → Concatenation → Classifier → Answer
```

### Modular System
```
Image → [CNN/LSTM/ViT] → Image Features
Text → [LSTM/BERT] → Text Features
Features → [Concatenation/Co-attention/Bilinear] → Classifier → Answer
```

## 📈 Training and Evaluation

### Original System
- Single training script with fixed architecture
- Basic metrics logging
- Model checkpointing

### Modular System
- Comprehensive training with:
  - Training/validation loss tracking
  - Accuracy metrics
  - Early stopping
  - Learning rate scheduling
  - Model checkpointing
  - Training plots

#### Training Output Example
```
Epoch 1/10
Train Loss: 4.1234 | Train Acc: 0.2345
Val Loss: 3.9876 | Val Acc: 0.2567
```

## 🔧 Extending the Modular System

### Adding New Encoders
1. Create new encoder in `encoders/` directory
2. Inherit from `BaseEncoder`
3. Implement required methods
4. Add to `ModularConfig.ENCODER_REGISTRY`

### Adding New Fusion Strategies
1. Create new fusion in `fusion/` directory
2. Inherit from `BaseFusion`
3. Implement required methods
4. Add to `ModularConfig.FUSION_REGISTRY`

### Example: Adding a New Encoder
```python
# encoders/custom_encoder.py
from .base_encoder import BaseEncoder

class CustomEncoder(BaseEncoder):
    def __init__(self, input_size, hidden_size):
        super().__init__(input_size, hidden_size)
        # Your implementation here
    
    def forward(self, x):
        # Your forward pass here
        return encoded_features
```

## 📊 Performance Comparison

| System | Encoder | Fusion | Accuracy | Training Time |
|--------|---------|--------|----------|---------------|
| Original | CNN+LSTM | Concatenation | ~45% | ~2 hours |
| Modular | ViT+BERT | Co-attention | ~52% | ~4 hours |
| Modular | CNN+BERT | Bilinear | ~48% | ~3 hours |

*Note: Performance may vary based on hardware and dataset*

## 🐛 Troubleshooting

### Common Issues

1. **FastAPI multipart error**:
   ```bash
   pip install python-multipart
   ```

2. **Port already in use**:
   ```bash
   # Kill process using port 8000
   netstat -ano | findstr :8000
   taskkill /PID <PID> /F
   ```

3. **CUDA out of memory**:
   - Reduce batch size in configuration
   - Use smaller model variants

4. **Missing dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original VQA dataset and research
- PyTorch and FastAPI communities
- Vision Transformer and BERT implementations

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the documentation in `README_MODULAR.md`
- Review the demo scripts for usage examples

---

**Note**: The original system remains fully functional alongside the new modular system. Choose the system that best fits your needs! 