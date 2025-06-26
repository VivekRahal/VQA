# VQA with Pretrained Encoders from Hugging Face

This project now supports **pretrained encoders** from Hugging Face, significantly improving VQA performance by leveraging models that have already learned powerful representations from large datasets.

## 🚀 What's New

### **Pretrained Models Available:**

#### **Image Encoders:**
- **ResNet Models**: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`
- **EfficientNet Models**: `efficientnet_b0`, `efficientnet_b1`, `efficientnet_b2`
- **Vision Transformer (ViT)**: `google/vit-base-patch16-224`, `google/vit-large-patch16-224`

#### **Text Encoders:**
- **BERT Models**: `bert-base-uncased`, `bert-large-uncased`, `distilbert-base-uncased`
- **LSTM with Pretrained Embeddings**: `glove-wiki-gigaword-50`

## 📋 Quick Start

### 1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 2. **Prepare Data** (if not done already)
```bash
python prepare_data.py
```

### 3. **Run Demo**
```bash
python demo_pretrained_vqa.py
```

### 4. **Train with Pretrained Models**
```bash
python main_modular.py
```

## 🎯 Available Presets

### **Pretrained Combinations:**

| Preset Name | Image Encoder | Text Encoder | Fusion | Description |
|-------------|---------------|--------------|---------|-------------|
| `pretrained_cnn_lstm` | ResNet50 | LSTM | Concatenation | Classic CNN + LSTM |
| `pretrained_cnn_bert` | ResNet50 | BERT | Concatenation | ResNet + BERT |
| `pretrained_cnn_bert_coattention` | ResNet50 | BERT | Co-attention | ResNet + BERT + Co-attention |
| `pretrained_cnn_bert_bilinear` | ResNet50 | BERT | Bilinear | ResNet + BERT + Bilinear |
| `pretrained_vit_bert` | ViT | BERT | Concatenation | Vision Transformer + BERT |
| `pretrained_vit_bert_coattention` | ViT | BERT | Co-attention | ViT + BERT + Co-attention |
| `pretrained_vit_bert_bilinear` | ViT | BERT | Bilinear | ViT + BERT + Bilinear |
| `efficient_vit_bert` | EfficientNet | BERT | Concatenation | EfficientNet + BERT |

### **Custom Combinations:**
| Preset Name | Image Encoder | Text Encoder | Fusion | Description |
|-------------|---------------|--------------|---------|-------------|
| `original` | Custom CNN | Custom LSTM | Concatenation | Original implementation |

## 🔧 How to Use

### **1. Change Preset in main_modular.py**
```python
# In main_modular.py, change this line:
config = ModularConfig().get_preset_config("pretrained_cnn_bert_bilinear")

# To try different combinations:
config = ModularConfig().get_preset_config("pretrained_vit_bert_coattention")
config = ModularConfig().get_preset_config("efficient_vit_bert")
```

### **2. Custom Configuration**
```python
from modular_config import ModularConfig

config = ModularConfig()

# Enable pretrained models
config.use_pretrained_models = True

# Choose specific models
config.image_encoder_type = "cnn"
config.cnn_model_name = "resnet101"  # Use ResNet101 instead of ResNet50
config.text_encoder_type = "bert"
config.bert_model_name = "bert-large-uncased"  # Use BERT-Large

# Set fusion strategy
config.fusion_type = "coattention"
```

## 📊 Model Specifications

### **ResNet Models:**
- **ResNet18/34**: 512D output
- **ResNet50/101/152**: 2048D output
- **EfficientNet-B0**: 1280D output
- **EfficientNet-B1**: 1280D output
- **EfficientNet-B2**: 1408D output

### **BERT Models:**
- **BERT-Base**: 768D output
- **BERT-Large**: 1024D output
- **DistilBERT**: 768D output

### **ViT Models:**
- **ViT-Base**: 768D output
- **ViT-Large**: 1024D output

## 🎯 Performance Benefits

### **Why Pretrained Models?**

1. **Better Representations**: Models trained on millions of images/text samples
2. **Faster Convergence**: Start with good features, learn task-specific patterns
3. **Higher Accuracy**: Leverage knowledge from large-scale pretraining
4. **Transfer Learning**: Apply knowledge from one domain to VQA

### **Expected Improvements:**
- **Training Speed**: 2-5x faster convergence
- **Accuracy**: 10-30% improvement over custom encoders
- **Robustness**: Better generalization to unseen data

## 🔧 Technical Details

### **Model Downloads:**
- **First Run**: Models are automatically downloaded from Hugging Face
- **Storage**: ~500MB for BERT, ~100MB for ResNet, ~200MB for ViT
- **Caching**: Models are cached locally for future use

### **Memory Requirements:**
- **ResNet50 + BERT**: ~2-3GB GPU memory
- **ViT + BERT**: ~3-4GB GPU memory
- **EfficientNet + BERT**: ~2-3GB GPU memory

### **Training Tips:**
```python
# For limited GPU memory:
config.batch_size = 4  # Reduce batch size
config.learning_rate = 0.0001  # Lower learning rate

# For faster training:
config.batch_size = 16  # Increase if memory allows
config.learning_rate = 0.0005  # Slightly higher learning rate
```

## 🚨 Important Notes

### **Image Size Requirements:**
- **ResNet/EfficientNet**: Expects 224x224 images
- **ViT**: Expects 224x224 images
- **Custom CNN**: Works with 64x64 images

### **Text Processing:**
- **BERT**: Uses BERT tokenizer, handles variable length sequences
- **LSTM**: Uses custom vocabulary, requires padding

### **Fine-tuning Options:**
```python
# Freeze pretrained models (faster training, less memory)
# Uncomment in encoder files:
# self.bert.requires_grad_(False)  # Freeze BERT
# self.cnn.requires_grad_(False)   # Freeze CNN
# self.vit.requires_grad_(False)   # Freeze ViT
```

## 📈 Monitoring Training

### **Key Metrics to Watch:**
- **Training Loss**: Should decrease steadily
- **Validation Loss**: Should follow training loss
- **GPU Memory**: Monitor usage with pretrained models
- **Download Progress**: First run downloads models

### **Troubleshooting:**
```bash
# If models fail to download:
pip install --upgrade transformers torch torchvision

# If out of memory:
# Reduce batch_size in config
# Use CPU: config.device = torch.device("cpu")

# If training is slow:
# Increase batch_size if memory allows
# Use smaller models (ResNet18 instead of ResNet50)
```

## 🎉 Next Steps

1. **Experiment**: Try different pretrained model combinations
2. **Fine-tune**: Adjust learning rates and batch sizes
3. **Evaluate**: Compare performance across different presets
4. **Customize**: Create your own encoder combinations

## 📚 Additional Resources

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PyTorch Vision Models](https://pytorch.org/vision/stable/models.html)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [ViT Paper](https://arxiv.org/abs/2010.11929)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)

---

**Happy experimenting with pretrained encoders! 🚀** 