Visual Question Answering (VQA) System
A modular, object-oriented Visual Question Answering system with support for multiple encoder and fusion strategies. This project includes both training capabilities and a modern web interface for real-time VQA interactions.

🚀 Live Demo
Try the VQA Chat Application on Hugging Face Spaces: Hugging Face Spaces

🎯 Features
Core VQA System
Modular Architecture: Easy experimentation with different encoder and fusion combinations
Multiple Encoders: CNN, LSTM, ViT, BERT with pretrained model support
Fusion Strategies: Concatenation, Co-attention, Bilinear fusion
Pretrained Models: Support for Hugging Face transformers and torchvision models
GPU Optimization: Efficient memory management and CUDA support
Web Application
Modern Chat UI: Beautiful, responsive interface with real-time messaging
WebSocket Support: Instant communication for seamless user experience
Image Upload: Drag-and-drop functionality with preview
Model Management: Easy model loading and status monitoring
Cross-platform: Works on desktop and mobile browsers
🏗️ Architecture
Modular Components
VQA System
├── Encoders/
│   ├── CNN Encoder (ResNet, EfficientNet, etc.)
│   ├── LSTM Encoder (Bidirectional, Attention)
│   ├── ViT Encoder (Vision Transformer)
│   └── BERT Encoder (Text Transformer)
├── Fusion/
│   ├── Concatenation Fusion
│   ├── Co-attention Fusion
│   └── Bilinear Fusion
├── Model/
│   └── Modular VQA Model
├── Training/
│   └── Modular Trainer
└── Web App/
    └── FastAPI Chat Interface
Supported Combinations
Image Encoder	Text Encoder	Fusion Strategy	Use Case
CNN	LSTM	Concatenation	Baseline
CNN	BERT	Bilinear	Enhanced Text Understanding
ViT	BERT	Co-attention	State-of-the-art
CNN	BERT	Concatenation	Balanced Performance
🛠️ Installation
Prerequisites
Python 3.8+
PyTorch 2.0+
CUDA (optional, for GPU acceleration)
Setup
# Clone the repository
git clone https://github.com/your-username/VQA.git
cd VQA

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
📚 Usage
1. Data Preparation
# Prepare data splits and image lists
python prepare_data.py
2. Training
# Train with default preset (ViT + BERT + Co-attention)
python main_modular.py

# Train with custom preset
python main_modular.py --preset cnn_bert_bilinear
3. Web Application
# Start the chat application
python app.py

# Or use uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
4. Testing
# Test the web application
python test_app.py

# Create demo model for testing
python create_demo_model.py
🔧 Configuration
Available Presets
The system comes with several predefined configurations:

# ViT + BERT + Co-attention (Recommended)
config = ModularConfig.get_preset("vit_bert_coattention")

# CNN + BERT + Bilinear
config = ModularConfig.get_preset("cnn_bert_bilinear")

# CNN + LSTM + Concatenation (Baseline)
config = ModularConfig.get_preset("cnn_lstm_concatenation")
Custom Configuration
from modular_config import ModularConfig

config = ModularConfig()
config.image_encoder_type = "vit"
config.text_encoder_type = "bert"
config.fusion_type = "coattention"
config.use_pretrained_models = True
config.batch_size = 16
config.learning_rate = 1e-4
🌐 Web Interface
Features
Real-time Chat: WebSocket-based communication
Image Upload: Support for JPG, PNG, GIF, BMP
Model Management: Easy loading and status monitoring
Responsive Design: Works on all devices
Modern UI: Beautiful gradient design with animations
Usage
Open the application in your browser
Click "Load Model" to initialize the VQA system
Upload an image using the file button
Ask questions about the image
Get instant AI-powered answers
Example Questions
"What color is the car?"
"How many people are in the image?"
"What is the main object in this picture?"
"Is there a dog in the image?"
🚀 Deployment
Hugging Face Spaces
This project is designed to be easily deployed on Hugging Face Spaces:

Fork the repository to your Hugging Face account
Create a new Space with the Gradio SDK
Upload your trained model to the Space
Configure environment variables for model paths
Deploy and share your VQA application
Local Deployment
# Production deployment
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4

# Docker deployment
docker build -t vqa-chat .
docker run -p 8000:8000 vqa-chat
📊 Performance
Model Performance
Training Time: ~2-4 hours on RTX 3080
Inference Time: ~100-200ms per prediction
Memory Usage: 2-4GB GPU memory
Accuracy: 60-70% on VQA dataset
Optimization Tips
Use smaller batch sizes for limited GPU memory
Enable gradient accumulation for larger effective batch sizes
Use mixed precision training for faster training
Optimize DataLoader with appropriate num_workers
🔍 Troubleshooting
Common Issues
CUDA Out of Memory

# Reduce batch size in config
config.batch_size = 1
Missing Dependencies

# Install missing packages
pip install -r requirements.txt
Port Conflicts

# Use different port
uvicorn app:app --port 8001
Model Loading Issues

# Create demo model for testing
python create_demo_model.py
🤝 Contributing
Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Hugging Face for the transformers library and Spaces platform
PyTorch for the deep learning framework
FastAPI for the web framework
VQA Research Community for inspiration and datasets
📞 Support
Issues: GitHub Issues
Discussions: GitHub Discussions
Email: your-email@example.com
Happy VQA-ing! 🎉
