# VQA Chat Application

A modern, real-time Visual Question Answering (VQA) chat application built with FastAPI and WebSocket technology. This application provides an intuitive chat interface for testing your trained VQA model with any image.

## 🚀 Features

- **Modern Chat UI**: Beautiful, responsive chat interface with real-time messaging
- **WebSocket Support**: Real-time communication for instant responses
- **Image Upload**: Drag-and-drop or click-to-upload image functionality
- **Model Management**: Easy model loading and status monitoring
- **REST API**: Standard REST endpoints for integration
- **Cross-platform**: Works on desktop and mobile browsers
- **Object-Oriented Design**: Clean, maintainable code following OOP principles

## 📋 Prerequisites

Before running the application, make sure you have:

1. **Trained Model**: A trained VQA model file (`best_modular_model.pth`)
2. **Vocabulary**: The vocabulary file (`vocab.pkl`)
3. **Answer Space**: Answer mapping file (optional, will create default if missing)
4. **Dependencies**: All required Python packages installed

## 🛠️ Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Model Files**:
   - `best_modular_model.pth` - Your trained model weights
   - `vocab.pkl` - Vocabulary file
   - `data/answer_space.txt` - Answer mapping (optional)

## 🚀 Running the Application

### Method 1: Direct Python Execution
```bash
python app.py
```

### Method 2: Using Uvicorn
```bash
uvicorn app:VQAChatApp().app --host 0.0.0.0 --port 8000 --reload
```

### Method 3: Production Deployment
```bash
uvicorn app:VQAChatApp().app --host 0.0.0.0 --port 8000 --workers 4
```

## 🌐 Accessing the Application

1. **Open your browser** and go to: `http://localhost:8000`
2. **Load the model** by clicking the "Load Model" button
3. **Upload an image** using the file upload button
4. **Ask questions** about the image in the chat interface

## 📱 Usage Guide

### Step 1: Load Model
- Click the "Load Model" button
- Wait for the success message
- The "Send" button will become enabled

### Step 2: Upload Image
- Click "📷 Upload Image" or drag-and-drop an image
- Supported formats: JPG, PNG, GIF, BMP
- The image will appear in the chat

### Step 3: Ask Questions
- Type your question in the input field
- Press Enter or click "Send"
- Get instant AI-powered answers

### Example Questions:
- "What color is the car?"
- "How many people are in the image?"
- "What is the main object in this picture?"
- "Is there a dog in the image?"

## 🔧 API Endpoints

### REST API

#### `GET /`
- **Description**: Main chat interface
- **Response**: HTML page with chat UI

#### `POST /api/load-model`
- **Description**: Load the VQA model
- **Response**: JSON with status and message

#### `POST /api/predict`
- **Description**: Predict answer for image and question
- **Parameters**:
  - `image`: Uploaded image file
  - `question`: Text question
- **Response**: JSON with answer and status

### WebSocket API

#### `WS /ws`
- **Description**: Real-time communication endpoint
- **Message Types**:
  - `load_model`: Load the model
  - `predict`: Get prediction for image and question

## 🏗️ Architecture

### Class Structure

```python
VQAChatApp
├── __init__()           # Initialize FastAPI app and components
├── _setup_middleware()  # Setup CORS and middleware
├── _setup_routes()      # Define API endpoints
├── _setup_static_files() # Setup static file serving
├── _load_model()        # Load trained VQA model
├── _predict_answer()    # Generate predictions
├── _preprocess_image()  # Image preprocessing
├── _preprocess_question() # Question preprocessing
├── _get_chat_html()     # Generate chat UI HTML
└── run()               # Start the application
```

### Key Components

1. **FastAPI Application**: Main web framework
2. **WebSocket Handler**: Real-time communication
3. **Model Manager**: VQA model loading and inference
4. **Image Processor**: Image preprocessing pipeline
5. **Chat UI**: Modern, responsive frontend

## 🎨 UI Features

### Design Principles
- **Modern Aesthetic**: Gradient backgrounds and smooth animations
- **Responsive Design**: Works on all screen sizes
- **Intuitive UX**: Clear visual hierarchy and feedback
- **Accessibility**: Keyboard navigation and screen reader support

### Visual Elements
- **Chat Bubbles**: User and bot messages with distinct styling
- **Image Preview**: Thumbnail display of uploaded images
- **Status Indicators**: Loading states and success/error messages
- **Interactive Buttons**: Hover effects and disabled states

## 🧪 Testing

Run the test suite to verify functionality:

```bash
python test_app.py
```

The test suite includes:
- Server connectivity tests
- Model loading verification
- API endpoint testing
- WebSocket functionality tests
- Prediction accuracy checks

## 🔍 Troubleshooting

### Common Issues

1. **Model Not Found**:
   ```
   Error: Model file best_modular_model.pth not found
   ```
   **Solution**: Ensure the model file exists in the project root

2. **Port Already in Use**:
   ```
   Error: [Errno 10048] Only one usage of each socket address
   ```
   **Solution**: Change port or kill existing process

3. **CUDA Out of Memory**:
   ```
   Error: CUDA out of memory
   ```
   **Solution**: Reduce batch size or use CPU mode

4. **Missing Dependencies**:
   ```
   ModuleNotFoundError: No module named 'fastapi'
   ```
   **Solution**: Install requirements: `pip install -r requirements.txt`

### Performance Optimization

1. **GPU Memory**: Monitor GPU usage and adjust batch sizes
2. **Image Size**: Large images are automatically resized to 224x224
3. **Caching**: Model is loaded once and reused for all predictions
4. **Async Processing**: Non-blocking prediction handling

## 🔒 Security Considerations

- **Input Validation**: All user inputs are validated
- **File Upload Limits**: Image size and format restrictions
- **CORS Configuration**: Configurable cross-origin settings
- **Error Handling**: Graceful error handling without exposing internals

## 📈 Monitoring

### Logs
- Application startup logs
- Model loading status
- Prediction requests and responses
- Error tracking and debugging

### Metrics
- Request/response times
- Model inference latency
- Memory usage monitoring
- Error rates and types

## 🚀 Deployment

### Development
```bash
python app.py
```

### Production
```bash
# Using Gunicorn
gunicorn app:VQAChatApp().app -w 4 -k uvicorn.workers.UvicornWorker

# Using Docker
docker build -t vqa-chat .
docker run -p 8000:8000 vqa-chat
```

### Environment Variables
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `DEBUG`: Debug mode (default: True)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- FastAPI for the excellent web framework
- PyTorch for deep learning capabilities
- The VQA research community for inspiration
- All contributors and users of this project

---

**Happy Chatting! 🎉** 