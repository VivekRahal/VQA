# Deploying VQA System on Hugging Face Spaces

This guide will walk you through deploying your Visual Question Answering (VQA) system on Hugging Face Spaces, making it accessible to anyone through a web interface.

## 🚀 Overview

Hugging Face Spaces is a platform for hosting ML applications with a web interface. Our VQA system is designed to work seamlessly with Spaces, providing:

- **Web Interface**: Modern chat UI for VQA interactions
- **Model Hosting**: Secure model storage and serving
- **Scalability**: Automatic scaling based on usage
- **Sharing**: Easy sharing with the community

## 📋 Prerequisites

Before deploying, ensure you have:

1. **Hugging Face Account**: Sign up at [huggingface.co](https://huggingface.co)
2. **Trained Model**: Your VQA model file (`best_modular_model.pth`)
3. **Vocabulary File**: The vocabulary pickle file (`vocab.pkl`)
4. **Answer Space**: Answer mapping file (`data/answer_space.txt`)
5. **Git Repository**: Your VQA project on GitHub

## 🛠️ Deployment Steps

### Step 1: Prepare Your Repository

1. **Fork/Clone** your VQA repository to your local machine
2. **Ensure all files** are committed and pushed to GitHub
3. **Verify dependencies** in `requirements.txt`

### Step 2: Create a Hugging Face Space

1. **Go to** [huggingface.co/spaces](https://huggingface.co/spaces)
2. **Click** "Create new Space"
3. **Choose settings**:
   - **Owner**: Your username
   - **Space name**: `vqa-chat-app` (or your preferred name)
   - **License**: MIT (or your choice)
   - **SDK**: **Gradio** (we'll use this for the interface)
   - **Python version**: 3.9 or 3.10
   - **Hardware**: CPU (or GPU if needed)

### Step 3: Upload Your Model

1. **Create a model repository** on Hugging Face:
   - Go to [huggingface.co/models](https://huggingface.co/models)
   - Click "Create new model"
   - Name it `your-username/vqa-model`

2. **Upload model files**:
   ```bash
   # Clone the model repository
   git clone https://huggingface.co/your-username/vqa-model
   cd vqa-model
   
   # Copy your model files
   cp /path/to/your/best_modular_model.pth .
   cp /path/to/your/vocab.pkl .
   cp /path/to/your/data/answer_space.txt ./data/
   
   # Commit and push
   git add .
   git commit -m "Add VQA model files"
   git push
   ```

### Step 4: Create Space Files

Create these files in your Space repository:

#### `app.py` (Main Application)
```python
import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from transformers import AutoTokenizer
import os
import pickle

from modular_model import ModularVQAModel
from modular_config import ModularConfig
from utils.helper import preprocess_question

class VQASpaceApp:
    def __init__(self):
        self.model = None
        self.config = None
        self.vocab = None
        self.answer_mapping = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """Load the VQA model"""
        try:
            # Load configuration
            self.config = ModularConfig()
            
            # Load vocabulary from Hugging Face model
            from huggingface_hub import hf_hub_download
            vocab_path = hf_hub_download(repo_id="your-username/vqa-model", filename="vocab.pkl")
            with open(vocab_path, 'rb') as f:
                self.vocab = pickle.load(f)
            
            # Load answer mapping
            answer_space_path = hf_hub_download(repo_id="your-username/vqa-model", filename="data/answer_space.txt")
            with open(answer_space_path, 'r', encoding='utf-8') as f:
                answers = [line.strip() for line in f.readlines()]
            self.answer_mapping = {i: answer for i, answer in enumerate(answers)}
            
            # Set config
            self.config.set_vocab_size(len(self.vocab))
            self.config.set_num_classes(len(self.answer_mapping))
            
            # Create model
            self.model = ModularVQAModel(config=self.config)
            
            # Load weights
            model_path = hf_hub_download(repo_id="your-username/vqa-model", filename="best_modular_model.pth")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Load tokenizer if using BERT
            if getattr(self.config, 'text_encoder_type', '').lower() == 'bert':
                bert_model_name = getattr(self.config, 'bert_model_name', 'bert-base-uncased')
                self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
            
            print(f"✅ Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def predict(self, image, question):
        """Predict answer for image and question"""
        try:
            if not self.model:
                return "❌ Model not loaded"
            
            # Preprocess image
            if image is None:
                return "❌ Please upload an image"
            
            image = image.resize((224, 224))
            image_array = np.array(image) / 255.0
            image_tensor = torch.from_numpy(image_array).float()
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor.permute(2, 0, 1)
            image_tensor = image_tensor.to(self.device)
            
            # Preprocess question
            question_tensor = preprocess_question(
                question,
                vocab=self.vocab if self.tokenizer is None else None,
                tokenizer=self.tokenizer,
                max_length=32
            )
            question_tensor = question_tensor.to(self.device)
            
            # Get prediction
            with torch.no_grad():
                logits = self.model(image_tensor.unsqueeze(0), question_tensor.unsqueeze(0))
                probabilities = F.softmax(logits, dim=1)
                predicted_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_idx].item()
            
            # Get answer
            answer = self.answer_mapping.get(predicted_idx, "Unknown")
            
            return f"Answer: {answer} (Confidence: {confidence:.2%})"
            
        except Exception as e:
            return f"❌ Prediction failed: {str(e)}"

# Create app instance
vqa_app = VQASpaceApp()

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="VQA Chat Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 VQA Chat Assistant")
        gr.Markdown("Upload an image and ask questions about it!")
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(label="Upload Image", type="pil")
                question_input = gr.Textbox(
                    label="Question",
                    placeholder="Ask a question about the image...",
                    lines=2
                )
                predict_btn = gr.Button("Ask Question", variant="primary")
            
            with gr.Column(scale=1):
                output = gr.Textbox(label="Answer", lines=3)
        
        # Example questions
        gr.Examples(
            examples=[
                ["What color is the car?", "How many people are in the image?"],
                ["What is the main object in this picture?", "Is there a dog in the image?"],
                ["What type of vehicle is this?", "What is the weather like?"]
            ],
            inputs=question_input
        )
        
        # Handle prediction
        predict_btn.click(
            fn=vqa_app.predict,
            inputs=[image_input, question_input],
            outputs=output
        )
        
        # Handle Enter key
        question_input.submit(
            fn=vqa_app.predict,
            inputs=[image_input, question_input],
            outputs=output
        )
    
    return demo

# Launch the app
if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
```

#### `requirements.txt`
```txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
gradio>=4.0.0
Pillow>=10.0.0
numpy>=1.24.0
huggingface-hub>=0.16.0
```

#### `README.md`
```markdown
# VQA Chat Assistant

A Visual Question Answering (VQA) system that can answer questions about images.

## Usage

1. Upload an image using the file upload button
2. Type your question in the text box
3. Click "Ask Question" or press Enter
4. Get an AI-powered answer about the image

## Example Questions

- "What color is the car?"
- "How many people are in the image?"
- "What is the main object in this picture?"
- "Is there a dog in the image?"

## Model Information

- **Architecture**: Modular VQA with CNN + BERT + Co-attention
- **Training Data**: VQA dataset
- **Performance**: ~65% accuracy on test set

## Technical Details

This Space uses:
- PyTorch for deep learning
- Transformers for BERT encoding
- Gradio for the web interface
- Hugging Face Hub for model hosting
```

### Step 5: Configure Environment Variables

In your Space settings, add these environment variables:

- `HF_TOKEN`: Your Hugging Face token (for private models)
- `MODEL_REPO_ID`: `your-username/vqa-model`
- `DEVICE`: `cpu` or `cuda`

### Step 6: Deploy and Test

1. **Commit and push** your Space files
2. **Wait for build** (usually 2-5 minutes)
3. **Test the interface** with sample images
4. **Share your Space** with the community

## 🔧 Advanced Configuration

### Custom Model Loading

For more complex model loading:

```python
def load_custom_model():
    """Load model with custom configuration"""
    config = ModularConfig()
    config.image_encoder_type = "vit"
    config.text_encoder_type = "bert"
    config.fusion_type = "coattention"
    config.use_pretrained_models = True
    
    # Load from multiple files
    model_files = {
        "model": "best_modular_model.pth",
        "vocab": "vocab.pkl",
        "answers": "data/answer_space.txt",
        "config": "model_config.json"
    }
    
    # Download all files
    for key, filename in model_files.items():
        path = hf_hub_download(repo_id="your-username/vqa-model", filename=filename)
        # Load and process files...
```

### Error Handling

```python
def robust_predict(image, question):
    """Robust prediction with error handling"""
    try:
        # Validate inputs
        if image is None:
            return "❌ Please upload an image"
        if not question.strip():
            return "❌ Please enter a question"
        
        # Check model status
        if not hasattr(vqa_app, 'model') or vqa_app.model is None:
            return "❌ Model not loaded. Please try again."
        
        # Make prediction
        result = vqa_app.predict(image, question)
        return result
        
    except Exception as e:
        return f"❌ Error: {str(e)}"
```

## 🚀 Performance Optimization

### Memory Management

```python
# Clear GPU memory after each prediction
import gc

def predict_with_cleanup(image, question):
    result = vqa_app.predict(image, question)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return result
```

### Caching

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_predict(question_hash, image_hash):
    """Cache predictions for repeated questions"""
    # Implementation...
```

## 🔒 Security Considerations

### Input Validation

```python
def validate_inputs(image, question):
    """Validate user inputs"""
    # Check image size
    if image and image.size[0] * image.size[1] > 10_000_000:  # 10MP limit
        return False, "Image too large"
    
    # Check question length
    if len(question) > 500:
        return False, "Question too long"
    
    # Check for malicious content
    if any(word in question.lower() for word in ["script", "javascript", "eval"]):
        return False, "Invalid question content"
    
    return True, "Valid inputs"
```

### Rate Limiting

```python
import time
from collections import defaultdict

class RateLimiter:
    def __init__(self, max_requests=10, window=60):
        self.max_requests = max_requests
        self.window = window
        self.requests = defaultdict(list)
    
    def is_allowed(self, user_id):
        now = time.time()
        user_requests = self.requests[user_id]
        
        # Remove old requests
        user_requests = [req for req in user_requests if now - req < self.window]
        self.requests[user_id] = user_requests
        
        # Check limit
        if len(user_requests) >= self.max_requests:
            return False
        
        # Add current request
        user_requests.append(now)
        return True
```

## 📊 Monitoring and Analytics

### Usage Tracking

```python
import json
from datetime import datetime

def log_prediction(image, question, answer, confidence):
    """Log prediction for analytics"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "confidence": confidence,
        "image_size": image.size if image else None
    }
    
    # Save to file or database
    with open("predictions.log", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
```

## 🤝 Community Features

### User Feedback

```python
def add_feedback_system():
    """Add feedback system to the interface"""
    with gr.Row():
        feedback_btn = gr.Button("👍 Correct Answer")
        feedback_btn_2 = gr.Button("👎 Wrong Answer")
    
    def handle_feedback(is_correct):
        # Save feedback
        return "Thank you for your feedback!"
    
    feedback_btn.click(fn=lambda: handle_feedback(True))
    feedback_btn_2.click(fn=lambda: handle_feedback(False))
```

## 🎯 Best Practices

### 1. Model Optimization
- Use model quantization for faster inference
- Implement batch processing for multiple questions
- Cache frequently asked questions

### 2. User Experience
- Provide clear error messages
- Add loading indicators
- Include example questions
- Support multiple image formats

### 3. Performance
- Monitor memory usage
- Implement request queuing
- Use CDN for static assets
- Optimize image preprocessing

### 4. Security
- Validate all inputs
- Implement rate limiting
- Monitor for abuse
- Keep dependencies updated

## 🚀 Going Live

Once your Space is working:

1. **Test thoroughly** with various images and questions
2. **Optimize performance** based on usage patterns
3. **Add documentation** for users
4. **Share on social media** and ML communities
5. **Collect feedback** and iterate

## 📈 Scaling Up

As your Space gains popularity:

1. **Upgrade to GPU** if needed
2. **Implement caching** for better performance
3. **Add analytics** to understand usage
4. **Consider monetization** options
5. **Build a community** around your VQA system

---

**Happy Deploying! 🚀**

Your VQA system is now ready to help people around the world understand images through natural language questions! 