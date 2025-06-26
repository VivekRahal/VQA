"""
Modern FastAPI VQA Chat Application
Provides a real-time chat interface for Visual Question Answering
"""

import os
import json
import base64
import io
from typing import Optional, Dict, Any
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from transformers import AutoTokenizer

from modular_model import ModularVQAModel
from modular_config import ModularConfig
from dataset import VQADataset
from utils.helper import load_vocab, preprocess_question


class VQAChatApp:
    """Main VQA Chat Application Class"""
    
    def __init__(self):
        self.app = FastAPI(
            title="VQA Chat Application",
            description="Real-time Visual Question Answering with Modern Chat UI",
            version="1.0.0"
        )
        self.model: Optional[ModularVQAModel] = None
        self.config: Optional[ModularConfig] = None
        self.vocab: Optional[Dict] = None
        self.answer_mapping: Optional[Dict] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None  # For BERT/ViT
        
        self._setup_middleware()
        self._setup_routes()
        self._setup_static_files()
    
    def _setup_middleware(self):
        """Setup CORS and other middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_static_files(self):
        """Setup static files for the chat UI"""
        # Create static directory if it doesn't exist
        static_dir = Path("static")
        static_dir.mkdir(exist_ok=True)
        
        # Mount static files
        self.app.mount("/static", StaticFiles(directory="static"), name="static")
    
    def _setup_routes(self):
        """Setup all API routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def get_chat_ui():
            """Serve the main chat UI"""
            return self._get_chat_html()
        
        @self.app.post("/api/load-model")
        async def load_model_endpoint():
            """Load the trained VQA model"""
            try:
                await self._load_model()
                return {"status": "success", "message": "Model loaded successfully"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
        
        @self.app.post("/api/predict")
        async def predict_endpoint(image: UploadFile = File(...), question: str = Form(...)):
            """Predict answer for image and question"""
            try:
                if not self.model:
                    raise HTTPException(status_code=400, detail="Model not loaded. Please load model first.")
                
                # Process image
                image_data = await image.read()
                pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
                
                # Get prediction
                answer = await self._predict_answer(pil_image, question)
                
                return {
                    "status": "success",
                    "answer": answer,
                    "question": question
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time chat"""
            await websocket.accept()
            
            try:
                while True:
                    # Receive message
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    if message["type"] == "load_model":
                        try:
                            await self._load_model()
                            await websocket.send_text(json.dumps({
                                "type": "model_status",
                                "status": "loaded",
                                "message": "Model loaded successfully"
                            }))
                        except Exception as e:
                            await websocket.send_text(json.dumps({
                                "type": "model_status",
                                "status": "error",
                                "message": f"Failed to load model: {str(e)}"
                            }))
                    
                    elif message["type"] == "predict":
                        try:
                            if not self.model:
                                await websocket.send_text(json.dumps({
                                    "type": "prediction",
                                    "status": "error",
                                    "message": "Model not loaded"
                                }))
                                continue
                            
                            # Decode base64 image
                            image_data = base64.b64decode(message["image"].split(",")[1])
                            pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
                            
                            # Get prediction
                            answer = await self._predict_answer(pil_image, message["question"])
                            
                            await websocket.send_text(json.dumps({
                                "type": "prediction",
                                "status": "success",
                                "answer": answer,
                                "question": message["question"]
                            }))
                        except Exception as e:
                            await websocket.send_text(json.dumps({
                                "type": "prediction",
                                "status": "error",
                                "message": f"Prediction failed: {str(e)}"
                            }))
            
            except WebSocketDisconnect:
                print("WebSocket client disconnected")
    
    async def _load_model(self):
        """Load the trained VQA model"""
        # Load configuration
        self.config = ModularConfig()
        
        # Load vocabulary
        self.vocab = load_vocab("vocab.pkl")
        
        # Load answer mapping
        answer_space_file = "data/answer_space.txt"
        if os.path.exists(answer_space_file):
            with open(answer_space_file, 'r', encoding='utf-8') as f:
                answers = [line.strip() for line in f.readlines()]
            self.answer_mapping = {i: answer for i, answer in enumerate(answers)}
        else:
            # Create default answer mapping if file doesn't exist
            self.answer_mapping = {i: f"answer_{i}" for i in range(1000)}
        
        # Set vocab size and num classes in config
        self.config.set_vocab_size(len(self.vocab))
        self.config.set_num_classes(len(self.answer_mapping))
        
        # Initialize model
        self.model = ModularVQAModel(config=self.config)
        
        # Load trained weights
        model_path = "best_modular_model.pth"
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise FileNotFoundError(f"Model file {model_path} not found")
        self.model.to(self.device)
        self.model.eval()
        
        # If using BERT, load tokenizer
        if getattr(self.config, 'text_encoder_type', '').lower() == 'bert':
            bert_model_name = getattr(self.config, 'bert_model_name', 'bert-base-uncased')
            self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
            print(f"Loaded BERT tokenizer: {bert_model_name}")
        else:
            self.tokenizer = None
        
        print(f"Model loaded successfully on {self.device}")
    
    async def _predict_answer(self, image: Image.Image, question: str) -> str:
        """Predict answer for given image and question"""
        with torch.no_grad():
            # Preprocess image
            image_tensor = self._preprocess_image(image)
            
            # Preprocess question
            question_tensor = preprocess_question(
                question,
                vocab=self.vocab if self.tokenizer is None else None,
                tokenizer=self.tokenizer,
                max_length=32
            )
            
            # Move to device
            image_tensor = image_tensor.to(self.device)
            question_tensor = question_tensor.to(self.device)
            
            # Get prediction
            logits = self.model(image_tensor.unsqueeze(0), question_tensor.unsqueeze(0))
            probabilities = F.softmax(logits, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            
            # Get answer
            answer = self.answer_mapping.get(predicted_idx, "Unknown")
            
            return answer
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model input"""
        # Resize to model input size
        image = image.resize((224, 224))
        
        # Convert to tensor and normalize
        image_array = np.array(image) / 255.0
        image_tensor = torch.from_numpy(image_array).float()
        
        # Convert to channels first format (C, H, W)
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.permute(2, 0, 1)
        
        return image_tensor
    
    def _preprocess_question(self, question: str) -> torch.Tensor:
        """Preprocess question for model input"""
        # Tokenize question
        tokens = preprocess_question(question, self.vocab)
        
        # Convert to tensor
        question_tensor = torch.tensor(tokens, dtype=torch.long)
        
        return question_tensor
    
    def _get_chat_html(self) -> str:
        """Generate the chat UI HTML"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VQA Chat Application</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .chat-container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            width: 100%;
            max-width: 800px;
            height: 600px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .chat-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        
        .chat-header h1 {
            font-size: 24px;
            margin-bottom: 5px;
        }
        
        .chat-header p {
            opacity: 0.9;
            font-size: 14px;
        }
        
        .chat-messages {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            background: #f8f9fa;
        }
        
        .message {
            margin-bottom: 15px;
            display: flex;
            align-items: flex-start;
        }
        
        .message.user {
            justify-content: flex-end;
        }
        
        .message-content {
            max-width: 70%;
            padding: 12px 16px;
            border-radius: 18px;
            word-wrap: break-word;
        }
        
        .message.user .message-content {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .message.bot .message-content {
            background: white;
            color: #333;
            border: 1px solid #e0e0e0;
        }
        
        .message-image {
            max-width: 200px;
            max-height: 200px;
            border-radius: 10px;
            margin-bottom: 10px;
        }
        
        .chat-input {
            padding: 20px;
            background: white;
            border-top: 1px solid #e0e0e0;
        }
        
        .input-group {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        
        .file-input {
            display: none;
        }
        
        .file-label {
            background: #667eea;
            color: white;
            padding: 10px 15px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.3s;
        }
        
        .file-label:hover {
            background: #5a6fd8;
        }
        
        .question-input {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            font-size: 14px;
            outline: none;
            transition: border-color 0.3s;
        }
        
        .question-input:focus {
            border-color: #667eea;
        }
        
        .send-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 14px;
            transition: transform 0.2s;
        }
        
        .send-btn:hover {
            transform: translateY(-2px);
        }
        
        .send-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .status {
            text-align: center;
            padding: 10px;
            font-size: 14px;
            color: #666;
        }
        
        .status.error {
            color: #e74c3c;
        }
        
        .status.success {
            color: #27ae60;
        }
        
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>VQA Chat Assistant</h1>
            <p>Ask questions about images and get intelligent answers</p>
        </div>
        
        <div class="chat-messages" id="chatMessages">
            <div class="message bot">
                <div class="message-content">
                    👋 Hello! I'm your VQA assistant. Please load the model first, then upload an image and ask me a question about it.
                </div>
            </div>
        </div>
        
        <div class="status" id="status">
            <button onclick="loadModel()" class="send-btn">Load Model</button>
        </div>
        
        <div class="chat-input">
            <div class="input-group">
                <input type="file" id="imageInput" class="file-input" accept="image/*">
                <label for="imageInput" class="file-label">📷 Upload Image</label>
                
                <input type="text" id="questionInput" class="question-input" placeholder="Ask a question about the image...">
                
                <button onclick="sendMessage()" class="send-btn" id="sendBtn" disabled>Send</button>
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let currentImage = null;
        let modelLoaded = false;
        
        // Initialize WebSocket connection
        function initWebSocket() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);
            
            ws.onopen = function() {
                console.log('WebSocket connected');
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            };
            
            ws.onclose = function() {
                console.log('WebSocket disconnected');
                setTimeout(initWebSocket, 1000);
            };
        }
        
        // Handle WebSocket messages
        function handleWebSocketMessage(data) {
            if (data.type === 'model_status') {
                if (data.status === 'loaded') {
                    modelLoaded = true;
                    document.getElementById('status').innerHTML = '<span class="status success">✅ Model loaded successfully</span>';
                    document.getElementById('sendBtn').disabled = false;
                } else {
                    document.getElementById('status').innerHTML = `<span class="status error">❌ ${data.message}</span>`;
                }
            } else if (data.type === 'prediction') {
                if (data.status === 'success') {
                    addBotMessage(`Answer: ${data.answer}`);
                } else {
                    addBotMessage(`Error: ${data.message}`);
                }
            }
        }
        
        // Load model
        function loadModel() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                document.getElementById('status').innerHTML = '<span class="loading"></span> Loading model...';
                ws.send(JSON.stringify({type: 'load_model'}));
            }
        }
        
        // Handle image upload
        document.getElementById('imageInput').addEventListener('change', function(event) {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    currentImage = e.target.result;
                    addUserMessage('📷 Image uploaded', currentImage);
                };
                reader.readAsDataURL(file);
            }
        });
        
        // Send message
        function sendMessage() {
            const question = document.getElementById('questionInput').value.trim();
            if (!question || !currentImage || !modelLoaded) return;
            
            addUserMessage(question);
            document.getElementById('questionInput').value = '';
            
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'predict',
                    question: question,
                    image: currentImage
                }));
            }
        }
        
        // Add user message
        function addUserMessage(text, image = null) {
            const messagesDiv = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message user';
            
            let content = `<div class="message-content">`;
            if (image) {
                content += `<img src="${image}" class="message-image"><br>`;
            }
            content += text;
            content += '</div>';
            
            messageDiv.innerHTML = content;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        // Add bot message
        function addBotMessage(text) {
            const messagesDiv = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message bot';
            messageDiv.innerHTML = `<div class="message-content">${text}</div>`;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        // Handle Enter key
        document.getElementById('questionInput').addEventListener('keypress', function(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        });
        
        // Initialize WebSocket on page load
        window.onload = function() {
            initWebSocket();
        };
    </script>
</body>
</html>
        """
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the FastAPI application"""
        uvicorn.run(self.app, host=host, port=port)


def main():
    """Main function to run the VQA Chat Application"""
    app = VQAChatApp()
    print("🚀 Starting VQA Chat Application...")
    print("📱 Open your browser and go to: http://localhost:8000")
    print("💡 First click 'Load Model' then upload an image and ask questions!")
    app.run()


if __name__ == "__main__":
    main() 