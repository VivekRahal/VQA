"""
Hugging Face Spaces VQA Application
Optimized for deployment on Hugging Face Spaces with Gradio interface
"""

import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from transformers import AutoTokenizer
import os
import pickle
import json
from typing import Optional, Dict, Any

# Import our VQA modules
from modular_model import ModularVQAModel
from modular_config import ModularConfig
from utils.helper import preprocess_question


class VQASpaceApp:
    """VQA Application optimized for Hugging Face Spaces"""
    
    def __init__(self):
        self.model: Optional[ModularVQAModel] = None
        self.config: Optional[ModularConfig] = None
        self.vocab: Optional[Dict] = None
        self.answer_mapping: Optional[Dict] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model_loaded = False
        
        # Load model on initialization
        self.load_model()
    
    def load_model(self) -> bool:
        """Load the VQA model from Hugging Face Hub"""
        try:
            print("🔄 Loading VQA model...")
            
            # Load configuration
            self.config = ModularConfig()
            
            # Try to load from Hugging Face Hub
            try:
                from huggingface_hub import hf_hub_download
                
                # Download model files
                model_path = hf_hub_download(
                    repo_id=os.getenv("MODEL_REPO_ID", "VivekRahal/vqa-model"),
                    filename="best_modular_model.pth"
                )
                
                vocab_path = hf_hub_download(
                    repo_id=os.getenv("MODEL_REPO_ID", "VivekRahal/vqa-model"),
                    filename="vocab.pkl"
                )
                
                answer_space_path = hf_hub_download(
                    repo_id=os.getenv("MODEL_REPO_ID", "VivekRahal/vqa-model"),
                    filename="data/answer_space.txt"
                )
                
                print("✅ Downloaded model files from Hugging Face Hub")
                
            except Exception as e:
                print(f"⚠️ Could not download from Hub: {e}")
                print("🔄 Creating demo model...")
                self._create_demo_model()
                return True
            
            # Load vocabulary
            with open(vocab_path, 'rb') as f:
                self.vocab = pickle.load(f)
            print(f"✅ Loaded vocabulary with {len(self.vocab)} tokens")
            
            # Load answer mapping
            with open(answer_space_path, 'r', encoding='utf-8') as f:
                answers = [line.strip() for line in f.readlines()]
            self.answer_mapping = {i: answer for i, answer in enumerate(answers)}
            print(f"✅ Loaded answer space with {len(self.answer_mapping)} answers")
            
            # Set configuration
            self.config.set_vocab_size(len(self.vocab))
            self.config.set_num_classes(len(self.answer_mapping))
            
            # Create and load model
            self.model = ModularVQAModel(config=self.config)
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Load tokenizer if using BERT
            if getattr(self.config, 'text_encoder_type', '').lower() == 'bert':
                bert_model_name = getattr(self.config, 'bert_model_name', 'bert-base-uncased')
                self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
                print(f"✅ Loaded BERT tokenizer: {bert_model_name}")
            
            self.model_loaded = True
            print(f"✅ Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self._create_demo_model()
            return False
    
    def _create_demo_model(self):
        """Create a demo model with random weights"""
        import torch.nn as nn
        
        print("🔄 Creating demo model...")
        
        # Create default vocabulary
        self.vocab = {
            "<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3,
            "what": 4, "is": 5, "the": 6, "color": 7, "of": 8, "this": 9,
            "image": 10, "how": 11, "many": 12, "people": 13, "are": 14,
            "in": 15, "picture": 16, "object": 17, "main": 18, "see": 19
        }
        
        # Create default answer space
        default_answers = [
            "yes", "no", "red", "blue", "green", "yellow", "black", "white",
            "one", "two", "three", "four", "five", "car", "person", "dog",
            "cat", "house", "tree", "book", "chair", "table", "unknown"
        ]
        self.answer_mapping = {i: answer for i, answer in enumerate(default_answers)}
        
        # Create configuration
        self.config = ModularConfig()
        self.config.set_vocab_size(len(self.vocab))
        self.config.set_num_classes(len(self.answer_mapping))
        self.config.image_encoder_type = "cnn"
        self.config.text_encoder_type = "lstm"
        self.config.fusion_type = "concatenation"
        self.config.use_pretrained_models = False
        
        # Create model
        self.model = ModularVQAModel(config=self.config)
        
        # Initialize with random weights
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        self.model.apply(init_weights)
        self.model.to(self.device)
        self.model.eval()
        
        self.model_loaded = True
        print("✅ Demo model created with random weights")
        print("⚠️ Note: This is a demo model - answers will be random!")
    
    def predict(self, image: Image.Image, question: str) -> str:
        """Predict answer for image and question"""
        try:
            # Validate inputs
            if image is None:
                return "❌ Please upload an image"
            
            if not question or not question.strip():
                return "❌ Please enter a question"
            
            if not self.model_loaded:
                return "❌ Model not loaded. Please try again."
            
            # Preprocess image
            image = image.resize((224, 224))
            image_array = np.array(image) / 255.0
            image_tensor = torch.from_numpy(image_array).float()
            
            # Convert to channels first format (C, H, W)
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
            
            # Format response
            if confidence > 0.5:
                return f"🤖 **Answer:** {answer}\n\n💯 **Confidence:** {confidence:.1%}"
            else:
                return f"🤖 **Answer:** {answer}\n\n⚠️ **Confidence:** {confidence:.1%} (Low confidence)"
            
        except Exception as e:
            return f"❌ **Error:** {str(e)}\n\nPlease try again with a different image or question."
    
    def get_model_info(self) -> str:
        """Get information about the loaded model"""
        if not self.model_loaded:
            return "❌ Model not loaded"
        
        info = f"""
🤖 **Model Information:**
- **Image Encoder:** {getattr(self.config, 'image_encoder_type', 'Unknown')}
- **Text Encoder:** {getattr(self.config, 'text_encoder_type', 'Unknown')}
- **Fusion Strategy:** {getattr(self.config, 'fusion_type', 'Unknown')}
- **Device:** {self.device}
- **Vocabulary Size:** {len(self.vocab) if self.vocab else 0}
- **Answer Classes:** {len(self.answer_mapping) if self.answer_mapping else 0}
- **Pretrained Models:** {getattr(self.config, 'use_pretrained_models', False)}
        """
        return info


# Create app instance
vqa_app = VQASpaceApp()


def create_interface():
    """Create the Gradio interface"""
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .example-questions {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    """
    
    with gr.Blocks(
        title="VQA Chat Assistant",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as demo:
        
        # Header
        with gr.Row():
            gr.HTML("""
            <div class="main-header">
                <h1>🤖 VQA Chat Assistant</h1>
                <p>Ask questions about images and get intelligent answers powered by AI!</p>
            </div>
            """)
        
        # Model status
        with gr.Row():
            status_box = gr.Textbox(
                value=vqa_app.get_model_info(),
                label="Model Status",
                interactive=False,
                lines=8
            )
        
        # Main interface
        with gr.Row():
            with gr.Column(scale=1):
                # Image upload
                image_input = gr.Image(
                    label="📷 Upload Image",
                    type="pil",
                    height=300
                )
                
                # Question input
                question_input = gr.Textbox(
                    label="❓ Question",
                    placeholder="Ask a question about the image...",
                    lines=3
                )
                
                # Predict button
                predict_btn = gr.Button(
                    "🤖 Ask Question",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                # Output
                output = gr.Markdown(
                    label="🤖 Answer",
                    value="Upload an image and ask a question to get started!"
                )
                
                # Confidence visualization
                confidence_plot = gr.Plot(
                    label="Confidence Distribution",
                    visible=False
                )
        
        # Example questions
        with gr.Row():
            gr.HTML("""
            <div class="example-questions">
                <h3>💡 Example Questions:</h3>
                <ul>
                    <li>"What color is the car?"</li>
                    <li>"How many people are in the image?"</li>
                    <li>"What is the main object in this picture?"</li>
                    <li>"Is there a dog in the image?"</li>
                    <li>"What type of vehicle is this?"</li>
                    <li>"What is the weather like?"</li>
                </ul>
            </div>
            """)
        
        # Quick question buttons
        with gr.Row():
            quick_questions = [
                "What color is the main object?",
                "How many objects are in the image?",
                "What type of scene is this?",
                "Is this indoors or outdoors?"
            ]
            
            for question in quick_questions:
                btn = gr.Button(question, size="sm")
                btn.click(
                    fn=lambda q=question: q,
                    outputs=question_input
                )
        
        # Handle prediction
        def predict_with_status(image, question):
            if not vqa_app.model_loaded:
                return "❌ Model not loaded. Please refresh the page and try again."
            return vqa_app.predict(image, question)
        
        predict_btn.click(
            fn=predict_with_status,
            inputs=[image_input, question_input],
            outputs=output
        )
        
        # Handle Enter key
        question_input.submit(
            fn=predict_with_status,
            inputs=[image_input, question_input],
            outputs=output
        )
        
        # Clear button
        def clear_inputs():
            return None, "", "Upload an image and ask a question to get started!"
        
        clear_btn = gr.Button("🗑️ Clear", variant="secondary")
        clear_btn.click(
            fn=clear_inputs,
            outputs=[image_input, question_input, output]
        )
        
        # Footer
        with gr.Row():
            gr.HTML("""
            <div style="text-align: center; margin-top: 2rem; color: #666;">
                <p>Built with ❤️ using PyTorch, Transformers, and Gradio</p>
                <p>Model: Modular VQA with CNN + BERT + Co-attention</p>
            </div>
            """)
    
    return demo


# Launch the app
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    ) 