"""
Create a demo model for testing the FastAPI VQA chat application.
This creates a simple model with random weights for demonstration purposes.
"""

import torch
import torch.nn as nn
import pickle
import os
from modular_model import ModularVQAModel
from modular_config import ModularConfig
from utils.helper import load_vocab

def create_demo_model():
    """Create a demo model with random weights for testing"""
    print("🎭 Creating demo model for VQA chat application...")
    
    # Load or create vocabulary
    if os.path.exists("vocab.pkl"):
        vocab = load_vocab("vocab.pkl")
        print(f"✅ Loaded existing vocabulary with {len(vocab)} tokens")
    else:
        # Create a simple demo vocabulary
        vocab = {
            '<PAD>': 0,
            '<UNK>': 1,
            'what': 2,
            'is': 3,
            'the': 4,
            'color': 5,
            'of': 6,
            'this': 7,
            'image': 8,
            'how': 9,
            'many': 10,
            'people': 11,
            'are': 12,
            'in': 13,
            'there': 14,
            'dog': 15,
            'cat': 16,
            'car': 17,
            'red': 18,
            'blue': 19,
            'green': 20,
            'yellow': 21,
            'black': 22,
            'white': 23,
            'one': 24,
            'two': 25,
            'three': 26,
            'four': 27,
            'five': 28,
            'yes': 29,
            'no': 30
        }
        # Save vocabulary
        with open("vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)
        print(f"✅ Created demo vocabulary with {len(vocab)} tokens")
    
    # Create demo answer mapping
    demo_answers = [
        "red", "blue", "green", "yellow", "black", "white",
        "one", "two", "three", "four", "five",
        "yes", "no", "car", "dog", "cat", "person", "people",
        "image", "picture", "photo", "object", "thing"
    ]
    
    # Save answer mapping
    os.makedirs("data", exist_ok=True)
    with open("data/answer_space.txt", "w", encoding="utf-8") as f:
        for answer in demo_answers:
            f.write(f"{answer}\n")
    print(f"✅ Created demo answer space with {len(demo_answers)} answers")
    
    # Create configuration
    config = ModularConfig()
    config.set_vocab_size(len(vocab))
    config.set_num_classes(len(demo_answers))
    
    # Use a simpler configuration for demo
    config.image_encoder_type = "cnn"
    config.text_encoder_type = "lstm"
    config.fusion_type = "concatenation"
    config.use_pretrained_models = False
    config.batch_size = 1
    
    print("✅ Using simple CNN+LSTM+Concatenation for demo")
    
    # Create model
    model = ModularVQAModel(config=config)
    
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
    
    model.apply(init_weights)
    
    # Create demo checkpoint
    checkpoint = {
        'epoch': 0,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': None,
        'train_loss': 0.0,
        'val_loss': 0.0,
        'train_acc': 0.0,
        'val_acc': 0.0,
        'config': config.get_config_dict()
    }
    
    # Save model
    torch.save(checkpoint, "best_modular_model.pth")
    print("✅ Demo model saved as 'best_modular_model.pth'")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Demo model has {total_params:,} parameters")
    print(f"🎯 Model expects {len(vocab)} vocabulary tokens and {len(demo_answers)} answer classes")
    
    print("\n🎉 Demo model created successfully!")
    print("💡 You can now use the FastAPI chat application to test with this demo model.")
    print("⚠️  Note: This is a demo model with random weights - answers will be random!")

if __name__ == "__main__":
    create_demo_model() 