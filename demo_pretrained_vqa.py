# demo_pretrained_vqa.py
"""
Demo script for VQA with pretrained encoders from Hugging Face.
This demonstrates how to use pretrained BERT, ViT, and ResNet models.
"""

import torch
from modular_config import ModularConfig
from modular_model import ModularVQAModel

def demo_pretrained_encoders():
    """Demo different pretrained encoder combinations."""
    
    print("🚀 VQA with Pretrained Encoders Demo")
    print("=" * 50)
    
    # Create configuration
    config = ModularConfig()
    
    # Demo 1: Pretrained ResNet50 + BERT
    print("\n📸 Demo 1: ResNet50 + BERT (Concatenation)")
    config.get_preset_config("pretrained_cnn_bert")
    config.num_epochs = 1  # Just for demo
    config.batch_size = 1  # Small batch for demo
    
    model = ModularVQAModel(config)
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Demo 2: Pretrained ViT + BERT
    print("\n🖼️  Demo 2: ViT + BERT (Concatenation)")
    config.get_preset_config("pretrained_vit_bert")
    
    model = ModularVQAModel(config)
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Demo 3: Pretrained ViT + BERT with Co-attention
    print("\n🔗 Demo 3: ViT + BERT (Co-attention)")
    config.get_preset_config("pretrained_vit_bert_coattention")
    
    model = ModularVQAModel(config)
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Demo 4: EfficientNet + BERT
    print("\n⚡ Demo 4: EfficientNet + BERT")
    config.get_preset_config("efficient_vit_bert")
    
    model = ModularVQAModel(config)
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    print("\n🎉 All pretrained encoder demos completed!")
    print("\n📋 Available Pretrained Presets:")
    print("  • pretrained_cnn_lstm - ResNet50 + LSTM")
    print("  • pretrained_cnn_bert - ResNet50 + BERT")
    print("  • pretrained_cnn_bert_coattention - ResNet50 + BERT + Co-attention")
    print("  • pretrained_cnn_bert_bilinear - ResNet50 + BERT + Bilinear")
    print("  • pretrained_vit_bert - ViT + BERT")
    print("  • pretrained_vit_bert_coattention - ViT + BERT + Co-attention")
    print("  • pretrained_vit_bert_bilinear - ViT + BERT + Bilinear")
    print("  • efficient_vit_bert - EfficientNet + BERT")

def test_model_forward_pass():
    """Test a forward pass with pretrained encoders."""
    
    print("\n🧪 Testing Forward Pass with Pretrained Encoders")
    print("=" * 50)
    
    config = ModularConfig()
    config.get_preset_config("pretrained_cnn_bert")
    config.num_epochs = 1
    config.batch_size = 2
    
    model = ModularVQAModel(config)
    model.eval()
    
    # Create dummy data
    batch_size = 2
    image = torch.randn(batch_size, 3, 224, 224)  # ResNet expects 224x224
    text = torch.randint(0, 1000, (batch_size, 20))  # 20 tokens
    attention_mask = torch.ones(batch_size, 20)  # All tokens are valid
    
    print(f"📊 Input shapes:")
    print(f"  Image: {image.shape}")
    print(f"  Text: {text.shape}")
    print(f"  Attention Mask: {attention_mask.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(image, text, attention_mask)
    
    print(f"📊 Output shape: {output.shape}")
    print(f"✅ Forward pass successful!")

if __name__ == "__main__":
    # Run demos
    demo_pretrained_encoders()
    test_model_forward_pass()
    
    print("\n🎯 Next Steps:")
    print("1. Run 'python main_modular.py' with a pretrained preset")
    print("2. Try different combinations by changing the preset")
    print("3. Experiment with different fusion strategies")
    print("4. Fine-tune the pretrained models for your specific VQA task") 