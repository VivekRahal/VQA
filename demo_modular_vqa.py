# demo_modular_vqa.py
"""
Demo script for the modular VQA system.
This script shows how to easily switch between different encoder and fusion combinations.
"""

from modular_config import ModularConfig
from modular_trainer import ModularTrainer

def demo_original_config():
    """Demo with original CNN + LSTM + Concatenation configuration."""
    print("\n" + "="*50)
    print("DEMO 1: Original Configuration (CNN + LSTM + Concatenation)")
    print("="*50)
    
    # Use original preset
    config = ModularConfig().get_preset_config("original")
    
    # Create trainer
    trainer = ModularTrainer(
        train_csv="data/data_train.csv",
        img_dir="data/images",
        answer_space_file="data/answer_space.txt",
        train_img_list="data/train_images_list.txt",
        config=config
    )
    
    # Print model info
    model_info = trainer.get_model_info()
    print(f"\nModel Information:")
    print(f"Total Parameters: {model_info['total_parameters']:,}")
    print(f"Trainable Parameters: {model_info['trainable_parameters']:,}")
    
    return trainer

def demo_vit_bert_coattention():
    """Demo with ViT + BERT + Co-attention configuration."""
    print("\n" + "="*50)
    print("DEMO 2: ViT + BERT + Co-attention Configuration")
    print("="*50)
    
    # Use ViT + BERT + Co-attention preset
    config = ModularConfig().get_preset_config("vit_bert_coattention")
    
    # Create trainer
    trainer = ModularTrainer(
        train_csv="data/data_train.csv",
        img_dir="data/images",
        answer_space_file="data/answer_space.txt",
        train_img_list="data/train_images_list.txt",
        config=config
    )
    
    # Print model info
    model_info = trainer.get_model_info()
    print(f"\nModel Information:")
    print(f"Total Parameters: {model_info['total_parameters']:,}")
    print(f"Trainable Parameters: {model_info['trainable_parameters']:,}")
    
    return trainer

def demo_vit_bert_bilinear():
    """Demo with ViT + BERT + Bilinear configuration."""
    print("\n" + "="*50)
    print("DEMO 3: ViT + BERT + Bilinear Configuration")
    print("="*50)
    
    # Use ViT + BERT + Bilinear preset
    config = ModularConfig().get_preset_config("vit_bert_bilinear")
    
    # Create trainer
    trainer = ModularTrainer(
        train_csv="data/data_train.csv",
        img_dir="data/images",
        answer_space_file="data/answer_space.txt",
        train_img_list="data/train_images_list.txt",
        config=config
    )
    
    # Print model info
    model_info = trainer.get_model_info()
    print(f"\nModel Information:")
    print(f"Total Parameters: {model_info['total_parameters']:,}")
    print(f"Trainable Parameters: {model_info['trainable_parameters']:,}")
    
    return trainer

def demo_custom_config():
    """Demo with custom configuration."""
    print("\n" + "="*50)
    print("DEMO 4: Custom Configuration (CNN + BERT + Co-attention)")
    print("="*50)
    
    # Create custom configuration
    config = ModularConfig()
    config.image_encoder_type = "cnn"
    config.text_encoder_type = "bert"
    config.fusion_type = "coattention"
    config.image_encoder_dim = 512
    config.text_encoder_dim = 256
    config.fusion_dim = 512
    config.bert_embed_dim = 256
    config.bert_num_layers = 4
    config.coattention_hidden_dim = 256
    
    # Create trainer
    trainer = ModularTrainer(
        train_csv="data/data_train.csv",
        img_dir="data/images",
        answer_space_file="data/answer_space.txt",
        train_img_list="data/train_images_list.txt",
        config=config
    )
    
    # Print model info
    model_info = trainer.get_model_info()
    print(f"\nModel Information:")
    print(f"Total Parameters: {model_info['total_parameters']:,}")
    print(f"Trainable Parameters: {model_info['trainable_parameters']:,}")
    
    return trainer

def main():
    """Main function to run all demos."""
    print("Modular VQA System Demo")
    print("This demo shows how to easily switch between different encoder and fusion combinations.")
    
    # Run all demos
    trainers = []
    
    try:
        # Demo 1: Original configuration
        trainer1 = demo_original_config()
        trainers.append(("Original (CNN+LSTM+Concat)", trainer1))
        
        # Demo 2: ViT + BERT + Co-attention
        trainer2 = demo_vit_bert_coattention()
        trainers.append(("ViT+BERT+CoAttention", trainer2))
        
        # Demo 3: ViT + BERT + Bilinear
        trainer3 = demo_vit_bert_bilinear()
        trainers.append(("ViT+BERT+Bilinear", trainer3))
        
        # Demo 4: Custom configuration
        trainer4 = demo_custom_config()
        trainers.append(("Custom (CNN+BERT+CoAttention)", trainer4))
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY OF ALL CONFIGURATIONS")
        print("="*60)
        
        for name, trainer in trainers:
            model_info = trainer.get_model_info()
            print(f"{name:30} | Parameters: {model_info['total_parameters']:>8,}")
        
        print("\n" + "="*60)
        print("To train any of these models, call:")
        print("trainer.train()")
        print("\nTo save a model:")
        print("trainer.save_model('my_model.pth')")
        print("\nTo load a model:")
        print("trainer.load_model('my_model.pth')")
        print("="*60)
        
    except Exception as e:
        print(f"Error during demo: {e}")
        print("Make sure all required data files are present in the data/ directory.")

if __name__ == "__main__":
    main() 