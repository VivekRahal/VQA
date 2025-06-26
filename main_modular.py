# main_modular.py
"""
Main training script for the modular VQA system.
This script trains the ViT + BERT + Co-attention model using pre-split data and image lists.
"""

from modular_config import ModularConfig
from modular_trainer import ModularTrainer

def main():
    print("="*60)
    print("TRAINING ViT + BERT + Co-attention VQA Model")
    print("="*60)
    
    # Use ViT + BERT + Co-attention preset
    # Change this line to try different combinations:
    # config = ModularConfig().get_preset_config("original")        # CNN + LSTM + Concatenation
    # config = ModularConfig().get_preset_config("vit_bert_coattention")  # ViT + BERT + Co-attention
    # config = ModularConfig().get_preset_config("vit_bert_bilinear")     # ViT + BERT + Bilinear
    # config = ModularConfig().get_preset_config("cnn_bert_coattention")  # CNN + BERT + Co-attention
    # config = ModularConfig().get_preset_config("cnn_bert_bilinear")     # CNN + BERT + Bilinear (NEW!)
    
    config = ModularConfig().get_preset_config("cnn_bert_bilinear")  # Try CNN + BERT + Bilinear
    
    # Training parameters
    config.num_epochs = 1
    config.batch_size = 32
    config.learning_rate = 0.001
    
    print(f"\nConfiguration:")
    print(f"Image Encoder: {config.image_encoder_type}")
    print(f"Text Encoder: {config.text_encoder_type}")
    print(f"Fusion: {config.fusion_type}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Learning Rate: {config.learning_rate}")
    
    # Use pre-split files (assume these are already created by prepare_data.py)
    train_csv = "data/train_split.csv"
    val_csv = "data/val_split.csv"
    test_csv = "data/test_split.csv"
    train_img_list = "data/train_images_split.txt"
    val_img_list = "data/val_images_split.txt"
    test_img_list = "data/test_images_split.txt"
    
    # Create trainer with validation data
    trainer = ModularTrainer(
        train_csv=train_csv,
        img_dir="data/images",
        answer_space_file="data/answer_space.txt",
        train_img_list=train_img_list,
        val_csv=val_csv,
        test_img_list=val_img_list,
        config=config
    )
    
    # Print model info
    model_info = trainer.get_model_info()
    print(f"\nModel Information:")
    print(f"Total Parameters: {model_info['total_parameters']:,}")
    print(f"Trainable Parameters: {model_info['trainable_parameters']:,}")
    
    # Start training
    print(f"\nStarting training for {config.num_epochs} epoch(s)...")
    print("="*60)
    
    try:
        trainer.train()
        model_path = "vit_bert_coattention_model.pth"
        trainer.save_model(model_path)
        print(f"\n✅ Training completed successfully!")
        print(f"✅ Model saved to: {model_path}")
        
        # Final evaluation
        print(f"\n" + "="*50)
        print("FINAL EVALUATION RESULTS")
        print("="*50)
        train_loss, train_acc = trainer.evaluate(validation=False)
        print(f"Training Set   - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        val_loss, val_acc = trainer.evaluate(validation=True)
        print(f"Validation Set - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        print(f"\nTest Set Evaluation:")
        print(f"Test data file: {test_csv}")
        print("Note: To evaluate on test set, create a separate test trainer")
        print(f"\nTraining completed! Check 'training_curves.png' for visual analysis.")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        print("Check if all data files are present and GPU memory is sufficient.")

if __name__ == "__main__":
    main() 