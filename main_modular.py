# main_modular.py
"""
Main training script for modular VQA system.

This script provides the main entry point for training the modular VQA model
with different encoder and fusion combinations. It handles data loading,
model training, and evaluation.
"""

import os
import sys
from typing import Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modular_config import ModularConfig
from modular_model import ModularVQAModel
from modular_trainer import ModularTrainer
from dataset import VQADataset
from utils.helper import load_vocab, build_answer_mapping, vqa_collate_fn
from utils.dataloader_utils import get_dataloader_config


def setup_data_loaders(config: ModularConfig) -> tuple[DataLoader, DataLoader]:
    """
    Set up training and validation data loaders.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (train_loader, val_loader)
        
    Raises:
        FileNotFoundError: If required data files are missing
        ValueError: If data cannot be loaded
    """
    # Check if data files exist
    required_files = [
        'data/train_split.csv',
        'data/val_split.csv', 
        'data/train_images_split.txt',
        'data/val_images_split.txt',
        'vocab.pkl'
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required data file not found: {file_path}")
    
    # Load vocabulary and answer mapping
    try:
        vocab = load_vocab('vocab.pkl')
        answer_mapping = build_answer_mapping('data/train_split.csv')
    except Exception as e:
        raise ValueError(f"Failed to load vocabulary or answer mapping: {e}")
    
    # Update config with vocabulary size and number of classes
    config.set_vocab_size(len(vocab))
    config.set_num_classes(len(answer_mapping))
    
    # Create datasets
    train_dataset = VQADataset(
        csv_file='data/train_split.csv',
        img_dir='data/images',
        img_list_file='data/train_images_split.txt'
    )
    
    val_dataset = VQADataset(
        csv_file='data/val_split.csv',
        img_dir='data/images',
        img_list_file='data/val_images_split.txt'
    )
    
    # Get optimal dataloader configuration
    dataloader_config = get_dataloader_config()
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=dataloader_config['num_workers'],
        pin_memory=dataloader_config['pin_memory'],
        persistent_workers=dataloader_config['persistent_workers'],
        collate_fn=vqa_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=dataloader_config['num_workers'],
        pin_memory=dataloader_config['pin_memory'],
        persistent_workers=dataloader_config['persistent_workers'],
        collate_fn=vqa_collate_fn
    )
    
    print(f"✅ Data loaders created:")
    print(f"   Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"   Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    
    return train_loader, val_loader


def create_model(config: ModularConfig) -> ModularVQAModel:
    """
    Create and configure the VQA model.
    
    Args:
        config: Configuration object
        
    Returns:
        Configured VQA model
        
    Raises:
        ValueError: If model cannot be created
    """
    try:
        model = ModularVQAModel(config)
        model = model.to(config.device)
        return model
    except Exception as e:
        raise ValueError(f"Failed to create model: {e}")


def create_optimizer(model: ModularVQAModel, config: ModularConfig) -> optim.Optimizer:
    """
    Create optimizer for the model.
    
    Args:
        model: The model to optimize
        config: Configuration object
        
    Returns:
        Configured optimizer
    """
    return optim.Adam(model.parameters(), lr=config.learning_rate)


def main(preset_name: str = "pretrained_vit_bert_coattention") -> None:
    """
    Main training function.
    
    Args:
        preset_name: Name of the preset configuration to use
        
    Raises:
        ValueError: If training fails
    """
    print("🚀 Starting Modular VQA Training")
    print("=" * 50)
    
    try:
        # Create and configure model
        config = ModularConfig()
        config.get_preset_config(preset_name)
        config.num_epochs = 1  # Set to 1 epoch for quick training
        config.batch_size = 1  # Reduce batch size to fit in GPU memory
        config.print_config()
        
        # Set up data loaders
        train_loader, val_loader = setup_data_loaders(config)
        
        # Create trainer
        trainer = ModularTrainer(
            train_csv='data/train_split.csv',
            img_dir='data/images',
            answer_space_file='data/answer_space.txt',
            train_img_list='data/train_images_split.txt',
            val_csv='data/val_split.csv',
            test_img_list='data/val_images_split.txt',
            config=config
        )
        
        # Train the model
        print(f"\n🎯 Training with preset: {preset_name}")
        trainer.train()
        
        print("\n✅ Training completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise ValueError(f"Training failed: {e}")


if __name__ == "__main__":
    # You can change the preset here or pass it as a command line argument
    import sys
    
    if len(sys.argv) > 1:
        preset_name = sys.argv[1]
    else:
        preset_name = "pretrained_vit_bert_coattention"
    
    main(preset_name) 