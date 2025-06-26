# modular_trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
import numpy as np
from dataset import VQADataset
from modular_model import ModularVQAModel
from modular_config import ModularConfig
from utils.helper import vqa_collate_fn

class ModularTrainer:
    """
    Modular trainer that supports different encoder and fusion combinations.
    """
    
    def __init__(self, train_csv, img_dir, answer_space_file=None, train_img_list=None, 
                 val_csv=None, test_img_list=None, config=None):
        self.config = config if config is not None else ModularConfig()
        
        # Create the training dataset
        self.train_dataset = VQADataset(train_csv, img_dir, answer_space_file, train_img_list)
        self.train_dataloader = DataLoader(
            self.train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            collate_fn=vqa_collate_fn
        )
        
        # Create the validation dataset if provided
        self.val_dataloader = None
        if val_csv is not None:
            self.val_dataset = VQADataset(val_csv, img_dir, answer_space_file, test_img_list)
            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=vqa_collate_fn
            )
        
        # Set vocabulary size and number of classes
        self.config.set_vocab_size(len(self.train_dataset.word2idx))
        if self.train_dataset.answer2idx is not None:
            self.config.set_num_classes(len(self.train_dataset.answer2idx))
        
        # Create the modular model
        self.model = ModularVQAModel(self.config).to(self.config.device)
        
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        
        # Training history tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.patience_counter = 0
        self.early_stopping_patience = 10
        
        # Save vocabulary for evaluation
        with open("vocab.pkl", "wb") as f:
            pickle.dump(self.train_dataset.word2idx, f)
        print("[DEBUG] Vocabulary saved to vocab.pkl")
        
        # Print configuration
        self.config.print_config()
    
    def calculate_accuracy(self, outputs, targets):
        """Calculate accuracy for the batch."""
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == targets).sum().item()
        return correct / targets.size(0)
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = len(self.train_dataloader)
        
        for batch_idx, (images, questions, answers) in enumerate(self.train_dataloader):
            images = images.to(self.config.device)
            questions = questions.to(self.config.device)
            answers = answers.to(self.config.device)
            
            # Create attention mask for BERT if needed
            attention_mask = None
            if self.config.text_encoder_type.lower() == "bert":
                attention_mask = (questions != 0).float().to(self.config.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images, questions, attention_mask)
            loss = self.criterion(outputs, answers)
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            accuracy = self.calculate_accuracy(outputs, answers)
            
            total_loss += loss.item()
            total_accuracy += accuracy
            
            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.config.num_epochs}], "
                      f"Batch [{batch_idx}/{num_batches}], "
                      f"Loss: {loss.item():.4f}, "
                      f"Accuracy: {accuracy:.4f}")
        
        avg_train_loss = total_loss / num_batches
        avg_train_accuracy = total_accuracy / num_batches
        
        return avg_train_loss, avg_train_accuracy
    
    def validate_epoch(self, epoch):
        """Validate for one epoch."""
        if self.val_dataloader is None:
            print("⚠️  No validation data provided - skipping validation")
            return 0.0, 0.0
            
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = len(self.val_dataloader)
        
        print(f"Validating on {num_batches} validation batches...")
        
        with torch.no_grad():
            for batch_idx, (images, questions, answers) in enumerate(self.val_dataloader):
                images = images.to(self.config.device)
                questions = questions.to(self.config.device)
                answers = answers.to(self.config.device)
                
                # Create attention mask for BERT if needed
                attention_mask = None
                if self.config.text_encoder_type.lower() == "bert":
                    attention_mask = (questions != 0).float().to(self.config.device)
                
                outputs = self.model(images, questions, attention_mask)
                loss = self.criterion(outputs, answers)
                accuracy = self.calculate_accuracy(outputs, answers)
                
                total_loss += loss.item()
                total_accuracy += accuracy
                
                # Print validation progress every 5 batches
                if batch_idx % 5 == 0:
                    print(f"  Val Batch [{batch_idx}/{num_batches}], Loss: {loss.item():.4f}, Acc: {accuracy:.4f}")
        
        avg_val_loss = total_loss / num_batches
        avg_val_accuracy = total_accuracy / num_batches
        
        return avg_val_loss, avg_val_accuracy
    
    def check_overfitting(self, train_loss, val_loss, train_acc, val_acc):
        """Check for overfitting and underfitting."""
        if val_loss > train_loss * 1.5:
            return "OVERFITTING: Validation loss is much higher than training loss"
        elif val_acc < train_acc * 0.8:
            return "OVERFITTING: Validation accuracy is much lower than training accuracy"
        elif train_acc < 0.3 and val_acc < 0.3:
            return "UNDERFITTING: Both training and validation accuracy are low"
        elif train_loss > 4.0 and val_loss > 4.0:
            return "UNDERFITTING: Both training and validation losses are high"
        else:
            return "GOOD: Model is learning well"
    
    def train(self):
        """Train the model with proper monitoring."""
        print(f"\n{'='*60}")
        print(f"STARTING TRAINING - {self.config.num_epochs} EPOCHS")
        print(f"{'='*60}")
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch [{epoch+1}/{self.config.num_epochs}]")
            print("-" * 40)
            
            # Training phase
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validation phase
            val_loss, val_acc = self.validate_epoch(epoch)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Print epoch summary
            print(f"\nEpoch [{epoch+1}/{self.config.num_epochs}] Summary:")
            print(f"Training   - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
            if self.val_dataloader is not None:
                print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
            
            # Check for overfitting/underfitting
            if self.val_dataloader is not None:
                status = self.check_overfitting(train_loss, val_loss, train_acc, val_acc)
                print(f"Status: {status}")
            
            # Save best model based on validation loss
            if self.val_dataloader is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_accuracy = val_acc
                self.patience_counter = 0
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'config': self.config,
                    'vocab': self.train_dataset.word2idx,
                    'answer_mapping': self.train_dataset.answer2idx,
                    'epoch': epoch + 1,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc
                }, "best_modular_model.pth")
                print(f"✅ New best model saved! (Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f})")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f"🛑 Early stopping triggered after {epoch+1} epochs")
                break
            
            # Learning rate scheduling
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Learning Rate: {current_lr:.6f}")
        
        # Final summary
        print(f"\n{'='*60}")
        print("TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"Best Validation Loss: {self.best_val_loss:.4f}")
        print(f"Best Validation Accuracy: {self.best_val_accuracy:.4f}")
        print(f"Final Training Loss: {self.train_losses[-1]:.4f}")
        print(f"Final Training Accuracy: {self.train_accuracies[-1]:.4f}")
        
        # Plot training curves if matplotlib is available
        try:
            self.plot_training_curves()
        except ImportError:
            print("Matplotlib not available for plotting training curves")
    
    def plot_training_curves(self):
        """Plot training and validation curves."""
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Loss curves
            ax1.plot(self.train_losses, label='Training Loss', color='blue')
            if self.val_losses:
                ax1.plot(self.val_losses, label='Validation Loss', color='red')
            ax1.set_title('Training and Validation Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Accuracy curves
            ax2.plot(self.train_accuracies, label='Training Accuracy', color='blue')
            if self.val_accuracies:
                ax2.plot(self.val_accuracies, label='Validation Accuracy', color='red')
            ax2.set_title('Training and Validation Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
            print("📊 Training curves saved to 'training_curves.png'")
            
        except ImportError:
            print("Matplotlib not available for plotting")
    
    def evaluate(self, validation=False):
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        dataloader = self.val_dataloader if validation else self.train_dataloader
        
        if len(dataloader) == 0:
            print("Warning: No data in dataloader, returning 0 loss")
            return 0.0, 0.0
        
        with torch.no_grad():
            for images, questions, answers in dataloader:
                images = images.to(self.config.device)
                questions = questions.to(self.config.device)
                answers = answers.to(self.config.device)
                
                # Create attention mask for BERT if needed
                attention_mask = None
                if self.config.text_encoder_type.lower() == "bert":
                    attention_mask = (questions != 0).float().to(self.config.device)
                
                outputs = self.model(images, questions, attention_mask)
                loss = self.criterion(outputs, answers)
                accuracy = self.calculate_accuracy(outputs, answers)
                
                total_loss += loss.item()
                total_accuracy += accuracy
        
        avg_loss = total_loss / len(dataloader)
        avg_accuracy = total_accuracy / len(dataloader)
        
        return avg_loss, avg_accuracy
    
    def get_model_info(self):
        """Get detailed information about the model."""
        return self.model.get_model_info()
    
    def save_model(self, filepath="modular_model.pth"):
        """Save the model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'vocab': self.train_dataset.word2idx,
            'answer_mapping': self.train_dataset.answer2idx,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath="modular_model.pth"):
        """Load the model."""
        checkpoint = torch.load(filepath, map_location=self.config.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            self.train_accuracies = checkpoint['train_accuracies']
            self.val_accuracies = checkpoint['val_accuracies']
        print(f"Model loaded from {filepath}") 