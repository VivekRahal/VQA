# train.py
import time  # >>> TUNING CHANGE: Import time for epoch timing
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle  # For saving vocabulary

from dataset import VQADataset
from model import VQAModel
from config import Config
from utils.helper import vqa_collate_fn
#from utils.training_stats import print_training_stats  # NEW: Import training stats helper

class Trainer:
    def __init__(self, train_csv, img_dir, answer_space_file=None, train_img_list=None, val_csv=None, test_img_list=None):
        self.config = Config()
        
        # Create training dataset.
        self.train_dataset = VQADataset(train_csv, img_dir, answer_space_file, train_img_list)
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=vqa_collate_fn
        )
        
        # Create validation dataset.
        self.val_dataloader = None
        if val_csv is not None:
            self.val_dataset = VQADataset(val_csv, img_dir, answer_space_file, test_img_list)
            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=vqa_collate_fn
            )
        
        self.vocab_size = len(self.train_dataset.word2idx)
        if self.train_dataset.answer2idx is not None:
            self.config.num_classes = len(self.train_dataset.answer2idx)
        
        # Initialize model.
        self.model = VQAModel(self.vocab_size, self.config).to(self.config.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        
        # >>> TUNING CHANGE: Save vocabulary mapping for consistency in evaluation.
        with open("vocab.pkl", "wb") as f:
            pickle.dump(self.train_dataset.word2idx, f)
        print("[DEBUG] Vocabulary saved to vocab.pkl")
        # <<< TUNING CHANGE
    
    def train(self):
        best_val_loss = float('inf')
        epoch_times = []  # >>> TUNING CHANGE: Record time per epoch
        
        for epoch in range(self.config.num_epochs):
            start_time = time.time()  # >>> TUNING CHANGE: Start epoch timer
            self.model.train()
            running_loss = 0.0
            
            for images, questions, answers in self.train_dataloader:
                images = images.to(self.config.device)
                questions = questions.to(self.config.device)
                answers = answers.to(self.config.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images, questions)
                loss = self.criterion(outputs, answers)
                loss.backward()
                
                # >>> TUNING CHANGE: Apply gradient clipping.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                # <<< TUNING CHANGE
                
                self.optimizer.step()
                running_loss += loss.item()
            
            epoch_time = time.time() - start_time  # >>> TUNING CHANGE: End epoch timer
            epoch_times.append(epoch_time)
            avg_train_loss = running_loss / len(self.train_dataloader)
            print(f"Epoch [{epoch+1}/{self.config.num_epochs}] completed in {epoch_time:.2f} sec, Training Loss: {avg_train_loss:.4f}")
            
            if self.val_dataloader is not None:
                val_loss = self.evaluate(validation=True)
                print(f"Epoch [{epoch+1}/{self.config.num_epochs}], Validation Loss: {val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), "best_model.pth")
            
            self.scheduler.step()
        
        # >>> TUNING CHANGE: Print training statistics after training.
        print_training_stats(epoch_times, self.model)
        # <<< TUNING CHANGE
    
    def evaluate(self, validation=False):
        self.model.eval()
        total_loss = 0.0
        dataloader = self.val_dataloader if validation else self.train_dataloader
        
        if len(dataloader) == 0:
            print("Warning: No data in dataloader, returning 0 loss")
            return 0.0
        
        with torch.no_grad():
            for images, questions, answers in dataloader:
                images = images.to(self.config.device)
                questions = questions.to(self.config.device)
                answers = answers.to(self.config.device)
                
                outputs = self.model(images, questions)
                loss = self.criterion(outputs, answers)
                total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        return avg_loss
