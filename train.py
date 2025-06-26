# trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
from dataset import VQADataset
from model import VQAModel
from config import Config
from utils.helper import vqa_collate_fn

class Trainer:
    def __init__(self, train_csv, img_dir, answer_space_file=None, train_img_list=None, val_csv=None, test_img_list=None):
        self.config = Config()
        
        # Create the training dataset using the training CSV and the training image list.
        self.train_dataset = VQADataset(train_csv, img_dir, answer_space_file, train_img_list)
        self.train_dataloader = DataLoader(
            self.train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            collate_fn=vqa_collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers
        )
        
        # Create the validation dataset using the validation CSV and the test image list.
        self.val_dataloader = None
        if val_csv is not None:
            self.val_dataset = VQADataset(val_csv, img_dir, answer_space_file, test_img_list)
            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=vqa_collate_fn,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                persistent_workers=self.config.persistent_workers
            )
        
        # Set vocabulary size and update number of answer classes (if applicable)
        self.vocab_size = len(self.train_dataset.word2idx)
        if self.train_dataset.answer2idx is not None:
            self.config.num_classes = len(self.train_dataset.answer2idx)
        
        # Create the model.
        self.model = VQAModel(self.vocab_size, self.config).to(self.config.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

        # changes for random samples:
        # Save the vocabulary mapping so that evaluation uses the same vocabulary.
        with open("vocab.pkl", "wb") as f:
            pickle.dump(self.train_dataset.word2idx, f)
        print("[DEBUG] Vocabulary saved to vocab.pkl")

    def train(self):
        best_val_loss = float('inf')
        for epoch in range(self.config.num_epochs):
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
                self.optimizer.step()
                
                running_loss += loss.item()
            avg_train_loss = running_loss / len(self.train_dataloader)
            print(f"Epoch [{epoch+1}/{self.config.num_epochs}], Training Loss: {avg_train_loss:.4f}")
            
            if self.val_dataloader is not None:
                val_loss = self.evaluate(validation=True)
                print(f"Epoch [{epoch+1}/{self.config.num_epochs}], Validation Loss: {val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), "best_model.pth")
            
            self.scheduler.step()
    
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
