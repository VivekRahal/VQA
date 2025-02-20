# evaluator.py
import torch
from torch.utils.data import DataLoader
import random
import difflib
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToPILImage

from dataset import VQADataset
from config import Config
from utils.helper import vqa_collate_fn

class Evaluator:
    def __init__(self, model, csv_file, img_dir, answer_space_file=None, eval_img_list=None):
        self.config = Config()
        # Create the evaluation dataset using the evaluation CSV and the evaluation image list.
        self.dataset = VQADataset(csv_file, img_dir, answer_space_file, eval_img_list)
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            collate_fn=vqa_collate_fn
        )
        self.model = model
        self.model.to(self.config.device)
        
    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, questions, answers in self.dataloader:
                images = images.to(self.config.device)
                questions = questions.to(self.config.device)
                answers = answers.to(self.config.device)
                
                outputs = self.model(images, questions)
                _, predicted = torch.max(outputs, 1)
                total += answers.size(0)
                correct += (predicted == answers).sum().item()
        accuracy = 100 * correct / total if total > 0 else 0
        print(f"Evaluation Accuracy: {accuracy:.2f}%")
        return accuracy
    
    def display_random_samples(self, num_samples=5):
        # Helper functions for similarity score and tokenization
        def similarity_score(predicted: str, actual: str) -> float:
            return difflib.SequenceMatcher(None, predicted, actual).ratio()
        
        def tokenize_question(question: str, word2idx: dict, max_len: int = 20):
            tokens = []
            for word in question.lower().split():
                tokens.append(word2idx.get(word, word2idx["<UNK>"]))
            if len(tokens) < max_len:
                tokens += [word2idx["<PAD>"]] * (max_len - len(tokens))
            else:
                tokens = tokens[:max_len]
            return tokens
        
        to_pil = ToPILImage()
        indices = random.sample(range(len(self.dataset)), num_samples)
        results = []
        
        # Invert the answer mapping if available.
        if self.dataset.answer2idx is not None:
            idx_to_answer = {v: k for k, v in self.dataset.answer2idx.items()}
        else:
            idx_to_answer = None
        
        for idx in indices:
            # Retrieve sample: (image tensor, tokenized question tensor, actual answer)
            img_tensor, question_tokens, actual_answer = self.dataset[idx]
            # Retrieve the original question text from the dataset DataFrame.
            question_text = self.dataset.data.iloc[idx]["question"]
            pil_image = to_pil(img_tensor.cpu())
            
            tokens = tokenize_question(question_text, self.dataset.word2idx, max_len=20)
            question_tensor = torch.tensor([tokens]).to(self.config.device)
            img_input = img_tensor.unsqueeze(0).to(self.config.device)
            
            with torch.no_grad():
                outputs = self.model(img_input, question_tensor)
                _, predicted = torch.max(outputs, 1)
            predicted_idx = predicted.item()
            
            if idx_to_answer is not None:
                predicted_answer = idx_to_answer.get(predicted_idx, "Unknown")
                actual_answer_text = idx_to_answer.get(actual_answer, "Unknown")
            else:
                predicted_answer = str(predicted_idx)
                actual_answer_text = str(actual_answer)
            
            sim = similarity_score(predicted_answer, actual_answer_text)
            results.append((pil_image, question_text, predicted_answer, actual_answer_text, sim))
        
        fig, axes = plt.subplots(1, num_samples, figsize=(20, 5))
        if num_samples == 1:
            axes = [axes]
        for ax, (img, question, pred_ans, actual_ans, sim) in zip(axes, results):
            ax.imshow(np.array(img))
            ax.axis("off")
            ax.set_title(f"Q: {question}\nPred: {pred_ans}\nActual: {actual_ans}\nSim: {sim*100:.1f}%", fontsize=8)
        plt.tight_layout()
        plt.show()
