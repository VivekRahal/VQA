# config.py
import torch

class Config:
    def __init__(self):
        self.batch_size = 2
        self.num_epochs = 5
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = 128   # for LSTM
        self.num_layers = 1      # for LSTM
        self.num_classes = 3     # assume 3 possible answers in our dummy dataset
