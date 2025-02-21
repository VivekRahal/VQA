# config.py
import torch

class Config:
    def __init__(self):
        self.batch_size = 16            # >>> TUNING CHANGE: Increased batch size from 2 to 16
        self.num_epochs = 50            # >>> TUNING CHANGE: Increased epochs from 20 to 50
        self.learning_rate = 0.0005     # >>> TUNING CHANGE: Reduced learning rate from 0.001 to 0.0005
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = 256          # >>> TUNING CHANGE: Increased LSTM hidden size from 128 to 256
        self.num_layers = 2             # >>> TUNING CHANGE: Increased LSTM layers from 1 to 2
        self.dropout = 0.5              # >>> TUNING CHANGE: Added dropout hyperparameter for LSTM
        self.num_classes = 0            # This will be updated based on your answer mapping
