# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class VQAModel(nn.Module):
    def __init__(self, vocab_size, config):
        super(VQAModel, self).__init__()
        self.config = config
        # Define a simple CNN for image feature extraction.
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.MaxPool2d(2),                                       # 32x32 -> 16x16
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 16x16 -> 8x8
            nn.ReLU(),
            nn.MaxPool2d(2),                                       # 8x8 -> 4x4
        )
        # The output of CNN is flattened; calculate its dimension.
        self.image_feature_dim = 32 * 4 * 4
        
        # Define the question model: an embedding layer followed by an LSTM.
        self.embedding = nn.Embedding(vocab_size, 50)  # Embedding dimension = 50
        self.lstm = nn.LSTM(50, config.hidden_size, config.num_layers, batch_first=True)
        
        # Combine image and question features and classify.
        self.fc = nn.Linear(self.image_feature_dim + config.hidden_size, config.num_classes)
        
    def forward(self, image, question):
        batch_size = image.size(0)
        # Process image through CNN.
        x_image = self.cnn(image)
        x_image = x_image.view(batch_size, -1)  # flatten the CNN output
        
        # Process question: embed and pass through LSTM.
        x_question = self.embedding(question)  # Shape: (batch, seq_len, embed_dim)
        lstm_out, (h_n, c_n) = self.lstm(x_question)
        # Take the hidden state from the last LSTM layer.
        x_question = h_n[-1]
        
        # Concatenate image and question features.
        x = torch.cat([x_image, x_question], dim=1)
        logits = self.fc(x)
        return logits
