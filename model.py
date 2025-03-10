# model.py
import torch
import torch.nn as nn
#from torchvision import models  # <<-- Old CNN-based code was here (commented out below)
from transformers import ViTModel, AutoImageProcessor
from PIL import Image

# >>> USING PRETRAINED ViT ENCODER INSTEAD OF CNN
class ViTEncoder(nn.Module):
    """
    A ViT-based image encoder that uses a pre-trained Vision Transformer (ViT)
    to extract image features. The [CLS] token from the last hidden state is used
    as the global image representation.
    """
    def __init__(self, model_name: str = "google/vit-base-patch16-224", use_pretrained: bool = True):
        super(ViTEncoder, self).__init__()
        if use_pretrained:
            self.vit = ViTModel.from_pretrained(model_name)
        else:
            self.vit = ViTModel.from_pretrained(model_name, pretrained=False)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
    
    def forward(self, images):
        # If images is a tensor, convert each sample to a PIL image.
        if isinstance(images, torch.Tensor):
            images = images.cpu()
            images = [Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype('uint8')) for img in images]
        inputs = self.processor(images=images, return_tensors="pt")
        # Move inputs to the same device as the ViT model.
        inputs = {k: v.to(self.vit.device) for k, v in inputs.items()}
        with torch.no_grad():  # Freeze ViT parameters if not fine-tuning.
            outputs = self.vit(**inputs)
        # Use the [CLS] token embedding from the last hidden state.
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # Shape: (B, hidden_dim) with hidden_dim=768 for vit-base.
        return cls_embeddings
# <<< USING PRETRAINED ViT ENCODER

class VQAModel(nn.Module):
    def __init__(self, vocab_size, config):
        super(VQAModel, self).__init__()
        self.config = config
        
        # >>> USING ViT ENCODER INSTEAD OF CNN
        self.vit_encoder = ViTEncoder()  # Pre-trained ViT encoder for image features.
        self.image_feature_dim = 768     # For vit-base patch16-224.
        # <<< End of ViT encoder usage.
        
        # >>> OLD CNN-BASED ENCODER (COMMENTED OUT)
        # self.cnn = nn.Sequential(
        #     ResidualBlock(3, 16, stride=2),
        #     nn.MaxPool2d(2),
        #     ResidualBlock(16, 32, stride=2),
        #     nn.MaxPool2d(2)
        # )
        # self.image_feature_dim = 32 * 4 * 4
        # <<< OLD CNN-BASED ENCODER
        
        # Text encoder: embedding and LSTM remain unchanged.
        self.embedding = nn.Embedding(vocab_size, 50)
        self.lstm = nn.LSTM(50, config.hidden_size, config.num_layers, 
                            batch_first=True, dropout=config.dropout)
        # Final classification layer fusing image and question features.
        self.fc = nn.Linear(self.image_feature_dim + config.hidden_size, config.num_classes)
        
    def forward(self, image, question):
        batch_size = image.size(0)
        # Extract image features using ViT.
        x_image = self.vit_encoder(image)  # Shape: (batch_size, 768)
        
        # Process the question using embedding and LSTM.
        x_question = self.embedding(question)  # (batch, seq_len, embed_dim)
        lstm_out, (h_n, c_n) = self.lstm(x_question)
        x_question = h_n[-1]  # Last layer's hidden state.
        
        # Fuse image and question features.
        x = torch.cat([x_image, x_question], dim=1)
        logits = self.fc(x)
        return logits
