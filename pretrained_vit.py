# pretrained_vit.py
import torch
import torch.nn as nn
from transformers import ViTModel, AutoImageProcessor
from PIL import Image

class ViTEncoder(nn.Module):
    """
    Pretrained Vision Transformer (ViT) encoder for VQA.
    This module loads a pretrained ViT (default: 'google/vit-base-patch16-224') to extract image features.
    """
    def __init__(self, model_name: str = "google/vit-base-patch16-224", use_pretrained: bool = True):
        super(ViTEncoder, self).__init__()
        if use_pretrained:
            self.vit = ViTModel.from_pretrained(model_name)
        else:
            self.vit = ViTModel.from_pretrained(model_name, pretrained=False)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        
    def forward(self, images):
        """
        Args:
            images: Either a list of PIL Images or a torch tensor (B, 3, H, W) with pixel values in [0,1].
        Returns:
            A tensor of shape (B, hidden_dim) containing image features.
        """
        # If images is a tensor, convert to list of PIL Images.
        if isinstance(images, torch.Tensor):
            images = [Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")) for img in images]
        
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.vit.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.vit(**inputs)
        # Use the [CLS] token embedding as the global feature vector.
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # shape: (B, hidden_dim)
        return cls_embeddings

# Example usage:
# from pretrained_vit import ViTEncoder
# vit_encoder = ViTEncoder().to(device)
# features = vit_encoder(list_of_pil_images)
