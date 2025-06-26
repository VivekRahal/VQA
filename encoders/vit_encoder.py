# encoders/vit_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_encoder import BaseEncoder

class PatchEmbedding(nn.Module):
    """Patch embedding layer for Vision Transformer."""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        Returns:
            Patches of shape (batch_size, n_patches, embed_dim)
        """
        x = self.proj(x)  # (batch_size, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2)  # (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)
        return x

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
    
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward network."""
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViTEncoder(BaseEncoder):
    """
    Vision Transformer (ViT) encoder for image feature extraction.
    """
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, 
                 embed_dim: int = 768, num_heads: int = 12, num_layers: int = 12, 
                 mlp_ratio: float = 4.0, dropout: float = 0.1, output_dim: int = 768):
        super().__init__(name="ViT")
        self.output_dim = output_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim))
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Projection to desired output dimension
        if embed_dim != output_dim:
            self.projection = nn.Linear(embed_dim, output_dim)
        else:
            self.projection = nn.Identity()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ViT encoder.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Encoded features of shape (batch_size, output_dim)
        """
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch_size, n_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, n_patches+1, embed_dim)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Layer norm
        x = self.norm(x)
        
        # Take class token representation
        x = x[:, 0]  # (batch_size, embed_dim)
        
        # Project to desired output dimension
        x = self.projection(x)
        
        return x 