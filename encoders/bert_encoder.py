# encoders/bert_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_encoder import BaseEncoder

class BERTEmbedding(nn.Module):
    """BERT-style embedding layer with token, position, and segment embeddings."""
    
    def __init__(self, vocab_size: int, embed_dim: int = 768, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.segment_embedding = nn.Embedding(2, embed_dim)  # 0 for question, 1 for answer (if needed)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, segment_ids=None):
        """
        Args:
            x: Input token indices of shape (batch_size, seq_len)
            segment_ids: Segment IDs of shape (batch_size, seq_len), defaults to zeros
        """
        seq_len = x.size(1)
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand_as(x)
        
        if segment_ids is None:
            segment_ids = torch.zeros_like(x)
        
        token_embeds = self.token_embedding(x)
        position_embeds = self.position_embedding(position_ids)
        segment_embeds = self.segment_embedding(segment_ids)
        
        embeddings = token_embeds + position_embeds + segment_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class BERTAttention(nn.Module):
    """BERT-style multi-head self-attention mechanism."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x, attention_mask=None):
        batch_size, seq_len, embed_dim = x.shape
        
        # Linear projections
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        # Softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # Output projection
        output = self.out_proj(context)
        
        return output

class BERTLayer(nn.Module):
    """BERT transformer layer with self-attention and feed-forward network."""
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = BERTAttention(embed_dim, num_heads, dropout)
        self.attention_norm = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.ff_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x, attention_mask=None):
        # Self-attention with residual connection
        attn_output = self.attention(x, attention_mask)
        x = self.attention_norm(x + attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.ff(x)
        x = self.ff_norm(x + ff_output)
        
        return x

class BERTEncoder(BaseEncoder):
    """
    BERT encoder for text feature extraction.
    Simplified BERT-like architecture for VQA tasks.
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 768, num_heads: int = 12, 
                 num_layers: int = 6, ff_dim: int = 3072, max_seq_len: int = 512, 
                 dropout: float = 0.1, output_dim: int = 768):
        super().__init__(name="BERT")
        self.output_dim = output_dim
        
        # Embedding layer
        self.embedding = BERTEmbedding(vocab_size, embed_dim, max_seq_len, dropout)
        
        # BERT layers
        self.layers = nn.ModuleList([
            BERTLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(embed_dim)
        
        # Pooler (for sentence-level representation)
        self.pooler = nn.Linear(embed_dim, embed_dim)
        
        # Projection to desired output dimension
        if embed_dim != output_dim:
            self.projection = nn.Linear(embed_dim, output_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through BERT encoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len) containing token indices
            attention_mask: Attention mask of shape (batch_size, seq_len), 1 for tokens to attend to, 0 for padding
            
        Returns:
            Encoded features of shape (batch_size, output_dim)
        """
        # Embeddings
        x = self.embedding(x)
        
        # Pass through BERT layers
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        # Final layer norm
        x = self.final_norm(x)
        
        # Pooling: take the first token ([CLS]) representation
        pooled_output = torch.tanh(self.pooler(x[:, 0]))
        
        # Project to desired output dimension
        output = self.projection(pooled_output)
        
        return output 