import torch
from torch import nn
import math
from transformer.components import MultiHeadAttention, FeedForwardNetwork, ResidualConnection, LayerNorm

class DecoderBlock(nn.Module):

    def __init__(self, 
                 features: int,
                 self_attention : MultiHeadAttention,
                 cross_attention : MultiHeadAttention,
                 FFN : FeedForwardNetwork,
                 dropout :float):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.FFN = FFN
        self.residual_conn = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])
    
    def forward(self, x, encoder_output, src_mask, trgt_mask):
        x = self.residual_conn[0](x, lambda x : self.self_attention(x, x, x, trgt_mask))
        x = self.residual_conn[1](x, lambda x : self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.residual_conn[2](x, self.FFN)
        return x

class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(features)
    
    def forward(self, x, encoder_output, src_mask, trgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, trgt_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):

    def __init__(self, d_model :int, vocab_size :int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch, deq_len, d_model) --> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1)