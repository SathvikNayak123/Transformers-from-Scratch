import torch
from torch import nn
import math
from transformer.components import MultiHeadAttention, FeedForwardNetwork, ResidualConnection, LayerNorm

class EncoderBlock(nn.Module):
     
    def __init__(self, 
                 features: int,
                 self_attention : MultiHeadAttention, 
                 FFN : FeedForwardNetwork, 
                 dropout :float):
        super().__init__()
        self.self_attention = self_attention
        self.FFN = FFN
        self.residual_conn = nn.ModuleList([ResidualConnection(features, dropout) for _ in range (2)])

    def forward(self, x, src_mask):
        x = self.residual_conn[0](x, lambda x : self.self_attention(x, x, x, src_mask))
        x = self.residual_conn[1](x, self.FFN)
        return x
    
class Encoder(nn.Module):

    def __init__(self, features: int, layers : nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
