import torch
from torch import nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model :int, vocab_size :int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncodings(nn.Module):
    
    def __init__(self, d_model :int, seq_len :int, dropout :float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # tensor of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # numerator is a tensor of shape (seq_len, 1)
        position = torch.arange(0,seq_len, dtype=torch.float).unsqueeze(1)
        # denominator in log space
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # sine for even d_model indices, and cos for odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # tensor of shape (1, seq_len, d_model) for batch of sentences
        pe = pe.unsqueeze(0)

        # save in module as pe is calculated only once in transformers
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad(False) #  tensor wont be learned
        return self.dropout(x)

class LayerNorm(nn.Module):

    def __init__(self, eps :float = 10 **-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # multiplied
        self.beta = nn.Parameter(torch.ones(1)) # added

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * ((x - mean)/(std + self.eps)) + self.beta
    

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model :int, d_ff :int, dropout :float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # 512 --> 2048 with W1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model) # 2048 --> 512 with W2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
    
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model :int, h :int,  dropout :float):
        super().__init__()
        self.d_model = d_model
        self.h = h # no. of heads

        assert d_model % h == 0, "d_model not divisible by h"
        self.d_k = d_model // h

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
    
    @staticmethod # can use method without creating class instance
    def attention(query, key, value, mask, dropout :nn.Dropout):
        d_k  = query.shape[0]

        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9) # -inf
        attention_scores = attention_scores.softmax(dim = -1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores
    
    def forward(self, q, k, v, mask):
        # get q, k, v
        query = self.W_q(q) # (batch, seq_len, d_model)
        key = self.W_k(k)
        value = self.W_v(v)

        # (batch, seq_len, d_model) -split to smaller matrices-> (batch, seq_length, h , d_k) -transpose-> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # (batch, h, seq_len, d_k) --> (batch, transpose, h, d_k) -concat all heads-> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (batch, seq_len, d_model)
        return self.W_o(x)
    
class ResidualConnection(nn.Module):

    def __init__(self, dropout :float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm()

    def forward(self, x, sublayer):
        # apply layer norm first then add sublayer
        return x + self.dropout(sublayer(self.norm(x)))
