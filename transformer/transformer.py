import torch
from torch import nn
import math
from transformer.components import InputEmbeddings, PositionalEncodings, MultiHeadAttention, FeedForwardNetwork, ResidualConnection, LayerNorm
from transformer.encoder import EncoderBlock, Encoder
from transformer.decoder import DecoderBlock, Decoder, ProjectionLayer

class Transformer(nn.Module):

    def __init__(self,
                 encoder : Encoder,
                 decoder : Decoder,
                 src_embed : InputEmbeddings,
                 trgt_embed : InputEmbeddings,
                 src_pos : PositionalEncodings,
                 trgt_pos : PositionalEncodings,
                 proj_layer : ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trgt_embed = trgt_embed
        self.src_pos = src_pos
        self.trgt_pos = trgt_pos
        self.proj_layer = proj_layer
    
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, trgt, trgt_mask):
        trgt = self.trgt_embed(trgt)
        trgt = self.trgt_pos(trgt)
        return self.decoder(trgt, encoder_output, src_mask, trgt_mask)
    
    def project(self, x):
        return self.proj_layer(x)

def build_model(src_vocab_size :int, trgt_vocab_size :int, src_seq_len :int, trgt_seq_len :int, d_model :int = 512, d_ff :int = 2048,N :int = 6, h :int = 8, dropout :float = 0.25):

    src_embed = InputEmbeddings(d_model, src_vocab_size)
    trgt_embed = InputEmbeddings(d_model, trgt_vocab_size)

    src_pos = PositionalEncodings(d_model, src_seq_len, dropout)
    trgt_pos = PositionalEncodings(d_model, trgt_seq_len, dropout)

    encoder_blocks = []
    for _ in range(N):
        self_attention = MultiHeadAttention(d_model, h, dropout)
        ffn = FeedForwardNetwork(d_model , d_ff, dropout)
        encoder_block = EncoderBlock(d_model, self_attention, ffn, dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for _ in range(N):
        self_attention = MultiHeadAttention(d_model, h, dropout)
        cross_attention = MultiHeadAttention(d_model, h, dropout)
        ffn = FeedForwardNetwork(d_model , d_ff, dropout)
        decoder_block = DecoderBlock(d_model, self_attention, cross_attention, ffn, dropout)
        decoder_blocks.append(decoder_block)

    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(d_model, trgt_vocab_size)

    transformer = Transformer(encoder, decoder, src_embed, trgt_embed, src_pos, trgt_pos, projection_layer)

    # initialize params using Xavier initialization
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer
    


