import torch
from torch import nn
from torch.utils.data import Dataset

class DataPrep(Dataset):

    def __init__(self, dataset, tokenizer_src, tokenizer_trgt, src_lang, trgt_lang, seq_len):
        super().__init__()
        self.dataset =dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_trgt = tokenizer_trgt
        self.src_lang = src_lang
        self.trgt_lang = trgt_lang
        self.seq_len = seq_len
        
        # Start of sentence, End of sentence and Padding tokens
        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype = torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype = torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype = torch.int64)

    def __len__(self):
       return len(self.dataset)
   
    def __getitem__(self, index):
        src_trgt_pair = self.dataset[index]
        src_text = src_trgt_pair['translation'][self.src_lang]
        trgt_text = src_trgt_pair['translation'][self.trgt_lang]

        encoder_input_tokens = self.tokenizer_src.encode(src_text).ids
        decoder_input_tokens = self.tokenizer_trgt.encode(trgt_text).ids

        encoder_padding_tokens = self.seq_len - len(encoder_input_tokens) - 2
        decoder_padding_tokens = self.seq_len - len(decoder_input_tokens) - 1

        if encoder_padding_tokens < 0 or decoder_padding_tokens < 0:
            raise ValueError('seq_len is too long')

        # Add SOS, EOS and padding to the encoder input tokens
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(encoder_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * encoder_padding_tokens, dtype = torch.int64)
            ],
            dim=0
        )
       # Add SOS and padding to the decoder input tokens
        decoder_input = torch.cat(
           [
               self.sos_token,
               torch.tensor(decoder_input_tokens, dtype=torch.int64),
               torch.tensor([self.pad_token] * decoder_padding_tokens, dtype = torch.int64)
           ],
            dim=0
        )

        # Ground Truth - output we expect from decoder
        label = torch.cat(
            [
                torch.tensor(decoder_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * decoder_padding_tokens, dtype = torch.int64)
            ],
            dim=0
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input" : encoder_input,
            "decoder_input" : decoder_input,
            "encoder_mask" : (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # ensures '[PAD]' tokens are not passed to self attention
            "decoder_mask" : (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),# ensures '[PAD]' and 'future' tokens are not passed to masked attention
            "label" : label,
            "src_text" : src_text,
            "trgt_text" : trgt_text
        }
    

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal = 1).type(torch.int) # makes everything above diagonal 1 and rest 0
    return mask == 0 # returns False for everything above diagonal and True for rest => masking 'future' tokens