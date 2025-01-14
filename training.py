import warnings
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from training.data_prep import DataPrep, causal_mask
from transformer.transformer import build_model
from config import get_config, get_weights_file_path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def get_all_sentences(dataset, lang):
    for item in dataset:
        yield item['translation'][lang]

def build_tokenizer(config, dataset, lang):
    # referred from HuggingFace docs
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token = '[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens = ['[UNK]','[PAD]','[EOS]','[SOS]'], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer = trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer

def get_dataset(config):
    raw_ds = load_dataset('Helsinki-NLP/opus_books', f'{config["lang_src"]}-{config["lang_trgt"]}', split = 'train')
    if raw_ds is None:
        print("----------------empty----------------")

    tokenizer_src = build_tokenizer(config, raw_ds, config['lang_src'])
    tokenizer_trgt = build_tokenizer(config, raw_ds, config['lang_trgt'])

    train_size = int(0.9 * len(raw_ds))
    train_ds_raw, val_ds_raw = random_split(raw_ds, [train_size, len(raw_ds)-train_size])

    train_ds = DataPrep(train_ds_raw, tokenizer_src, tokenizer_trgt, config['lang_src'], config['lang_src'], config['seq_len'])
    val_ds = DataPrep(val_ds_raw, tokenizer_src, tokenizer_trgt, config['lang_src'], config['lang_src'], config['seq_len'])

    # get max seq_len for src and trgt sentences
    max_len_src = 0
    max_len_trgt = 0

    for item in raw_ds:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        trgt_ids = tokenizer_src.encode(item['translation'][config['lang_trgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_trgt = max_len_trgt, len(trgt_ids)
    
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(train_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_trgt

def get_model(config, vocab_src_len, vocab_trgt_len):
    model = build_model(vocab_src_len, vocab_trgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model

def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_trgt = get_dataset(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_trgt.get_vocab_size()).to(device)

    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'preloading {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device) # (batch_size, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (batch_size, seq_len)
            
            encoder_mask = batch['encoder_mask'].to(device) # (batch_size, 1, 1, seq_len) as it only masks [PAD] tokens
            decoder_mask = batch['decoder_mask'].to(device) # (batch_size, 1, seq_len, seq_len) it masks [PAD] and 'future' tokens

            # transformer output
            encoder_output = model.encode(encoder_input, encoder_mask) # (batch_size, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            
            # predictions
            proj_output = model.project(decoder_output)
            # ground truth
            label = batch['label'].to(device)

            # (batch_size, trgt_vocab_size) --> (B * seq_len, trgt_vocab_size)
            loss = loss_fn(proj_output.view(-1, tokenizer_trgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss" : f"{loss.item():6.3f}"})

            # logging
            writer.add_scalar("train loss", loss.item(), global_step)
            writer.flush()

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
        
        # save model at end of every epoch
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch' : epoch,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'global_step' : global_step
        }, model_filename)

if __name__=="__main__":
    #warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)