from .model import Transformer
import sentencepiece as spm
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
import json
import matplotlib.pyplot as plt
from IPython.display import clear_output
from datetime import datetime
import os
from typing import List


class CorpusDataset(Dataset):
    def __init__(self, corpus_path, tokenizer, context_window=128):
        self.tokenizer = tokenizer
        self.context_window = context_window        
        self.tokens = self._tokenize_corpus(corpus_path)

    def _tokenize_corpus(self, corpus_path):
        with open(corpus_path, 'r') as file:
            corpus = file.read()

        tokens = self.tokenizer.encode(corpus)
        tokens = torch.tensor(tokens)
        return tokens

    def __len__(self,):
        return len(self.tokens) - self.context_window - 1
        

    def __getitem__(self, idx):
        x = self.tokens[idx:idx+self.context_window]
        y = self.tokens[idx+1:idx+self.context_window+1]
        return x, y


class GPT:
    def __init__(self, device='mps', tokenizer=None, n_vocab=10000, chan_dim=1024, n_heads=8, inner_mult=4, Nx=16, max_context=10000, dropout=.3):
        self.init_args = {k:v for k, v in locals().items() if k != 'self'}

        self.device = device
        
        self.model = torch.jit.script(Transformer(n_vocab=n_vocab, chan_dim=chan_dim, n_heads=n_heads, 
                                         inner_mult=inner_mult, Nx=Nx, max_context=max_context, 
                                         dropout=dropout, device=device))
        if tokenizer:
            self.tokenizer = spm.SentencePieceProcessor(model_file=tokenizer)

        self.loss_history = []

    def train_tokenizer(self, corpus, output_path, vocab_size):
        '''
        Args:
            corpus (str): 'path/to/corpus/.txt'
                the corpus must be a document with one sentence per line
            output_path (str): 'path/to/file/prefix'
            vocab_size (int): size of vocab to train
        '''
        spm.SentencePieceTrainer.train(model_prefix=output_path, input=corpus, vocab_size=vocab_size)

    def train(self, corpus_path='./data/corpus.txt', context_window=128, epochs=5, batch_size=16, grad_acc_steps=4, lr=1e-5, num_workers=12):
        capture = ['corpus_path', 'context_window', 'epochs', 'batch_size', 'grad_acc_steps', 'lr', 'num_workers']
        self.train_args = {k:v for k, v in locals().items() if k in capture}
        self.dataset = CorpusDataset(corpus_path, self.tokenizer, context_window)
        if not hasattr(self, 'optimizer'):
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        total_steps = round((len(self.dataset) * epochs) / batch_size / grad_acc_steps)
        self.model.train()

        for epoch in range(epochs):
            for i, (x, y_true) in enumerate(dataloader):
                x = x.to(self.device)
                y_true = y_true.to(self.device)
                
                y_pred = self.model(x)
                loss = F.cross_entropy(y_pred.permute(0,2,1), y_true)
                loss = loss / grad_acc_steps
                loss.backward()

                if (i + 1) % grad_acc_steps == 0:
                    this_loss = loss.cpu().item()
                    self.loss_history.append(this_loss)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    clear_output(wait=True)
                    plt.title(f'Loss: {this_loss:.04f}   Step: {round(i/grad_acc_steps)}/{total_steps}')
                    plt.plot(self.loss_history)
                    plt.show() 

    def save_train(self, path):
        name = datetime.now().strftime('%Y-%m-%d__%H_%M_%S')
        save_dir = path + '/' + name
        os.mkdir(save_dir)
        training_history = {
            'init':self.init_args,
            'train':self.train_args,
            'history':self.loss_history,
        }
        with open(save_dir + '/history.json', 'w') as file:
            json.dump(training_history, file)

        torch.save(self.model.state_dict(), save_dir + '/weights.pt')

# @torch.jit.script
# def train_loop(epochs, dataloader, model, optimizer, loss_history, device):
#     for epoch in range(int(epochs)):
#         for i, (x, y_true) in enumerate(dataloader):
#             x = x.to(device)
#             y_true = y_true.to(device)

#             y_pred = model(x)
#             loss = F.cross_entropy(y_pred.permute(0,2,1), y_true)
#             loss = loss / grad_acc_steps
#             loss.backward()

#             if (i + 1) % grad_acc_steps == 0:
#                 this_loss = loss.cpu().item()
#                 loss_history.append(this_loss)
#                 optimizer.step()
#                 optimizer.zero_grad()

                # clear_output(wait=True)
                # plt.title(f'Loss: {this_loss:.04f}   Step: {round(i/grad_acc_steps)}/{total_steps}')
                # plt.plot(loss_history)
                # plt.show()

# @torch.jit.script
# def train_loop(epochs: int, dataloader: DataLoader, model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss_history: List[float], device: torch.device, grad_acc_steps: int):
#     for epoch in range(epochs):
#         for i, (x, y_true) in enumerate(dataloader):
#             x = x.to(device)
#             y_true = y_true.to(device)

#             y_pred = model(x)
#             loss = F.cross_entropy(y_pred, y_true)
#             loss = loss / grad_acc_steps
#             loss.backward()

#             if (i + 1) % grad_acc_steps == 0:
#                 this_loss = loss.item()
#                 loss_history.append(this_loss)
#                 optimizer.step()
#                 optimizer.zero_grad()
