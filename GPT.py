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
import random
import numpy as np


class CorpusDataset(Dataset):
    def __init__(self, corpus_path, tokenizer, context_window=128):
        self.tokenizer = tokenizer
        self.context_window = context_window        
        self.tokens = self._tokenize_corpus(corpus_path)
        self.padding_token = self.tokenizer.piece_to_id('[PAD]')
        self.padding = torch.tensor([self.padding_token] * self.context_window, dtype=torch.long)

    def _tokenize_corpus(self, corpus_path):
        with open(corpus_path, 'r') as file:
            corpus = file.read()

        tokens = self.tokenizer.encode(corpus)
        tokens = torch.tensor(tokens, dtype=torch.long)
        return tokens

    def __len__(self,):
        return len(self.tokens) - self.context_window - 1
        
    def __getitem__(self, idx):
        padding_len = random.randint(0, self.context_window)
            
        padding = self.padding[:padding_len]
        x_tokens = self.tokens[idx  :idx+self.context_window-padding_len  ]
        y_tokens = self.tokens[idx+1:idx+self.context_window-padding_len+1]
        
        x = torch.concat([padding, x_tokens], dim=-1)
        y = torch.concat([padding, y_tokens], dim=-1)
        return x, y


class GPT:
    def __init__(self, device='mps', tokenizer=None, n_vocab=10000, chan_dim=1024, n_heads=8, inner_mult=4, Nx=16, max_context=10000, dropout=.3, context_window=128):
        self.init_args = {k:v for k, v in locals().items() if k != 'self'}

        self.n_heads = n_heads
        self.device = device
        self.context_window = context_window

        if device != 'mps':
            print('using torch.compile')
            self.model = torch.compile(Transformer(n_vocab=n_vocab, chan_dim=chan_dim, n_heads=n_heads, 
                                             inner_mult=inner_mult, Nx=Nx, max_context=max_context, 
                                             dropout=dropout, device=device))
        if device == 'mps':
            print('NOT using torch.compile')
            self.model = Transformer(n_vocab=n_vocab, chan_dim=chan_dim, n_heads=n_heads, 
                                     inner_mult=inner_mult, Nx=Nx, max_context=max_context, 
                                     dropout=dropout, device=device)
        if tokenizer:
            self.tokenizer = spm.SentencePieceProcessor(model_file=tokenizer)

        self.padding_token = self.tokenizer.piece_to_id('[PAD]')

        self.loss_history = []

    def train_tokenizer(self, corpus, output_path, vocab_size):
        '''
        Args:
            corpus (str): 'path/to/corpus.txt'
                the corpus must be a document with one sentence per line
            output_path (str): 'path/to/file/prefix'
            vocab_size (int): size of vocab to train
        '''
        spm.SentencePieceTrainer.train(model_prefix=output_path, input=corpus, vocab_size=vocab_size, user_defined_symbols='[PAD]')

    def train(self, corpus_path='./data/corpus.txt', epochs=5, batch_size=16, grad_acc_steps=1, 
              lr=1e-5, num_workers=12, pin_memory=False, break_at=False, show_every=100, time_limit_hours=12):
        start = time.time()
        self.keep_training = True
        capture = ['corpus_path', 'epochs', 'batch_size', 'grad_acc_steps', 'lr', 'num_workers']
        self.train_args = {k:v for k, v in locals().items() if k in capture}
        self.dataset = CorpusDataset(corpus_path, self.tokenizer, self.context_window)
        if not hasattr(self, 'optimizer'):
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        total_steps = (len(self.dataset) * epochs) / batch_size
        epoch_len = len(self.dataset) / batch_size
        self.model.train()
        # loss_accum = []

        for epoch in range(epochs):
            if self.keep_training:
                for i, (x, y_true) in enumerate(dataloader):
                    x = x.to(self.device)
                    y_true = y_true.to(self.device)
                    
                    y_pred = self.model(x)
                    loss = F.cross_entropy(y_pred.permute(0,2,1), y_true, ignore_index=self.dataset.padding_token)
                    # loss = loss / grad_acc_steps
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
    
                    if i % show_every == 0:
                        this_loss = loss.cpu().item()
                        self.loss_history.append(this_loss)
    
                        this_step = (epoch * epoch_len) + i
    
                        clear_output(wait=True)
                        plt.title(f'Loss: {this_loss:.04f}   Step: {this_step}/{total_steps}')
                        plt.plot(self.loss_history)
                        plt.show()
    
                    if break_at:
                        if i > break_at:
                            self.keep_training = False
                            break
    
                    if self.time_limit(start, time_limit_hours):
                        self.keep_training = False
                        break
                    

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
        torch.save(self.optimizer.state_dict(), save_dir + '/optimizer.pt')
        

    def load_weights(self, state_dict_path, load_to_obj):
        state_dict = torch.load(state_dict_path, map_location=self.device)

        if self.device == 'mps':
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('_orig_mod.'):
                    new_key = k[len('_orig_mod.'):]
                    new_state_dict[new_key] = v
                else:
                    new_state_dict[k] = v
            
            load_to_obj.load_state_dict(new_state_dict)

        else:
            load_to_obj.load_state_dict(state_dict)

    def load_checkpoint(self, folder):
        self.load_weights(folder + '/weights.pt', self.model)
        print('model weights loaded')
        
        if not hasattr(self, 'optimizer'):
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
            
        self.load_weights(folder + '/optimizer.pt', self.optimizer)
        print('optimizer weights loaded')

        with open(folder + '/history.json', 'r') as file:
            history = json.load(file)

        self.loss_history = history['history']
        print('training history loaded')
        print('\nready to continue training 😎\n')

    def generate(self, text_lists, n_gen, top_p=0, temperature=1):
        self.model.eval()
        
        batch_size = len(text_lists)
        all_tokens = [[]] * batch_size


        # init input tensor and fill with [PAD]
        input_tensor = torch.empty((batch_size, self.context_window), dtype=torch.int32).to(self.device)
        input_tensor[:, :] = self.tokenizer.piece_to_id('[PAD]')

        # fill with text padded on left side
        for batch_idx, t in enumerate(text_lists):
            tokens = self.tokenizer.encode(t)
            all_tokens[batch_idx] = tokens
            input_tensor[batch_idx, -len(tokens):] = torch.tensor(tokens)
            
        # generate tokens
        for n in tqdm(range(n_gen)):
            y = self.model(input_tensor)
            probabilities = F.softmax(y[:, -1, :] / temperature, dim=-1)
            next_tokens = self.top_p_sample(probabilities, top_p)

            # slide values in input_tensor
            input_tensor = torch.roll(input_tensor, -1, -1)

            # add new tokens
            input_tensor[:, -1:] = next_tokens

            # 
            for batch_idx, token in enumerate(next_tokens.tolist()):
                all_tokens[batch_idx].extend(token)
                

        # decode
        text_out = []
        for batch_idx, tokens in enumerate(all_tokens):
            # not_padding = tokens != self.tokenizer.piece_to_id('[PAD]')
            text = self.tokenizer.decode(tokens)
            text_out.append(text)

        return text_out
    

    def top_p_sample(self, probabilities, top_p):
        sorted_probs, sorted_tokens = torch.sort(probabilities, dim=-1, descending=True)
        sorted_cum_prob = torch.cumsum(sorted_probs, dim=-1)
        not_sample = sorted_cum_prob >= top_p
        # always keep the first choice
        not_sample[:, 0] = False
        sorted_probs[not_sample] = 0.0
        chosen_prob = torch.multinomial(sorted_probs, 1)
        chosen_tokens = torch.gather(sorted_tokens, -1, chosen_prob)
        return chosen_tokens

    @staticmethod
    def time_limit(start, hours_limit):
        end = time.time()
        seconds = end - start
        hours = seconds / 60 / 60
        if hours >= hours_limit:
            return True
        else:
            return False


# TEST Train
if __name__ == '__main__':
    gpt = GPT(tokenizer='./tokenizers/potter_5k_padding.model', n_vocab=5_000,
             chan_dim=512, n_heads=4, inner_mult=4, Nx=16)
    gpt.train(num_workers=0, epochs=1, break_at=100)

# TEST top p
if __name__ == '__main__':
    t0 = torch.tensor([[.2, .1, .2, .4, .05, .05], [.7, .1, .1, .03, .03, .04]])
    display(pd.DataFrame(torch.concat([top_p_sample(t0, 1) for _ in range(20000)], dim=-1).T).value_counts(0, normalize=True))
    display(pd.DataFrame(torch.concat([top_p_sample(t0, 1) for _ in range(20000)], dim=-1).T).value_counts(1, normalize=True))

# TEST Generate
if __name__ == '__main__':
    gpt = GPT(tokenizer='./tokenizers/potter_5k_padding.model', n_vocab=5_000)
    print(gpt.generate(['test', 'multiple'], 10))
