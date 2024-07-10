# ***
# This is a ground up build of a GPT style model for academic purposes.
# ***

# +
import torch.nn as nn
import torch.nn.functional as F
import torch

from torch import empty, zeros
from torch.nn.init import xavier_normal_, kaiming_normal_


# -

class FeedForward(nn.Module):
    '''
    Feed Forward Block from from  "Attention is all you Need" (Vaswani et al., 2017)
    implemented as FFN(x) = max(0, xW1 + b1 )W2 + b2
    '''
    def __init__(self, chan_dim=512, inner_mult=4, dropout=.2, device='cpu', **kwargs):
        super().__init__(**kwargs)
        self.W1 = nn.Parameter(kaiming_normal_(empty(chan_dim, chan_dim*inner_mult ))).to(device)
        self.b1 = nn.Parameter(zeros(chan_dim*inner_mult)).to(device)
        self.W2 = nn.Parameter(xavier_normal_(empty(chan_dim*inner_mult, chan_dim))).to(device)
        self.b2 = nn.Parameter(zeros(chan_dim)).to(device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # FFN(x) = max(0, xW1 + b1 )W2 + b2
        x = torch.einsum('btc,ci->bti', x, self.W1) + self.b1 # upward projection
        x = F.relu(x)
        x = torch.einsum('bti,ic->btc', x, self.W2) + self.b2 # downwards projection
        x = self.dropout(x)
        return x


class MuliHeadedMaskedSelfAttention(nn.Module):
    '''from  "Attention is all you Need" (Vaswani et al., 2017)'''
    
    def __init__(self, chan_dim=512, n_heads=4, dropout=.2, device='cpu', **kwargs):
        '''
        Args:
            n_heads: integer
                number of heads
        '''
        super().__init__(**kwargs)

        # d_k = d_v = d_model / h - 3.2.2 page 5
        self.d_k = chan_dim // n_heads

        self.Q = nn.Parameter(xavier_normal_(empty(chan_dim, n_heads, self.d_k))).to(device)
        self.K = nn.Parameter(xavier_normal_(empty(chan_dim, n_heads, self.d_k))).to(device)
        self.V = nn.Parameter(xavier_normal_(empty(chan_dim, n_heads, self.d_k))).to(device)
        self.O = nn.Parameter(xavier_normal_(empty(self.d_k * n_heads, chan_dim))).to(device)


        self.dropout1   = nn.Dropout(dropout) # REVIEW IF NEEDED
        self.dropout2   = nn.Dropout(dropout)
        

    def forward(self, x):
        '''
        Args:
            x: tensor
                of shape [B, T, C]

        Returns:
            x: tensor
                of shape [B, T, C]
        '''
        
        # calculate k, q, v, for all batches and heads
        q = torch.einsum('btc,chd->bhtd', x, self.Q)
        k = torch.einsum('btc,chd->bhtd', x, self.K)
        v = torch.einsum('btc,chd->bhtd', x, self.V)

        # scaled dot prod step 1 [ q @ kT / d_k**.5 ] - 3.2.1 page 4
        w = torch.einsum('bhtd,bhzd->bhtz', q, k) * self.d_k**0.5

        # masking - 3.2.3 page 5
        # keep the lower triangle
        mask = torch.tril(torch.ones_like(w, dtype=torch.bool))
        # set upper tri to -inf
        w = torch.where(mask, w, torch.tensor(float('-inf')))

        # scaled dot prod step 2 [ softmax ]- 3.2.1 page 4
        w = F.softmax(w, dim=-1)

        # dropout
        # w = self.dropout1(w)

        # scaled dot prod step 3 [ @ v ]- 3.2.1 page 4
        x = torch.einsum('bhtz,bhzd->bhtd', w, v)

        # from [B, H, T, d_k] to [B, T, H*d_k] - 3.2.2 page 5
        x = torch.permute(x, (0, 2, 1, 3))
        B, T, H, d_k = x.shape
        x = torch.reshape(x, (B, T, H*d_k))

        # linear - 3.3 page 5
        # transform back to main transformer channel dim
        x = torch.einsum('bti,ic->btc', x, self.O)

        # dropout
        x = self.dropout2(x)
        
        return x


class PositionalEncoding(nn.Module):
    '''
    Adds positional encodings to embedding outputs    
    '''

    def __init__(self, chan_dim=512, max_context=10000, device='cpu', **kwargs):
        '''
        Args:
            max_pos: integer
                maximum len of sequence that can be positionally encoded
        '''
        super().__init__(**kwargs)
        # self.pos   = float(max_pos)
        self.t_pos = torch.arange(float(max_context), dtype=torch.float32).to(device)
        self.c_pos = torch.arange(float(chan_dim), dtype=torch.float32).to(device) # notated as i in original equation

    def forward(self, x):
        '''
        PE(pos, 2i)   = sin(pos/10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        '''
        B, T, C = x.shape
                
        val = self.t_pos[:T, None] / torch.pow(10000., (2. * self.c_pos[None, :]) / C)
        sin_mat = torch.sin(val)
        cos_mat = torch.cos(val)

        mask = torch.tile((self.c_pos % 2 == 0), [T, 1])

        pos_enc = torch.where(mask, sin_mat, cos_mat)

        return x + pos_enc


class TransformerBlock(nn.Module):
    '''from  "Attention is all you Need" (Vaswani et al., 2017)'''
    def __init__(self, chan_dim=512, n_heads=4, inner_mult=4, dropout=.2, device='cpu', **kwargs):
        '''
        Args:
            n_heads: int
                number of transformer heads
            dropout: float
                dropout_rate, padded to dropout layers
        '''
        super().__init__(**kwargs)
        self.att = MuliHeadedMaskedSelfAttention(chan_dim, n_heads, dropout, device)
        self.ffd = FeedForward(chan_dim, inner_mult, dropout, device)
        self.ln1 = nn.LayerNorm(chan_dim).to(device)
        self.ln2 = nn.LayerNorm(chan_dim).to(device)
        
    def forward(self, x):
        '''
        Args:
            x: tensor
                shape [B, T, C]
        Returns:
            x: tensor
                shape [B, T, C]
        '''
        # residual connections
        x = x + self.att(self.ln1(x))
        x = x + self.ffd(self.ln2(x))
        return x


class Transformer(nn.Module):
    '''
    Generative Pretrained Transformer Architecture
    '''
    def __init__(self, n_vocab, chan_dim=512, n_heads=4, inner_mult=4, Nx=16, max_context=10000, dropout=.3, device='cpu', **kwargs):
        '''
        Args:
        '''
        super().__init__(**kwargs)

        # layers
        self.emb = nn.Embedding(n_vocab, chan_dim, device=device)
        self.pos = PositionalEncoding(chan_dim, max_context, device)

        self.stack = nn.ModuleList([TransformerBlock(chan_dim, n_heads, inner_mult, dropout, device) for i in range(Nx)])

        self.ln = nn.LayerNorm(chan_dim, device=device)
        self.to_vocab = nn.Linear(chan_dim, n_vocab, device=device)


    def forward(self, x):
        # embedding
        x = self.emb(x)

        # positional encoding
        x = self.pos(x)

        # transformer stack
        for T in self.stack:
            x = T(x)

        # output
        x = self.to_vocab(self.ln(x))

        return x
