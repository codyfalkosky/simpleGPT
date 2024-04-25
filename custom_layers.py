# ***
# This is a ground up build of a GPT style model for academic purposes.
# ***

import tensorflow as tf
from tensorflow.keras.layers import Dense, Softmax, Dropout, LayerNormalization, Input, Embedding
from tensorflow.keras.initializers import GlorotNormal


class FeedForward(tf.keras.layers.Layer):
    '''
    Feed Forward Block from from  "Attention is all you Need" (Vaswani et al., 2017)
    implemented as FFN(x) = max(0, xW1 + b1 )W2 + b2
    '''

    def __init__(self, inner_mult=4, dropout=.2, **kwargs):
        '''
        Args:
            inner_mult: integer
                W1 is mxn, W2 is nxp, m = p, n = m * inner_mult
                the mulitplier of the outer dim to the inner
            dropout: float
                dropout rate passed to dropout layers
    
        Call Arguments:
            x: tensor
                of shape [B, T, C]
    
        Returns:
            x: tensor
            of shape [B, T, C]
        '''
        super().__init__(**kwargs)
        self.inner_mult = inner_mult
        
        self.relu       = tf.keras.layers.ReLU()
        self.dropout    = Dropout(dropout)

    def build(self, input_shape):
        b, t, c = input_shape
        
        self.linear1_m = self.add_weight(
            shape = [c, int(c*self.inner_mult)],
            initializer = 'glorot_normal',
            name = 'linear1_m',
            trainable = True
        )
        self.linear1_b = self.add_weight(
            shape = [int(c*self.inner_mult)],
            initializer = 'zeros',
            name = 'linear1_b',
            trainable = True       
        )
        self.linear2_m = self.add_weight(
            shape = [int(c*self.inner_mult), c],
            initializer = 'glorot_normal',
            name = 'linear2_m',
            trainable = True
        )
        self.linear2_b = self.add_weight(
            shape = [c],
            initializer = 'zeros',
            name = 'linear2_b',
            trainable = True
        )
        

    def call(self, x, training=False):
        # FFN(x) = max(0, xW1 + b1 )W2 + b2
        x = tf.einsum('btc,ci->bti', x, self.linear1_m) + self.linear1_b
        x = self.relu(x)
        x = tf.einsum('bti,io->bto', x, self.linear2_m) + self.linear2_b
        x = self.dropout(x, training=training)

        return x


class MuliHeadedMaskedSelfAttention(tf.keras.layers.Layer):
    '''from  "Attention is all you Need" (Vaswani et al., 2017)'''
    
    def __init__(self, n_heads, dropout=.2, **kwargs):
        '''
        Args:
            n_heads: integer
                number of heads
        '''
        super().__init__(**kwargs)

        # store instance params
        self.n_heads  = n_heads

        self.softmax    = tf.keras.layers.Softmax()
        self.dropout1   = Dropout(dropout)
        self.dropout2   = Dropout(dropout)
        

    def build(self, input_shape):
        b, t, c = input_shape

        # d_k = d_v = d_model / h - 3.2.2 page 5
        self.d_k      = c // self.n_heads
        
        self.key_mat = self.add_weight(
            shape = [c,  self.n_heads, self.d_k],
            initializer = 'glorot_normal',
            name = 'key_matrix',
            trainable = True
        )
        self.query_mat = self.add_weight(
            shape = [c,  self.n_heads, self.d_k],
            initializer = 'glorot_normal',
            name = 'query_matrix',
            trainable = True       
        )
        self.value_mat = self.add_weight(
            shape = [c,  self.n_heads, self.d_k],
            initializer = 'glorot_normal',
            name = 'value_matrix',
            trainable = True
        )
        self.linear_m1 = self.add_weight(
            shape = [int(self.n_heads * self.d_k), c],
            initializer = 'glorot_normal',
            name = 'linear_m1',
            trainable = True
        )

    def call(self, x, training=False):
        '''
        Args:
            x: tensor
                of shape [B, T, C]

        Returns:
            x: tensor
                of shape [B, T, C]
        '''
        
        # calculate k, q, v, for all batches and heads
        q = tf.einsum('btc,chd->bhtd', x, self.query_mat)
        k = tf.einsum('btc,chd->bhtd', x, self.key_mat)
        v = tf.einsum('btc,chd->bhtd', x, self.value_mat)

        # scaled dot prod step 1 [ q @ kT / d_k**.5 ] - 3.2.1 page 4
        w = tf.einsum('bhtd,bhzd->bhtz', q, k) * self.d_k**0.5

        # masking - 3.2.3 page 5
        mask = tf.linalg.band_part(tf.ones_like(w, dtype=tf.bool), -1, 0)        
        w = tf.where(mask, w, float('-inf'))

        # scaled dot prod step 2 [ softmax ]- 3.2.1 page 4
        w = self.softmax(w)

        # dropout
        w = self.dropout1(w, training=training)

        # scaled dot prod step 3 [ @ v ]- 3.2.1 page 4
        x = tf.einsum('bhtz,bhzd->bhtd', w, v)

        # concat from [B, H, T, d_k] to [B, T, out_chan] - 3.2.2 page 5
        x    = tf.transpose(x, perm=[0, 2, 1, 3])
        x_sh = tf.shape(x)
        x    = tf.reshape(x, [x_sh[0], x_sh[1], tf.reduce_prod(x_sh[2:])])

        # linear - 3.3 page 5
        x = tf.einsum('btc,cz->btz', x, self.linear_m1)

        # dropout
        x = self.dropout2(x, training=training)
        
        return x


class PositionalEncoding(tf.keras.layers.Layer):
    '''
    Adds positional encodings to embedding outputs    
    '''

    def __init__(self, max_pos=10000, **kwargs):
        '''
        Args:
            max_pos: integer
                maximum len of sequence that can be positionally encoded
        '''
        super().__init__(**kwargs)
        self.pos   = float(max_pos)
        self.t_pos = tf.range(float(max_pos), dtype=tf.float32)

    def call(self, x):
        x_shape = tf.shape(x)
        C   = tf.cast(x_shape[2], tf.float32)
        T   = x_shape[1]
        
        t_i = tf.range(C, dtype=tf.float32)
        val = self.t_pos[:T, None] / tf.pow(10000., (2. * t_i[None, :]) / C)
        sin_mat = tf.sin(val)
        cos_mat = tf.cos(val)

        mask = tf.tile((t_i % 2 == 0)[None, :], [T, 1])

        pos_enc = tf.where(mask, sin_mat, cos_mat)

        return x + pos_enc  

# +
# class PositionalEncoding(tf.keras.layers.Layer):
#     '''
#     Adds positional encodings to embedding outputs    
#     '''

#     def __init__(self, max_pos=10000, **kwargs):
#         '''
#         Args:
#             max_pos: integer
#                 maximum len of sequence that can be positionally encoded
#         '''
#         super().__init__(**kwargs)
#         self.max_pos = max_pos
#         self.t_pos = tf.range(float(max_pos), dtype=tf.float32)

#     def build(self, input_shape):
#         b, t, c = input_shape

#         t_i = tf.range(c, dtype=tf.float32)
#         val = self.t_pos[:, None] / tf.pow(10000., (2. * t_i[None, :]) / c)
#         sin_mat = tf.sin(val)
#         cos_mat = tf.cos(val)

#         mask = tf.tile((t_i % 2 == 0)[None, :], [self.max_pos, 1])
#         self.pos_enc = tf.where(mask, sin_mat, cos_mat)
                

#     def call(self, x):
#         '''
#         Args:
#             x: tensor
#                 raw sequence, shape B, T, C
#         Returns:
#             x: tensor
#                 positionally encoded sequence, shape B, T, C
#         '''
#         x_shape = tf.shape(x)
#         T   = x_shape[1]

#         return x + self.pos_enc[:T, :]
# -

class TransformerBlock(tf.keras.layers.Layer):
    '''from  "Attention is all you Need" (Vaswani et al., 2017)'''
    def __init__(self, n_heads, dropout=.2, **kwargs):
        '''
        Args:
            n_heads: int
                number of transformer heads
            dropout: float
                dropout_rate, padded to dropout layers
        '''
        super().__init__(**kwargs)
        self.att = MuliHeadedMaskedSelfAttention(n_heads, dropout=dropout, name='mha')
        self.ffd = FeedForward(dropout=dropout, name='ff')
        self.ln1 = LayerNormalization(name='ln1')
        self.ln2 = LayerNormalization(name='ln2')

    def build(self, input_shape):
        super().build(input_shape)
        
    def call(self, x, training=False):
        '''
        Args:
            x: tensor
                shape [B, T, C]
        Returns:
            x: tensor
                shape [B, T, C]
        '''
        x = x + self.att(self.ln1(x), training=training)
        x = x + self.ffd(self.ln2(x), training=training)
        return x


class GPTModel(tf.keras.models.Model):
    '''
    Generative Pretrained Transformer Architecture
    '''
    def __init__(self, n_emb, n_heads, n_blocks, n_vocab, dropout=.3, **kwargs):
        '''
        Args:
            n_emb: int
                dimension of word embedding
            n_head: int
                number of heads for each transformer block
            n_blocks: int
                number of transformer blocks
            n_vocab: int
                total vocab size
            dropout: float
                dropout_rate, passed to dropout layers
        '''
        super().__init__(**kwargs)

        # constants
        self.n_emb = n_emb
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.n_vocab = n_vocab
        self.dropout = dropout

        # layers
        self.emb = Embedding(n_vocab, n_emb, name='emb')
        self.pos = PositionalEncoding(name='pos_enc')

        self.transformers = [TransformerBlock(n_heads, dropout, name=f'T{i}') for i in range(n_blocks)]

        self.ln = LayerNormalization(name='ln_out')
        self.voc = Dense(n_vocab, name='voc')

    def build(self, input_shape):
        super().build(input_shape)

    # @tf.function(jit_compile=True) will be called in train_step
    def call(self, x, training=False):
        # embedding
        x = self.emb(x)

        # positional encoding
        x = self.pos(x)

        # transformer stack
        for T in self.transformers:
            x = T(x, training=training)

        # output
        x = self.ln(x)
        x = self.voc(x)

        return x

# Eager: 6.13ms  
# Graph: 336µs  
# Jit:   252µs  
