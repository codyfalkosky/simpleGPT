# ***
# This is a ground up build of a GPT style model for academic purposes.
# ***

import tensorflow as tf
from tensorflow.keras.layers import Dense, Softmax, Dropout, LayerNormalization, Input, Embedding
from tensorflow.keras.initializers import GlorotNormal


class FeedForward(tf.keras.layers.Layer):
    '''from  "Attention is all you Need" (Vaswani et al., 2017)'''

    def __init__(self, in_chan, out_chan, inner_mult=4, dropout=.5):
        super().__init__()
        self.in_chan    = in_chan
        self.out_chan   = out_chan
        self.inner_mult = inner_mult
        
        self.relu       = tf.keras.layers.ReLU()
        self.dropout    = Dropout(dropout)

    def build(self, input_shape):
        self.linear1_m = self.add_weight(
            shape = [self.in_chan, int(self.out_chan*self.inner_mult)],
            initializer = 'glorot_normal',
            name = 'linear1_m',
            trainable = True
        )
        self.linear1_b = self.add_weight(
            shape = [int(self.out_chan*self.inner_mult)],
            initializer = 'zeros',
            name = 'linear1_b',
            trainable = True       
        )
        self.linear2_m = self.add_weight(
            shape = [int(self.out_chan*self.inner_mult), self.out_chan],
            initializer = 'glorot_normal',
            name = 'linear2_m',
            trainable = True
        )
        self.linear2_b = self.add_weight(
            shape = [self.out_chan],
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
    
    def __init__(self, in_chan, n_heads, out_chan, dropout):
        '''
        Args:
            in_chan: integer
                channel depth of inputs
            n_heads: integer
                number of heads
            out_chan: integer
                channel depth of output

        Call arguments:
            x: tensor
                of shape [B, T, in_chan]

        Returns:
            output_activated: tensor
                of shape [B, T, out_chan]
        '''
        super().__init__()

        # store instance params
        self.in_chan  = in_chan
        self.n_heads  = n_heads
        self.out_chan = out_chan

        # d_k = d_v = d_model / h - 3.2.2 page 5
        self.d_k      = out_chan // n_heads

        self.softmax    = tf.keras.layers.Softmax()
        self.dropout1   = Dropout(dropout)
        
        # self.linear_m1  = tf.Variable(GlorotNormal()([int(self.n_heads * self.d_k), self.out_chan]),name='linear_m1')
        self.dropout2   = Dropout(dropout)

    def build(self, input_shape):
        self.key_mat = self.add_weight(
            shape = [self.in_chan,  self.n_heads, self.d_k],
            initializer = 'glorot_normal',
            name = 'key_matrix',
            trainable = True
        )
        self.query_mat = self.add_weight(
            shape = [self.in_chan,  self.n_heads, self.d_k],
            initializer = 'glorot_normal',
            name = 'query_matrix',
            trainable = True       
        )
        self.value_mat = self.add_weight(
            shape = [self.in_chan,  self.n_heads, self.d_k],
            initializer = 'glorot_normal',
            name = 'linear2_m',
            trainable = True
        )
        self.linear_m1 = self.add_weight(
            shape = [int(self.n_heads * self.d_k), self.out_chan],
            initializer = 'glorot_normal',
            name = 'linear_m1',
            trainable = True
        )

    def call(self, x, training=False):
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

    Args:
        max_pos: integer
            maximum 
    
    '''

    def __init__(self, max_pos=10000):
        super().__init__()
        self.pos   = float(max_pos)
        self.t_pos = tf.range(float(max_pos), dtype=tf.float32)

    def call(self, x):
        C   = tf.cast(tf.shape(x)[2], tf.float32)
        T   = tf.shape(x)[1]
        
        t_i = tf.range(C, dtype=tf.float32)
        val = self.t_pos[:T, None] / tf.pow(10000., (2. * t_i[None, :]) / C)
        sin_mat = tf.sin(val)
        cos_mat = tf.cos(val)

        mask = tf.tile((t_i % 2 == 0)[None, :], [T, 1])

        pos_enc = tf.where(mask, sin_mat, cos_mat)

        return x + pos_enc  


# class TransformerBlock(tf.keras.layers.Layer):
#     '''from  "Attention is all you Need" (Vaswani et al., 2017)'''

#     def __init__(self, in_chan, n_heads, out_chan, dropout):
#         super().__init__()

#         self.att = MuliHeadedMaskedSelfAttention(in_chan, n_heads, out_chan, dropout)
#         self.ffd = FeedForward(out_chan, out_chan, dropout)
#         self.ln1 = LayerNormalization()
#         self.ln2 = LayerNormalization()

#     def build(self, input_shape):
#         # self.att.build(input_shape)
#         # self.ffd.build(input_shape)
#         # self.self.ln1.build(input_shape)
#         # self.self.ln2.build(input_shape)
#         pass


#     def call(self, x, training=False):
#         x = x + self.att(self.ln1(x), training=training)
#         x = x + self.ffd(self.ln1(x), training=training)
#         return x


def build_gpt(n_emb, n_heads, n_blocks, n_vocab, dropout=.3):

    x_in = Input([None,])

    x = Embedding(n_vocab, n_emb)(x_in)
    x = PositionalEncoding()(x)

    for _ in range(n_blocks):
        lnx = LayerNormalization()(x)
        x   = x + MuliHeadedMaskedSelfAttention(n_emb, n_heads, n_emb, dropout)(lnx)

        lnx = LayerNormalization()(x)
        x   = x + FeedForward(n_emb, n_emb, dropout)(lnx)

    x = LayerNormalization()(x)
    x = Dense(n_vocab)(x)

    return tf.keras.Model(inputs=x_in, outputs=x)
    
