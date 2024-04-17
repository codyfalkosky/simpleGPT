import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, Softmax, Embedding
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Mean
from .custom_layers import PositionalEncoding, TransformerBlock
from IPython.display import clear_output
import tiktoken
import matplotlib.pyplot as plt


class GPT(tf.keras.Model):

    def __init__(self, n_emb, n_heads, n_blocks, dropout=.3, block_size=8, batch_size=1, valid_split=.05):
        super().__init__()

        # CONSTANTS
        self.block_size  = block_size
        self.batch_size  = batch_size
        self.valid_split = valid_split

        # LAYERS
        self.tokens    = tiktoken.get_encoding('gpt2')
        self.embedding = Embedding(self.tokens.n_vocab, n_emb)
        self.pos_emb   = PositionalEncoding()
        self.loss_fn   = SparseCategoricalCrossentropy(from_logits=True)
        self.transf    = [TransformerBlock(n_emb, n_heads, n_emb, dropout) for _ in range(n_blocks)]
        self.layern    = LayerNormalization()
        self.to_voc    = Dense(self.tokens.n_vocab)
        self.softmax   = Softmax()

        # METRICS
        self.train_loss = Mean()
        self.valid_loss = Mean()
        self.history = {}
        self.history['train_loss'] = []
        self.history['valid_loss'] = []

    @tf.function
    def call(self, x, targets=None, training=False):
        x = self.embedding(x) # [B, T, vocab_size]
        x = self.pos_emb(x)

        for T in self.transf:
            x = T(x, training=training)

        x       = self.layern(x)
        logits  = self.to_voc(x)
        
        if targets is None:
            loss   = None
        else:
            loss   = self.loss_fn(targets, logits)

        return logits, loss

    def generate(self, idx, max_new_tokens=10):

        for _ in range(max_new_tokens):
            # predictions
            logits, loss = self(idx)
    
            # get the new word predictions
            logits = logits[:, -1, :] # [B, C]
    
            # choose new word
            new_word = tf.random.categorical(logits, 1, dtype=tf.int32) # [B, 1]
    
            # append to sentence
            idx = tf.concat([idx, new_word], axis=-1) # [B, T+1]
    
        return idx

    @tf.function
    def train_step(self, batch):
        with tf.GradientTape() as tape:
            logits, loss = self(batch[0], targets=batch[1], training=True)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.train_loss.update_state(loss)
    @tf.function
    def valid_step(self, batch):
        logits, loss = self(batch[0], targets=batch[1], training=False)
        self.valid_loss.update_state(loss)

    def report_and_clear(self):
        self.history['train_loss'].append(self.train_loss.result().numpy())
        self.history['valid_loss'].append(self.valid_loss.result().numpy())

        self.train_loss.reset_state()
        self.valid_loss.reset_state()

        clear_output(wait=True)
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['train_loss'])
        plt.plot(self.history['valid_loss'])
        plt.show()

        

    def fit(self, n_epochs, dataset_path=None, optimizer=None, n_train_steps=50, n_valid_steps=10):
        if optimizer:
            self.optimizer = optimizer
        
        if not hasattr(self, 'dataset'):
            print('building dataset')
            self.dataset_from_path(dataset_path)

        for _ in range(n_epochs):
            
            for _ in range(n_train_steps):
                batch = next(self.dataset['train'])
                self.train_step(batch)
    
            for _ in range(n_valid_steps):
                batch = next(self.dataset['valid'])
                self.valid_step(batch)
    
            self.report_and_clear()


    def dataset_from_path(self, path):
        'path/to/corpus.txt -> tf.data.Dataset'

        # read in path
        with open(path, 'r') as file:
            corpus = file.read()

        tokens  = self.tokens.encode(corpus)
        split   = int(len(tokens)*self.valid_split)
        self.dataset = {}

        for name, (start, end) in zip(['train', 'valid'],[[None, -split], [-split, None]]):
            self.dataset[name] = tf.data.Dataset.from_tensor_slices(tokens[start:end])
            self.dataset[name] = self.dataset[name].window(size=self.block_size+1, shift=1)
            self.dataset[name] = self.dataset[name].flat_map(lambda x: x.batch(self.block_size+1))
            self.dataset[name] = self.dataset[name].map(lambda x: (x[:-1], x[1:]))
            self.dataset[name] = self.dataset[name].cache()
            self.dataset[name] = self.dataset[name].repeat()
            self.dataset[name] = self.dataset[name].shuffle(10000)
            self.dataset[name] = self.dataset[name].batch(self.batch_size)
            self.dataset[name] = iter(self.dataset[name])

if __name__ == '__main__':
    potterGPT = GPT(8, 2, 2)

    fit_params = dict(
        n_epochs=10, 
        dataset_path='/Users/codyfalkosky/Documents/hidden_desktop/potterGPT/data/corpus.txt', 
        optimizer=tf.keras.optimizers.legacy.Adam(), 
        n_train_steps=100,
        n_valid_steps=10
    )
    
    potterGPT.fit(**fit_params)
