import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, Softmax, Embedding
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.metrics import Mean
from .custom_layers import build_gpt
from IPython.display import clear_output
import tiktoken
import matplotlib.pyplot as plt


class GPT:

    def __init__(self, n_emb, n_heads, n_blocks, dropout=.3, block_size=8, batch_size=1, valid_split=.05, tpu=False):
        
        # self.strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
        if tpu:
            self.start_tpu()

        # CONSTANTS
        self.block_size  = block_size
        self.batch_size  = batch_size
        self.valid_split = valid_split

        # LAYERS
        self.tokens    = tiktoken.get_encoding('gpt2')
    
        # with self.strategy.scope():
        self.model = build_gpt(n_emb, n_heads, n_blocks, self.tokens.n_vocab, dropout)

        # METRICS
        self.train_loss = Mean()
        self.valid_loss = Mean()
        
        self.history = {}
        self.history['train_loss'] = []
        self.history['valid_loss'] = []

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

        # def step_fn(batch):           
        with tf.GradientTape() as tape:
            logits = self.model(batch[0], training=True)
            loss   = sparse_categorical_crossentropy(batch[1], logits, from_logits=True)
            loss   = tf.reduce_mean(loss)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.train_loss.update_state(loss)

        # self.strategy.run(step_fn, args=(next(iterator),))
        
    @tf.function
    def valid_step(self, batch):

        # def step_fn(batch):
        logits = self.model(batch[0], training=False)
        loss   = sparse_categorical_crossentropy(batch[1], logits, from_logits=True)
        loss   = tf.reduce_mean(loss)
        self.valid_loss.update_state(loss)

        # self.strategy.run(step_fn, args=(next(iterator), ))

    def report_and_clear(self):
        self.history['train_loss'].append(self.train_loss.result().numpy())
        self.history['valid_loss'].append(self.valid_loss.result().numpy())

        print(f'{self.train_loss.result().numpy()}, {self.valid_loss.result().numpy()}')

        self.train_loss.reset_state()
        self.valid_loss.reset_state()

        clear_output(wait=True)
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['train_loss'])
        plt.plot(self.history['valid_loss'])
        plt.show()

        

    def fit(self, n_epochs, dataset_path=None, optimizer=None, optimizer_params={},
            n_train_steps=50, n_valid_steps=10):
        if optimizer:
            # with self.strategy.scope():
            self.optimizer = optimizer(**optimizer_params)
        
        if not hasattr(self, 'dataset'):
            print('building dataset')
            self.dataset_from_path(dataset_path)

        for _ in range(n_epochs):
            
            for _ in range(n_train_steps):
                self.train_step(next(self.dataset['train']))
    
            for _ in range(n_valid_steps):
                self.valid_step(next(self.dataset['valid']))
    
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
            self.dataset[name] = self.dataset[name].batch(self.batch_size) # * self.strategy.num_replicas_in_sync)
            self.dataset[name] = self.dataset[name].prefetch(tf.data.AUTOTUNE)
            # self.dataset[name] = self.strategy.experimental_distribute_dataset(self.dataset[name])
            self.dataset[name] = iter(self.dataset[name])

    def start_tpu(self):
        self.resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(self.resolver)
        # This is the TPU initialization code that has to be at the beginning.
        tf.tpu.experimental.initialize_tpu_system(self.resolver)
        print('TPUs available:')
        for device in tf.config.list_logical_devices('TPU'):
            print(device)

        self.strategy = tf.distribute.TPUStrategy(self.resolver)

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
