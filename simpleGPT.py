import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, Softmax, Embedding
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.metrics import Mean
from .custom_layers import GPTModel
from IPython.display import clear_output
import tiktoken
import matplotlib.pyplot as plt
import datetime
import os


class GPT:
    'Container for GPT, training, initalization, saving and loading'

    def __init__(self, n_emb, n_heads, n_blocks, log_dir=None, dropout=.3, block_size=8, batch_size=1, valid_split=.05, tpu=False, mixed_precision=None):
        if mixed_precision == 'f16':
            tf.keras.mixed_precision.set_global_policy('mixed_float16')        
        elif mixed_precision == 'b16':
            tf.keras.mixed_precision.set_global_policy('bfloat_float16')
         
        self.strategy = tf.distribute.get_strategy()
        
        if tpu:
            self.start_tpu()

        # CONSTANTS
        self.n_emb    = n_emb
        self.n_heads  = n_heads
        self.n_blocks = n_blocks
        
        self.block_size  = block_size
        self.batch_size  = batch_size
        self.valid_split = valid_split
        self.log_dir     = log_dir

        # TOKENIZER
        self.tokens = tiktoken.get_encoding('gpt2')
    
        with self.strategy.scope():
            self.model = GPTModel(n_emb, n_heads, n_blocks, self.tokens.n_vocab, dropout)
            self.loss  = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

            self.train_metric = tf.keras.metrics.Mean()
            self.valid_metric = tf.keras.metrics.Mean()

        # tensorboard
        if log_dir:
            current_day = datetime.datetime.now().strftime('%m%d')
            run = 1
            while os.path.exists(log_dir + '/' + current_day + f'_run_{run}'):
                run += 1
            
            train_log_dir = log_dir + '/' + current_day + f'_run_{run}' + '/train'
            valid_log_dir = log_dir + '/' + current_day + f'_run_{run}' + '/valid'
            self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
            self.valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)

        self.saved_params = False



    @tf.function(jit_compile=True)
    def train_step_fn(self, x, y_true):
        print('TRACING: train_step_fn')
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss   = self.loss(y_true, y_pred)
            loss   = tf.nn.compute_average_loss(loss, global_batch_size=self.batch_size*self.block_size)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(list(zip(grads, self.model.trainable_variables)))
        self.train_metric.update_state(loss * self.strategy.num_replicas_in_sync)
           


    @tf.function(jit_compile=True)
    def valid_step_fn(self, x, y_true):
        print('TRACING: valid_step_fn')
        y_pred = self.model(x, training=False)
        loss   = self.loss(y_true, y_pred)
        loss   = tf.nn.compute_average_loss(loss, global_batch_size=self.batch_size*self.block_size)

        self.valid_metric.update_state(loss * self.strategy.num_replicas_in_sync)

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

    def save_metrics_and_clear(self):
        # record train metrics
        step = self.optimizer.iterations.numpy()
        
        with self.train_summary_writer.as_default():
            tf.summary.scalar('loss', self.train_metric.result(), step=step)

        with self.valid_summary_writer.as_default():
            tf.summary.scalar('val loss', self.valid_metric.result(), step=step)

        self.train_metric.reset_state()
        self.valid_metric.reset_state()

    def save_params(self):
        params = {}
        params['n_emb']      = self.n_emb
        params['n_heads']    = self.n_heads
        params['n_blocks']   = self.n_blocks
        params['block_size'] = self.block_size
        params['batch_size'] = self.batch_size
        params['n_steps_fused'] = self.n_steps_fused
        params['optimizer']  = self.optimizer.get_config()

        params = str(params)
        
        with self.train_summary_writer.as_default():
            tf.summary.text('params', params, step=0)

        self.saved_params = True
            
            

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
                b = next(self.dataset['train'])
                self.strategy.run(self.train_step_fn, args=(b[0], b[1],))
    
            for _ in range(n_valid_steps):
                b = next(self.dataset['valid'])
                self.strategy.run(self.valid_step_fn, args=(b[0], b[1],))
            
            if self.log_dir:
                self.save_metrics_and_clear()

            if self.log_dir and not self.saved_params:
                self.save_params()


    def dataset_from_path(self, path):
        'path/to/corpus.txt -> tf.data.Dataset'

        # read in path
        with open(path, 'r') as file:
            corpus = file.read()

        tokens  = self.tokens.encode(corpus)
        split   = int(len(tokens)*self.valid_split)
        self.dataset = {}

        for name, (start, end) in zip(['train', 'valid'],[[None, -split], [-split, None]]):
            self.dataset[name] = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(tokens[start:end]))\
                                    .window(size=self.block_size+1, shift=1)\
                                    .flat_map(lambda x: x.batch(self.block_size+1, drop_remainder=True))\
                                    .map(lambda x: (x[:-1], x[1:]))\
                                    .repeat()\
                                    .shuffle(10000)\
                                    .batch(self.batch_size)\
                                    .prefetch(tf.data.AUTOTUNE)
            
            self.dataset[name] = self.strategy.experimental_distribute_dataset(self.dataset[name])
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
