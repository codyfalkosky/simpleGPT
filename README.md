<div align="center">
    <img src="./images/pytorch.png" width="75px" style="vertical-align: middle; padding-right: 10px"></img>
    <img src="./images/GPT.png" width="150px" style="vertical-align: middle; padding-right: 10px"></img>
    <!-- <img src="./images/plus.png" width="50px" style="vertical-align: middle;"></img> -->
</div>

# "Attention Is All You Need" PyTorch Implementation
***

[**Overview**](#1)
| [**Download**](#2)
| [**Potter**](#3)
| [**Generate**](#4)
| [**Train Your Own Model**](#5)
<div id='1'></div>

## Overview
***
This is a from-scratch implementation of the "Attention Is All You Need" paper in PyTorch.  If you want to see a simple implementation of a GPT you are in the right place!  The entire pre-training, generating, tokenizing, training a tokenizer, loading, and saving has been built into a class for convenience.  This class is simpleGPT.  Additionally, this package contains a pre-trained model on Harry Potter books as proof of concept!  

This is an academic exercise that is essentially an overfitting task. The right way to train an LLM is to use trillions of tokens and millions of GPU hours and then fine-tune it for the specific task.

https://github.com/user-attachments/assets/cb7ed472-8754-4b92-a8be-13718f16e9e9

<div align="center">
    <!-- <video src='https://storage.googleapis.com/codyfalkosky_public/weights/potter_gen.mp4' width=800 autoplay loop title='actual model output from this model!'> </video> -->
</div>

<br>
<div id='2'></div>

## Download
***
To download, clone my repo!

```bash
git clone https://github.com/codyfalkosky/simpleGPT.git
```

<div id='3'></div>

## Potter
***
Try out the Harry Potter GPT.

```python
from simpleGPT import GPT

gpt = GPT('potter') 

```

<div id='4'></div>

## Generate
***
Supports batch generation automatically!  Since this model is pre-trained on a corpus, there's no "stopping token".  It will generate infinitely until you run out of RAM. Set `top_p=0` for greedy decoding, or choose a float up to 1 for more decoding possibilities!  Raise the temperature for more randomness in your generations.

```python
gpt.generate(text_lists=['And then Harry said', 'Ron quickly took out his wand', 'Hermione knew'],
             n_gen=10,      # generate 10 new tokens per batch
             top_p=0.0,     # for greedy, increase to 1.0 for more possibilities!
             temperature=1  # for no effect, increase to promote rarer word choice
)

```

<div id='5'></div>
<br><br>

<div id='4'></div>

## Train Your Model
***
Training your model is super easy and simpleGPT has everything you need to try out and learn the GPT pre-training process.  All the steps are laid out in this notebook!

### Step 1: Train a Tokenizer

simpleGPT uses sentencepiece to train a tokenizer
1. data must be in 1 large txt file
2. each sentence on its own line

like this: [tokenizer training text example](https://raw.githubusercontent.com/codyfalkosky/simpleGPT/main/data/corpus_line.txt)

```python
# init a gpt instance
gpt = GPT()

# set params and train
gpt.train_tokenizer(corpus='path/to/corpus.txt',
                    output_path='path/to/file/tokenizer',
                    vocab_size=10_000)


# tokenizer saved to `output_path`
```
### Step 2: Train a Model

Model data is in a different format than tokenizer training data.

Model data format:
1. data must be in 1 large txt file (like tokenizer training)
2. but normal formatting, not a new sentence every line (unlike tokenizer training)

like this: [model training text example](https://raw.githubusercontent.com/codyfalkosky/simpleGPT/main/data/corpus.txt)

```python
gpt = GPT(device='cuda',                              # cuda, mps or cpu
          tokenizer='path/to/file/tokenizer.model',   # dont forget .model!
          n_vocab=5_000,                             # must match vocab_size from tokenizer
          chan_dim=1024,                               # internal model channel depth
          n_heads=4,                                  # number of heads per transformer block
          inner_mult=4,                               # upscaling of internal layer in FeedForward
          Nx=8,                                       # number of transformer layers
          max_context=10_000,                         # dont touch (unless your context is > 10,000
          dropout=.3,                                 # dropout to use for training
          context_window=256,)                        # length of model context window

gpt.train(corpus_path='/content/simpleGPT/data/corpus.txt',  # the corpus
          epochs=100,                                          # total training epochs
          batch_size=64,                                    # batch size
          grad_acc_steps=1,                                  # number of gradient accumulation steps
          lr=1e-5,                                           # learning rate
          num_workers=12,                                    # for dataloading, 12 is good for A100
          pin_memory=True,                                   # recommended True for GPU 
          break_at=None,                                     # for testing params! stop at break_at steps
          time_limit_hours=15)                               # optional for colab timeouts

gpt.save_train('/path/to/savefile')  # saves model & training history & params

```


