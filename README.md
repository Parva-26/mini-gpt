
# Mini GPT – Transformer Language Model From Scratch

This project is my implementation of a decoder-only Transformer language model built completely from scratch in PyTorch.

It was inspired by Andrej Karpathy’s “GPT from scratch” series, but instead of following the notebook format, I structured the implementation as a modular Python project to better understand how large language models are engineered in practice.

The goal of this project was not just to copy the architecture, but to deeply understand:

- Multi-head self-attention
- Causal masking
- Residual connections
- Layer normalization
- Feed-forward transformer blocks
- Autoregressive text generation
- Training loop structure and gradient stability


## Model Architecture

- Decoder-only Transformer
- 4–6 Transformer blocks
- Multi-head masked self-attention
- Feed-forward networks (4x expansion)
- Token + positional embeddings
- Final linear projection head

The model is trained at character-level on a Shakespeare-style dataset.


## Training Details

- Optimizer: AdamW
- Gradient clipping applied
- Cross entropy loss
- Trained on Google Colab T4 GPU
- Block size: 128
- Embedding size: 256–384 (configurable)

This implementation focuses on clarity and correctness rather than scaling to billions of parameters.


## Example Generation

Prompt: To be, or not to be

Generated sample:


To be, or not to be as the days:
That I will thee thy stoops of the name,
And I'll be a sweet to speeding.

CAMILLO:
I am hear your boy: I would not might
I am prove and that we there impeace. For the feet
And be with him to be the sound and this drunker'd
The thing Rospers, how now confess: come to him!
I think said my lenity, so will they was my royal
Who are with the disproachious wife in my lord,
Who are his new of fine? What caste heard thine.

QUEEN ELIZABETH:
Let's friends mister to the render persuade of a daughter
As the need; and all be one your grace.

DUCHESS OF YORK:
Your come to your Lucio
I comes to the great of the state,
Are souls given for the comes and a presences
You may be a cup, and common from your services,
No: my lord, and and the hate been in my life.

MENENIUS:
The armour eyes her with our father's all the most
words of the and particule, powers
Out appoting stranged too make the orders,
And rich have crave with my love is news.

Second Servant:
Peace to do them her beat wear t


(Output varies depending on training time.)



## How to Run

### Train
```bash
python train.py
```
### Generate Text
```bash
python generate.py
```

## What I Learned

This project helped me understand:

- Why attention scores are scaled by sqrt(d_k)
- How masking prevents future token leakage
- How residual connections stabilize training
- How transformer blocks stack hierarchically
- The difference between architectural understanding and large-scale training


## Future Improvements

- Switch to BPE tokenizer
- Implement learning rate scheduler
- Add checkpointing system
- Implement GPT-2 scale architecture
- Experiment with fine-tuning


This is part of my journey toward understanding large-scale language models at a systems level.
