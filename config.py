
import torch

class Config:
    batch_size = 64
    block_size = 128
    max_iters = 4000
    eval_interval = 300
    learning_rate = 3e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_embd = 256
    n_head = 4
    n_layer = 4
    dropout = 0.2
