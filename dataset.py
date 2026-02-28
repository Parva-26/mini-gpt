
import torch

class TextDataset:
    def __init__(self, text, block_size):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch:i for i,ch in enumerate(chars)}
        self.itos = {i:ch for i,ch in enumerate(chars)}
        self.encode = lambda s: [self.stoi[c] for c in s]
        self.decode = lambda l: ''.join([self.itos[i] for i in l])

        data = torch.tensor(self.encode(text), dtype=torch.long)
        n = int(0.9*len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]
        self.block_size = block_size

    def get_batch(self, split, batch_size):
        data = self.train_data if split=='train' else self.val_data
        ix = torch.randint(len(data)-self.block_size, (batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        return x, y
