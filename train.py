
import torch
import torch.nn as nn
from torch.optim import AdamW
from model import GPT
from dataset import TextDataset
from config import Config
import math

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

dataset = TextDataset(text, Config.block_size)
model = GPT(dataset.vocab_size).to(Config.device)

optimizer = AdamW(model.parameters(), lr=Config.learning_rate)

# Estimating Loss 

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(200)
        for k in range(200):
            xb, yb = dataset.get_batch(split, Config.batch_size)
            xb, yb = xb.to(Config.device), yb.to(Config.device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Training Loop

for iter in range(Config.max_iters):

    if iter % Config.eval_interval == 0:
        losses = estimate_loss()
        train_loss = losses['train']
        val_loss = losses['val']
        print(f"Step {iter}: "
              f"Train Loss {train_loss:.4f}    "
              f"Val Loss {val_loss:.4f} ")

    xb, yb = dataset.get_batch('train', Config.batch_size)
    xb, yb = xb.to(Config.device), yb.to(Config.device)

    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()

# Save Checkpoint

torch.save(model.state_dict(), "mini_gpt.pt")
print("Training complete. Model saved.")
