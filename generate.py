
import torch
import torch.nn.functional as F
from model import GPT
from dataset import TextDataset
from config import Config


with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

dataset = TextDataset(text, Config.block_size)

# Trained model

model = GPT(dataset.vocab_size).to(Config.device)
model.load_state_dict(torch.load("mini_gpt.pt", map_location=Config.device))
model.eval()

# Sampling 

def generate_text(prompt="", max_new_tokens=1000, temperature=1.0, top_k=None):
    if prompt == "":
        context = torch.zeros((1, 1), dtype=torch.long, device=Config.device)
    else:
        encoded = torch.tensor([dataset.encode(prompt)], dtype=torch.long).to(Config.device)
        context = encoded

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = context[:, -Config.block_size:]
            logits = model(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                values, _ = torch.topk(logits, top_k)
                logits[logits < values[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, next_token), dim=1)

    return dataset.decode(context[0].tolist())


if __name__ == "__main__":
    output = generate_text(
        prompt="To be, or not to be",
        max_new_tokens=1000,
        temperature=0.8,
        top_k=35
    )
    print(output)
