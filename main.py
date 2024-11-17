import torch
from model import BigramLanguageModel
from utils import encode, decode, get_batch
from train import train

# Hyperparameters
batch_size = 16
block_size = 32
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

# Load data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

data = torch.tensor(encode(text, stoi), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Initialize model
model = BigramLanguageModel(vocab_size, n_embd, n_head, n_layer, block_size, dropout).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Train the model
train(model, optimizer, train_data, val_data, max_iters, eval_interval, eval_iters, block_size, batch_size, device)

# Generate text
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=200, block_size=block_size)[0].tolist(), itos))
