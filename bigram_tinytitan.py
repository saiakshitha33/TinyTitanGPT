import torch
import torch.nn as nn
import torch.nn.functional as F

# Load training data
with open("input.txt", "r") as f:
    text = f.read()

# Build character-level vocab
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Encode data
data = torch.tensor(encode(text), dtype=torch.long)
block_size = 8
x = data[:-1]
y = data[1:]

# Bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        if targets is None:
            return logits, None
        B, T = idx.shape
        logits = logits.view(B * T, vocab_size)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Train
model = BigramLanguageModel(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for step in range(100):
    xb = x[:block_size].unsqueeze(0)
    yb = y[:block_size].unsqueeze(0)
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 10 == 0:
        print(f"Step {step}: loss = {loss.item():.4f}")

# Generate text
print("\nTinyTitan Output:")
print(decode(model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))
