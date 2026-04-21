import torch
import random
from src.model import Transformer
from src.sample import sample
import time

data = "data/input.txt"
device = torch.device("mps")

with open(data) as f:
        lines = f.readlines()

text = "\n".join(lines)

seq_len = 128
iterations = 10000000
batch_size = 64

transformer = Transformer().to(device)
optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-4)



def grab_chunk() -> torch.Tensor:

    start = random.randint(0, len(text) - seq_len)
    chunk = text[start:start + seq_len]
    return torch.tensor(transformer.encoder.encode(chunk)).to(device)

def add_noise(input: list[int], t: float) -> str:
    mask = (torch.rand(batch_size, seq_len, device=device) < mask_prob).long()                                                                                                                                                            
    masked_input = input * (1 - mask)  
    return masked_input, mask     

start = time.time()
for i in range(0, iterations):
    batch = chunks = torch.stack([grab_chunk() for _ in range(batch_size)])

    mask_prob = random.uniform(0, 1)

    masked_input, mask = add_noise(batch, mask_prob)

    predictions = transformer.forward(masked_input, torch.tensor([mask_prob], device=device))

    loss = torch.nn.functional.cross_entropy(predictions[mask == 1], batch[mask == 1])

    if i % 1000 == 0:
        elapsed = time.time() - start
        print(f"step {i}, loss: {loss.item():.4f}, it/s: {i / elapsed:.1f}")
        sample(transformer, "To be, ", 64, device)
        torch.save(transformer.state_dict(), "checkpoint.pt")

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    