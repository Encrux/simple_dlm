import torch
import random
from transformer.transformer import Transformer
import time

data = "data/input.txt"
device = torch.device("mps")

with open(data) as f:
        lines = f.readlines()

text = "\n".join(lines)

seq_len = 128
iterations = 10000000

transformer = Transformer().to(device)
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.001)



def grab_chunk() -> torch.Tensor:

    start = random.randint(0, len(text) - seq_len)
    chunk = text[start:start + seq_len]
    return torch.tensor(transformer.encoder.encode(chunk)).to(device)

def add_noise(input: list[int], t: float) -> str:
    mask = (torch.rand(seq_len) < mask_prob).long().to(device)                                                                                                                                                            
    masked_input = input * (1 - mask)  
    return masked_input, mask
     
start = time.time()
for i in range(0, iterations):
    chunk = grab_chunk()

    mask_prob = random.uniform(0, 1)

    masked_input, mask = add_noise(chunk, mask_prob)

    predictions = transformer.forward(masked_input, mask_prob)

    loss = torch.nn.functional.cross_entropy(predictions[mask == 1], chunk[mask == 1])

    if i % 1000 == 0:
        elapsed = time.time() - start
        print(f"step {i}, loss: {loss.item():.4f}, it/s: {i / elapsed:.1f}") 

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    