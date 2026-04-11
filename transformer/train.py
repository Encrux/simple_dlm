from torch import nn, rand, optim, Tensor, tensor
import random
from transformer.transformer import Transformer



data = "data/input.txt"

with open(data) as f:
        lines = f.readlines()

text = "\n".join(lines)

seq_len = 128
iterations = 10000000

transformer = Transformer()
optimizer =optim.Adam(transformer.parameters(), lr=0.001)



def grab_chunk() -> Tensor:

    start = random.randint(0, len(text) - seq_len)
    chunk = text[start:start + seq_len]
    return tensor(transformer.encoder.encode(chunk))

def add_noise(input: list[int], t: float) -> str:
    mask = (rand(seq_len) < mask_prob).long()                                                                                                                                                               
    masked_input = input * (1 - mask)  
    return masked_input, mask
     

for i in range(0, iterations):
    chunk = grab_chunk()

    mask_prob = random.random(0, 1)

    masked_input, mask = add_noise(chunk, mask_prob)

    predictions = transformer.forward(masked_input, mask_prob)

    loss = nn.functional.cross_entropy(predictions[mask == 1], chunk[mask == 1])

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    