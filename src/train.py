import argparse
import os
import torch
import random
from src.model import Transformer
from src.sample import sample
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="mps")
    args = p.parse_args()

    os.makedirs("checkpoints", exist_ok=True)
    data = "data/input.txt"
    device = torch.device(args.device)

    with open(data) as f:
            lines = f.readlines()

    text = "\n".join(lines)

    seq_len = 128
    iterations = 10000000
    batch_size = 64

    transformer = Transformer().to(device)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-4)

    corpus = torch.tensor(transformer.encoder.encode(text), dtype=torch.long, device=device)
    corpus_len = corpus.shape[0]
    arange_seq = torch.arange(seq_len, device=device)

    def grab_batch() -> torch.Tensor:
        starts = torch.randint(0, corpus_len - seq_len, (batch_size,), device=device)
        return corpus[starts[:, None] + arange_seq[None, :]]

    def add_noise(input: list[int], t: float) -> str:
        mask = (torch.rand(batch_size, seq_len, device=device) < mask_prob).long()
        masked_input = input * (1 - mask)
        return masked_input, mask

    start = time.time()
    for i in range(0, iterations):
        batch = grab_batch()

        mask_prob = random.uniform(0, 1)

        masked_input, mask = add_noise(batch, mask_prob)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            predictions = transformer.forward(masked_input, torch.tensor([mask_prob], device=device))
            per_token = torch.nn.functional.cross_entropy(
                predictions.reshape(-1, predictions.shape[-1]),
                batch.reshape(-1),
                reduction="none",
            ).reshape(batch_size, seq_len)
            loss = (per_token * mask).sum() / mask.sum().clamp(min=1)

        if i % 1000 == 0:
            elapsed = time.time() - start
            print(f"step {i}, loss: {loss.item():.4f}, it/s: {i / elapsed:.1f}")
            sample(transformer, "To be, ", 64, device)
            torch.save(transformer.state_dict(), "checkpoints/checkpoint.pt")

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        

if __name__ == "__main__": 
    main()