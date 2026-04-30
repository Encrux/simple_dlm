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
    p.add_argument("--data", default="data/input.txt")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--resume", default=None, help="path to checkpoint.pt to resume from")
    args = p.parse_args()

    os.makedirs("checkpoints", exist_ok=True)
    data = args.data
    device = torch.device(args.device)

    with open(data) as f:
        text = f.read()

    seq_len = args.seq_len
    iterations = 10000000
    batch_size = args.batch_size

    transformer = Transformer(data_path=data).to(device)
    if args.resume:
        transformer.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"resumed weights from {args.resume}")
    optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-4)

    corpus_np = transformer.encoder.encode_array(text)
    corpus = torch.from_numpy(corpus_np).to(device=device, dtype=torch.int32)
    corpus_len = corpus.shape[0]
    arange_seq = torch.arange(seq_len, device=device, dtype=torch.int64)

    def grab_batch() -> torch.Tensor:
        starts = torch.randint(0, corpus_len - seq_len, (batch_size,), device=device, dtype=torch.int64)
        return corpus[starts[:, None] + arange_seq[None, :]].long()

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