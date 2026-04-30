import argparse
import random

import torch

from src.model import Transformer


def sample(model, query, length, device):
    tokens = model.encoder.encode(query)
    x = torch.zeros(length, dtype=torch.long, device=device)        
    x[:len(tokens)] = torch.tensor(tokens, device=device)                                                                                                                           

    fixed = (x != 0)                                                                                                                                                                                                                
    with torch.no_grad():                                                                                                                                                                                         
        for step in range(20):

            mask_prob = torch.tensor([1.0 - step / 20], device=device)
            predictions = model.forward(x, mask_prob)
            probs = torch.softmax(predictions, dim=-1)
                                                                                                                                                                                                                
            mask_positions = (x == 0) & ~fixed                                                                                                                                                                            
            if not mask_positions.any():                                                                                                                                                                          
                break                                                                                                                                                                                             
                                                                                                                                                                                                                
            for pos in mask_positions.nonzero():                                                                                                                                                                  
                if random.random() < 1 / (20 - step):
                    x[pos] = torch.multinomial(probs[pos], 1)                                                                                                                                                     
                                            
        print(''.join(model.encoder.decode(x.tolist())))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/checkpoint.pt")
    p.add_argument("--query", default="To be, ")
    p.add_argument("--length", type=int, default=64)
    p.add_argument("--device", default="mps")
    p.add_argument("--data", default="data/input.txt")
    args = p.parse_args()

    device = torch.device(args.device)
    model = Transformer(data_path=args.data).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    sample(model, args.query, args.length, device)


if __name__ == "__main__":
    main()
