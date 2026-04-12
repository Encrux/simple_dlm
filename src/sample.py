import torch
import random


def sample(model, query, length, device):
    tokens = model.encoder.encode(query)
    x = torch.zeros(length, dtype=torch.long, device=device)        
    x[:len(tokens)] = torch.tensor(tokens, device=device)                                                                                                                           

    fixed = (x != 0)                                                                                                                                                                                                                
    with torch.no_grad():                                                                                                                                                                                         
        for step in range(20):

            predictions = model.forward(x, 1.0 - step / 20)                                                                                                                                                 
            probs = torch.softmax(predictions, dim=-1)
                                                                                                                                                                                                                
            mask_positions = (x == 0) & ~fixed                                                                                                                                                                            
            if not mask_positions.any():                                                                                                                                                                          
                break                                                                                                                                                                                             
                                                                                                                                                                                                                
            for pos in mask_positions.nonzero():                                                                                                                                                                  
                if random.random() < 1 / (20 - step):
                    x[pos] = torch.multinomial(probs[pos], 1)                                                                                                                                                     
                                            
        print(''.join(model.encoder.decode(x.tolist())))  
