from torch import nn, Tensor, softmax, sqrt, randn, layer_norm, tensor
from transformer.encoder import Encoder

class Transformer(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.encoder = Encoder("data/input.txt")

        vocab_size = len(self.encoder.vocab())
        max_seq_len = 1024
        embed_dim = 256
        hidden_dim = 4 * embed_dim
        num_layers = 10

        self.token_emb = nn.Parameter(randn(vocab_size, embed_dim))                                                                                                                                             
        self.pos_emb = nn.Parameter(randn(max_seq_len, embed_dim))
        self.mask_prob_emb = self.mask_prob_emb = nn.Linear(1, embed_dim)

        self.layers = nn.ModuleList([                                                                                                                                                                                 
            nn.ModuleList([
                nn.LayerNorm(embed_dim),        
                nn.Linear(embed_dim, embed_dim), #W_Q
                nn.Linear(embed_dim, embed_dim), #W_K
                nn.Linear(embed_dim, embed_dim), #W_V

                #FFN
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(), 
                nn.Linear(hidden_dim, embed_dim)])
            for _ in range(num_layers)                                                                                                                                                                                
        ])    

        self.output = nn.Linear(embed_dim, vocab_size)

    def attention(self, Q: Tensor, K: Tensor, V: Tensor):
        d_k = K.shape[-1]
        return softmax(Q @ K.T / (d_k ** 0.5), dim=-1) @ V 
    
    def self_attention(self, X: Tensor):
        return self.attention(X, X, X)
    
    def embedding(self, x: Tensor, mask_prob: float):
        t_emb = self.mask_prob_emb(tensor([mask_prob], device=x.device))
        return self.token_emb[x] + self.pos_emb[:len(x)] + t_emb
    
    def forward(self, x: Tensor, mask_prob: float) -> Tensor:
        x = self.embedding(x, mask_prob)
        for ln1, W_Q, W_K, W_V, ln2, linear1, relu, linear2 in self.layers:
            x = x + self.attention(W_Q(ln1(x)), W_K(ln1(x)), W_V(ln1(x)))
            x = x + linear2(relu(linear1(ln2(x))))
            
        return self.output(x)