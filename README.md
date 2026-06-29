# My Own Diffusion Language Model
Free-Range, Organic, Hand-Crafted. No AI used.

### Noteworthy Gibberish

step 67000, loss: 1.2239, it/s: 0.7:

```
To be, and be of men?



Prown AMEN:

O yout aboars of

Ra':

Un
```

step 77000, loss: 1.0891, it/s: 0.8:
```
To be, fo hend!



First her sense ountier to Jupits,

be horse.
```

Wiser words have never been spoken. Trained on an M2 Air 16GB for... a while, idk. Be horse.

# Setup

install dependencies via uv

```bash
uv sync
```

add training corpus (single .txt file) in /data and call it input.txt. For example, the [tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset:

```sh
curl -o data/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

# Running the code

Models are saved in checkpoints/checkpoint.pt by default. 

## Training

Old models are overwritten during training.

```bash
uv run train --device cuda (or mps/cpu)
````

## Sampling

```bash
uv run sample --query "To be, "
```

## Export to ONNX

```bash
uv run export-onnx --checkpoint checkpoints/checkpoint
```

