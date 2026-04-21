# My own Diffusion Language Model
Free-Range, Organic, Hand-Crafted

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

