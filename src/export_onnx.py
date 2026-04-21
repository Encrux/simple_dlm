import argparse
import json
from pathlib import Path

import torch

from src.model import Transformer


def export(checkpoint: str, out_dir: str, seq_len: int):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = Transformer()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    tokens = torch.zeros(seq_len, dtype=torch.long)
    mask_prob = torch.tensor([0.5], dtype=torch.float32)

    torch.onnx.export(
        model,
        (tokens, mask_prob),
        out / "model.onnx",
        input_names=["tokens", "mask_prob"],
        output_names=["logits"],
        dynamic_axes={"tokens": {0: "seq"}, "logits": {0: "seq"}},
        opset_version=17,
    )

    vocab = list(model.encoder.vocab())
    (out / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False))

    print(f"wrote model.onnx and vocab.json to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoint.pt")
    p.add_argument("--out", default="web/public")
    p.add_argument("--seq-len", type=int, default=128)
    args = p.parse_args()
    export(args.checkpoint, args.out, args.seq_len)
