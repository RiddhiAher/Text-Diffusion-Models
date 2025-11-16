"""Generate samples from a trained checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from d3pm.data import Text8Dataset, VOCAB_SIZE
from d3pm.diffusion import ConciseD3PM, DiffusionConfig
from d3pm.model import ModelConfig, TextDiffusionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model_cfg = ModelConfig(vocab_size=VOCAB_SIZE, seq_len=args.seq_len)
    model = TextDiffusionModel(model_cfg)
    model.load_state_dict(ckpt["model"])
    model.to(args.device)
    model.eval()

    diffusion = ConciseD3PM(DiffusionConfig(vocab_size=VOCAB_SIZE, timesteps=args.timesteps)).to(args.device)

    samples = diffusion.sample(model, torch.Size((args.num_samples, args.seq_len))).cpu()
    decoded = Text8Dataset.decode(samples)
    print("\n".join(decoded))


if __name__ == "__main__":
    main()

