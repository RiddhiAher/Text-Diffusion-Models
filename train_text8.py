"""Train the concise D3PM on the Text8 dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from d3pm.data import Text8Dataset, VOCAB_SIZE
from d3pm.diffusion import ConciseD3PM, DiffusionConfig
from d3pm.model import ModelConfig, TextDiffusionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    dataset = Text8Dataset(root=str(args.root), seq_len=args.seq_len, download=args.download)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model_cfg = ModelConfig(vocab_size=VOCAB_SIZE, seq_len=args.seq_len)
    model = TextDiffusionModel(model_cfg).to(args.device)

    diffusion = ConciseD3PM(DiffusionConfig(vocab_size=VOCAB_SIZE, timesteps=args.timesteps)).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    global_step = 0
    args.save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in progress:
            batch = batch.to(args.device)
            loss = diffusion.training_losses(model, batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            progress.set_postfix(loss=loss.item())
            global_step += 1

        checkpoint_path = args.save_dir / "latest.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": global_step,
                "config": {
                    "model": model_cfg.__dict__,
                    "timesteps": args.timesteps,
                    "seq_len": args.seq_len,
                },
            },
            checkpoint_path,
        )
        print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    train(parse_args())

