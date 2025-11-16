"""Utilities for working with the Text8 dataset."""
from __future__ import annotations

import argparse
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import torch
from torch.utils.data import Dataset
import requests

TEXT8_URL = "http://mattmahoney.net/dc/text8.zip"
ALPHABET = " abcdefghijklmnopqrstuvwxyz"  # Text8 already lowercases and strips punctuation.
VOCAB_SIZE = len(ALPHABET)
TOKEN_TO_ID = {ch: idx for idx, ch in enumerate(ALPHABET)}
ID_TO_TOKEN = {idx: ch for ch, idx in TOKEN_TO_ID.items()}


def ensure_text8(root: Path) -> Path:
    """Ensure the raw `text8` file exists inside *root*.

    The function looks for `text8` or `text8.zip`.  When `text8.zip` is found it will be
    extracted in-place.  If nothing exists we raise an informative error suggesting to
    run the module with `--download`.
    """

    root.mkdir(parents=True, exist_ok=True)
    raw_file = root / "text8"
    archive = root / "text8.zip"

    if raw_file.exists():
        return raw_file

    if archive.exists():
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(root)
        if raw_file.exists():
            return raw_file

    raise FileNotFoundError(
        "Could not locate Text8. Run `python -m d3pm.data --download --root <dir>` first."
    )


def download_text8(root: Path) -> Path:
    """Download and unzip the Text8 dataset."""

    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / "text8.zip"
    raw_path = root / "text8"

    if raw_path.exists():
        return raw_path

    print(f"Downloading Text8 (~100MB) to {archive_path} ...")
    response = requests.get(TEXT8_URL, stream=True, timeout=60)
    response.raise_for_status()
    with open(archive_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1 << 20):
            if chunk:
                f.write(chunk)

    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(root)

    return raw_path


def chunk_tokens(sequence: Iterable[int], seq_len: int) -> List[torch.Tensor]:
    """Split a flat sequence of token ids into contiguous chunks."""

    data = torch.tensor(list(sequence), dtype=torch.long)
    usable_len = (len(data) // seq_len) * seq_len
    data = data[:usable_len]
    return data.view(-1, seq_len).split(1)


@dataclass
class Text8Dataset(Dataset):
    """Simple contiguous chunks of Text8 tokens."""

    root: str
    seq_len: int = 256
    download: bool = False

    def __post_init__(self) -> None:
        root = Path(self.root)
        if self.download:
            path = download_text8(root)
        else:
            path = ensure_text8(root)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        tokens = [TOKEN_TO_ID.get(ch, 0) for ch in text]
        self.data = torch.tensor(tokens, dtype=torch.long)
        usable_len = (len(self.data) // self.seq_len) * self.seq_len
        self.data = self.data[:usable_len].view(-1, self.seq_len)

    def __len__(self) -> int:
        return self.data.size(0)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]

    @staticmethod
    def decode(batch: torch.Tensor) -> List[str]:
        """Turn a batch of token ids back into raw text."""

        if batch.ndim == 1:
            batch = batch.unsqueeze(0)
        sentences = []
        for row in batch:
            sentences.append("".join(ID_TO_TOKEN[int(token)] for token in row))
        return sentences


def main() -> None:
    parser = argparse.ArgumentParser(description="Text8 utilities")
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    if args.download:
        path = download_text8(args.root)
    else:
        path = ensure_text8(args.root)

    ds = Text8Dataset(root=str(args.root), seq_len=args.seq_len)
    print(f"Stored {len(ds)} sequences of length {args.seq_len} at {path}")


if __name__ == "__main__":
    main()

