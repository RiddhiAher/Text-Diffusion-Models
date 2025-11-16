"""Model components for the concise D3PM baseline."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int, max_steps: int = 10_000):
        super().__init__()
        self.dim = dim
        self.max_steps = max_steps

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.dim // 2
        freqs = torch.exp(
            torch.arange(half_dim, device=device, dtype=torch.float32)
            * -(math.log(10_000.0) / (half_dim - 1))
        )
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.size(-1) < self.dim:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


@dataclass
class ModelConfig:
    vocab_size: int
    seq_len: int
    dim: int = 256
    depth: int = 4
    heads: int = 4
    dropout: float = 0.1


class TextDiffusionModel(nn.Module):
    """A lightweight Transformer encoder that predicts the clean token logits."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.dim)
        self.pos_emb = nn.Parameter(torch.randn(1, config.seq_len, config.dim) * 0.01)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.dim,
            nhead=config.heads,
            dim_feedforward=config.dim * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.depth)
        self.time_emb = SinusoidalTimeEmbedding(config.dim)
        self.to_logits = nn.Sequential(
            nn.LayerNorm(config.dim),
            nn.Linear(config.dim, config.vocab_size),
        )

    def forward(self, tokens: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = tokens.shape
        assert seq_len == self.config.seq_len, "Sequence length mismatch"
        tok = self.token_emb(tokens)
        time = self.time_emb(timesteps).unsqueeze(1)
        hidden = tok + time + self.pos_emb[:, :seq_len]
        encoded = self.encoder(hidden)
        logits = self.to_logits(encoded)
        return logits

