"""Concise discrete diffusion process for language modeling."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class DiffusionConfig:
    vocab_size: int
    timesteps: int = 1000
    beta_start: float = 1e-3
    beta_end: float = 5e-2


class ConciseD3PM(torch.nn.Module):
    """Implements a simple D3PM variant where corruption mixes with a uniform distribution."""

    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        betas = torch.linspace(config.beta_start, config.beta_end, config.timesteps)
        self.register_buffer("betas", betas)
        alphas = 1.0 - betas
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", torch.cumprod(alphas, dim=0))

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        return torch.randint(0, self.config.timesteps, (batch_size,), device=self.betas.device)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Sample `x_t` given the clean tokens `x_0` and timestep `t`."""

        noise = torch.randint(0, self.config.vocab_size, x_start.shape, device=x_start.device)
        alphas_cumprod = self.alpha_bars[t].view(-1, *([1] * (x_start.ndim - 1)))
        keep_mask = torch.bernoulli(alphas_cumprod.expand_as(x_start).float()).to(torch.bool)
        return torch.where(keep_mask, x_start, noise)

    def training_losses(self, model, x_start: torch.Tensor) -> torch.Tensor:
        batch_size = x_start.size(0)
        t = self.sample_timesteps(batch_size)
        x_t = self.q_sample(x_start, t)
        logits = model(x_t, t)
        loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), x_start.view(-1))
        return loss

    @torch.no_grad()
    def p_sample(self, model, x_t: torch.Tensor, t: int) -> torch.Tensor:
        logits = model(x_t, torch.full((x_t.size(0),), t, device=x_t.device, dtype=torch.long))
        probs = torch.softmax(logits, dim=-1)
        tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view_as(x_t)
        return tokens

    @torch.no_grad()
    def sample(self, model, shape: torch.Size) -> torch.Tensor:
        device = self.betas.device
        x_t = torch.randint(0, self.config.vocab_size, shape, device=device)
        for step in reversed(range(self.config.timesteps)):
            x_t = self.p_sample(model, x_t, step)
        return x_t

