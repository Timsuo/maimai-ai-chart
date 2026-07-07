"""Positional encodings for frame-level chart models."""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """Classic fixed sinusoidal positional encoding for [B, T, D] tensors."""

    def __init__(self, d_model: int, max_len: int = 10000, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive.")

        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_terms = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(positions * div_terms)
        pe[:, 1::2] = torch.cos(positions * div_terms[: pe[:, 1::2].shape[1]])

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape [B, T, D], got {tuple(x.shape)}.")
        if x.size(1) > self.pe.size(1):
            raise ValueError(
                f"Sequence length {x.size(1)} exceeds max_len {self.pe.size(1)}."
            )
        return self.dropout(x + self.pe[:, : x.size(1), :].to(dtype=x.dtype))


class LearnedPositionalEncoding(nn.Module):
    """Learned positional embedding for [B, T, D] tensors."""

    def __init__(self, d_model: int, max_len: int = 10000, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        self.embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape [B, T, D], got {tuple(x.shape)}.")
        positions = torch.arange(x.size(1), device=x.device)
        position_features = self.embedding(positions).unsqueeze(0).to(dtype=x.dtype)
        return self.dropout(x + position_features)
