"""V2.5 frame-level Transformer baseline."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from maichart.models.positional_encoding import (
    LearnedPositionalEncoding,
    SinusoidalPositionalEncoding,
)


@dataclass(slots=True)
class MaichartTransformerV25Config:
    input_dim: int
    num_note_types: int
    num_start_pattern_types: int = 11
    num_chord_size_classes: int = 3
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 4
    dropout: float = 0.1
    dim_feedforward: int = 1024
    positional_encoding: str = "sinusoidal"
    max_len: int = 10000
    density_nonnegative: bool = False


class MaichartTransformerV25(nn.Module):
    """Minimal multi-task frame predictor for V2.5 experiments."""

    def __init__(
        self,
        input_dim: int,
        num_note_types: int,
        *,
        num_start_pattern_types: int = 11,
        num_chord_size_classes: int = 3,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
        dim_feedforward: int = 1024,
        positional_encoding: str = "sinusoidal",
        max_len: int = 10000,
        density_nonnegative: bool = False,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if num_note_types <= 0:
            raise ValueError("num_note_types must be positive.")
        if num_start_pattern_types <= 0:
            raise ValueError("num_start_pattern_types must be positive.")
        if num_chord_size_classes <= 0:
            raise ValueError("num_chord_size_classes must be positive.")

        self.config = MaichartTransformerV25Config(
            input_dim=input_dim,
            num_note_types=num_note_types,
            num_start_pattern_types=num_start_pattern_types,
            num_chord_size_classes=num_chord_size_classes,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            positional_encoding=positional_encoding,
            max_len=max_len,
            density_nonnegative=density_nonnegative,
        )
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = _make_positional_encoding(
            positional_encoding,
            d_model=d_model,
            max_len=max_len,
            dropout=dropout,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)

        self.note_presence_head = nn.Linear(d_model, 1)
        self.button_head = nn.Linear(d_model, 8)
        self.type_head = nn.Linear(d_model, num_note_types)
        self.density_head = nn.Linear(d_model, 1)
        self.note_start_head = nn.Linear(d_model, 1)
        self.pattern_start_head = nn.Linear(d_model, num_start_pattern_types)
        self.chord_size_head = nn.Linear(d_model, num_chord_size_classes)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        *,
        difficulty_id: torch.Tensor | None = None,
        level: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run frame prediction.

        ``difficulty_id`` and ``level`` are reserved for future conditioning and
        are accepted so dataset/training APIs can grow without changing callers.
        """

        del difficulty_id, level
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape [B, T, F], got {tuple(x.shape)}.")
        if x.size(-1) != self.config.input_dim:
            raise ValueError(
                f"Expected input_dim={self.config.input_dim}, got {x.size(-1)}."
            )
        if padding_mask is not None:
            if padding_mask.shape != x.shape[:2]:
                raise ValueError(
                    "padding_mask must have shape [B, T], "
                    f"got {tuple(padding_mask.shape)} for x {tuple(x.shape)}."
                )
            padding_mask = padding_mask.to(device=x.device, dtype=torch.bool)

        hidden = self.input_projection(x)
        hidden = self.positional_encoding(hidden)
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        hidden = self.final_norm(hidden)

        density_pred = self.density_head(hidden)
        if self.config.density_nonnegative:
            density_pred = F.softplus(density_pred)

        return {
            "note_presence_logits": self.note_presence_head(hidden),
            "button_logits": self.button_head(hidden),
            "type_logits": self.type_head(hidden),
            "density_pred": density_pred,
            "note_start_logits": self.note_start_head(hidden),
            "pattern_start_logits": self.pattern_start_head(hidden),
            "chord_size_logits": self.chord_size_head(hidden),
        }


def _make_positional_encoding(
    kind: str,
    *,
    d_model: int,
    max_len: int,
    dropout: float,
) -> nn.Module:
    normalized = kind.strip().lower()
    if normalized in {"sin", "sinusoidal", "fixed"}:
        return SinusoidalPositionalEncoding(d_model, max_len=max_len, dropout=dropout)
    if normalized in {"learned", "embedding"}:
        return LearnedPositionalEncoding(d_model, max_len=max_len, dropout=dropout)
    raise ValueError(f"Unknown positional_encoding: {kind!r}.")
