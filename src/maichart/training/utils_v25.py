"""Shared helpers for V2.5 train/evaluate commands."""

from __future__ import annotations

from collections import Counter
from typing import Any

import torch

from maichart.training.dataset_v25 import MaichartV25Dataset, TrainingDataError


def resolve_device(name: str) -> torch.device:
    normalized = name.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise TrainingDataError("--device cuda was requested, but CUDA is not available.")
        return torch.device("cuda")
    raise TrainingDataError("--device must be one of: auto, cpu, cuda.")


def select_sample_index(
    dataset: MaichartV25Dataset,
    *,
    sample_index: int | None = None,
    song_id: str | None = None,
    difficulty_index: int | None = None,
) -> int:
    if sample_index is not None:
        if sample_index < 0 or sample_index >= len(dataset):
            raise TrainingDataError(
                f"--sample-index {sample_index} is out of range for {len(dataset)} samples."
            )
        return sample_index

    if song_id is None and difficulty_index is None:
        return 0
    if song_id is None or difficulty_index is None:
        raise TrainingDataError("--song-id and --difficulty-index must be provided together.")

    for index, ref in enumerate(dataset.samples):
        if ref.song_id == song_id and ref.difficulty_index == difficulty_index:
            return index
    raise TrainingDataError(
        f"No usable sample found for song_id={song_id!r}, "
        f"difficulty_index={difficulty_index}."
    )


def assert_sample_is_finite(sample: dict[str, Any]) -> None:
    tensors = {"x": sample["x"], **{f"y.{key}": value for key, value in sample["y"].items()}}
    bad = [
        name
        for name, tensor in tensors.items()
        if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all()
    ]
    if bad:
        joined = ", ".join(bad)
        raise TrainingDataError(f"Sample contains NaN or Inf values in: {joined}.")


def sample_sanity_stats(sample: dict[str, Any], note_type_vocab: tuple[str, ...]) -> dict[str, Any]:
    x = sample["x"]
    y = sample["y"]
    note_presence = y["note_presence"]
    buttons = y["buttons"]
    note_type = y["note_type"]
    density = y["density"]
    counts = Counter(int(value) for value in note_type.tolist())
    return {
        "song_id": sample["meta"].get("song_id"),
        "difficulty_index": sample["meta"].get("difficulty_index"),
        "level": sample["meta"].get("level"),
        "x_shape": list(x.shape),
        "note_positive_ratio": float(note_presence.mean().item()),
        "button_positive_ratio": float(buttons.mean().item()),
        "note_type_distribution": {
            note_type_vocab[index] if 0 <= index < len(note_type_vocab) else str(index): count
            for index, count in sorted(counts.items())
        },
        "density_min": float(density.min().item()),
        "density_max": float(density.max().item()),
        "density_mean": float(density.mean().item()),
        "has_nan_or_inf": bool(
            any(
                torch.is_floating_point(tensor) and not torch.isfinite(tensor).all()
                for tensor in [x, note_presence, buttons, density]
            )
        ),
    }


def print_sample_sanity(sample: dict[str, Any], note_type_vocab: tuple[str, ...]) -> None:
    stats = sample_sanity_stats(sample, note_type_vocab)
    print("Training sample sanity:", flush=True)
    print(f"  song_id={stats['song_id']}", flush=True)
    print(f"  difficulty_index={stats['difficulty_index']}", flush=True)
    print(f"  level={stats['level']}", flush=True)
    print(f"  x_shape={stats['x_shape']}", flush=True)
    print(f"  note_positive_ratio={stats['note_positive_ratio']:.6f}", flush=True)
    print(f"  button_positive_ratio={stats['button_positive_ratio']:.6f}", flush=True)
    print(f"  note_type_distribution={stats['note_type_distribution']}", flush=True)
    print(
        "  density="
        f"min:{stats['density_min']:.6f} "
        f"max:{stats['density_max']:.6f} "
        f"mean:{stats['density_mean']:.6f}",
        flush=True,
    )
    print(f"  has_nan_or_inf={stats['has_nan_or_inf']}", flush=True)
