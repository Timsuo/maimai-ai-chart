"""Batch collation for variable-length V2.5 samples."""

from __future__ import annotations

from typing import Any

import torch


def collate_v25(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("collate_v25 requires at least one sample.")

    batch_size = len(samples)
    lengths = [int(sample["x"].size(0)) for sample in samples]
    max_len = max(lengths)
    input_dim = int(samples[0]["x"].size(1))

    x = samples[0]["x"].new_zeros((batch_size, max_len, input_dim))
    padding_mask = torch.ones((batch_size, max_len), dtype=torch.bool)
    loss_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    y = {
        "note_presence": samples[0]["y"]["note_presence"].new_zeros((batch_size, max_len, 1)),
        "buttons": samples[0]["y"]["buttons"].new_zeros((batch_size, max_len, 8)),
        "note_type": samples[0]["y"]["note_type"].new_zeros((batch_size, max_len)),
        "density": samples[0]["y"]["density"].new_zeros((batch_size, max_len, 1)),
    }

    for index, sample in enumerate(samples):
        length = lengths[index]
        x[index, :length] = sample["x"]
        padding_mask[index, :length] = False
        loss_mask[index, :length] = True
        for key in y:
            y[key][index, :length] = sample["y"][key]

    return {
        "x": x,
        "padding_mask": padding_mask,
        "loss_mask": loss_mask,
        "y": y,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "meta": [sample.get("meta", {}) for sample in samples],
    }
