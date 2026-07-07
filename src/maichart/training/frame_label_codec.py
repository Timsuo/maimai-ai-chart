"""Conversion from frame-label JSON records to V2.5 training targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

DEFAULT_NOTE_TYPES = ("none", "tap", "break", "hold", "slide", "touch", "touch_hold")


@dataclass(slots=True)
class FrameLabelCodec:
    """Encode existing frame-label records into model target tensors."""

    note_types: tuple[str, ...] = DEFAULT_NOTE_TYPES

    @property
    def token_to_id(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.note_types)}

    @property
    def id_to_token(self) -> dict[int, str]:
        return {index: token for index, token in enumerate(self.note_types)}

    @property
    def num_note_types(self) -> int:
        return len(self.note_types)

    def encode_frames(self, frames: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not frames:
            raise ValueError("frame_labels.json contains no frames.")

        note_presence: list[list[float]] = []
        buttons: list[list[float]] = []
        note_type: list[int] = []
        density: list[list[float]] = []

        for frame in frames:
            labels = frame.get("labels")
            if not isinstance(labels, dict):
                raise ValueError("Each frame label must contain a 'labels' object.")
            note_presence.append([1.0 if labels.get("has_note") else 0.0])
            buttons.append(_encode_buttons(labels))
            note_type.append(self.token_to_id[_primary_note_type(labels)])
            density.append([float(labels.get("note_count") or 0.0)])

        return {
            "note_presence": torch.tensor(note_presence, dtype=torch.float32),
            "buttons": torch.tensor(buttons, dtype=torch.float32),
            "note_type": torch.tensor(note_type, dtype=torch.long),
            "density": torch.tensor(density, dtype=torch.float32),
        }


def _encode_buttons(labels: dict[str, Any]) -> list[float]:
    encoded = [0.0] * 8
    positions = labels.get("positions") or []
    for position in positions:
        index = _button_index(position)
        if index is not None:
            encoded[index] = 1.0
    return encoded


def _button_index(position: Any) -> int | None:
    text = str(position).strip()
    if text.isdigit():
        value = int(text)
        if 1 <= value <= 8:
            return value - 1
    return None


def _primary_note_type(labels: dict[str, Any]) -> str:
    if not labels.get("has_note"):
        return "none"
    if int(labels.get("break_count") or 0) > 0:
        return "break"
    if int(labels.get("touch_hold_start_count") or 0) > 0:
        return "touch_hold"
    if int(labels.get("slide_start_count") or 0) > 0:
        return "slide"
    if int(labels.get("hold_start_count") or 0) > 0:
        return "hold"
    if int(labels.get("touch_count") or 0) > 0:
        return "touch"
    if int(labels.get("tap_count") or 0) > 0:
        return "tap"
    if int(labels.get("slide_active_count") or 0) > 0:
        return "slide"
    if int(labels.get("hold_active_count") or 0) > 0:
        return "hold"
    note_types = labels.get("note_types") or []
    for note_type in note_types:
        if note_type in DEFAULT_NOTE_TYPES:
            return str(note_type)
    return "none"
