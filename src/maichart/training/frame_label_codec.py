"""Conversion from frame-label JSON records to V2.5 training targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

DEFAULT_NOTE_TYPES = ("none", "tap", "break", "hold", "slide", "touch", "touch_hold")
FRAME_PATTERN_TYPES = (
    "none",
    "active_hold",
    "active_slide",
    "single_tap",
    "double_tap",
    "multi_tap",
    "break",
    "single_hold",
    "hold_chord",
    "single_slide",
    "slide_chord",
    "tap_slide_mix",
    "touch",
    "touch_hold",
    "other_mix",
)
START_PATTERN_TYPES = (
    "single_tap",
    "double_tap",
    "multi_tap",
    "break",
    "single_hold",
    "hold_chord",
    "single_slide",
    "slide_chord",
    "tap_slide_mix",
    "touch",
    "touch_hold",
)
START_PATTERN_IGNORE_INDEX = -100
START_PATTERN_TO_ID = {token: index for index, token in enumerate(START_PATTERN_TYPES)}
CHORD_SIZE_START_CLASSES = ("single", "double", "three_plus")


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

    @property
    def start_pattern_vocab(self) -> tuple[str, ...]:
        return START_PATTERN_TYPES

    @property
    def num_start_pattern_types(self) -> int:
        return len(START_PATTERN_TYPES)

    @property
    def num_chord_size_start_classes(self) -> int:
        return len(CHORD_SIZE_START_CLASSES)

    def encode_frames(self, frames: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not frames:
            raise ValueError("frame_labels.json contains no frames.")

        note_presence: list[list[float]] = []
        buttons: list[list[float]] = []
        note_type: list[int] = []
        density: list[list[float]] = []
        note_start: list[list[float]] = []
        pattern_start: list[int] = []
        chord_size_start: list[int] = []

        for frame in frames:
            labels = frame.get("labels")
            if not isinstance(labels, dict):
                raise ValueError("Each frame label must contain a 'labels' object.")
            derived = derive_frame_pattern(labels)
            note_start_presence = bool(derived["note_start_presence"])
            note_presence.append([1.0 if labels.get("has_note") else 0.0])
            buttons.append(_encode_buttons(labels))
            note_type.append(self.token_to_id[_primary_note_type(labels)])
            density.append([float(labels.get("note_count") or 0.0)])
            note_start.append([1.0 if note_start_presence else 0.0])
            pattern_start.append(_start_pattern_id(str(derived["pattern_type"])))
            chord_size_start.append(_chord_size_start_class(int(derived["chord_size"])))

        return {
            "note_presence": torch.tensor(note_presence, dtype=torch.float32),
            "buttons": torch.tensor(buttons, dtype=torch.float32),
            "note_type": torch.tensor(note_type, dtype=torch.long),
            "density": torch.tensor(density, dtype=torch.float32),
            "note_start": torch.tensor(note_start, dtype=torch.float32),
            "pattern_start": torch.tensor(pattern_start, dtype=torch.long),
            "chord_size_start": torch.tensor(chord_size_start, dtype=torch.long),
        }


def derive_frame_pattern(labels: dict[str, Any]) -> dict[str, bool | int | str]:
    """Derive coarse frame-level pattern labels from one frame ``labels`` object."""

    note_count = _int_label(labels, "note_count")
    tap_count = _int_label(labels, "tap_count")
    break_count = _int_label(labels, "break_count")
    hold_start_count = _int_label(labels, "hold_start_count")
    hold_active_count = _int_label(labels, "hold_active_count")
    slide_start_count = _int_label(labels, "slide_start_count")
    slide_active_count = _int_label(labels, "slide_active_count")
    touch_count = _int_label(labels, "touch_count")
    touch_hold_start_count = _int_label(labels, "touch_hold_start_count")

    if note_count <= 0:
        if slide_active_count > 0:
            pattern_type = "active_slide"
        elif hold_active_count > 0:
            pattern_type = "active_hold"
        else:
            pattern_type = "none"
    elif touch_hold_start_count > 0:
        pattern_type = "touch_hold"
    elif touch_count > 0:
        pattern_type = "touch"
    elif tap_count > 0 and slide_start_count > 0:
        pattern_type = "tap_slide_mix"
    elif break_count > 0:
        pattern_type = "break"
    elif slide_start_count > 0:
        pattern_type = "single_slide" if slide_start_count == 1 else "slide_chord"
    elif hold_start_count > 0:
        pattern_type = "single_hold" if hold_start_count == 1 else "hold_chord"
    elif tap_count > 0:
        if note_count == 1:
            pattern_type = "single_tap"
        elif note_count == 2:
            pattern_type = "double_tap"
        else:
            pattern_type = "multi_tap"
    else:
        pattern_type = "other_mix"

    return {
        "activity_presence": bool(labels.get("has_note")),
        "note_start_presence": note_count > 0,
        "chord_size": max(0, note_count),
        "chord_size_class": _chord_size_class(note_count),
        "pattern_type": pattern_type,
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


def _int_label(labels: dict[str, Any], key: str) -> int:
    try:
        return int(labels.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def _chord_size_class(note_count: int) -> int:
    if note_count <= 0:
        return 0
    if note_count == 1:
        return 1
    if note_count == 2:
        return 2
    return 3


def _start_pattern_id(pattern_type: str) -> int:
    return START_PATTERN_TO_ID.get(pattern_type, START_PATTERN_IGNORE_INDEX)


def _chord_size_start_class(note_count: int) -> int:
    if note_count <= 0:
        return START_PATTERN_IGNORE_INDEX
    if note_count == 1:
        return 0
    if note_count == 2:
        return 1
    return 2


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
