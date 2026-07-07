"""Timing-only tokenizer for raw Maidata ``inote_x`` blocks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from maichart.maidata import RawDifficultyBlock

TICKS_PER_BEAT = 1920


@dataclass(slots=True)
class TimingDirective:
    """A timing directive found at the start of a raw timing point."""

    directive_type: str
    raw: str
    value: float | int | str | None = None


@dataclass(slots=True)
class RawTimingPoint:
    """One comma-delimited timing point with opaque note text."""

    index: int
    raw: str
    note_text: str
    beat: Fraction
    tick: int
    time_sec: float | None
    division: int
    bpm: float
    directives: list[TimingDirective] = field(default_factory=list)


def tokenize_inote_timing(
    raw_inote: str,
    initial_bpm: float = 120.0,
    initial_division: int = 4,
) -> list[RawTimingPoint]:
    """Tokenize raw ``inote_x`` text into timing points.

    This recognizes only timing structure: comma-separated points, grid
    directives, BPM directives, and the ``E`` end marker. Note text remains
    opaque.
    """

    if initial_division <= 0:
        raise ValueError("initial_division must be a positive integer.")
    if initial_bpm <= 0:
        raise ValueError("initial_bpm must be positive.")

    points: list[RawTimingPoint] = []
    beat = Fraction(0, 1)
    time_sec = 0.0
    bpm = float(initial_bpm)
    division = int(initial_division)

    for index, raw_point in enumerate(_split_top_level_commas(raw_inote)):
        directives, note_text = _parse_point_directives(raw_point)
        point_bpm = bpm
        point_division = division

        for directive in directives:
            if directive.directive_type == "bpm":
                point_bpm = float(directive.value)
            elif directive.directive_type == "grid":
                point_division = int(directive.value)

        points.append(
            RawTimingPoint(
                index=index,
                raw=raw_point,
                note_text=note_text,
                beat=beat,
                tick=_beat_to_tick(beat),
                time_sec=time_sec,
                division=point_division,
                bpm=point_bpm,
                directives=directives,
            )
        )

        step_beats = Fraction(4, point_division)
        time_sec += float(step_beats) * 60.0 / point_bpm
        beat += step_beats
        bpm = point_bpm
        division = point_division

    return points


def tokenize_difficulty_timing(
    difficulty: RawDifficultyBlock,
    initial_bpm: float = 120.0,
    initial_division: int = 4,
) -> list[RawTimingPoint]:
    """Tokenize the raw ``inote`` field from a difficulty block."""

    if difficulty.inote is None:
        return []
    return tokenize_inote_timing(
        difficulty.inote,
        initial_bpm=initial_bpm,
        initial_division=initial_division,
    )


def timing_point_to_dict(point: RawTimingPoint) -> dict[str, Any]:
    """Convert a raw timing point to JSON-compatible primitives."""

    return {
        "index": point.index,
        "raw": point.raw,
        "note_text": point.note_text,
        "beat": _format_fraction(point.beat),
        "tick": point.tick,
        "time_sec": point.time_sec,
        "division": point.division,
        "bpm": point.bpm,
        "directives": [
            {
                "directive_type": directive.directive_type,
                "raw": directive.raw,
                "value": directive.value,
            }
            for directive in point.directives
        ],
    }


def timing_points_to_json(points: list[RawTimingPoint], *, indent: int = 2) -> str:
    """Serialize raw timing points to JSON."""

    return json.dumps(
        [timing_point_to_dict(point) for point in points],
        ensure_ascii=False,
        indent=indent,
    )


def _split_top_level_commas(raw_inote: str) -> list[str]:
    points: list[str] = []
    current: list[str] = []
    square_depth = 0

    for char in raw_inote:
        if char == "[":
            square_depth += 1
        elif char == "]" and square_depth > 0:
            square_depth -= 1

        if char == "," and square_depth == 0:
            points.append("".join(current))
            current = []
        else:
            current.append(char)

    points.append("".join(current))
    return points


def _parse_point_directives(raw_point: str) -> tuple[list[TimingDirective], str]:
    text = raw_point.strip()
    directives: list[TimingDirective] = []
    cursor = 0

    while cursor < len(text):
        cursor = _skip_whitespace(text, cursor)
        if cursor >= len(text):
            break

        char = text[cursor]
        if char == "{":
            end = text.find("}", cursor + 1)
            if end == -1:
                raise ValueError(f"Malformed grid directive in timing point: {raw_point!r}")
            raw = text[cursor : end + 1]
            value_text = text[cursor + 1 : end].strip()
            try:
                value = int(value_text)
            except ValueError as exc:
                raise ValueError(f"Invalid grid division {raw!r}.") from exc
            if value <= 0:
                raise ValueError(f"Grid division must be positive in {raw!r}.")
            directives.append(TimingDirective("grid", raw, value))
            cursor = end + 1
            continue

        if char == "(":
            end = text.find(")", cursor + 1)
            if end == -1:
                raise ValueError(f"Malformed BPM directive in timing point: {raw_point!r}")
            raw = text[cursor : end + 1]
            value_text = text[cursor + 1 : end].strip()
            try:
                value = float(value_text)
            except ValueError as exc:
                raise ValueError(f"Invalid BPM value {raw!r}.") from exc
            if value <= 0:
                raise ValueError(f"BPM must be positive in {raw!r}.")
            directives.append(TimingDirective("bpm", raw, value))
            cursor = end + 1
            continue

        if char == "E" and text[cursor + 1 :].strip() == "":
            directives.append(TimingDirective("end", "E", "E"))
            cursor = len(text)
            break

        break

    note_text = text[cursor:].strip()
    return directives, note_text


def _skip_whitespace(text: str, cursor: int) -> int:
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    return cursor


def _beat_to_tick(beat: Fraction) -> int:
    tick = beat * TICKS_PER_BEAT
    if tick.denominator != 1:
        raise ValueError(f"Beat {beat} does not map to an integer tick.")
    return tick.numerator


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"
