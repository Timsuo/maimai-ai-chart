"""Unified duration parsing for Maidata-like note syntax."""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from maichart.timing import TICKS_PER_BEAT

GRID_DURATION_RE = re.compile(
    r"^(?P<division>[1-9][0-9]*):(?P<count>[+-]?[0-9]+)$"
)
SECONDS_DURATION_RE = re.compile(r"^#(?P<seconds>[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))$")
TIMING_PAIR_RE = re.compile(
    r"^(?P<first>[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))"
    r"##"
    r"(?P<second>[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))$"
)


@dataclass(slots=True)
class DurationExpr:
    """A raw-preserving parsed duration expression."""

    raw: str
    kind: str
    beats: Fraction | None = None
    seconds: float | None = None
    ticks: int | None = None
    values: list[float] | None = None


def parse_duration_expr(raw: str, bpm: float | None = None) -> DurationExpr:
    """Parse a Maidata duration block such as ``[8:1]`` or ``[#0.8057]``."""

    text = raw.strip()
    if not (text.startswith("[") and text.endswith("]")):
        return DurationExpr(raw=raw, kind="unknown")

    body = text[1:-1].strip()

    grid_match = GRID_DURATION_RE.fullmatch(body)
    if grid_match is not None:
        beats = Fraction(
            4 * int(grid_match.group("count")),
            int(grid_match.group("division")),
        )
        return DurationExpr(
            raw=raw,
            kind="grid_fraction",
            beats=beats,
            seconds=_beats_to_seconds(beats, bpm),
            ticks=duration_to_ticks_or_none(beats),
        )

    seconds_match = SECONDS_DURATION_RE.fullmatch(body)
    if seconds_match is not None:
        seconds = float(seconds_match.group("seconds"))
        beats = _seconds_to_beats(seconds, bpm)
        return DurationExpr(
            raw=raw,
            kind="seconds",
            beats=beats,
            seconds=seconds,
            ticks=duration_to_ticks_or_none(beats) if beats is not None else None,
        )

    pair_match = TIMING_PAIR_RE.fullmatch(body)
    if pair_match is not None:
        values = [float(pair_match.group("first")), float(pair_match.group("second"))]
        # V1 preserves both timing-pair values and uses the second value as the
        # conservative playable duration in seconds until the full semantics are known.
        seconds = values[1]
        beats = _seconds_to_beats(seconds, bpm)
        return DurationExpr(
            raw=raw,
            kind="timing_pair",
            beats=beats,
            seconds=seconds,
            ticks=duration_to_ticks_or_none(beats) if beats is not None else None,
            values=values,
        )

    return DurationExpr(raw=raw, kind="unknown")


def resolve_duration_expr(expr: DurationExpr, bpm: float | None) -> DurationExpr:
    """Return a copy of a duration expression with beat/second fields filled."""

    beats = expr.beats
    seconds = expr.seconds

    if beats is None and seconds is not None:
        beats = _seconds_to_beats(seconds, bpm)
    if seconds is None and beats is not None:
        seconds = _beats_to_seconds(beats, bpm)

    return DurationExpr(
        raw=expr.raw,
        kind=expr.kind,
        beats=beats,
        seconds=seconds,
        ticks=duration_to_ticks_or_none(beats) if beats is not None else None,
        values=list(expr.values) if expr.values is not None else None,
    )


def duration_expr_to_dict(expr: DurationExpr | None) -> dict[str, Any] | None:
    """Convert a duration expression to JSON-friendly primitives."""

    if expr is None:
        return None
    return {
        "raw": expr.raw,
        "kind": expr.kind,
        "beats": _format_fraction(expr.beats) if expr.beats is not None else None,
        "seconds": expr.seconds,
        "ticks": expr.ticks,
        "values": expr.values,
    }


def duration_expr_from_dict(data: dict[str, Any] | None) -> DurationExpr | None:
    """Load a duration expression from JSON-friendly primitives."""

    if data is None:
        return None
    beats_data = data.get("beats")
    beats = Fraction(beats_data) if beats_data is not None else None
    values_data = data.get("values")
    return DurationExpr(
        raw=str(data.get("raw", "")),
        kind=str(data.get("kind", "unknown")),
        beats=beats,
        seconds=data.get("seconds"),
        ticks=data.get("ticks"),
        values=list(values_data) if values_data is not None else None,
    )


def duration_to_ticks_or_none(duration_beats: Fraction) -> int | None:
    """Convert exact beats to ticks when the result is integral."""

    ticks = duration_beats * TICKS_PER_BEAT
    if ticks.denominator != 1:
        return None
    return ticks.numerator


def _seconds_to_beats(seconds: float, bpm: float | None) -> Fraction | None:
    if bpm is None or bpm <= 0:
        return None
    return Fraction(str(seconds)) * Fraction(str(bpm)) / 60


def _beats_to_seconds(beats: Fraction, bpm: float | None) -> float | None:
    if bpm is None or bpm <= 0:
        return None
    return float(beats) * 60.0 / bpm


def _format_fraction(value: Fraction | None) -> str | None:
    if value is None:
        return None
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"
