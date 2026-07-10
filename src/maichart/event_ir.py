"""Event-level chart intermediate representation.

This module defines the first EventIR shape used beside the legacy ChartIR.
It is intentionally additive: parsers, exporters, frame labels, and training
code continue to use ChartIR by default.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, TypeAlias

from maichart.ir import ChartMetadata, DifficultyMetadata, UnknownChartToken

EventType: TypeAlias = Literal[
    "tap",
    "hold",
    "slide",
    "touch",
    "touch_hold",
]


@dataclass(slots=True)
class TapEvent:
    """A button tap event."""

    tick: int
    position: str
    is_break: bool = False
    is_ex: bool = False
    raw_notation: str | None = None
    modifiers: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HoldEvent:
    """A button hold event."""

    head_tick: int
    position: str
    duration_ticks: int | None
    duration_raw: str | None = None
    duration_kind: str | None = None
    raw_notation: str | None = None
    modifiers: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SlideSegment:
    """One motion segment in a slide event."""

    start_position: str
    path_type: str
    path_args: list[str]
    end_position: str | None
    travel_duration_ticks: int | None
    duration_raw: str | None = None
    duration_kind: str | None = None
    raw_notation: str | None = None
    path_parts: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class SlideEvent:
    """A slide with a star-head timing point and motion payload."""

    head_tick: int
    start_position: str
    launch_offset_ticks: int | None
    travel_duration_ticks: int | None
    segments: list[SlideSegment]
    end_position: str | None
    raw_notation: str | None = None
    head_modifiers: dict[str, Any] = field(default_factory=dict)
    duration_raw: str | None = None
    duration_kind: str | None = None
    timing_pair_values: list[float] | None = None


@dataclass(slots=True)
class TouchEvent:
    """A touch tap event."""

    tick: int
    area: str
    position: str
    firework: bool = False
    raw_notation: str | None = None
    modifiers: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TouchHoldEvent:
    """A touch hold event."""

    head_tick: int
    area: str
    position: str
    duration_ticks: int | None
    duration_raw: str | None = None
    duration_kind: str | None = None
    raw_notation: str | None = None
    modifiers: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TimingEvent:
    """A tempo event at a chart tick."""

    tick: int
    bpm: float
    beat: float | None = None
    time_sec: float | None = None
    raw_notation: str | None = None


@dataclass(slots=True)
class MeterEvent:
    """A meter event reserved for future non-4/4 support."""

    tick: int
    numerator: int
    denominator: int
    ticks_per_measure: int | None = None
    beat: float | None = None
    time_sec: float | None = None
    raw_notation: str | None = None


ChartEvent: TypeAlias = TapEvent | HoldEvent | SlideEvent | TouchEvent | TouchHoldEvent


@dataclass(slots=True)
class ChartEventIR:
    """Top-level event-level representation for one chart difficulty."""

    schema_version: int = 1
    metadata: ChartMetadata = field(default_factory=ChartMetadata)
    difficulty: DifficultyMetadata = field(default_factory=DifficultyMetadata)
    timing_events: list[TimingEvent] = field(default_factory=list)
    meter_events: list[MeterEvent] = field(default_factory=list)
    events: list[ChartEvent] = field(default_factory=list)
    unknown_tokens: list[UnknownChartToken] = field(default_factory=list)
    raw: str | None = None


def event_ir_to_dict(chart: ChartEventIR) -> dict[str, Any]:
    """Convert EventIR to JSON-compatible primitives."""

    return {
        "schema_version": chart.schema_version,
        "metadata": asdict(chart.metadata),
        "difficulty": asdict(chart.difficulty),
        "timing_events": [asdict(event) for event in chart.timing_events],
        "meter_events": [asdict(event) for event in chart.meter_events],
        "events": [_event_to_dict(event) for event in chart.events],
        "unknown_tokens": [asdict(token) for token in chart.unknown_tokens],
        "raw": chart.raw,
    }


def event_ir_from_dict(payload: dict[str, Any]) -> ChartEventIR:
    """Load EventIR from JSON-compatible primitives."""

    return ChartEventIR(
        schema_version=int(payload.get("schema_version", 1)),
        metadata=ChartMetadata(**(payload.get("metadata") or {})),
        difficulty=DifficultyMetadata(**(payload.get("difficulty") or {})),
        timing_events=[
            TimingEvent(**event)
            for event in payload.get("timing_events", [])
        ],
        meter_events=[
            MeterEvent(**event)
            for event in payload.get("meter_events", [])
        ],
        events=[
            _event_from_dict(event)
            for event in payload.get("events", [])
        ],
        unknown_tokens=[
            UnknownChartToken(**token)
            for token in payload.get("unknown_tokens", [])
        ],
        raw=payload.get("raw"),
    )


def _event_to_dict(event: ChartEvent) -> dict[str, Any]:
    data = asdict(event)
    data["event_type"] = _event_type(event)
    return data


def _event_from_dict(data: dict[str, Any]) -> ChartEvent:
    event_type = data.get("event_type")
    payload = {key: value for key, value in data.items() if key != "event_type"}

    if event_type == "tap":
        return TapEvent(**payload)
    if event_type == "hold":
        return HoldEvent(**payload)
    if event_type == "slide":
        segments = [
            SlideSegment(**segment)
            for segment in payload.get("segments", [])
        ]
        return SlideEvent(**{**payload, "segments": segments})
    if event_type == "touch":
        return TouchEvent(**payload)
    if event_type == "touch_hold":
        return TouchHoldEvent(**payload)

    raise ValueError(f"Unsupported EventIR event_type: {event_type!r}")


def _event_type(event: ChartEvent) -> EventType:
    if isinstance(event, TapEvent):
        return "tap"
    if isinstance(event, HoldEvent):
        return "hold"
    if isinstance(event, SlideEvent):
        return "slide"
    if isinstance(event, TouchEvent):
        return "touch"
    if isinstance(event, TouchHoldEvent):
        return "touch_hold"
    raise TypeError(f"Unsupported EventIR event object: {type(event).__name__}")
