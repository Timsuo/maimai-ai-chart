"""Intermediate representation for Maidata/Simai-like charts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

NoteType: TypeAlias = Literal["tap", "hold", "slide", "touch", "touch_hold"]


@dataclass(slots=True)
class ChartMetadata:
    """Song-level metadata shared by one or more chart difficulties."""

    title: str | None = None
    artist: str | None = None
    audio_file: str | None = None
    background_file: str | None = None
    offset: float | None = None
    raw: str | None = None


@dataclass(slots=True)
class DifficultyMetadata:
    """Metadata for a single playable difficulty."""

    index: int | None = None
    name: str | None = None
    level: str | None = None
    designer: str | None = None
    raw: str | None = None


@dataclass(slots=True)
class TimedRaw:
    """Shared timing fields for events and notes."""

    time_sec: float | None = None
    beat: float | None = None
    tick: int | None = None
    raw: str | None = None


@dataclass(slots=True)
class BpmEvent(TimedRaw):
    """A tempo change at a chart position."""

    bpm: float | None = None


@dataclass(slots=True)
class GridEvent(TimedRaw):
    """A local grid/division change used while reading chart tokens."""

    division: int | None = None


@dataclass(slots=True)
class HSpeedEvent(TimedRaw):
    """A scroll-speed change at a chart position."""

    speed: float | None = None


@dataclass(slots=True)
class TimingData:
    """Timing lane for BPM, grid, and hspeed events."""

    bpms: list[BpmEvent] = field(default_factory=list)
    grids: list[GridEvent] = field(default_factory=list)
    hspeeds: list[HSpeedEvent] = field(default_factory=list)


@dataclass(slots=True)
class Note(TimedRaw):
    """A normalized note with enough optional fields for V1 chart parsing."""

    note_type: NoteType = "tap"
    position: str | None = None
    end_position: str | None = None
    duration_sec: float | None = None
    duration_beats: float | None = None
    duration_ticks: int | None = None
    duration: dict[str, Any] | None = None
    path: list[str] = field(default_factory=list)
    segments: list[dict[str, Any]] = field(default_factory=list)
    modifiers: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class UnknownChartToken(TimedRaw):
    """A raw token preserved because V1 parser does not understand it yet."""

    timing_index: int | None = None
    raw_timing_point: str | None = None
    raw_token: str | None = None
    reason: str | None = None


@dataclass(slots=True)
class ChartIR:
    """Top-level intermediate representation for one chart difficulty."""

    metadata: ChartMetadata = field(default_factory=ChartMetadata)
    difficulty: DifficultyMetadata = field(default_factory=DifficultyMetadata)
    timing: TimingData = field(default_factory=TimingData)
    notes: list[Note] = field(default_factory=list)
    unknown_tokens: list[UnknownChartToken] = field(default_factory=list)
    raw: str | None = None
    schema_version: int = 1
