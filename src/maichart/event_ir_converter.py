"""Converters from legacy ChartIR to EventIR."""

from __future__ import annotations

from typing import Any

from maichart.event_ir import (
    ChartEvent,
    ChartEventIR,
    HoldEvent,
    SlideEvent,
    SlideSegment,
    TapEvent,
    TimingEvent,
    TouchEvent,
    TouchHoldEvent,
)
from maichart.ir import ChartIR, Note


def chart_ir_to_event_ir(chart: ChartIR) -> ChartEventIR:
    """Derive a first-pass EventIR view from the legacy ChartIR."""

    return ChartEventIR(
        schema_version=1,
        metadata=chart.metadata,
        difficulty=chart.difficulty,
        timing_events=[
            TimingEvent(
                tick=int(event.tick),
                bpm=float(event.bpm),
                beat=event.beat,
                time_sec=event.time_sec,
                raw_notation=event.raw,
            )
            for event in chart.timing.bpms
            if event.tick is not None and event.bpm is not None
        ],
        meter_events=[],
        events=[
            event
            for note in chart.notes
            for event in [_note_to_event(note)]
            if event is not None
        ],
        unknown_tokens=list(chart.unknown_tokens),
        raw=chart.raw,
    )


def _note_to_event(note: Note) -> ChartEvent | None:
    if note.tick is None or note.position is None:
        return None

    tick = int(note.tick)
    position = str(note.position)
    modifiers = _copy_modifiers(note.modifiers)

    if note.note_type == "tap":
        return TapEvent(
            tick=tick,
            position=position,
            is_break=bool(modifiers.get("break")),
            is_ex=bool(modifiers.get("ex")),
            raw_notation=note.raw,
            modifiers=modifiers,
        )

    if note.note_type == "hold":
        return HoldEvent(
            head_tick=tick,
            position=position,
            duration_ticks=note.duration_ticks,
            duration_raw=_duration_raw(note.duration),
            duration_kind=_duration_kind(note.duration),
            raw_notation=note.raw,
            modifiers=modifiers,
        )

    if note.note_type == "slide":
        segments = _slide_segments(note)
        end_position = _final_slide_end_position(segments, note)
        return SlideEvent(
            head_tick=tick,
            start_position=position,
            launch_offset_ticks=None,
            travel_duration_ticks=note.duration_ticks,
            segments=segments,
            end_position=end_position,
            raw_notation=note.raw,
            head_modifiers=_slide_head_modifiers(modifiers),
            duration_raw=_duration_raw(note.duration),
            duration_kind=_duration_kind(note.duration),
            timing_pair_values=_duration_values(note.duration),
        )

    if note.note_type == "touch":
        return TouchEvent(
            tick=tick,
            area=_touch_area(note, position),
            position=position,
            firework=bool(modifiers.get("firework")),
            raw_notation=note.raw,
            modifiers=modifiers,
        )

    if note.note_type == "touch_hold":
        return TouchHoldEvent(
            head_tick=tick,
            area=_touch_area(note, position),
            position=position,
            duration_ticks=note.duration_ticks,
            duration_raw=_duration_raw(note.duration),
            duration_kind=_duration_kind(note.duration),
            raw_notation=note.raw,
            modifiers=modifiers,
        )

    return None


def _slide_segments(note: Note) -> list[SlideSegment]:
    segments: list[SlideSegment] = []
    inferred_start = str(note.position) if note.position is not None else ""

    for raw_segment in note.segments:
        segment = dict(raw_segment)
        trajectory = segment.get("trajectory")
        if isinstance(trajectory, dict):
            trajectory_data = trajectory
        else:
            trajectory_data = {}

        start_position = _optional_str(
            segment.get("start_position")
            or trajectory_data.get("start_position")
            or inferred_start
        )
        path_type = str(
            segment.get("pattern")
            or trajectory_data.get("pattern")
            or "unknown"
        )
        path_args = _path_args(segment, trajectory_data)
        end_position = _optional_str(
            segment.get("end_position")
            or trajectory_data.get("end_position")
            or _last_or_none(path_args)
        )
        duration = segment.get("duration") if isinstance(segment.get("duration"), dict) else None
        travel_duration_ticks = _int_or_none(
            segment.get("duration_ticks")
            if segment.get("duration_ticks") is not None
            else (duration or {}).get("ticks")
        )

        segments.append(
            SlideSegment(
                start_position=start_position or "",
                path_type=path_type,
                path_args=path_args,
                end_position=end_position,
                travel_duration_ticks=travel_duration_ticks,
                duration_raw=_duration_raw(duration),
                duration_kind=_duration_kind(duration),
                raw_notation=_optional_str(segment.get("raw")),
                path_parts=_copy_path_parts(segment.get("path_parts")),
            )
        )

        if end_position is not None:
            inferred_start = end_position

    return segments


def _final_slide_end_position(segments: list[SlideSegment], note: Note) -> str | None:
    for segment in reversed(segments):
        if segment.end_position is not None:
            return segment.end_position
    return str(note.end_position) if note.end_position is not None else None


def _duration_raw(duration: dict[str, Any] | None) -> str | None:
    if isinstance(duration, dict) and duration.get("raw") is not None:
        return str(duration["raw"])
    return None


def _duration_kind(duration: dict[str, Any] | None) -> str | None:
    if isinstance(duration, dict) and duration.get("kind") is not None:
        return str(duration["kind"])
    return None


def _duration_values(duration: dict[str, Any] | None) -> list[float] | None:
    if not isinstance(duration, dict):
        return None
    values = duration.get("values")
    if values is None:
        return None
    return [float(value) for value in values]


def _path_args(segment: dict[str, Any], trajectory: dict[str, Any]) -> list[str]:
    values = segment.get("path_args")
    if values is None:
        values = trajectory.get("path_args")
    if values is None:
        return []
    return [str(value) for value in values]


def _copy_path_parts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(part) for part in value if isinstance(part, dict)]


def _copy_modifiers(modifiers: dict[str, Any] | None) -> dict[str, Any]:
    return dict(modifiers or {})


def _slide_head_modifiers(modifiers: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in modifiers.items()
        if key != "slide_segments"
    }


def _touch_area(note: Note, position: str) -> str:
    area = note.modifiers.get("touch_area") if note.modifiers else None
    if area is not None:
        return str(area)
    return position[:1]


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _last_or_none(values: list[str]) -> str | None:
    return values[-1] if values else None
