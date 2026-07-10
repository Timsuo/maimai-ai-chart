"""Frame-label v2 builder for EventIR.

This is a side-path builder used for EventIR validation. It does not replace
the legacy ChartIR frame-label pipeline.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any, Literal

from maichart.event_ir import (
    ChartEventIR,
    HoldEvent,
    SlideEvent,
    TapEvent,
    TouchEvent,
    TouchHoldEvent,
)
from maichart.duration import parse_duration_expr
from maichart.timing import TICKS_PER_BEAT

EVENT_FRAME_LABELS_SCHEMA = "maichart-frame-labels-v2"
SlideLaunchPolicy = Literal["legacy", "unknown", "default_one_beat"]


def build_frame_labels_from_event_ir(
    chart: ChartEventIR,
    *,
    division: int = 16,
    slide_launch_policy: SlideLaunchPolicy = "legacy",
) -> dict[str, Any]:
    """Convert EventIR into side-path frame labels v2."""

    if slide_launch_policy not in {"legacy", "unknown", "default_one_beat"}:
        raise ValueError(
            "slide_launch_policy must be one of: legacy, unknown, default_one_beat."
        )

    ticks_per_frame = _ticks_per_frame(division)
    max_tick = _max_label_tick(chart, slide_launch_policy)
    frame_count = max(1, max_tick // ticks_per_frame + 1)
    frames = [
        {
            "frame_index": index,
            "beat": _format_beat(index * ticks_per_frame),
            "tick": index * ticks_per_frame,
            "time_sec": _time_for_tick(chart, index * ticks_per_frame),
            "labels": _empty_labels(),
        }
        for index in range(frame_count)
    ]

    for event in chart.events:
        _apply_event_to_frames(event, chart, frames, ticks_per_frame, slide_launch_policy)

    for frame in frames:
        labels = frame["labels"]
        labels["has_note"] = (
            labels["note_count"] > 0
            or labels["hold_active_count"] > 0
            or labels["slide_active_count"] > 0
            or labels["slide_motion_active_count"] > 0
        )

    return {
        "schema": EVENT_FRAME_LABELS_SCHEMA,
        "song_id": None,
        "difficulty": chart.difficulty.index,
        "slide_launch_policy": slide_launch_policy,
        "grid": {
            "division": division,
            "ticks_per_beat": TICKS_PER_BEAT,
            "ticks_per_frame": ticks_per_frame,
        },
        "frames": frames,
    }


def _empty_labels() -> dict[str, Any]:
    return {
        "has_note": False,
        "note_count": 0,
        "tap_count": 0,
        "break_count": 0,
        "hold_start_count": 0,
        "hold_active_count": 0,
        "slide_start_count": 0,
        "slide_active_count": 0,
        "touch_count": 0,
        "touch_hold_start_count": 0,
        "note_types": [],
        "positions": [],
        "slide_patterns": [],
        "duration_kinds": [],
        "has_validation_warning": False,
        "warning_codes": [],
        "note_start_count": 0,
        "slide_head_count": 0,
        "slide_motion_start_count": 0,
        "slide_motion_active_count": 0,
        "slide_motion_end_count": 0,
        "slide_launch_offset_kinds": [],
        "slide_travel_duration_kinds": [],
        "slide_unknown_launch_offset_count": 0,
    }


def _apply_event_to_frames(
    event: Any,
    chart: ChartEventIR,
    frames: list[dict[str, Any]],
    ticks_per_frame: int,
    slide_launch_policy: SlideLaunchPolicy,
) -> None:
    if isinstance(event, TapEvent):
        labels = _labels_for_tick(frames, event.tick, ticks_per_frame)
        if labels is None:
            return
        _apply_note_start(labels, "tap", event.position)
        labels["tap_count"] += 1
        if event.is_break:
            labels["break_count"] += 1
        return

    if isinstance(event, HoldEvent):
        labels = _labels_for_tick(frames, event.head_tick, ticks_per_frame)
        if labels is None:
            return
        _apply_note_start(labels, "hold", event.position)
        _apply_break_modifier(labels, event.modifiers)
        labels["hold_start_count"] += 1
        _append_known_kind(labels["duration_kinds"], event.duration_kind)
        duration_ticks = _duration_ticks_for_event(
            chart=chart,
            tick=event.head_tick,
            duration_ticks=event.duration_ticks,
            duration_raw=event.duration_raw,
            timing_pair_values=None,
        )
        _apply_active_range(
            frames,
            start_tick=event.head_tick,
            duration_ticks=duration_ticks,
            ticks_per_frame=ticks_per_frame,
            counter_name="hold_active_count",
        )
        return

    if isinstance(event, SlideEvent):
        _apply_slide_event(event, chart, frames, ticks_per_frame, slide_launch_policy)
        return

    if isinstance(event, TouchEvent):
        labels = _labels_for_tick(frames, event.tick, ticks_per_frame)
        if labels is None:
            return
        _apply_note_start(labels, "touch", event.position)
        _apply_break_modifier(labels, event.modifiers)
        labels["touch_count"] += 1
        return

    if isinstance(event, TouchHoldEvent):
        labels = _labels_for_tick(frames, event.head_tick, ticks_per_frame)
        if labels is None:
            return
        _apply_note_start(labels, "touch_hold", event.position)
        _apply_break_modifier(labels, event.modifiers)
        labels["touch_hold_start_count"] += 1
        _append_known_kind(labels["duration_kinds"], event.duration_kind)
        duration_ticks = _duration_ticks_for_event(
            chart=chart,
            tick=event.head_tick,
            duration_ticks=event.duration_ticks,
            duration_raw=event.duration_raw,
            timing_pair_values=None,
        )
        _apply_active_range(
            frames,
            start_tick=event.head_tick,
            duration_ticks=duration_ticks,
            ticks_per_frame=ticks_per_frame,
            counter_name="hold_active_count",
        )


def _apply_slide_event(
    event: SlideEvent,
    chart: ChartEventIR,
    frames: list[dict[str, Any]],
    ticks_per_frame: int,
    slide_launch_policy: SlideLaunchPolicy,
) -> None:
    head_labels = _labels_for_tick(frames, event.head_tick, ticks_per_frame)
    if head_labels is None:
        return

    _apply_note_start(head_labels, "slide", event.start_position)
    _apply_break_modifier(head_labels, event.head_modifiers)
    head_labels["slide_start_count"] += 1
    head_labels["slide_head_count"] += 1
    _append_known_kind(head_labels["duration_kinds"], event.duration_kind)
    for segment in event.segments:
        _append_unique(head_labels["slide_patterns"], segment.path_type)
        _append_known_kind(head_labels["duration_kinds"], segment.duration_kind)
    _append_unique_kind(
        head_labels["slide_travel_duration_kinds"],
        _travel_duration_kind(event),
    )

    launch = _resolve_slide_launch(event, slide_launch_policy)
    _append_unique(head_labels["slide_launch_offset_kinds"], launch["kind"])
    if launch["unknown"]:
        head_labels["slide_unknown_launch_offset_count"] += 1
        return

    motion_start_tick = event.head_tick + int(launch["ticks"])
    motion_duration_ticks = _slide_motion_duration_ticks(event, chart)
    if motion_duration_ticks is None:
        return

    start_labels = _labels_for_tick(frames, motion_start_tick, ticks_per_frame)
    if start_labels is not None:
        start_labels["slide_motion_start_count"] += 1
        _append_unique(start_labels["slide_launch_offset_kinds"], launch["kind"])

    _apply_active_range(
        frames,
        start_tick=motion_start_tick,
        duration_ticks=motion_duration_ticks,
        ticks_per_frame=ticks_per_frame,
        counter_name="slide_motion_active_count",
    )
    _apply_active_range(
        frames,
        start_tick=motion_start_tick,
        duration_ticks=motion_duration_ticks,
        ticks_per_frame=ticks_per_frame,
        counter_name="slide_active_count",
    )

    end_tick = motion_start_tick + motion_duration_ticks
    end_labels = _labels_for_tick(frames, end_tick, ticks_per_frame)
    if end_labels is not None:
        end_labels["slide_motion_end_count"] += 1


def _slide_motion_duration_ticks(
    event: SlideEvent,
    chart: ChartEventIR,
) -> int | None:
    duration_ticks = _duration_ticks_for_event(
        chart=chart,
        tick=event.head_tick,
        duration_ticks=event.travel_duration_ticks,
        duration_raw=event.duration_raw,
        timing_pair_values=event.timing_pair_values,
    )
    if duration_ticks is not None:
        return duration_ticks

    segment_ticks: list[int] = []
    for segment in event.segments:
        segment_duration_ticks = _duration_ticks_for_event(
            chart=chart,
            tick=event.head_tick,
            duration_ticks=segment.travel_duration_ticks,
            duration_raw=segment.duration_raw,
            timing_pair_values=None,
        )
        if segment_duration_ticks is None:
            return None
        segment_ticks.append(segment_duration_ticks)
    return sum(segment_ticks) if segment_ticks else None


def _apply_note_start(labels: dict[str, Any], note_type: str, position: str) -> None:
    labels["note_count"] += 1
    labels["note_start_count"] += 1
    _append_unique(labels["note_types"], note_type)
    _append_unique(labels["positions"], position)


def _apply_break_modifier(labels: dict[str, Any], modifiers: dict[str, Any]) -> None:
    if modifiers.get("break"):
        labels["break_count"] += 1


def _resolve_slide_launch(
    event: SlideEvent,
    policy: SlideLaunchPolicy,
) -> dict[str, int | str | bool]:
    if event.launch_offset_ticks is not None:
        return {
            "ticks": int(event.launch_offset_ticks),
            "kind": "explicit",
            "unknown": False,
        }
    if policy == "legacy":
        return {"ticks": 0, "kind": "legacy_zero", "unknown": False}
    if policy == "default_one_beat":
        return {"ticks": TICKS_PER_BEAT, "kind": "default_one_beat", "unknown": False}
    return {"ticks": 0, "kind": "unknown", "unknown": True}


def _travel_duration_kind(event: SlideEvent) -> str:
    if event.duration_kind:
        return event.duration_kind
    if event.travel_duration_ticks is not None:
        return "ticks"
    return "unknown"


def _max_label_tick(chart: ChartEventIR, policy: SlideLaunchPolicy) -> int:
    max_tick = 0
    for event in chart.events:
        if isinstance(event, TapEvent):
            max_tick = max(max_tick, event.tick)
        elif isinstance(event, HoldEvent):
            max_tick = max(max_tick, event.head_tick)
            duration_ticks = event.duration_ticks
            if duration_ticks is not None and duration_ticks > 0:
                max_tick = max(max_tick, event.head_tick + duration_ticks)
        elif isinstance(event, SlideEvent):
            max_tick = max(max_tick, event.head_tick)
            launch = _resolve_slide_launch(event, policy)
            duration_ticks = event.travel_duration_ticks
            if not launch["unknown"] and duration_ticks is not None:
                motion_end = (
                    event.head_tick
                    + int(launch["ticks"])
                    + max(0, int(duration_ticks))
                )
                max_tick = max(max_tick, motion_end)
        elif isinstance(event, TouchEvent):
            max_tick = max(max_tick, event.tick)
        elif isinstance(event, TouchHoldEvent):
            max_tick = max(max_tick, event.head_tick)
            duration_ticks = event.duration_ticks
            if duration_ticks is not None and duration_ticks > 0:
                max_tick = max(max_tick, event.head_tick + duration_ticks)
    return max_tick


def _apply_active_range(
    frames: list[dict[str, Any]],
    *,
    start_tick: int,
    duration_ticks: int | None,
    ticks_per_frame: int,
    counter_name: str,
) -> None:
    if duration_ticks is None or duration_ticks <= 0:
        return
    start_frame = _frame_index_for_tick(start_tick, ticks_per_frame)
    end_tick_exclusive = start_tick + duration_ticks
    end_frame = _frame_index_for_tick(max(start_tick, end_tick_exclusive - 1), ticks_per_frame)
    for frame_index in range(max(0, start_frame), min(len(frames), end_frame + 1)):
        frames[frame_index]["labels"][counter_name] += 1


def _labels_for_tick(
    frames: list[dict[str, Any]],
    tick: int,
    ticks_per_frame: int,
) -> dict[str, Any] | None:
    frame_index = _frame_index_for_tick(tick, ticks_per_frame)
    if frame_index < 0 or frame_index >= len(frames):
        return None
    return frames[frame_index]["labels"]


def _time_for_tick(chart: ChartEventIR, tick: int) -> float | None:
    bpms = sorted(chart.timing_events, key=lambda event: event.tick)
    anchor_tick = 0
    anchor_time = 0.0
    bpm = 120.0
    if bpms:
        bpm = float(bpms[0].bpm)

    for event in bpms:
        if event.tick > tick:
            break
        anchor_tick = event.tick
        if event.time_sec is not None:
            anchor_time = float(event.time_sec)
        else:
            anchor_time = (event.tick / TICKS_PER_BEAT) * 60.0 / bpm
        bpm = float(event.bpm)

    return anchor_time + ((tick - anchor_tick) / TICKS_PER_BEAT) * 60.0 / bpm


def _duration_ticks_for_event(
    *,
    chart: ChartEventIR | None,
    tick: int,
    duration_ticks: int | None,
    duration_raw: str | None,
    timing_pair_values: list[float] | None,
) -> int | None:
    if duration_ticks is not None:
        return max(0, int(duration_ticks))

    if timing_pair_values:
        seconds = float(timing_pair_values[-1])
        return _seconds_to_ticks(seconds, _bpm_for_tick(chart, tick))

    if duration_raw is None:
        return None

    duration = parse_duration_expr(duration_raw)
    if duration.ticks is not None:
        return max(0, int(duration.ticks))
    if duration.beats is not None:
        return max(0, int(round(float(duration.beats) * TICKS_PER_BEAT)))
    if duration.seconds is not None:
        return _seconds_to_ticks(float(duration.seconds), _bpm_for_tick(chart, tick))
    return None


def _seconds_to_ticks(seconds: float, bpm: float) -> int:
    return max(0, int(round(seconds * bpm / 60.0 * TICKS_PER_BEAT)))


def _bpm_for_tick(chart: ChartEventIR | None, tick: int) -> float:
    bpm = 120.0
    if chart is None:
        return bpm
    for event in sorted(chart.timing_events, key=lambda timing: timing.tick):
        if event.tick > tick:
            break
        bpm = float(event.bpm)
    return bpm


def _ticks_per_frame(division: int) -> int:
    if division <= 0:
        raise ValueError("division must be positive.")
    ticks = Fraction(TICKS_PER_BEAT * 4, division)
    if ticks.denominator != 1:
        raise ValueError(
            f"division {division} does not map to an integer tick frame."
        )
    return ticks.numerator


def _frame_index_for_tick(tick: int, ticks_per_frame: int) -> int:
    return math.floor(tick / ticks_per_frame)


def _format_beat(tick: int) -> str:
    beat = Fraction(tick, TICKS_PER_BEAT)
    if beat.denominator == 1:
        return str(beat.numerator)
    return f"{beat.numerator}/{beat.denominator}"


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _append_unique_kind(values: list[str], value: str | None) -> None:
    _append_unique(values, value or "unknown")


def _append_known_kind(values: list[str], value: str | None) -> None:
    if value is not None:
        _append_unique(values, value)
