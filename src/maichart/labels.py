"""Frame label builder for ChartIR machine-learning datasets."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

from maichart.ir import ChartIR, Note
from maichart.serialization import load_chart_json
from maichart.timing import TICKS_PER_BEAT

FRAME_LABELS_SCHEMA = "maichart-frame-labels-v1"


@dataclass(slots=True)
class FrameGrid:
    """Fixed beat-grid metadata for frame labels."""

    division: int
    ticks_per_beat: int
    ticks_per_frame: int


@dataclass(slots=True)
class FrameLabels:
    """Labels attached to one fixed-grid frame."""

    has_note: bool = False
    note_count: int = 0
    tap_count: int = 0
    break_count: int = 0
    hold_start_count: int = 0
    hold_active_count: int = 0
    slide_start_count: int = 0
    slide_active_count: int = 0
    touch_count: int = 0
    touch_hold_start_count: int = 0
    note_types: list[str] = field(default_factory=list)
    positions: list[str] = field(default_factory=list)
    slide_patterns: list[str] = field(default_factory=list)
    duration_kinds: list[str] = field(default_factory=list)
    has_validation_warning: bool = False
    warning_codes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FrameLabel:
    """One frame in a frame-label sequence."""

    frame_index: int
    beat: str
    tick: int
    time_sec: float | None
    labels: FrameLabels = field(default_factory=FrameLabels)


@dataclass(slots=True)
class FrameLabelSet:
    """Top-level frame-label JSON payload for one chart difficulty."""

    schema: str
    song_id: str | None
    difficulty: int | None
    grid: FrameGrid
    frames: list[FrameLabel] = field(default_factory=list)


def build_frame_labels_from_chart_ir(
    chart_ir: ChartIR,
    division: int = 16,
    *,
    song_id: str | None = None,
) -> FrameLabelSet:
    """Convert one ChartIR into fixed beat-grid frame labels."""

    ticks_per_frame = _ticks_per_frame(division)
    max_tick = _max_label_tick(chart_ir.notes)
    frame_count = max(1, max_tick // ticks_per_frame + 1)
    frames = [
        FrameLabel(
            frame_index=index,
            beat=_format_beat(index * ticks_per_frame),
            tick=index * ticks_per_frame,
            time_sec=_time_for_tick(chart_ir, index * ticks_per_frame),
        )
        for index in range(frame_count)
    ]

    for note in chart_ir.notes:
        _apply_note_to_frames(note, frames, ticks_per_frame, chart_ir)

    for frame in frames:
        labels = frame.labels
        labels.has_note = (
            labels.note_count > 0
            or labels.hold_active_count > 0
            or labels.slide_active_count > 0
        )

    return FrameLabelSet(
        schema=FRAME_LABELS_SCHEMA,
        song_id=song_id,
        difficulty=chart_ir.difficulty.index,
        grid=FrameGrid(
            division=division,
            ticks_per_beat=TICKS_PER_BEAT,
            ticks_per_frame=ticks_per_frame,
        ),
        frames=frames,
    )


def frame_labels_to_dict(labels: FrameLabelSet) -> dict[str, Any]:
    """Convert frame labels to JSON-compatible primitives."""

    return asdict(labels)


def frame_labels_to_json(labels: FrameLabelSet, *, indent: int = 2) -> str:
    """Serialize frame labels to JSON."""

    return json.dumps(frame_labels_to_dict(labels), ensure_ascii=False, indent=indent)


def frame_labels_from_dict(data: dict[str, Any]) -> FrameLabelSet:
    """Load frame labels from JSON-compatible primitives."""

    grid_data = data.get("grid") or {}
    return FrameLabelSet(
        schema=str(data.get("schema", FRAME_LABELS_SCHEMA)),
        song_id=data.get("song_id"),
        difficulty=data.get("difficulty"),
        grid=FrameGrid(
            division=int(grid_data.get("division", 16)),
            ticks_per_beat=int(grid_data.get("ticks_per_beat", TICKS_PER_BEAT)),
            ticks_per_frame=int(grid_data.get("ticks_per_frame", _ticks_per_frame(16))),
        ),
        frames=[
            FrameLabel(
                frame_index=int(frame.get("frame_index", 0)),
                beat=str(frame.get("beat", "0")),
                tick=int(frame.get("tick", 0)),
                time_sec=frame.get("time_sec"),
                labels=FrameLabels(**(frame.get("labels") or {})),
            )
            for frame in data.get("frames", [])
        ],
    )


def frame_labels_from_json(payload: str) -> FrameLabelSet:
    """Deserialize frame labels from JSON text."""

    data = json.loads(payload)
    if not isinstance(data, dict):
        raise TypeError("Frame labels JSON must decode to an object.")
    return frame_labels_from_dict(data)


def save_frame_labels_json(labels: FrameLabelSet, path: str | Path) -> None:
    """Write frame labels JSON."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(frame_labels_to_json(labels), encoding="utf-8")


def load_frame_labels_json(path: str | Path) -> FrameLabelSet:
    """Read frame labels JSON."""

    return frame_labels_from_json(Path(path).read_text(encoding="utf-8"))


def build_frame_labels_for_dataset_manifest(
    manifest_path: str | Path,
    *,
    out_dir: str | Path,
    division: int = 16,
    output_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build frame-label files for every ChartIR cache entry in a dataset manifest."""

    manifest_file = Path(manifest_path).resolve()
    manifest_base = manifest_file.parent
    output_path = Path(output_manifest_path).resolve() if output_manifest_path else manifest_file
    labels_dir = Path(out_dir).resolve()

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    errors = manifest.setdefault("errors", [])
    label_count = 0

    for song in manifest.get("songs", []):
        song_id = str(song.get("song_id") or "unknown")
        for difficulty in song.get("difficulties", []):
            try:
                chart_ir_path = _resolve_manifest_path(
                    difficulty["chart_ir_path"],
                    manifest_base,
                )
                chart = load_chart_json(chart_ir_path)
                labels = build_frame_labels_from_chart_ir(
                    chart,
                    division=division,
                    song_id=song_id,
                )
                difficulty_index = difficulty.get("index", labels.difficulty)
                labels_path = labels_dir / song_id / f"difficulty_{difficulty_index}.frame_labels.json"
                save_frame_labels_json(labels, labels_path)
                difficulty["frame_labels_path"] = _display_path(labels_path, manifest_base)
                label_count += 1
            except Exception as exc:  # noqa: BLE001 - keep batch builds non-fatal.
                errors.append(
                    {
                        "stage": "frame_labels",
                        "message": str(exc),
                        "song_id": song_id,
                        "difficulty_index": difficulty.get("index"),
                        "chart_ir_path": difficulty.get("chart_ir_path"),
                        "exception_type": type(exc).__name__,
                    }
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "manifest": manifest,
        "label_count": label_count,
        "error_count": len(errors),
    }


def _apply_note_to_frames(
    note: Note,
    frames: list[FrameLabel],
    ticks_per_frame: int,
    chart: ChartIR,
) -> None:
    if note.tick is None:
        return
    start_frame = _frame_index_for_tick(note.tick, ticks_per_frame)
    if start_frame < 0 or start_frame >= len(frames):
        return

    labels = frames[start_frame].labels
    note_type = note.note_type
    labels.note_count += 1
    _append_unique(labels.note_types, note_type)
    if note.position is not None:
        _append_unique(labels.positions, str(note.position))
    for duration_kind in _duration_kinds(note):
        _append_unique(labels.duration_kinds, duration_kind)

    is_break = bool(note.modifiers.get("break")) if note.modifiers else False
    if is_break:
        labels.break_count += 1

    if note_type == "tap":
        labels.tap_count += 1
    elif note_type == "hold":
        labels.hold_start_count += 1
        _apply_active_range(
            frames,
            start_tick=note.tick,
            duration_ticks=_duration_ticks(note, chart),
            ticks_per_frame=ticks_per_frame,
            counter_name="hold_active_count",
        )
    elif note_type == "slide":
        labels.slide_start_count += 1
        for pattern in _slide_patterns(note):
            _append_unique(labels.slide_patterns, pattern)
        _apply_active_range(
            frames,
            start_tick=note.tick,
            duration_ticks=_duration_ticks(note, chart),
            ticks_per_frame=ticks_per_frame,
            counter_name="slide_active_count",
        )
    elif note_type == "touch":
        labels.touch_count += 1
    elif note_type == "touch_hold":
        labels.touch_hold_start_count += 1
        _apply_active_range(
            frames,
            start_tick=note.tick,
            duration_ticks=_duration_ticks(note, chart),
            ticks_per_frame=ticks_per_frame,
            counter_name="hold_active_count",
        )


def _apply_active_range(
    frames: list[FrameLabel],
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
        labels = frames[frame_index].labels
        setattr(labels, counter_name, getattr(labels, counter_name) + 1)


def _max_label_tick(notes: list[Note]) -> int:
    max_tick = 0
    for note in notes:
        if note.tick is None:
            continue
        max_tick = max(max_tick, note.tick)
        if note.duration_ticks is not None and note.duration_ticks > 0:
            max_tick = max(max_tick, note.tick + note.duration_ticks)
    return max_tick


def _duration_ticks(note: Note, chart: ChartIR) -> int | None:
    if note.duration_ticks is not None:
        return max(0, int(note.duration_ticks))
    if note.duration_beats is not None:
        return max(0, int(round(float(note.duration_beats) * TICKS_PER_BEAT)))
    if note.duration_sec is not None and note.tick is not None:
        bpm = _bpm_for_tick(chart, note.tick)
        return max(0, int(round(float(note.duration_sec) * bpm / 60.0 * TICKS_PER_BEAT)))
    return None


def _duration_kinds(note: Note) -> list[str]:
    kinds: list[str] = []
    if isinstance(note.duration, dict) and note.duration.get("kind"):
        kinds.append(str(note.duration["kind"]))
    for segment in note.segments:
        duration = segment.get("duration")
        if isinstance(duration, dict) and duration.get("kind"):
            kinds.append(str(duration["kind"]))
    return kinds


def _slide_patterns(note: Note) -> list[str]:
    patterns: list[str] = []
    for segment in note.segments:
        pattern = segment.get("pattern")
        if pattern is not None:
            patterns.append(str(pattern))
    if not patterns and note.modifiers.get("slide_pattern") is not None:
        patterns.append(str(note.modifiers["slide_pattern"]))
    return patterns


def _time_for_tick(chart: ChartIR, tick: int) -> float | None:
    bpms = sorted(
        [event for event in chart.timing.bpms if event.tick is not None and event.bpm],
        key=lambda event: int(event.tick or 0),
    )
    anchor_tick = 0
    anchor_time = 0.0
    bpm = 120.0
    if bpms:
        bpm = float(bpms[0].bpm or bpm)

    for event in bpms:
        event_tick = int(event.tick or 0)
        if event_tick > tick:
            break
        anchor_tick = event_tick
        if event.time_sec is not None:
            anchor_time = float(event.time_sec)
        else:
            anchor_time = _time_for_tick_without_event_time(
                chart,
                event_tick,
                fallback_bpm=bpm,
            )
        bpm = float(event.bpm or bpm)

    return anchor_time + ((tick - anchor_tick) / TICKS_PER_BEAT) * 60.0 / bpm


def _time_for_tick_without_event_time(
    chart: ChartIR,
    tick: int,
    *,
    fallback_bpm: float,
) -> float:
    note_times = [
        note.time_sec
        for note in chart.notes
        if note.tick == tick and note.time_sec is not None
    ]
    if note_times:
        return float(note_times[0])
    return (tick / TICKS_PER_BEAT) * 60.0 / fallback_bpm


def _bpm_for_tick(chart: ChartIR, tick: int) -> float:
    bpm = 120.0
    for event in sorted(
        [event for event in chart.timing.bpms if event.tick is not None and event.bpm],
        key=lambda event: int(event.tick or 0),
    ):
        if int(event.tick or 0) > tick:
            break
        bpm = float(event.bpm or bpm)
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


def _resolve_manifest_path(path: str | Path, base_path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (base_path / candidate).resolve()


def _display_path(path: Path, base_path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(base_path.resolve()).as_posix()
    except ValueError:
        return str(resolved)
