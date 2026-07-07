"""Statistics helpers for V1 Maidata parser output."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field

from maichart.maidata import RawDifficultyBlock, RawMaidataChart
from maichart.notes import parse_timing_points_notes
from maichart.timing import tokenize_difficulty_timing


@dataclass(slots=True)
class DifficultyStats:
    """Statistics for one parsed difficulty."""

    difficulty_index: int
    level: str | None
    designer: str | None
    timing_points: int
    note_count: int
    unknown_token_count: int
    parsed_token_count: int
    total_token_count: int
    parse_coverage: float
    type_counts: dict[str, int] = field(default_factory=dict)
    type_ratios: dict[str, float] = field(default_factory=dict)
    duration_kind_counts: dict[str, int] = field(default_factory=dict)
    slide_pattern_counts: dict[str, int] = field(default_factory=dict)
    bpm_min: float | None = None
    bpm_max: float | None = None
    duration_beats: float | None = None
    density_notes_per_beat: float | None = None
    per_measure_counts: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class ChartStats:
    """Statistics for a Maidata chart."""

    title: str | None
    artist: str | None
    wholebpm: str | None
    difficulty_count: int
    difficulties: list[DifficultyStats] = field(default_factory=list)


def compute_raw_maidata_stats(
    chart: RawMaidataChart,
    *,
    difficulty_index: int | None = None,
) -> ChartStats:
    """Compute note and timing statistics for parsed Maidata data."""

    difficulties = [
        difficulty
        for difficulty in chart.difficulties
        if difficulty_index is None or difficulty.index == difficulty_index
    ]
    return ChartStats(
        title=chart.title,
        artist=chart.artist,
        wholebpm=chart.wholebpm,
        difficulty_count=len(chart.difficulties),
        difficulties=[
            _compute_difficulty_stats(chart, difficulty)
            for difficulty in difficulties
        ],
    )


def chart_stats_to_dict(stats: ChartStats) -> dict[str, object]:
    """Convert chart stats to JSON-compatible primitives."""

    return asdict(stats)


def chart_stats_to_json(stats: ChartStats, *, indent: int = 2) -> str:
    """Serialize chart stats to JSON."""

    return json.dumps(chart_stats_to_dict(stats), ensure_ascii=False, indent=indent)


def _compute_difficulty_stats(
    chart: RawMaidataChart,
    difficulty: RawDifficultyBlock,
) -> DifficultyStats:
    points = tokenize_difficulty_timing(
        difficulty,
        initial_bpm=_parse_initial_bpm(chart.wholebpm),
    )
    parsed_points = parse_timing_points_notes(points)
    notes = [note for point in parsed_points for note in point.notes]
    unknown_count = sum(len(point.unknown_tokens) for point in parsed_points)
    type_counts = Counter(note.note_type for note in notes)
    duration_kind_counts = Counter(
        note.duration.kind
        for note in notes
        if note.duration is not None
    )
    slide_pattern_counts = Counter(
        str(note.slide_pattern)
        for note in notes
        if note.note_type == "slide" and note.slide_pattern is not None
    )
    note_count = len(notes)
    parsed_token_count = note_count
    total_token_count = parsed_token_count + unknown_count
    duration_beats = float(points[-1].beat) if points else None

    bpms = [point.bpm for point in points]
    per_measure_counts: Counter[str] = Counter()
    for note in notes:
        measure = int(note.beat // 4)
        per_measure_counts[str(measure)] += 1

    return DifficultyStats(
        difficulty_index=difficulty.index,
        level=difficulty.level,
        designer=difficulty.designer,
        timing_points=len(points),
        note_count=note_count,
        unknown_token_count=unknown_count,
        parsed_token_count=parsed_token_count,
        total_token_count=total_token_count,
        parse_coverage=(
            parsed_token_count / total_token_count
            if total_token_count
            else 1.0
        ),
        type_counts=dict(type_counts),
        type_ratios={
            note_type: count / note_count
            for note_type, count in type_counts.items()
        }
        if note_count
        else {},
        duration_kind_counts=dict(duration_kind_counts),
        slide_pattern_counts=dict(slide_pattern_counts),
        bpm_min=min(bpms) if bpms else None,
        bpm_max=max(bpms) if bpms else None,
        duration_beats=duration_beats,
        density_notes_per_beat=(
            note_count / duration_beats
            if duration_beats not in (None, 0.0)
            else None
        ),
        per_measure_counts=dict(sorted(per_measure_counts.items(), key=lambda item: int(item[0]))),
    )


def _parse_initial_bpm(value: str | None) -> float:
    if value is None:
        return 120.0
    try:
        parsed = float(value)
    except ValueError:
        return 120.0
    return parsed if parsed > 0 else 120.0
