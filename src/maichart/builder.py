"""Helpers for building ChartIR from parsed Maidata structures."""

from __future__ import annotations

from maichart.ir import (
    BpmEvent,
    ChartIR,
    ChartMetadata,
    DifficultyMetadata,
    GridEvent,
    TimingData,
    UnknownChartToken,
)
from maichart.maidata import RawDifficultyBlock, RawMaidataChart
from maichart.notes import (
    parse_timing_points_notes,
    parsed_timing_notes_to_chart_notes,
)
from maichart.timing import RawTimingPoint, tokenize_difficulty_timing


def build_chart_ir_for_difficulty(
    chart: RawMaidataChart,
    difficulty: RawDifficultyBlock,
    *,
    initial_division: int = 4,
) -> ChartIR:
    """Build a single-difficulty ChartIR from raw Maidata parser output."""

    initial_bpm = _parse_initial_bpm(chart.wholebpm)
    timing_points = tokenize_difficulty_timing(
        difficulty,
        initial_bpm=initial_bpm,
        initial_division=initial_division,
    )
    parsed_points = parse_timing_points_notes(timing_points)

    return ChartIR(
        metadata=ChartMetadata(
            title=chart.title,
            artist=chart.artist,
            offset=_parse_float(chart.first),
            raw=chart.raw,
        ),
        difficulty=DifficultyMetadata(
            index=difficulty.index,
            level=difficulty.level,
            designer=difficulty.designer,
            raw=_join_raw_difficulty(difficulty),
        ),
        timing=_build_timing_data(timing_points),
        notes=parsed_timing_notes_to_chart_notes(parsed_points),
        unknown_tokens=_build_unknown_tokens(parsed_points),
        raw=difficulty.raw_inote,
    )


def build_chart_ir_by_difficulty_index(
    chart: RawMaidataChart,
    difficulty_index: int,
    *,
    initial_division: int = 4,
) -> ChartIR:
    """Build ChartIR for one difficulty index from a raw Maidata chart."""

    difficulty = next(
        (
            candidate
            for candidate in chart.difficulties
            if candidate.index == difficulty_index
        ),
        None,
    )
    if difficulty is None:
        raise ValueError(f"No difficulty {difficulty_index} found.")
    return build_chart_ir_for_difficulty(
        chart,
        difficulty,
        initial_division=initial_division,
    )


def _build_timing_data(points: list[RawTimingPoint]) -> TimingData:
    timing = TimingData()
    seen_bpm: set[tuple[int, float]] = set()
    seen_grid: set[tuple[int, int]] = set()

    for point in points:
        for directive in point.directives:
            if directive.directive_type == "bpm":
                key = (point.tick, float(directive.value))
                if key not in seen_bpm:
                    timing.bpms.append(
                        BpmEvent(
                            bpm=float(directive.value),
                            time_sec=point.time_sec,
                            beat=float(point.beat),
                            tick=point.tick,
                            raw=directive.raw,
                        )
                    )
                    seen_bpm.add(key)
            elif directive.directive_type == "grid":
                key = (point.tick, int(directive.value))
                if key not in seen_grid:
                    timing.grids.append(
                        GridEvent(
                            division=int(directive.value),
                            time_sec=point.time_sec,
                            beat=float(point.beat),
                            tick=point.tick,
                            raw=directive.raw,
                        )
                    )
                    seen_grid.add(key)

    return timing


def _build_unknown_tokens(parsed_points) -> list[UnknownChartToken]:
    unknown_tokens: list[UnknownChartToken] = []
    for point in parsed_points:
        for token in point.unknown_tokens:
            unknown_tokens.append(
                UnknownChartToken(
                    timing_index=point.timing_index,
                    raw_timing_point=point.raw,
                    raw_token=token.raw,
                    reason=token.reason,
                    time_sec=point.time_sec,
                    beat=float(point.beat),
                    tick=point.tick,
                    raw=token.raw,
                )
            )
    return unknown_tokens


def _join_raw_difficulty(difficulty: RawDifficultyBlock) -> str | None:
    parts = [
        difficulty.raw_level,
        difficulty.raw_designer,
        difficulty.raw_inote,
    ]
    return "\n".join(part for part in parts if part is not None) or None


def _parse_initial_bpm(value: str | None) -> float:
    if value is None:
        return 120.0
    try:
        parsed = float(value)
    except ValueError:
        return 120.0
    return parsed if parsed > 0 else 120.0


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None
