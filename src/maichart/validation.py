"""Validation helpers for V1 Maidata parser output."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Literal

from maichart.maidata import RawDifficultyBlock, RawMaidataChart
from maichart.notes import parse_timing_points_notes
from maichart.duration import parse_duration_expr
from maichart.timing import RawTimingPoint, tokenize_difficulty_timing

Severity = Literal["error", "warning"]


@dataclass(slots=True)
class ValidationIssue:
    """One validation issue found in a raw chart or parsed difficulty."""

    severity: Severity
    code: str
    message: str
    difficulty_index: int | None = None
    timing_index: int | None = None
    raw: str | None = None


@dataclass(slots=True)
class ValidationReport:
    """Validation summary for a Maidata file."""

    ok: bool
    errors: int
    warnings: int
    issues: list[ValidationIssue] = field(default_factory=list)


def validate_raw_maidata_chart(
    chart: RawMaidataChart,
    *,
    difficulty_index: int | None = None,
    strict: bool = False,
) -> ValidationReport:
    """Validate parsed Maidata metadata and selected raw difficulty blocks."""

    issues: list[ValidationIssue] = []
    _validate_metadata(chart, issues)

    difficulties = _select_difficulties(chart, difficulty_index, issues)
    for difficulty in difficulties:
        _validate_difficulty(chart, difficulty, issues, strict=strict)

    errors = sum(1 for issue in issues if issue.severity == "error")
    warnings = sum(1 for issue in issues if issue.severity == "warning")
    return ValidationReport(
        ok=errors == 0,
        errors=errors,
        warnings=warnings,
        issues=issues,
    )


def validation_report_to_dict(report: ValidationReport) -> dict[str, object]:
    """Convert a validation report to JSON-compatible primitives."""

    return asdict(report)


def validation_report_to_json(report: ValidationReport, *, indent: int = 2) -> str:
    """Serialize a validation report to JSON."""

    return json.dumps(validation_report_to_dict(report), ensure_ascii=False, indent=indent)


def _validate_metadata(chart: RawMaidataChart, issues: list[ValidationIssue]) -> None:
    if not chart.title:
        issues.append(ValidationIssue("warning", "missing-title", "Missing &title field."))
    if not chart.artist:
        issues.append(ValidationIssue("warning", "missing-artist", "Missing &artist field."))
    if not chart.difficulties:
        issues.append(
            ValidationIssue("error", "missing-difficulties", "No difficulty blocks found.")
        )
    if chart.wholebpm is not None:
        try:
            if float(chart.wholebpm) <= 0:
                raise ValueError
        except ValueError:
            issues.append(
                ValidationIssue(
                    "error",
                    "invalid-wholebpm",
                    f"&wholebpm must be positive, got {chart.wholebpm!r}.",
                )
            )


def _select_difficulties(
    chart: RawMaidataChart,
    difficulty_index: int | None,
    issues: list[ValidationIssue],
) -> list[RawDifficultyBlock]:
    if difficulty_index is None:
        return chart.difficulties

    difficulty = next(
        (
            candidate
            for candidate in chart.difficulties
            if candidate.index == difficulty_index
        ),
        None,
    )
    if difficulty is None:
        issues.append(
            ValidationIssue(
                "error",
                "missing-difficulty",
                f"Difficulty {difficulty_index} is not present.",
                difficulty_index=difficulty_index,
            )
        )
        return []
    return [difficulty]


def _validate_difficulty(
    chart: RawMaidataChart,
    difficulty: RawDifficultyBlock,
    issues: list[ValidationIssue],
    *,
    strict: bool,
) -> None:
    if difficulty.level is None:
        issues.append(
            ValidationIssue(
                "warning",
                "missing-level",
                "Difficulty level is missing.",
                difficulty_index=difficulty.index,
            )
        )
    if difficulty.inote is None:
        issues.append(
            ValidationIssue(
                "error",
                "missing-inote",
                "Difficulty has no inote block.",
                difficulty_index=difficulty.index,
            )
        )
        return

    points = tokenize_difficulty_timing(
        difficulty,
        initial_bpm=_parse_initial_bpm(chart.wholebpm),
    )
    _validate_timing_points(difficulty, points, issues)
    _validate_notes(difficulty, points, issues, strict=strict)


def _validate_timing_points(
    difficulty: RawDifficultyBlock,
    points: list[RawTimingPoint],
    issues: list[ValidationIssue],
) -> None:
    previous_tick = -1
    previous_time = -1.0
    for point in points:
        if point.tick < previous_tick:
            issues.append(
                ValidationIssue(
                    "error",
                    "non-monotonic-tick",
                    "Timing ticks must be monotonic.",
                    difficulty_index=difficulty.index,
                    timing_index=point.index,
                    raw=point.raw,
                )
            )
        if point.time_sec is not None and point.time_sec < previous_time:
            issues.append(
                ValidationIssue(
                    "error",
                    "non-monotonic-time",
                    "Timing seconds must be monotonic.",
                    difficulty_index=difficulty.index,
                    timing_index=point.index,
                    raw=point.raw,
                )
            )
        previous_tick = point.tick
        previous_time = point.time_sec if point.time_sec is not None else previous_time


def _validate_notes(
    difficulty: RawDifficultyBlock,
    points: list[RawTimingPoint],
    issues: list[ValidationIssue],
    *,
    strict: bool,
) -> None:
    parsed_points = parse_timing_points_notes(points)
    for point in parsed_points:
        for note in point.notes:
            if note.note_type in {"tap", "hold", "slide"} and not _is_valid_button_position(note.position):
                issues.append(
                    ValidationIssue(
                        "error",
                        "invalid-position",
                        "Note position must be 1 through 8.",
                        difficulty_index=difficulty.index,
                        timing_index=point.timing_index,
                        raw=note.raw,
                    )
                )
            if note.note_type in {"touch", "touch_hold"} and not _is_valid_touch_position(note.position):
                issues.append(
                    ValidationIssue(
                        "error",
                        "invalid-touch-position",
                        "Touch position must be C or an A/B/D/E lane with optional 1 through 8 index.",
                        difficulty_index=difficulty.index,
                        timing_index=point.timing_index,
                        raw=note.raw,
                    )
                )
            if note.end_position is not None and not _is_valid_position_value(note.end_position):
                issues.append(
                    ValidationIssue(
                        "error",
                        "invalid-end-position",
                        "Slide end position must be 1 through 8.",
                        difficulty_index=difficulty.index,
                        timing_index=point.timing_index,
                        raw=note.raw,
                    )
                )
            _validate_note_duration(difficulty, point, note, issues)
        for token in point.unknown_tokens:
            _validate_unknown_token_duration(difficulty, point, token.raw, issues)
            issues.append(
                ValidationIssue(
                    "error" if strict else "warning",
                    "unknown-token",
                    "Token is preserved as raw unknown syntax.",
                    difficulty_index=difficulty.index,
                    timing_index=point.timing_index,
                    raw=token.raw,
                )
            )


def _parse_initial_bpm(value: str | None) -> float:
    if value is None:
        return 120.0
    try:
        parsed = float(value)
    except ValueError:
        return 120.0
    return parsed if parsed > 0 else 120.0


def _is_valid_position_value(value: int | str) -> bool:
    text = str(value)
    return all(char in "12345678" for char in text)


def _is_valid_button_position(value: int | str) -> bool:
    text = str(value)
    return len(text) == 1 and text in "12345678"


def _is_valid_touch_position(value: int | str) -> bool:
    text = str(value)
    if text == "C":
        return True
    if len(text) == 2 and text[0] in "ABDE" and text[1] in "12345678":
        return True
    if len(text) == 1 and text in "ABDE":
        return True
    return False


def _validate_note_duration(
    difficulty: RawDifficultyBlock,
    point,
    note,
    issues: list[ValidationIssue],
) -> None:
    duration_kind = note.duration.kind if note.duration is not None else None

    if duration_kind == "unknown":
        issues.append(
            ValidationIssue(
                "warning",
                "unknown-duration",
                "Duration expression is preserved but not interpreted.",
                difficulty_index=difficulty.index,
                timing_index=point.timing_index,
                raw=note.raw,
            )
        )

    if note.duration_sec is not None and note.duration_sec < 0:
        issues.append(
            ValidationIssue(
                "error",
                "invalid-duration",
                "Note duration seconds must be positive.",
                difficulty_index=difficulty.index,
                timing_index=point.timing_index,
                raw=note.raw,
            )
        )
    if note.duration_ticks is not None and note.duration_ticks < 0:
        issues.append(
            ValidationIssue(
                "error",
                "invalid-duration",
                "Note duration ticks must be positive.",
                difficulty_index=difficulty.index,
                timing_index=point.timing_index,
                raw=note.raw,
            )
        )
    if note.duration_beats is not None and note.duration_beats < 0:
        issues.append(
            ValidationIssue(
                "error",
                "invalid-duration",
                "Note duration beats must be positive.",
                difficulty_index=difficulty.index,
                timing_index=point.timing_index,
                raw=note.raw,
            )
        )
    elif note.duration_beats is not None and note.duration_beats == 0:
        issues.append(
            ValidationIssue(
                "warning",
                "zero-duration-note",
                "Note duration is zero beats.",
                difficulty_index=difficulty.index,
                timing_index=point.timing_index,
                raw=note.raw,
            )
        )
    elif (
        note.duration_beats is not None
        and note.duration_ticks is None
        and duration_kind in {"grid_fraction", "compound"}
    ):
        issues.append(
            ValidationIssue(
                "warning",
                "non-integer-duration-tick",
                "Note duration does not map to an integer tick.",
                difficulty_index=difficulty.index,
                timing_index=point.timing_index,
                raw=note.raw,
            )
        )

    if note.duration is not None and note.duration.values is not None:
        if any(value < 0 for value in note.duration.values):
            issues.append(
                ValidationIssue(
                    "error",
                    "invalid-duration",
                    "Duration timing values must be positive.",
                    difficulty_index=difficulty.index,
                    timing_index=point.timing_index,
                    raw=note.raw,
                )
            )


def _validate_unknown_token_duration(
    difficulty: RawDifficultyBlock,
    point,
    raw: str,
    issues: list[ValidationIssue],
) -> None:
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end < start:
        return

    duration = parse_duration_expr(raw[start : end + 1])
    if duration.kind != "unknown":
        return

    issues.append(
        ValidationIssue(
            "error",
            "invalid-duration",
            "Token contains a duration block that V1 cannot parse.",
            difficulty_index=difficulty.index,
            timing_index=point.timing_index,
            raw=raw,
        )
    )
