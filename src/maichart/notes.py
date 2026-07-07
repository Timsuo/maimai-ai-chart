"""Basic tap, break, hold, and slide parser for raw timing points."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from maichart.duration import (
    DurationExpr,
    duration_expr_to_dict,
    duration_to_ticks_or_none,
    resolve_duration_expr,
    parse_duration_expr,
)
from maichart.ir import Note
from maichart.slides import get_slide_pattern_definition, is_supported_slide_pattern
from maichart.timing import RawTimingPoint

BASIC_TAP_RE = re.compile(r"^(?P<position>[1-8])(?P<break>b?)(?P<ex>x?)$")
BASIC_HOLD_RE = re.compile(
    r"^(?P<position>[1-8])h(?P<modifiers>[bx]*)(?P<duration>\[[^\]]+\])$"
)
BASIC_SLIDE_RE = re.compile(
    r"^(?P<position>[1-8])(?P<head_modifiers>[bx]*)(?P<pattern>pp|qq|[-<>pqszvwV])(?P<end_position>[1-8]{1,2})"
    r"(?P<duration>\[[^\]]+\])(?P<tail_modifiers>[bx]*)$"
)
CONTINUED_SLIDE_RE = re.compile(
    r"^(?P<pattern>pp|qq|[-<>pqszvwV])(?P<end_position>[1-8]{1,2})"
    r"(?P<duration>\[[^\]]+\])(?P<tail_modifiers>[bx]*)$"
)
COMPOUND_SLIDE_RE = re.compile(
    r"^(?P<position>[1-8])(?P<head_modifiers>[bx]*)(?P<path_text>[1-8<>pqszvwV-]+)"
    r"(?P<duration>\[[^\]]+\])(?P<tail_modifiers>[bx]*)$"
)
CONTINUED_COMPOUND_SLIDE_RE = re.compile(
    r"^(?P<path_text>[1-8<>pqszvwV-]+)(?P<duration>\[[^\]]+\])(?P<tail_modifiers>[bx]*)$"
)
BASIC_TOUCH_RE = re.compile(
    r"^(?P<area>[ABCDE])(?P<position>[1-8]?)(?P<firework>f?)$"
)
BASIC_TOUCH_HOLD_RE = re.compile(
    r"^(?P<area>[ABCDE])(?P<position>[1-8]?)h(?P<firework>f?)(?P<duration>\[[^\]]+\])$"
)


@dataclass(slots=True)
class ParsedNote:
    """A V1 parsed note."""

    note_type: str
    position: int | str
    is_break: bool
    raw: str
    time_sec: float | None = None
    beat: Fraction = Fraction(0, 1)
    tick: int = 0
    group_id: str | None = None
    duration_beats: Fraction | None = None
    duration_ticks: int | None = None
    duration_sec: float | None = None
    end_position: int | str | None = None
    slide_pattern: str | None = None
    slide_segments: list[dict[str, Any]] = field(default_factory=list)
    duration: DurationExpr | None = None
    segments: list[dict[str, Any]] = field(default_factory=list)
    modifiers: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class UnknownNoteToken:
    """A raw note token intentionally left unparsed."""

    raw: str
    reason: str


@dataclass(slots=True)
class ParsedTimingPointNotes:
    """Parsed and unknown note tokens for one raw timing point."""

    timing_index: int
    raw: str
    note_text: str
    beat: Fraction
    tick: int
    time_sec: float | None
    notes: list[ParsedNote] = field(default_factory=list)
    unknown_tokens: list[UnknownNoteToken] = field(default_factory=list)


def parse_basic_tap_break_token(token: str) -> ParsedNote | None:
    """Parse one complete basic tap or break token.

    The token must fully match ``^[1-8]b?$``. Complex note syntax is not
    partially parsed.
    """

    candidate = token.strip()
    match = BASIC_TAP_RE.fullmatch(candidate)
    if match is None:
        return None

    return ParsedNote(
        note_type="tap",
        position=int(match.group("position")),
        is_break=match.group("break") == "b",
        raw=token,
        modifiers=_note_modifiers(
            is_break=match.group("break") == "b",
            is_ex=match.group("ex") == "x",
        ),
    )


def parse_basic_hold_token(token: str) -> ParsedNote | None:
    """Parse one complete basic hold token.

    Supported hold syntax is ``^[1-8]h\\[[1-9][0-9]*:[1-9][0-9]*\\]$``.
    Duration is measured with the Maidata-style measure fraction where
    ``[4:1]`` is one beat and ``[8:3]`` is one and a half beats.
    """

    candidate = token.strip()
    match = BASIC_HOLD_RE.fullmatch(candidate)
    if match is None:
        return None

    duration = parse_duration_expr(match.group("duration"))
    if duration.kind == "unknown":
        return None
    modifiers = _parse_note_modifier_text(match.group("modifiers"))
    return ParsedNote(
        note_type="hold",
        position=int(match.group("position")),
        is_break=bool(modifiers["break"]),
        raw=token,
        duration_beats=duration.beats,
        duration_ticks=duration.ticks,
        duration_sec=duration.seconds,
        duration=duration,
        modifiers=modifiers,
    )


def parse_basic_slide_token(token: str) -> ParsedNote | None:
    """Parse one complete basic single-segment slide token.

    Supported slide syntax is one start button, optional break marker, one or
    more path segments, and duration blocks, such as ``1-4[8:1]``,
    ``3b-1[8:1]``, ``7pp1[8:1]``, or ``1-4[8:1]*-6[8:1]``.
    """

    candidate = token.strip()
    raw_segments = _split_top_level_stars(candidate)
    if not raw_segments:
        return None

    parsed_segments: list[dict[str, Any]] = []
    for index, raw_segment in enumerate(raw_segments):
        regex = BASIC_SLIDE_RE if index == 0 else CONTINUED_SLIDE_RE
        match = regex.fullmatch(raw_segment)
        if match is None:
            compound_regex = COMPOUND_SLIDE_RE if index == 0 else CONTINUED_COMPOUND_SLIDE_RE
            compound_match = compound_regex.fullmatch(raw_segment)
            if compound_match is None:
                return None
            try:
                parsed_segments.append(
                    _compound_slide_segment_from_match(compound_match, raw_segment)
                )
            except ValueError:
                return None
            continue
        try:
            parsed_segments.append(_slide_segment_from_match(match, raw_segment))
        except ValueError:
            return None

    _attach_slide_trajectory_refs(parsed_segments)
    first_segment = parsed_segments[0]
    duration = _combine_segment_durations(parsed_segments, token)
    return ParsedNote(
        note_type="slide",
        position=int(first_segment["position"]),
        is_break=bool(first_segment["is_break"]),
        raw=token,
        duration_beats=duration.beats,
        duration_ticks=duration.ticks,
        duration_sec=duration.seconds,
        duration=duration,
        end_position=first_segment["end_position"],
        slide_pattern=str(first_segment.get("pattern", "compound")),
        slide_segments=parsed_segments,
        segments=parsed_segments,
        modifiers=_slide_note_modifiers(parsed_segments),
    )


def parse_basic_touch_token(token: str) -> ParsedNote | None:
    """Parse one complete touch tap or touch hold token."""

    candidate = token.strip()

    hold_match = BASIC_TOUCH_HOLD_RE.fullmatch(candidate)
    if hold_match is not None:
        duration = parse_duration_expr(hold_match.group("duration"))
        if duration.kind == "unknown":
            return None
        position = _touch_position(
            hold_match.group("area"),
            hold_match.group("position"),
        )
        return ParsedNote(
            note_type="touch_hold",
            position=position,
            is_break=False,
            raw=token,
            duration_beats=duration.beats,
            duration_ticks=duration.ticks,
            duration_sec=duration.seconds,
            duration=duration,
            modifiers={
                "touch_area": hold_match.group("area"),
                "firework": hold_match.group("firework") == "f",
            },
        )

    tap_match = BASIC_TOUCH_RE.fullmatch(candidate)
    if tap_match is None:
        return None
    position = _touch_position(tap_match.group("area"), tap_match.group("position"))
    return ParsedNote(
        note_type="touch",
        position=position,
        is_break=False,
        raw=token,
        modifiers={
            "touch_area": tap_match.group("area"),
            "firework": tap_match.group("firework") == "f",
        },
    )


def parse_timing_point_notes(point: RawTimingPoint) -> ParsedTimingPointNotes:
    """Parse basic tap, break, hold, and slide notes from one timing point."""

    parsed = ParsedTimingPointNotes(
        timing_index=point.index,
        raw=point.raw,
        note_text=point.note_text,
        beat=point.beat,
        tick=point.tick,
        time_sec=point.time_sec,
    )

    if point.note_text == "":
        return parsed

    tokens = _split_top_level_slashes(point.note_text)
    for token in tokens:
        if token.strip() == "":
            parsed.unknown_tokens.append(UnknownNoteToken(token, "empty token"))
            continue

        note = parse_basic_tap_break_token(token)
        if note is None:
            note = parse_basic_hold_token(token)
        if note is None:
            note = parse_basic_slide_token(token)
        if note is None:
            note = parse_basic_touch_token(token)
        if note is None:
            parsed.unknown_tokens.append(
                UnknownNoteToken(token, "unsupported note syntax")
            )
            continue

        note.beat = point.beat
        note.tick = point.tick
        note.time_sec = point.time_sec
        _apply_duration_context(note, point.bpm)
        parsed.notes.append(note)

    if len(parsed.notes) > 1:
        group_id = f"timing-{point.index}"
        for note in parsed.notes:
            note.group_id = group_id

    return parsed


def parse_timing_points_notes(points: list[RawTimingPoint]) -> list[ParsedTimingPointNotes]:
    """Parse basic tap, break, hold, and slide notes from timing points."""

    return [parse_timing_point_notes(point) for point in points]


def parsed_note_to_chart_note(note: ParsedNote) -> Note:
    """Convert a parsed tap, break, or hold note to the current ChartIR shape."""

    modifiers: dict[str, Any] = {"break": note.is_break}
    modifiers.update(note.modifiers)
    if note.group_id is not None:
        modifiers["group_id"] = note.group_id
    if note.slide_pattern is not None:
        modifiers["slide_pattern"] = note.slide_pattern
    if note.slide_segments:
        legacy_segments = [_json_safe_slide_segment(segment) for segment in note.slide_segments]
        modifiers["slide_segments"] = legacy_segments

    return Note(
        note_type=note.note_type,
        position=str(note.position),
        end_position=(
            str(note.end_position) if note.end_position is not None else None
        ),
        time_sec=note.time_sec,
        beat=float(note.beat),
        tick=note.tick,
        raw=note.raw,
        duration_sec=note.duration_sec,
        duration_beats=(
            float(note.duration_beats) if note.duration_beats is not None else None
        ),
        duration_ticks=note.duration_ticks,
        duration=duration_expr_to_dict(note.duration),
        path=(
            _slide_path(note)
            if note.slide_pattern is not None and note.end_position is not None
            else []
        ),
        segments=[_json_safe_slide_segment(segment) for segment in note.segments],
        modifiers=modifiers,
    )


def parsed_timing_notes_to_chart_notes(
    parsed_points: list[ParsedTimingPointNotes],
) -> list[Note]:
    """Flatten parsed timing-point notes into ChartIR Note objects."""

    return [
        parsed_note_to_chart_note(note)
        for point in parsed_points
        for note in point.notes
    ]


def parsed_timing_point_notes_to_dict(point: ParsedTimingPointNotes) -> dict[str, Any]:
    """Convert parsed timing-point notes to JSON-compatible primitives."""

    return {
        "timing_index": point.timing_index,
        "raw": point.raw,
        "note_text": point.note_text,
        "beat": _format_fraction(point.beat),
        "tick": point.tick,
        "time_sec": point.time_sec,
        "notes": [
            {
                "note_type": note.note_type,
                "position": note.position,
                "is_break": note.is_break,
                "raw": note.raw,
                "time_sec": note.time_sec,
                "beat": _format_fraction(note.beat),
                "tick": note.tick,
                "group_id": note.group_id,
                "duration_beats": (
                    _format_fraction(note.duration_beats)
                    if note.duration_beats is not None
                    else None
                ),
                "duration_ticks": note.duration_ticks,
                "duration_sec": note.duration_sec,
                "duration": duration_expr_to_dict(note.duration),
                "end_position": note.end_position,
                "slide_pattern": note.slide_pattern,
                "slide_segments": [
                    _json_safe_slide_segment(segment)
                    for segment in note.slide_segments
                ],
                "segments": [
                    _json_safe_slide_segment(segment)
                    for segment in note.segments
                ],
                "modifiers": note.modifiers,
            }
            for note in point.notes
        ],
        "unknown_tokens": [
            {
                "raw": token.raw,
                "reason": token.reason,
            }
            for token in point.unknown_tokens
        ],
    }


def parsed_timing_points_notes_to_json(
    parsed_points: list[ParsedTimingPointNotes],
    *,
    indent: int = 2,
) -> str:
    """Serialize parsed timing-point notes to JSON."""

    return json.dumps(
        [parsed_timing_point_notes_to_dict(point) for point in parsed_points],
        ensure_ascii=False,
        indent=indent,
    )


def _split_top_level_slashes(note_text: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    square_depth = 0

    for char in note_text:
        if char == "[":
            square_depth += 1
        elif char == "]" and square_depth > 0:
            square_depth -= 1

        if char == "/" and square_depth == 0:
            tokens.append("".join(current))
            current = []
        else:
            current.append(char)

    tokens.append("".join(current))
    return tokens


def _split_top_level_stars(note_text: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    square_depth = 0

    for char in note_text:
        if char == "[":
            square_depth += 1
        elif char == "]" and square_depth > 0:
            square_depth -= 1

        if char == "*" and square_depth == 0:
            tokens.append("".join(current))
            current = []
        else:
            current.append(char)

    tokens.append("".join(current))
    return tokens


def _slide_segment_from_match(match: re.Match[str], raw: str) -> dict[str, Any]:
    duration = parse_duration_expr(match.group("duration"))
    if duration.kind == "unknown":
        raise ValueError("Unsupported slide duration.")

    pattern = match.group("pattern")
    if not is_supported_slide_pattern(pattern):
        raise ValueError(f"Unsupported slide pattern {pattern!r}.")

    end_position = match.group("end_position")
    path_args = _slide_path_args(pattern, end_position)
    segment: dict[str, Any] = {
        "pattern": pattern,
        "end_position": _slide_end_position(pattern, path_args),
        "path_args": path_args,
        "duration_beats": duration.beats,
        "duration_ticks": duration.ticks,
        "duration_sec": duration.seconds,
        "duration": duration,
        "modifiers": _parse_note_modifier_text(match.group("tail_modifiers")),
        "raw": raw,
    }
    if "position" in match.groupdict():
        segment["position"] = int(match.group("position"))
        head_modifiers = _parse_note_modifier_text(match.group("head_modifiers"))
        tail_modifiers = segment["modifiers"]
        segment["head_modifiers"] = head_modifiers
        segment["is_break"] = bool(head_modifiers["break"] or tail_modifiers["break"])
    return segment


def _compound_slide_segment_from_match(match: re.Match[str], raw: str) -> dict[str, Any]:
    duration = parse_duration_expr(match.group("duration"))
    if duration.kind == "unknown":
        raise ValueError("Unsupported slide duration.")

    path_text = match.group("path_text")
    path_parts = _parse_compound_slide_path(path_text)
    if not path_parts:
        raise ValueError("Unsupported compound slide path.")
    segment: dict[str, Any] = {
        "pattern": "compound",
        "path_text": path_text,
        "path_parts": path_parts,
        "end_position": _compound_slide_end_position(path_parts),
        "path_args": [
            arg
            for part in path_parts
            for arg in part["path_args"]
        ],
        "duration_beats": duration.beats,
        "duration_ticks": duration.ticks,
        "duration_sec": duration.seconds,
        "duration": duration,
        "modifiers": _parse_note_modifier_text(match.group("tail_modifiers")),
        "raw": raw,
    }
    if "position" in match.groupdict():
        head_modifiers = _parse_note_modifier_text(match.group("head_modifiers"))
        tail_modifiers = segment["modifiers"]
        segment["position"] = int(match.group("position"))
        segment["head_modifiers"] = head_modifiers
        segment["is_break"] = bool(head_modifiers["break"] or tail_modifiers["break"])
    return segment


def _json_safe_slide_segment(segment: dict[str, Any]) -> dict[str, Any]:
    safe = dict(segment)
    duration = safe.get("duration")
    if isinstance(duration, DurationExpr):
        safe["duration"] = duration_expr_to_dict(duration)
    duration_beats = safe.get("duration_beats")
    if isinstance(duration_beats, Fraction):
        safe["duration_beats"] = _format_fraction(duration_beats)
    trajectory = safe.get("trajectory")
    if isinstance(trajectory, dict):
        safe["trajectory"] = dict(trajectory)
    return safe


def _slide_path(note: ParsedNote) -> list[str]:
    if not note.slide_segments:
        return [str(note.position), str(note.slide_pattern), str(note.end_position)]

    path = [str(note.position)]
    for index, segment in enumerate(note.slide_segments):
        if index > 0:
            path.append("*")
        if "path_text" in segment:
            path.append(str(segment["path_text"]))
        else:
            path.append(str(segment["pattern"]))
            path.append(str(segment["end_position"]))
    return path


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _note_modifiers(*, is_break: bool = False, is_ex: bool = False) -> dict[str, Any]:
    return {
        "break": is_break,
        "ex": is_ex,
    }


def _parse_note_modifier_text(text: str | None) -> dict[str, Any]:
    text = text or ""
    return _note_modifiers(
        is_break="b" in text,
        is_ex="x" in text,
    )


def _slide_note_modifiers(segments: list[dict[str, Any]]) -> dict[str, Any]:
    is_break = any(bool(segment.get("is_break")) for segment in segments)
    is_ex = any(
        bool(segment.get("head_modifiers", {}).get("ex"))
        or bool(segment.get("modifiers", {}).get("ex"))
        for segment in segments
    )
    return _note_modifiers(is_break=is_break, is_ex=is_ex)


def _touch_position(area: str, position: str) -> str:
    return f"{area}{position}" if position else area


def _slide_path_args(pattern: str, end_position: str) -> list[int]:
    expected_count = get_slide_pattern_definition(pattern).path_arg_count if get_slide_pattern_definition(pattern) else 1
    args = [int(char) for char in end_position]
    if isinstance(expected_count, int) and len(args) != expected_count:
        raise ValueError(
            f"Slide pattern {pattern!r} expects {expected_count} path arg(s), got {len(args)}."
        )
    return args


def _slide_end_position(pattern: str, path_args: list[int]) -> int | None:
    if pattern == "V":
        return path_args[-1] if path_args else None
    return path_args[0] if len(path_args) == 1 else None


def _attach_slide_trajectory_refs(segments: list[dict[str, Any]]) -> None:
    current_start: int | None = None
    for index, segment in enumerate(segments):
        if index == 0:
            current_start = int(segment["position"])
        elif current_start is None:
            current_start = _segment_terminal_position(segments[index - 1])

        if current_start is None:
            continue

        segment["start_position"] = current_start
        segment["trajectory"] = _slide_trajectory_ref(segment, current_start)
        terminal = _segment_terminal_position(segment)
        current_start = terminal if terminal is not None else current_start


def _slide_trajectory_ref(segment: dict[str, Any], start_position: int) -> dict[str, Any]:
    pattern = str(segment["pattern"])
    path_args = [int(value) for value in segment.get("path_args", [])]
    raw_path = _raw_slide_path(segment, start_position)
    return {
        "trajectory_id": raw_path,
        "pattern": pattern,
        "start_position": start_position,
        "end_position": segment.get("end_position"),
        "path_args": path_args,
        "raw": raw_path,
    }


def _raw_slide_path(segment: dict[str, Any], start_position: int) -> str:
    if "path_text" in segment:
        return f"{start_position}{segment['path_text']}"
    return f"{start_position}{segment['pattern']}{''.join(str(arg) for arg in segment.get('path_args', []))}"


def _segment_terminal_position(segment: dict[str, Any]) -> int | None:
    end_position = segment.get("end_position")
    if isinstance(end_position, int):
        return end_position
    path_args = segment.get("path_args")
    if path_args:
        return int(path_args[-1])
    return None


def _parse_compound_slide_path(path_text: str) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    cursor = 0
    patterns = ("pp", "qq", "-", "<", ">", "p", "q", "s", "z", "v", "V", "w")

    while cursor < len(path_text):
        pattern = next(
            (candidate for candidate in patterns if path_text.startswith(candidate, cursor)),
            None,
        )
        if pattern is None:
            return []
        cursor += len(pattern)

        definition = get_slide_pattern_definition(pattern)
        if definition is None or not definition.supported:
            return []

        if definition.path_arg_count == "variable":
            digits = []
            while cursor < len(path_text) and path_text[cursor].isdigit():
                digits.append(int(path_text[cursor]))
                cursor += 1
            if not digits:
                return []
        else:
            end = cursor + definition.path_arg_count
            digit_text = path_text[cursor:end]
            if len(digit_text) != definition.path_arg_count or not digit_text.isdigit():
                return []
            digits = [int(char) for char in digit_text]
            cursor = end

        parts.append(
            {
                "pattern": pattern,
                "path_args": digits,
                "end_position": _slide_end_position(pattern, digits),
            }
        )

    return parts


def _compound_slide_end_position(parts: list[dict[str, Any]]) -> int | None:
    if not parts:
        return None
    return _segment_terminal_position(parts[-1])


def _combine_segment_durations(
    segments: list[dict[str, Any]],
    raw: str,
) -> DurationExpr:
    durations = [
        segment["duration"]
        for segment in segments
        if isinstance(segment.get("duration"), DurationExpr)
    ]
    beats = (
        sum((duration.beats for duration in durations), Fraction(0, 1))
        if durations and all(duration.beats is not None for duration in durations)
        else None
    )
    seconds = (
        sum((duration.seconds for duration in durations), 0.0)
        if durations and all(duration.seconds is not None for duration in durations)
        else None
    )
    kind = durations[0].kind if len(durations) == 1 else "compound"
    values = (
        list(durations[0].values)
        if len(durations) == 1 and durations[0].values is not None
        else None
    )
    return DurationExpr(
        raw=durations[0].raw if len(durations) == 1 else raw,
        kind=kind,
        beats=beats,
        seconds=seconds,
        ticks=duration_to_ticks_or_none(beats) if beats is not None else None,
        values=values,
    )


def _apply_duration_context(note: ParsedNote, bpm: float) -> None:
    if note.slide_segments:
        for segment in note.slide_segments:
            duration = segment.get("duration")
            if isinstance(duration, DurationExpr):
                resolved = resolve_duration_expr(duration, bpm)
                segment["duration"] = resolved
                segment["duration_beats"] = resolved.beats
                segment["duration_ticks"] = resolved.ticks
                segment["duration_sec"] = resolved.seconds
        note.duration = _combine_segment_durations(note.slide_segments, note.raw)
    elif note.duration is not None:
        note.duration = resolve_duration_expr(note.duration, bpm)

    if note.duration is not None:
        note.duration_beats = note.duration.beats
        note.duration_ticks = note.duration.ticks
        note.duration_sec = note.duration.seconds
