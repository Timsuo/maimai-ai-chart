"""Maidata exporter for V1 ChartIR objects."""

from __future__ import annotations

from collections import defaultdict

from maichart.ir import ChartIR, Note


def export_chart_ir_to_maidata(chart: ChartIR) -> str:
    """Export one single-difficulty ChartIR object to Maidata-like text."""

    difficulty_index = chart.difficulty.index or 1
    lines = [
        f"&title={chart.metadata.title or ''}",
        f"&artist={chart.metadata.artist or ''}",
        f"&first={_format_number(chart.metadata.offset or 0)}",
        f"&wholebpm={_format_number(_initial_bpm(chart))}",
        f"&lv_{difficulty_index}={chart.difficulty.level or ''}",
        f"&des_{difficulty_index}={chart.difficulty.designer or ''}",
    ]

    raw_inote = _extract_raw_inote(chart.raw)
    if raw_inote is not None:
        lines.append(raw_inote.rstrip("\r\n"))
    else:
        lines.append(f"&inote_{difficulty_index}={_render_notes_as_inote(chart.notes)}")

    return "\n".join(lines) + "\n"


def _extract_raw_inote(raw: str | None) -> str | None:
    if raw is None:
        return None
    stripped = raw.lstrip()
    if stripped.startswith("&inote_"):
        return stripped
    return None


def _render_notes_as_inote(notes: list[Note]) -> str:
    if not notes:
        return "E"

    by_tick: dict[int, list[Note]] = defaultdict(list)
    for note in notes:
        by_tick[note.tick or 0].append(note)

    tokens: list[str] = []
    for tick in sorted(by_tick):
        point_notes = sorted(by_tick[tick], key=lambda note: note.raw or "")
        tokens.append("/".join(_render_note(note) for note in point_notes))
    tokens.append("E")
    return ",".join(tokens)


def _render_note(note: Note) -> str:
    if note.raw:
        return note.raw
    if note.note_type == "tap":
        suffix = "b" if note.modifiers.get("break") else ""
        return f"{note.position or ''}{suffix}"
    if note.note_type == "hold":
        return f"{note.position or ''}h[4:1]"
    if note.note_type == "slide":
        pattern = note.modifiers.get("slide_pattern", "-")
        return f"{note.position or ''}{pattern}{note.end_position or ''}[4:1]"
    return note.raw or ""


def _initial_bpm(chart: ChartIR) -> float:
    if chart.timing.bpms:
        bpm = chart.timing.bpms[0].bpm
        if bpm is not None:
            return bpm
    return 120.0


def _format_number(value: float | int) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return str(numeric)
