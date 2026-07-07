"""JSON serialization helpers for ChartIR."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from maichart.ir import (
    BpmEvent,
    ChartIR,
    ChartMetadata,
    DifficultyMetadata,
    GridEvent,
    HSpeedEvent,
    Note,
    TimingData,
    UnknownChartToken,
)


def chart_to_dict(chart: ChartIR) -> dict[str, Any]:
    """Convert a chart IR object into JSON-compatible primitives."""

    return asdict(chart)


def chart_from_dict(data: dict[str, Any]) -> ChartIR:
    """Load a chart IR object from JSON-compatible primitives."""

    metadata_data = data.get("metadata") or {}
    difficulty_data = data.get("difficulty") or {}
    timing_data = data.get("timing") or {}

    timing = TimingData(
        bpms=[BpmEvent(**event) for event in timing_data.get("bpms", [])],
        grids=[GridEvent(**event) for event in timing_data.get("grids", [])],
        hspeeds=[HSpeedEvent(**event) for event in timing_data.get("hspeeds", [])],
    )

    return ChartIR(
        schema_version=int(data.get("schema_version", 1)),
        metadata=ChartMetadata(**metadata_data),
        difficulty=DifficultyMetadata(**difficulty_data),
        timing=timing,
        notes=[Note(**note) for note in data.get("notes", [])],
        unknown_tokens=[
            UnknownChartToken(**token)
            for token in data.get("unknown_tokens", [])
        ],
        raw=data.get("raw"),
    )


def chart_to_json(chart: ChartIR, *, indent: int = 2) -> str:
    """Serialize a chart IR object to a JSON string."""

    return json.dumps(chart_to_dict(chart), ensure_ascii=False, indent=indent)


def chart_from_json(payload: str) -> ChartIR:
    """Deserialize a chart IR object from a JSON string."""

    data = json.loads(payload)
    if not isinstance(data, dict):
        raise TypeError("ChartIR JSON must decode to an object.")
    return chart_from_dict(data)


def save_chart_json(chart: ChartIR, path: str | Path, *, indent: int = 2) -> None:
    """Write a chart IR object to a JSON file."""

    Path(path).write_text(chart_to_json(chart, indent=indent), encoding="utf-8")


def load_chart_json(path: str | Path) -> ChartIR:
    """Read a chart IR object from a JSON file."""

    return chart_from_json(Path(path).read_text(encoding="utf-8"))
