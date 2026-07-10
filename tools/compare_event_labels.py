"""Compare legacy ChartIR frame labels with EventIR frame labels v2."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from maichart.event_ir_converter import chart_ir_to_event_ir  # noqa: E402
from maichart.event_labels import build_frame_labels_from_event_ir  # noqa: E402
from maichart.labels import build_frame_labels_from_chart_ir, frame_labels_to_dict  # noqa: E402
from maichart.serialization import load_chart_json  # noqa: E402


DIFF_FIELDS = (
    "note_count",
    "positions",
    "slide_start_count",
    "slide_active_count",
    "hold_active_count",
    "slide_patterns",
)
SUMMARY_FIELDS = (
    "manifest",
    "slide_launch_policy",
    "limit",
    "sample_count",
    "total_frames",
    "note_count_diff_frames",
    "positions_diff_frames",
    "slide_start_count_diff_frames",
    "slide_active_count_diff_frames",
    "hold_active_count_diff_frames",
    "pattern_diff_frames",
    "v2_slide_head_count",
    "v2_slide_motion_start_count",
    "v2_slide_motion_active_count",
    "v2_slide_motion_end_count",
    "v2_slide_unknown_launch_offset_count",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare legacy frame labels with EventIR frame_labels_v2.",
    )
    parser.add_argument("manifest", help="dataset or training manifest JSON")
    parser.add_argument("--cache-dir", default="cache", help="cache root for fallback ChartIR paths")
    parser.add_argument("--out", required=True, help="output report directory")
    parser.add_argument("--division", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="maximum usable difficulties to compare")
    parser.add_argument(
        "--slide-launch-policy",
        choices=("legacy", "unknown", "default_one_beat"),
        default="legacy",
    )
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest).resolve()
    manifest = _load_json(manifest_path)
    manifest_base = manifest_path.parent
    cache_dir = _resolve_path(args.cache_dir, manifest_base)
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    per_sample_rows: list[dict[str, Any]] = []
    lifecycle_rows: list[dict[str, Any]] = []
    diff_detail_rows: list[dict[str, Any]] = []
    summary = _empty_summary(
        manifest_path=manifest_path,
        slide_launch_policy=args.slide_launch_policy,
        limit=args.limit,
    )

    for sample in _iter_usable_difficulties(
        manifest,
        manifest_base,
        cache_dir,
        limit=args.limit,
        slide_launch_policy=args.slide_launch_policy,
    ):
        chart = load_chart_json(sample["chart_ir_path"])
        event_ir = chart_ir_to_event_ir(chart)
        v1 = frame_labels_to_dict(
            build_frame_labels_from_chart_ir(chart, division=args.division)
        )
        v2 = build_frame_labels_from_event_ir(
            event_ir,
            division=args.division,
            slide_launch_policy=args.slide_launch_policy,
        )
        row, details = _compare_sample(sample, chart, event_ir, v1, v2)
        lifecycle_row = _slide_lifecycle_row(sample, v2)
        per_sample_rows.append(row)
        diff_detail_rows.extend(details)
        lifecycle_rows.append(lifecycle_row)
        _accumulate_summary(summary, row, lifecycle_row)

    summary["sample_count"] = len(per_sample_rows)
    _write_json(out_dir / "summary.json", summary)
    _write_csv(out_dir / "summary.csv", [summary], SUMMARY_FIELDS)
    _write_csv(out_dir / "per_sample_diff.csv", per_sample_rows, _per_sample_fields())
    _write_csv(out_dir / "diff_details.csv", diff_detail_rows, _diff_detail_fields())
    _write_csv(
        out_dir / "slide_lifecycle_stats.csv",
        lifecycle_rows,
        _lifecycle_fields(),
    )

    return 0


def _iter_usable_difficulties(
    manifest: dict[str, Any],
    manifest_base: Path,
    cache_dir: Path,
    *,
    limit: int | None,
    slide_launch_policy: str,
):
    songs = manifest.get("songs")
    if not isinstance(songs, list):
        raise ValueError("Manifest must contain a 'songs' list.")

    yielded = 0
    for song in songs:
        if not isinstance(song, dict):
            continue
        song_id = str(song.get("song_id") or "unknown")
        for difficulty in song.get("difficulties") or []:
            if not isinstance(difficulty, dict):
                continue
            if difficulty.get("usable_for_training") is False:
                continue
            difficulty_index = int(difficulty.get("difficulty_index", difficulty.get("index", 0)))
            fallback = (
                cache_dir
                / "chart_ir"
                / song_id
                / f"difficulty_{difficulty_index}.chart_ir.json"
            )
            chart_ir_path = _difficulty_path(
                difficulty,
                "chart_ir_path",
                fallback,
                manifest_base,
            )
            if not chart_ir_path.is_file():
                continue
            if limit is not None and yielded >= limit:
                return
            yielded += 1
            yield {
                "song_id": song_id,
                "difficulty_index": difficulty_index,
                "chart_ir_path": chart_ir_path,
                "slide_launch_policy": slide_launch_policy,
            }


def _compare_sample(
    sample: dict[str, Any],
    chart: Any,
    event_ir: Any,
    v1: dict[str, Any],
    v2: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    v1_frames = v1.get("frames") or []
    v2_frames = v2.get("frames") or []
    frame_count = max(len(v1_frames), len(v2_frames))
    row = {
        "song_id": sample["song_id"],
        "difficulty_index": sample["difficulty_index"],
        "slide_launch_policy": sample["slide_launch_policy"],
        "chart_ir_path": str(sample["chart_ir_path"]),
        "total_frames": frame_count,
        "note_count_diff_frames": 0,
        "positions_diff_frames": 0,
        "slide_start_count_diff_frames": 0,
        "slide_active_count_diff_frames": 0,
        "hold_active_count_diff_frames": 0,
        "pattern_diff_frames": 0,
    }
    details: list[dict[str, Any]] = []

    for index in range(frame_count):
        v1_labels = _frame_labels(v1_frames, index)
        v2_labels = _frame_labels(v2_frames, index)
        if _value(v1_labels, "note_count") != _value(v2_labels, "note_count"):
            row["note_count_diff_frames"] += 1
            details.append(_diff_detail(sample, chart, event_ir, index, "note_count", v1_labels, v2_labels))
        if _list(v1_labels.get("positions")) != _list(v2_labels.get("positions")):
            row["positions_diff_frames"] += 1
            details.append(_diff_detail(sample, chart, event_ir, index, "positions", v1_labels, v2_labels))
        if _value(v1_labels, "slide_start_count") != _value(v2_labels, "slide_start_count"):
            row["slide_start_count_diff_frames"] += 1
            details.append(_diff_detail(sample, chart, event_ir, index, "slide_start_count", v1_labels, v2_labels))
        if _value(v1_labels, "slide_active_count") != _value(v2_labels, "slide_active_count"):
            row["slide_active_count_diff_frames"] += 1
            details.append(_diff_detail(sample, chart, event_ir, index, "slide_active_count", v1_labels, v2_labels))
        if _value(v1_labels, "hold_active_count") != _value(v2_labels, "hold_active_count"):
            row["hold_active_count_diff_frames"] += 1
            details.append(_diff_detail(sample, chart, event_ir, index, "hold_active_count", v1_labels, v2_labels))
        if _list(v1_labels.get("slide_patterns")) != _list(v2_labels.get("slide_patterns")):
            row["pattern_diff_frames"] += 1
            details.append(_diff_detail(sample, chart, event_ir, index, "slide_patterns", v1_labels, v2_labels))

    return row, details


def _slide_lifecycle_row(sample: dict[str, Any], v2: dict[str, Any]) -> dict[str, Any]:
    row = {
        "song_id": sample["song_id"],
        "difficulty_index": sample["difficulty_index"],
        "slide_launch_policy": sample["slide_launch_policy"],
        "total_frames": len(v2.get("frames") or []),
        "v2_slide_head_count": 0,
        "v2_slide_motion_start_count": 0,
        "v2_slide_motion_active_count": 0,
        "v2_slide_motion_end_count": 0,
        "v2_slide_unknown_launch_offset_count": 0,
    }
    for frame in v2.get("frames") or []:
        labels = frame.get("labels") or {}
        row["v2_slide_head_count"] += int(labels.get("slide_head_count") or 0)
        row["v2_slide_motion_start_count"] += int(labels.get("slide_motion_start_count") or 0)
        row["v2_slide_motion_active_count"] += int(labels.get("slide_motion_active_count") or 0)
        row["v2_slide_motion_end_count"] += int(labels.get("slide_motion_end_count") or 0)
        row["v2_slide_unknown_launch_offset_count"] += int(
            labels.get("slide_unknown_launch_offset_count") or 0
        )
    return row


def _accumulate_summary(
    summary: dict[str, Any],
    diff_row: dict[str, Any],
    lifecycle_row: dict[str, Any],
) -> None:
    for field in (
        "total_frames",
        "note_count_diff_frames",
        "positions_diff_frames",
        "slide_start_count_diff_frames",
        "slide_active_count_diff_frames",
        "hold_active_count_diff_frames",
        "pattern_diff_frames",
    ):
        summary[field] += int(diff_row[field])
    for field in (
        "v2_slide_head_count",
        "v2_slide_motion_start_count",
        "v2_slide_motion_active_count",
        "v2_slide_motion_end_count",
        "v2_slide_unknown_launch_offset_count",
    ):
        summary[field] += int(lifecycle_row[field])


def _empty_summary(
    *,
    manifest_path: Path,
    slide_launch_policy: str,
    limit: int | None,
) -> dict[str, Any]:
    summary = {field: 0 for field in SUMMARY_FIELDS}
    summary["manifest"] = str(manifest_path)
    summary["slide_launch_policy"] = slide_launch_policy
    summary["limit"] = "" if limit is None else limit
    return summary


def _diff_detail(
    sample: dict[str, Any],
    chart: Any,
    event_ir: Any,
    frame_index: int,
    field_name: str,
    v1_labels: dict[str, Any],
    v2_labels: dict[str, Any],
) -> dict[str, Any]:
    tick = frame_index * 480
    return {
        "song_id": sample["song_id"],
        "difficulty_index": sample["difficulty_index"],
        "slide_launch_policy": sample["slide_launch_policy"],
        "frame_index": frame_index,
        "field_name": field_name,
        "v1_value": json.dumps(v1_labels.get(field_name), ensure_ascii=False),
        "v2_value": json.dumps(v2_labels.get(field_name), ensure_ascii=False),
        "nearby_note_raw": ";".join(_nearby_note_raw(chart, tick)),
        "nearby_event_raw": ";".join(_nearby_event_raw(event_ir, tick)),
    }


def _nearby_note_raw(chart: Any, tick: int) -> list[str]:
    values: list[str] = []
    for note in getattr(chart, "notes", []):
        note_tick = getattr(note, "tick", None)
        if note_tick is None or abs(int(note_tick) - tick) > 960:
            continue
        values.append(str(getattr(note, "raw", None) or getattr(note, "note_type", "")))
    return values[:8]


def _nearby_event_raw(event_ir: Any, tick: int) -> list[str]:
    values: list[str] = []
    for event in getattr(event_ir, "events", []):
        event_tick = getattr(event, "tick", getattr(event, "head_tick", None))
        if event_tick is None or abs(int(event_tick) - tick) > 960:
            continue
        values.append(str(getattr(event, "raw_notation", None) or type(event).__name__))
    return values[:8]


def _frame_labels(frames: list[dict[str, Any]], index: int) -> dict[str, Any]:
    if index >= len(frames):
        return {}
    labels = frames[index].get("labels")
    return labels if isinstance(labels, dict) else {}


def _difficulty_path(
    difficulty: dict[str, Any],
    key: str,
    fallback: Path,
    manifest_base: Path,
) -> Path:
    return _resolve_path(difficulty.get(key) or fallback, manifest_base)


def _resolve_path(path: str | Path, manifest_base: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (manifest_base / candidate).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _per_sample_fields() -> tuple[str, ...]:
    return (
        "song_id",
        "difficulty_index",
        "slide_launch_policy",
        "chart_ir_path",
        "total_frames",
        "note_count_diff_frames",
        "positions_diff_frames",
        "slide_start_count_diff_frames",
        "slide_active_count_diff_frames",
        "hold_active_count_diff_frames",
        "pattern_diff_frames",
    )


def _lifecycle_fields() -> tuple[str, ...]:
    return (
        "song_id",
        "difficulty_index",
        "slide_launch_policy",
        "total_frames",
        "v2_slide_head_count",
        "v2_slide_motion_start_count",
        "v2_slide_motion_active_count",
        "v2_slide_motion_end_count",
        "v2_slide_unknown_launch_offset_count",
    )


def _diff_detail_fields() -> tuple[str, ...]:
    return (
        "song_id",
        "difficulty_index",
        "slide_launch_policy",
        "frame_index",
        "field_name",
        "v1_value",
        "v2_value",
        "nearby_note_raw",
        "nearby_event_raw",
    )


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _value(labels: dict[str, Any], field_name: str) -> Any:
    return labels.get(field_name, 0)


if __name__ == "__main__":
    raise SystemExit(main())
