"""Build side-path EventIR and frame_labels_v2 cache files for a manifest."""

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

from maichart.event_ir import event_ir_to_dict  # noqa: E402
from maichart.event_ir_converter import chart_ir_to_event_ir  # noqa: E402
from maichart.event_labels import (  # noqa: E402
    EVENT_FRAME_LABELS_SCHEMA,
    build_frame_labels_from_event_ir,
)
from maichart.serialization import load_chart_json  # noqa: E402


SUMMARY_FIELDS = (
    "manifest",
    "sample_count",
    "written_event_ir_count",
    "written_frame_labels_v2_count",
    "skipped_count",
    "error_count",
    "slide_launch_policy",
    "total_frames",
    "total_slide_heads",
    "total_slide_unknown_launch_offset_count",
    "unknown_launch_offset_per_slide_head",
)
ERROR_FIELDS = (
    "song_id",
    "difficulty",
    "chart_ir_path",
    "error_type",
    "error_message",
)
POLICIES = ("legacy", "unknown", "default_one_beat")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build EventIR and frame_labels_v2 side-path cache files.",
    )
    parser.add_argument("manifest", help="training manifest JSON")
    parser.add_argument(
        "--cache-dir",
        default="cache",
        help="legacy cache root used as a fallback for ChartIR paths",
    )
    parser.add_argument(
        "--out-cache-dir",
        required=True,
        help="output cache root for event_ir and frame_labels_v2",
    )
    parser.add_argument(
        "--out-manifest",
        required=True,
        help="output manifest copy with event frame-label fields",
    )
    parser.add_argument("--division", type=int, default=16)
    parser.add_argument(
        "--slide-launch-policy",
        choices=POLICIES,
        default="legacy",
        help="policy for EventIR slides with unknown launch offsets",
    )
    args = parser.parse_args(argv)

    result = build_event_frame_labels_cache(
        manifest_path=args.manifest,
        cache_dir=args.cache_dir,
        out_cache_dir=args.out_cache_dir,
        out_manifest_path=args.out_manifest,
        division=args.division,
        slide_launch_policy=args.slide_launch_policy,
    )
    print(
        "Event frame-label cache build complete: "
        f"samples={result['summary']['sample_count']} "
        f"event_ir={result['summary']['written_event_ir_count']} "
        f"frame_labels_v2={result['summary']['written_frame_labels_v2_count']} "
        f"errors={result['summary']['error_count']}"
    )
    return 0


def build_event_frame_labels_cache(
    *,
    manifest_path: str | Path,
    cache_dir: str | Path,
    out_cache_dir: str | Path,
    out_manifest_path: str | Path,
    division: int = 16,
    slide_launch_policy: str = "legacy",
) -> dict[str, Any]:
    if slide_launch_policy not in POLICIES:
        raise ValueError(f"slide_launch_policy must be one of: {', '.join(POLICIES)}")

    manifest_file = Path(manifest_path).resolve()
    manifest_base = manifest_file.parent
    cache_root = _resolve_path(cache_dir, manifest_base)
    out_cache_root = _resolve_path(out_cache_dir, Path.cwd())
    out_manifest_file = Path(out_manifest_path).resolve()
    out_manifest_base = out_manifest_file.parent

    manifest = _load_json(manifest_file)
    summary = _empty_summary(manifest_file, slide_launch_policy)
    errors: list[dict[str, Any]] = []

    songs = manifest.get("songs")
    if not isinstance(songs, list):
        raise ValueError("Manifest must contain a 'songs' list.")

    for song in songs:
        if not isinstance(song, dict):
            continue
        song_id = str(song.get("song_id") or "unknown")
        for difficulty in song.get("difficulties") or []:
            if not isinstance(difficulty, dict):
                continue
            difficulty_index = _difficulty_index(difficulty)
            if difficulty.get("usable_for_training") is not True:
                summary["skipped_count"] += 1
                continue
            summary["sample_count"] += 1

            chart_ir_path = _chart_ir_path(
                difficulty=difficulty,
                song_id=song_id,
                difficulty_index=difficulty_index,
                manifest_base=manifest_base,
                cache_root=cache_root,
            )
            try:
                chart = load_chart_json(chart_ir_path)
                event_ir = chart_ir_to_event_ir(chart)
                event_payload = event_ir_to_dict(event_ir)
                labels = build_frame_labels_from_event_ir(
                    event_ir,
                    division=division,
                    slide_launch_policy=slide_launch_policy,
                )
                labels["song_id"] = song_id

                event_ir_path = (
                    out_cache_root
                    / "event_ir"
                    / song_id
                    / f"difficulty_{difficulty_index}.event_ir.json"
                )
                labels_path = (
                    out_cache_root
                    / "frame_labels_v2"
                    / song_id
                    / f"difficulty_{difficulty_index}.frame_labels_v2.json"
                )
                _write_json(event_ir_path, event_payload)
                _write_json(labels_path, labels)

                difficulty["event_frame_labels_path"] = _display_path(
                    labels_path,
                    out_manifest_base,
                )
                difficulty["event_frame_labels_schema"] = EVENT_FRAME_LABELS_SCHEMA
                difficulty["event_frame_labels_policy"] = slide_launch_policy

                summary["written_event_ir_count"] += 1
                summary["written_frame_labels_v2_count"] += 1
                _accumulate_label_summary(summary, labels)
            except Exception as exc:  # noqa: BLE001 - batch builds should continue.
                errors.append(
                    {
                        "song_id": song_id,
                        "difficulty": difficulty_index,
                        "chart_ir_path": str(chart_ir_path),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )

    summary["error_count"] = len(errors)
    summary["unknown_launch_offset_per_slide_head"] = _ratio(
        summary["total_slide_unknown_launch_offset_count"],
        summary["total_slide_heads"],
    )

    out_manifest_file.parent.mkdir(parents=True, exist_ok=True)
    out_manifest_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    _write_json(out_cache_root / "event_frame_labels_build_summary.json", summary)
    _write_csv(
        out_cache_root / "event_frame_labels_build_summary.csv",
        [summary],
        SUMMARY_FIELDS,
    )
    errors_path = out_cache_root / "event_frame_labels_build_errors.csv"
    if errors:
        _write_csv(errors_path, errors, ERROR_FIELDS)
    elif errors_path.exists():
        errors_path.unlink()

    return {
        "manifest": manifest,
        "summary": summary,
        "errors": errors,
    }


def _chart_ir_path(
    *,
    difficulty: dict[str, Any],
    song_id: str,
    difficulty_index: int,
    manifest_base: Path,
    cache_root: Path,
) -> Path:
    fallback = (
        cache_root
        / "chart_ir"
        / song_id
        / f"difficulty_{difficulty_index}.chart_ir.json"
    )
    explicit = difficulty.get("chart_ir_path")
    if explicit is None:
        return fallback

    candidate = _resolve_path(explicit, manifest_base)
    if candidate.is_file():
        return candidate
    if fallback.is_file():
        return fallback
    return candidate


def _accumulate_label_summary(summary: dict[str, Any], labels: dict[str, Any]) -> None:
    frames = labels.get("frames") or []
    summary["total_frames"] += len(frames)
    for frame in frames:
        frame_labels = frame.get("labels") if isinstance(frame, dict) else None
        if not isinstance(frame_labels, dict):
            continue
        summary["total_slide_heads"] += int(frame_labels.get("slide_head_count") or 0)
        summary["total_slide_unknown_launch_offset_count"] += int(
            frame_labels.get("slide_unknown_launch_offset_count") or 0
        )


def _empty_summary(manifest_file: Path, slide_launch_policy: str) -> dict[str, Any]:
    summary = {field: 0 for field in SUMMARY_FIELDS}
    summary["manifest"] = str(manifest_file)
    summary["slide_launch_policy"] = slide_launch_policy
    summary["unknown_launch_offset_per_slide_head"] = 0.0
    return summary


def _difficulty_index(difficulty: dict[str, Any]) -> int:
    value = difficulty.get("difficulty_index", difficulty.get("index"))
    if value is None:
        return 0
    return int(value)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _resolve_path(path: str | Path, base_path: Path) -> Path:
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


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


if __name__ == "__main__":
    raise SystemExit(main())
