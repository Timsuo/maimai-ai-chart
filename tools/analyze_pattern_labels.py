from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from maichart.training.frame_label_codec import (  # noqa: E402
    FRAME_PATTERN_TYPES,
    derive_frame_pattern,
)

CHORD_SIZE_CLASS_NAMES = {
    0: "none",
    1: "single",
    2: "double",
    3: "three_or_more",
}
TOUCH_PATTERNS = ("touch", "touch_hold")
SLIDE_PATTERNS = ("active_slide", "single_slide", "slide_chord", "tap_slide_mix")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    source = Path(args.input).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    refs = _collect_frame_label_refs(
        source,
        include_unusable=args.include_unusable,
        cache_dir=Path(args.cache_dir).resolve() if args.cache_dir else None,
    )
    if not refs:
        raise SystemExit(f"No frame-label files found in {source}.")

    stats = analyze_frame_label_refs(refs, long_tail_min_count=args.long_tail_min_count)
    _write_outputs(out_dir, stats)
    print(
        "Pattern label stats: "
        f"samples={stats['sample_count']} "
        f"frames={stats['total_frames']} "
        f"activity={stats['activity_count']} "
        f"note_starts={stats['note_start_count']} "
        f"other_mix={stats['other_mix_count']}",
        flush=True,
    )
    print(f"Wrote pattern label stats to {out_dir}", flush=True)
    return 0


def analyze_frame_label_refs(
    refs: list[dict[str, Any]],
    *,
    long_tail_min_count: int = 10,
) -> dict[str, Any]:
    pattern_counts: Counter[str] = Counter()
    chord_counts: Counter[int] = Counter()
    sample_summaries: list[dict[str, Any]] = []
    total_frames = 0
    activity_count = 0
    note_start_count = 0

    for ref in refs:
        payload = _load_json(Path(ref["path"]))
        frames = payload.get("frames")
        if not isinstance(frames, list):
            continue

        sample_patterns: Counter[str] = Counter()
        sample_chords: Counter[int] = Counter()
        sample_activity = 0
        sample_note_starts = 0
        for frame in frames:
            labels = frame.get("labels") if isinstance(frame, dict) else None
            if not isinstance(labels, dict):
                continue
            derived = derive_frame_pattern(labels)
            pattern_type = str(derived["pattern_type"])
            chord_size_class = int(derived["chord_size_class"])

            pattern_counts[pattern_type] += 1
            chord_counts[chord_size_class] += 1
            sample_patterns[pattern_type] += 1
            sample_chords[chord_size_class] += 1
            total_frames += 1
            if bool(derived["activity_presence"]):
                activity_count += 1
                sample_activity += 1
            if bool(derived["note_start_presence"]):
                note_start_count += 1
                sample_note_starts += 1

        sample_total = sum(sample_patterns.values())
        if sample_total == 0:
            continue
        summary = {
            "song_id": ref.get("song_id") or payload.get("song_id"),
            "difficulty_index": ref.get("difficulty_index") or payload.get("difficulty"),
            "level": ref.get("level"),
            "path": str(ref["path"]),
            "total_frames": sample_total,
            "activity_count": sample_activity,
            "activity_ratio": _ratio(sample_activity, sample_total),
            "note_start_count": sample_note_starts,
            "note_start_ratio": _ratio(sample_note_starts, sample_total),
            "other_mix_count": sample_patterns["other_mix"],
            "other_mix_ratio": _ratio(sample_patterns["other_mix"], sample_total),
            "touch_count": sum(sample_patterns[name] for name in TOUCH_PATTERNS),
            "touch_ratio": _ratio(sum(sample_patterns[name] for name in TOUCH_PATTERNS), sample_total),
            "slide_count": sum(sample_patterns[name] for name in SLIDE_PATTERNS),
            "slide_ratio": _ratio(sum(sample_patterns[name] for name in SLIDE_PATTERNS), sample_total),
        }
        for pattern_type in FRAME_PATTERN_TYPES:
            summary[f"pattern_{pattern_type}"] = sample_patterns[pattern_type]
        for chord_class in sorted(CHORD_SIZE_CLASS_NAMES):
            summary[f"chord_size_class_{chord_class}"] = sample_chords[chord_class]
        sample_summaries.append(summary)

    long_tail_patterns = [
        {
            "pattern_type": pattern_type,
            "count": pattern_counts[pattern_type],
            "ratio": _ratio(pattern_counts[pattern_type], total_frames),
        }
        for pattern_type in FRAME_PATTERN_TYPES
        if 0 < pattern_counts[pattern_type] < long_tail_min_count
    ]
    other_mix_count = pattern_counts["other_mix"]
    touch_count = sum(pattern_counts[name] for name in TOUCH_PATTERNS)
    slide_count = sum(pattern_counts[name] for name in SLIDE_PATTERNS)

    return {
        "sample_count": len(sample_summaries),
        "total_frames": total_frames,
        "activity_count": activity_count,
        "activity_ratio": _ratio(activity_count, total_frames),
        "note_start_count": note_start_count,
        "note_start_ratio": _ratio(note_start_count, total_frames),
        "pattern_distribution": [
            {
                "pattern_type": pattern_type,
                "count": pattern_counts[pattern_type],
                "ratio": _ratio(pattern_counts[pattern_type], total_frames),
            }
            for pattern_type in FRAME_PATTERN_TYPES
        ],
        "chord_size_distribution": [
            {
                "chord_size_class": chord_class,
                "name": CHORD_SIZE_CLASS_NAMES[chord_class],
                "count": chord_counts[chord_class],
                "ratio": _ratio(chord_counts[chord_class], total_frames),
            }
            for chord_class in sorted(CHORD_SIZE_CLASS_NAMES)
        ],
        "long_tail_min_count": int(long_tail_min_count),
        "long_tail_pattern_count": len(long_tail_patterns),
        "long_tail_patterns": long_tail_patterns,
        "other_mix_count": other_mix_count,
        "other_mix_ratio": _ratio(other_mix_count, total_frames),
        "touch_pattern_count": touch_count,
        "touch_pattern_ratio": _ratio(touch_count, total_frames),
        "slide_pattern_count": slide_count,
        "slide_pattern_ratio": _ratio(slide_count, total_frames),
        "sample_summaries": sample_summaries,
    }


def _collect_frame_label_refs(
    source: Path,
    *,
    include_unusable: bool,
    cache_dir: Path | None,
) -> list[dict[str, Any]]:
    if source.is_dir():
        return [
            {"path": path}
            for path in sorted(source.rglob("*.frame_labels.json"))
            if path.is_file()
        ]
    manifest = _load_json(source)
    manifest_base = source.parent
    refs: list[dict[str, Any]] = []
    for song in manifest.get("songs") or []:
        if not isinstance(song, dict):
            continue
        song_id = str(song.get("song_id") or "")
        for difficulty in song.get("difficulties") or []:
            if not isinstance(difficulty, dict):
                continue
            if not include_unusable and difficulty.get("usable_for_training") is False:
                continue
            difficulty_index = _difficulty_index(difficulty)
            path = _resolve_frame_labels_path(
                difficulty.get("frame_labels_path"),
                manifest_base=manifest_base,
                manifest_cache_dir=manifest.get("cache_dir"),
                cache_dir=cache_dir,
                song_id=song_id,
                difficulty_index=difficulty_index,
            )
            if path is None:
                continue
            refs.append(
                {
                    "path": path,
                    "song_id": song_id,
                    "difficulty_index": difficulty_index,
                    "level": difficulty.get("level"),
                }
            )
    return refs


def _resolve_frame_labels_path(
    value: Any,
    *,
    manifest_base: Path,
    manifest_cache_dir: Any,
    cache_dir: Path | None,
    song_id: str,
    difficulty_index: int,
) -> Path | None:
    candidates: list[Path] = []
    if value:
        raw = Path(str(value))
        candidates.append(raw if raw.is_absolute() else (manifest_base / raw).resolve())
    for root in _cache_roots(manifest_cache_dir, cache_dir):
        candidates.append(root / "frame_labels" / song_id / f"difficulty_{difficulty_index}.frame_labels.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _cache_roots(manifest_cache_dir: Any, cache_dir: Path | None) -> list[Path]:
    roots: list[Path] = []
    if cache_dir is not None:
        roots.append(cache_dir)
    if manifest_cache_dir:
        raw = Path(str(manifest_cache_dir))
        if raw.is_absolute():
            roots.append(raw)
            roots.append(ROOT / raw.name)
        else:
            roots.append((ROOT / raw).resolve())
    roots.extend([ROOT / "cache_qc_fixed", ROOT / "cache"])

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(root.resolve())
    return unique


def _write_outputs(out_dir: Path, stats: dict[str, Any]) -> None:
    _write_json(out_dir / "pattern_distribution.json", stats)
    _write_csv(
        out_dir / "pattern_distribution.csv",
        stats["pattern_distribution"],
        ["pattern_type", "count", "ratio"],
    )
    _write_csv(
        out_dir / "chord_size_distribution.csv",
        stats["chord_size_distribution"],
        ["chord_size_class", "name", "count", "ratio"],
    )
    sample_fields = [
        "song_id",
        "difficulty_index",
        "level",
        "path",
        "total_frames",
        "activity_count",
        "activity_ratio",
        "note_start_count",
        "note_start_ratio",
        "other_mix_count",
        "other_mix_ratio",
        "touch_count",
        "touch_ratio",
        "slide_count",
        "slide_ratio",
    ]
    sample_fields.extend(f"pattern_{pattern_type}" for pattern_type in FRAME_PATTERN_TYPES)
    sample_fields.extend(f"chord_size_class_{chord_class}" for chord_class in sorted(CHORD_SIZE_CLASS_NAMES))
    _write_csv(out_dir / "sample_pattern_summary.csv", stats["sample_summaries"], sample_fields)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _difficulty_index(difficulty: dict[str, Any]) -> int:
    value = difficulty.get("difficulty_index", difficulty.get("index"))
    return int(value) if value is not None else 0


def _ratio(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze derived frame-level pattern labels.")
    parser.add_argument(
        "input",
        help="Training manifest JSON path or a cache/frame_labels directory.",
    )
    parser.add_argument("--out", required=True, help="Output directory for JSON/CSV stats.")
    parser.add_argument(
        "--cache-dir",
        help="Optional cache root used to resolve manifest entries with stale absolute paths.",
    )
    parser.add_argument(
        "--include-unusable",
        action="store_true",
        help="Include manifest difficulties marked usable_for_training=false.",
    )
    parser.add_argument(
        "--long-tail-min-count",
        type=int,
        default=10,
        help="A nonzero pattern count below this value is considered long-tail.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
