"""V2 dataset QC, cache, training manifest, and split builders."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from maichart.alignment import (
    alignment_report_to_dict,
    build_alignment_report,
    load_alignment_report_json,
    save_alignment_report_json,
)
from maichart.audio import (
    AudioFeatureDependencyError,
    analyze_audio_file,
    audio_features_to_dict,
    load_audio_features_json,
    save_audio_features_json,
)
from maichart.builder import build_chart_ir_by_difficulty_index
from maichart.labels import (
    build_frame_labels_from_chart_ir,
    frame_labels_to_dict,
    load_frame_labels_json,
    save_frame_labels_json,
)
from maichart.maidata import RawMaidataChart, parse_maidata_file
from maichart.serialization import load_chart_json, save_chart_json
from maichart.stats import DifficultyStats, compute_raw_maidata_stats
from maichart.validation import validate_raw_maidata_chart

RAW_SAMPLE_MANIFEST_SCHEMA = "maichart-raw-sample-manifest-v1"
TRAINING_MANIFEST_SCHEMA = "maichart-training-manifest-v1"
SPLIT_SUMMARY_SCHEMA = "maichart-split-summary-v1"

AUDIO_FILENAMES = ("track.mp3", "track.wav", "track.ogg")
BACKGROUND_FILENAMES = ("bg.png", "bg.jpg", "bg.jpeg")
DIFFICULTY_NAMES = {
    1: "easy",
    2: "basic",
    3: "advanced",
    4: "expert",
    5: "master",
    6: "remaster",
}
DIFFICULTY_WEIGHTS = {
    "easy": 0.25,
    "basic": 0.35,
    "advanced": 0.6,
    "expert": 1.0,
    "master": 1.1,
    "remaster": 1.2,
}
MIN_TRAINING_NOTES = 20


@dataclass(slots=True)
class RawSample:
    """Raw sample directory QC entry."""

    sample_id: str
    song_id: str
    sample_dir: str
    convert_report_path: str | None
    maidata_path: str | None
    audio_path: str | None
    background_path: str | None
    has_convert_report: bool
    has_maidata: bool
    has_audio: bool
    conversion_status: str | None
    dataset_usable: bool
    dataset_usable_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RawSampleManifest:
    """Raw sample scan manifest."""

    schema: str
    source_root: str
    sample_count: int
    samples: list[RawSample] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TrainingManifest:
    """Training manifest payload."""

    schema: str
    source_root: str
    cache_dir: str
    song_count: int
    difficulty_count: int
    usable_difficulty_count: int
    songs: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class SplitManifest:
    """Train/val/test split output summary."""

    schema: str
    seed: int
    split_by_song: bool
    train: dict[str, int]
    val: dict[str, int]
    test: dict[str, int]
    level_distribution: dict[str, dict[str, int]]
    difficulty_name_distribution: dict[str, dict[str, int]]
    output_dir: str


def build_raw_sample_manifest(
    raw_root: str | Path,
    *,
    output_path: str | Path | None = None,
    path_base: str | Path | None = None,
) -> RawSampleManifest:
    """Scan raw sample directories and record lightweight dataset usability."""

    root = Path(raw_root).resolve()
    base_path = Path(path_base).resolve() if path_base is not None else _base_for_output(output_path)
    sample_dirs = _discover_sample_dirs(root)
    warnings: list[str] = []
    errors: list[str] = []
    song_id_counts: Counter[str] = Counter()
    planned_song_ids: list[str] = []

    for sample_dir in sample_dirs:
        report = _read_convert_report(sample_dir / "convert_report.json", errors, base_path)
        song_id = _raw_song_id(sample_dir, report)
        planned_song_ids.append(song_id)
        song_id_counts[song_id] += 1

    seen_song_ids: Counter[str] = Counter()
    samples: list[RawSample] = []
    for sample_dir, planned_song_id in zip(sample_dirs, planned_song_ids):
        sample_warnings: list[str] = []
        sample_errors: list[str] = []
        report_path = sample_dir / "convert_report.json"
        report = _read_convert_report(report_path, sample_errors, base_path)
        song_id = planned_song_id
        if song_id_counts[song_id] > 1:
            seen_song_ids[song_id] += 1
            sample_warnings.append(f"duplicate-song-id:{song_id}")
            warnings.append(f"Duplicate song_id {song_id!r} found in {sample_dir}.")
            song_id = f"{song_id}__dup{seen_song_ids[song_id]}"

        maidata_path = sample_dir / "maidata.txt"
        audio_path = _find_first_existing(sample_dir, AUDIO_FILENAMES)
        background_path = _find_first_existing(sample_dir, BACKGROUND_FILENAMES)
        has_maidata = maidata_path.is_file()
        has_audio = audio_path is not None
        reasons: list[str] = []
        if has_maidata:
            reasons.append("maidata_exists")
        else:
            sample_errors.append("missing_maidata")
        if has_audio:
            reasons.append("audio_exists")
        else:
            sample_warnings.append("missing_audio")
        dataset_usable = has_maidata and has_audio

        samples.append(
            RawSample(
                sample_id=_safe_path_segment(sample_dir.name),
                song_id=song_id,
                sample_dir=_display_path(sample_dir, base_path),
                convert_report_path=_display_optional(report_path if report_path.is_file() else None, base_path),
                maidata_path=_display_optional(maidata_path if has_maidata else None, base_path),
                audio_path=_display_optional(audio_path, base_path),
                background_path=_display_optional(background_path, base_path),
                has_convert_report=report_path.is_file(),
                has_maidata=has_maidata,
                has_audio=has_audio,
                conversion_status=_conversion_status(report),
                dataset_usable=dataset_usable,
                dataset_usable_reasons=reasons,
                warnings=sample_warnings,
                errors=sample_errors,
            )
        )

    manifest = RawSampleManifest(
        schema=RAW_SAMPLE_MANIFEST_SCHEMA,
        source_root=_display_path(root, base_path),
        sample_count=len(samples),
        samples=samples,
        warnings=warnings,
        errors=errors,
    )
    if output_path is not None:
        _write_json(raw_sample_manifest_to_dict(manifest), output_path)
    return manifest


def build_training_manifest(
    raw_root: str | Path,
    *,
    cache_dir: str | Path,
    output_path: str | Path | None = None,
    division: int = 16,
    force: bool = False,
    encoding: str | None = None,
    limit: int | None = None,
    skip_audio: bool = False,
    skip_alignment: bool = False,
    sample_rate: int = 22050,
) -> TrainingManifest:
    """Build QC caches and a supervised-learning training manifest."""

    root = Path(raw_root).resolve()
    cache_root = Path(cache_dir).resolve()
    base_path = _base_for_output(output_path)
    raw_manifest = build_raw_sample_manifest(root, output_path=None, path_base=base_path)
    raw_samples = raw_manifest.samples[:limit] if limit is not None else raw_manifest.samples
    songs: list[dict[str, Any]] = []
    warnings = list(raw_manifest.warnings)
    errors: list[dict[str, Any]] = []
    parse_failure_count = 0
    audio_failure_count = 0
    alignment_failure_count = 0

    for index, sample in enumerate(raw_samples, start=1):
        print(f"[preprocess] {index}/{len(raw_samples)} {sample.song_id}")
        song_errors: list[dict[str, Any]] = []
        chart: RawMaidataChart | None = None
        if sample.maidata_path:
            try:
                chart = parse_maidata_file(_resolve_display_path(sample.maidata_path, base_path), encoding=encoding)
            except Exception as exc:  # noqa: BLE001 - batch processing must continue.
                parse_failure_count += 1
                error = _error_dict(
                    "parse_maidata",
                    exc,
                    song_id=sample.song_id,
                    path=sample.maidata_path,
                )
                errors.append(error)
                song_errors.append(error)
        else:
            parse_failure_count += 1

        metadata_bpm = _parse_level(chart.wholebpm)[0] if chart and chart.wholebpm else None
        audio_info, audio_features = _build_song_audio_qc(
            sample,
            cache_root=cache_root,
            base_path=base_path,
            division=division,
            force=force,
            skip_audio=skip_audio,
            sample_rate=sample_rate,
            metadata_bpm=metadata_bpm,
            errors=errors,
        )
        if audio_info.get("readable") is False:
            audio_failure_count += 1

        song = {
            "song_id": sample.song_id,
            "title": chart.title if chart else None,
            "artist": chart.artist if chart else None,
            "maidata_path": sample.maidata_path,
            "audio_path": sample.audio_path,
            "convert_report_path": sample.convert_report_path,
            "conversion_status": sample.conversion_status,
            "dataset_usable": sample.dataset_usable,
            "audio": audio_info,
            "difficulties": [],
        }
        if song_errors:
            song["processing_errors"] = song_errors

        if chart is not None:
            for difficulty in chart.difficulties:
                difficulty_entry = _process_difficulty(
                    chart,
                    sample=sample,
                    difficulty_index=difficulty.index,
                    cache_root=cache_root,
                    base_path=base_path,
                    division=division,
                    force=force,
                    audio_readable=bool(audio_info.get("readable")),
                    audio_features=audio_features,
                    skip_audio=skip_audio,
                    skip_alignment=skip_alignment,
                    errors=errors,
                )
                if difficulty_entry.get("alignment_status") == "failed":
                    alignment_failure_count += 1
                song["difficulties"].append(difficulty_entry)

        songs.append(song)

    summary = _build_preprocess_summary(
        raw_sample_count=len(raw_samples),
        songs=songs,
        audio_failure_count=audio_failure_count,
        parse_failure_count=parse_failure_count,
        alignment_failure_count=alignment_failure_count,
    )
    manifest = TrainingManifest(
        schema=TRAINING_MANIFEST_SCHEMA,
        source_root=_display_path(root, base_path),
        cache_dir=_display_path(cache_root, base_path),
        song_count=len(songs),
        difficulty_count=sum(len(song.get("difficulties", [])) for song in songs),
        usable_difficulty_count=sum(
            1
            for song in songs
            for difficulty in song.get("difficulties", [])
            if difficulty.get("usable_for_training")
        ),
        songs=songs,
        summary=summary,
        warnings=warnings,
        errors=errors,
    )
    if output_path is not None:
        _write_json(training_manifest_to_dict(manifest), output_path)
    return manifest


def build_dataset_splits(
    training_manifest_path: str | Path,
    *,
    output_dir: str | Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    split_by_song: bool = True,
) -> SplitManifest:
    """Build reproducible train/val/test manifests from usable difficulties."""

    _validate_ratios(train_ratio, val_ratio, test_ratio)
    source_path = Path(training_manifest_path).resolve()
    out_dir = Path(output_dir).resolve()
    manifest = json.loads(source_path.read_text(encoding="utf-8"))
    rng = random.Random(seed)

    if split_by_song:
        groups = [
            [song]
            for song in _songs_with_usable_difficulties(manifest.get("songs", []))
        ]
    else:
        groups = _difficulty_groups(manifest.get("songs", []))
    rng.shuffle(groups)
    train_groups, val_groups, test_groups = _partition_groups(
        groups,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    split_groups = {
        "train": train_groups,
        "val": val_groups,
        "test": test_groups,
    }
    split_manifests: dict[str, dict[str, Any]] = {}
    for split_name, selected_groups in split_groups.items():
        split_manifest = _subset_training_manifest(manifest, selected_groups)
        split_manifests[split_name] = split_manifest
        _write_json(split_manifest, out_dir / f"{split_name}_manifest.json")

    summary_data = {
        "schema": SPLIT_SUMMARY_SCHEMA,
        "seed": seed,
        "split_by_song": split_by_song,
        "train": _split_counts(split_manifests["train"]),
        "val": _split_counts(split_manifests["val"]),
        "test": _split_counts(split_manifests["test"]),
        "level_distribution": {
            name: _level_distribution(split_manifests[name])
            for name in ("train", "val", "test")
        },
        "difficulty_name_distribution": {
            name: _difficulty_name_distribution(split_manifests[name])
            for name in ("train", "val", "test")
        },
    }
    _write_json(summary_data, out_dir / "split_summary.json")
    return SplitManifest(
        schema=SPLIT_SUMMARY_SCHEMA,
        seed=seed,
        split_by_song=split_by_song,
        train=summary_data["train"],
        val=summary_data["val"],
        test=summary_data["test"],
        level_distribution=summary_data["level_distribution"],
        difficulty_name_distribution=summary_data["difficulty_name_distribution"],
        output_dir=str(out_dir),
    )


def raw_sample_manifest_to_dict(manifest: RawSampleManifest) -> dict[str, Any]:
    """Convert a raw sample manifest to JSON-compatible primitives."""

    return asdict(manifest)


def training_manifest_to_dict(manifest: TrainingManifest) -> dict[str, Any]:
    """Convert a training manifest to JSON-compatible primitives."""

    return asdict(manifest)


def split_manifest_to_dict(manifest: SplitManifest) -> dict[str, Any]:
    """Convert a split manifest summary to JSON-compatible primitives."""

    return asdict(manifest)


def save_raw_sample_manifest(manifest: RawSampleManifest, path: str | Path) -> None:
    """Write a raw sample manifest JSON file."""

    _write_json(raw_sample_manifest_to_dict(manifest), path)


def save_training_manifest(manifest: TrainingManifest, path: str | Path) -> None:
    """Write a training manifest JSON file."""

    _write_json(training_manifest_to_dict(manifest), path)


def _process_difficulty(
    chart: RawMaidataChart,
    *,
    sample: RawSample,
    difficulty_index: int,
    cache_root: Path,
    base_path: Path,
    division: int,
    force: bool,
    audio_readable: bool,
    audio_features,
    skip_audio: bool,
    skip_alignment: bool,
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    raw_difficulty = next(
        candidate for candidate in chart.difficulties if candidate.index == difficulty_index
    )
    stats = _stats_for_difficulty(chart, difficulty_index)
    report = validate_raw_maidata_chart(chart, difficulty_index=difficulty_index)
    warning_codes = [issue.code for issue in report.issues if issue.severity == "warning"]
    difficulty_name = DIFFICULTY_NAMES.get(difficulty_index, f"difficulty_{difficulty_index}")
    level, level_is_numeric = _parse_level(stats.level)
    qc_tags = _qc_tags(level_is_numeric=level_is_numeric)
    qc_warnings = _qc_warnings(level_is_numeric=level_is_numeric)
    chart_ir_path = cache_root / "chart_ir" / sample.song_id / f"difficulty_{difficulty_index}.chart_ir.json"
    labels_path = cache_root / "frame_labels" / sample.song_id / f"difficulty_{difficulty_index}.frame_labels.json"
    alignment_path = cache_root / "alignment_reports" / sample.song_id / f"difficulty_{difficulty_index}.alignment_report.json"

    chart_ir = None
    frame_labels = None
    chart_ir_ok = False
    frame_labels_ok = False
    alignment_summary = {
        "onset_hit_rate_50ms": 0.0,
        "nearest_onset_mean_delta_ms": None,
    }
    alignment_status = "skipped" if skip_alignment else "pending"
    has_existing_chart_ir = chart_ir_path.exists() and not force
    has_existing_frame_labels = labels_path.exists() and not force
    has_existing_alignment = alignment_path.exists() and not force

    chart_ir_ok = has_existing_chart_ir
    frame_labels_ok = has_existing_frame_labels

    def ensure_chart_ir():
        nonlocal chart_ir, chart_ir_ok
        if chart_ir is not None:
            return chart_ir
        try:
            chart_ir = _load_or_build_chart_ir(
                chart,
                difficulty_index=difficulty_index,
                path=chart_ir_path,
                force=force,
            )
            chart_ir_ok = True
        except Exception as exc:  # noqa: BLE001
            chart_ir_ok = False
            errors.append(_error_dict("chart_ir", exc, song_id=sample.song_id, difficulty_index=difficulty_index))
        return chart_ir

    def ensure_frame_labels():
        nonlocal frame_labels, frame_labels_ok
        if frame_labels is not None:
            return frame_labels
        current_chart_ir = ensure_chart_ir()
        if current_chart_ir is None:
            return None
        try:
            frame_labels = _load_or_build_frame_labels(
                current_chart_ir,
                song_id=sample.song_id,
                path=labels_path,
                division=division,
                force=force,
            )
            frame_labels_ok = True
        except Exception as exc:  # noqa: BLE001
            frame_labels_ok = False
            errors.append(_error_dict("frame_labels", exc, song_id=sample.song_id, difficulty_index=difficulty_index))
        return frame_labels

    if not chart_ir_ok:
        ensure_chart_ir()
    if chart_ir_ok and not frame_labels_ok:
        ensure_frame_labels()

    if not skip_alignment and has_existing_alignment:
        try:
            alignment_report = _load_alignment_report_summary(alignment_path)
            alignment_status = "processed"
            alignment_summary = {
                "onset_hit_rate_50ms": alignment_report.summary.onset_hit_rate_50ms,
                "nearest_onset_mean_delta_ms": alignment_report.summary.nearest_onset_mean_delta_ms,
            }
        except Exception as exc:  # noqa: BLE001
            alignment_status = "failed"
            errors.append(_error_dict("alignment", exc, song_id=sample.song_id, difficulty_index=difficulty_index))
    elif not skip_alignment:
        current_chart_ir = ensure_chart_ir()
        current_frame_labels = ensure_frame_labels()
        if current_chart_ir is not None and current_frame_labels is not None and audio_features is not None:
            try:
                alignment_report = _load_or_build_alignment_report(
                    current_chart_ir,
                    current_frame_labels,
                    audio_features,
                    path=alignment_path,
                    force=force,
                )
                alignment_status = "processed"
                alignment_summary = {
                    "onset_hit_rate_50ms": alignment_report.summary.onset_hit_rate_50ms,
                    "nearest_onset_mean_delta_ms": alignment_report.summary.nearest_onset_mean_delta_ms,
                }
            except Exception as exc:  # noqa: BLE001
                alignment_status = "failed"
                errors.append(_error_dict("alignment", exc, song_id=sample.song_id, difficulty_index=difficulty_index))
        else:
            alignment_status = "skipped"

    filter_reasons = _filter_reasons(
        has_chart=bool((raw_difficulty.inote or "").strip()),
        note_count=stats.note_count,
        validate_errors=report.errors,
        parse_coverage=stats.parse_coverage,
        unknown_token_count=stats.unknown_token_count,
        audio_readable=audio_readable,
        frame_labels_ok=frame_labels_ok,
        audio_features_ok=(audio_features is not None and not skip_audio),
    )
    if skip_audio:
        filter_reasons.append("audio_skipped")
    if not skip_alignment:
        if alignment_status == "failed":
            filter_reasons.append("alignment_failed")
    warning_codes.extend(
        _qc_warning_codes(
            difficulty_name=difficulty_name,
            level_is_numeric=level_is_numeric,
            validate_warnings=report.warnings,
            alignment=alignment_summary,
            alignment_status=alignment_status,
        )
    )

    return {
        "difficulty_index": difficulty_index,
        "difficulty_name": difficulty_name,
        "level_raw": stats.level,
        "level": level,
        "designer": stats.designer,
        "has_chart": bool((raw_difficulty.inote or "").strip()),
        "chart_ir_path": _display_path(chart_ir_path, base_path),
        "frame_labels_path": _display_path(labels_path, base_path),
        "alignment_report_path": _display_path(alignment_path, base_path),
        "note_count": stats.note_count,
        "type_counts": stats.type_counts,
        "parse_coverage": stats.parse_coverage,
        "unknown_token_count": stats.unknown_token_count,
        "validate_errors": report.errors,
        "validate_warnings": report.warnings,
        "qc_tags": qc_tags,
        "warnings": qc_warnings,
        "duration_kind_counts": stats.duration_kind_counts,
        "slide_pattern_counts": stats.slide_pattern_counts,
        "alignment": alignment_summary,
        "alignment_status": alignment_status,
        "chart_ir_status": "processed" if chart_ir_ok else "failed",
        "frame_labels_status": "processed" if frame_labels_ok else "failed",
        "usable_for_training": not filter_reasons,
        "training_weight": _training_weight(difficulty_name, report.warnings, warning_codes),
        "filter_reasons": filter_reasons,
        "warning_codes": sorted(set(warning_codes)),
    }


def _build_song_audio_qc(
    sample: RawSample,
    *,
    cache_root: Path,
    base_path: Path,
    division: int,
    force: bool,
    skip_audio: bool,
    sample_rate: int,
    metadata_bpm: float | None,
    errors: list[dict[str, Any]],
) -> tuple[dict[str, Any], Any | None]:
    audio_features_path = cache_root / "audio_features" / f"{sample.song_id}.audio_features.json"
    info = {
        "readable": False,
        "duration_sec": None,
        "estimated_bpm": None,
        "metadata_bpm": metadata_bpm,
        "bpm_delta": None,
        "audio_features_path": _display_path(audio_features_path, base_path),
        "status": "skipped" if skip_audio else "pending",
    }
    if skip_audio:
        return info, None
    if not sample.audio_path:
        info["status"] = "missing"
        return info, None

    source = _resolve_display_path(sample.audio_path, base_path)
    try:
        features = _load_or_build_audio_features(
            source,
            path=audio_features_path,
            division=division,
            sample_rate=sample_rate,
            force=force,
        )
    except AudioFeatureDependencyError as exc:
        info["status"] = "dependency_missing"
        errors.append(_error_dict("audio_features", exc, song_id=sample.song_id, path=sample.audio_path))
        return info, None
    except Exception as exc:  # noqa: BLE001
        info["status"] = "failed"
        errors.append(_error_dict("audio_features", exc, song_id=sample.song_id, path=sample.audio_path))
        return info, None

    estimated_bpm = float(features.tempo_bpm)
    info.update(
        {
            "readable": True,
            "duration_sec": float(features.duration_sec),
            "estimated_bpm": estimated_bpm,
            "bpm_delta": abs(estimated_bpm - metadata_bpm) if metadata_bpm is not None else None,
            "status": "processed",
        }
    )
    return info, features


def _load_or_build_chart_ir(chart: RawMaidataChart, *, difficulty_index: int, path: Path, force: bool):
    if path.exists() and not force:
        try:
            return load_chart_json(path)
        except Exception:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    chart_ir = build_chart_ir_by_difficulty_index(chart, difficulty_index)
    save_chart_json(chart_ir, path)
    return chart_ir


def _load_or_build_frame_labels(chart_ir, *, song_id: str, path: Path, division: int, force: bool):
    if path.exists() and not force:
        try:
            return load_frame_labels_json(path)
        except Exception:
            pass
    labels = build_frame_labels_from_chart_ir(chart_ir, division=division, song_id=song_id)
    save_frame_labels_json(labels, path)
    return labels


def _load_or_build_audio_features(source: Path, *, path: Path, division: int, sample_rate: int, force: bool):
    if path.exists() and not force:
        try:
            return load_audio_features_json(path)
        except Exception:
            pass
    features = analyze_audio_file(source, division=division, sample_rate=sample_rate)
    save_audio_features_json(features, path)
    return features


def _load_or_build_alignment_report(chart_ir, labels, audio_features, *, path: Path, force: bool):
    if path.exists() and not force:
        try:
            return _load_alignment_report_summary(path)
        except Exception:
            pass
    report = build_alignment_report(chart_ir, labels, audio_features)
    save_alignment_report_json(report, path)
    return report


def _load_alignment_report_summary(path: Path):
    data = _load_alignment_report_header(path)
    summary = data.get("summary")
    if not isinstance(summary, dict):
        return load_alignment_report_json(path)
    return SimpleNamespace(
        summary=SimpleNamespace(
            onset_hit_rate_50ms=summary.get("onset_hit_rate_50ms"),
            nearest_onset_mean_delta_ms=summary.get("nearest_onset_mean_delta_ms"),
        )
    )


def _load_alignment_report_header(path: Path) -> dict[str, Any]:
    prefix_lines: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if any(marker in line for marker in ('"by_note_type"', '"density"', '"frames"')):
                break
            prefix_lines.append(line)
    prefix = "".join(prefix_lines).rstrip()
    if prefix.endswith(","):
        prefix = prefix[:-1]
    return json.loads(prefix + "\n}")


def _stats_for_difficulty(chart: RawMaidataChart, difficulty_index: int) -> DifficultyStats:
    stats = compute_raw_maidata_stats(chart, difficulty_index=difficulty_index)
    if not stats.difficulties:
        raise ValueError(f"No stats produced for difficulty {difficulty_index}.")
    return stats.difficulties[0]


def _filter_reasons(
    *,
    has_chart: bool,
    note_count: int,
    validate_errors: int,
    parse_coverage: float,
    unknown_token_count: int,
    audio_readable: bool,
    frame_labels_ok: bool,
    audio_features_ok: bool,
) -> list[str]:
    reasons: list[str] = []
    if not has_chart:
        reasons.append("missing_chart")
    if note_count < MIN_TRAINING_NOTES:
        reasons.append("note_count_too_low")
    if validate_errors > 0:
        reasons.append("validate_errors")
    if parse_coverage < 1.0:
        reasons.append("parse_coverage_below_1")
    if unknown_token_count > 0:
        reasons.append("unknown_tokens")
    if not audio_readable:
        reasons.append("audio_unreadable")
    if not frame_labels_ok:
        reasons.append("frame_labels_failed")
    if not audio_features_ok:
        reasons.append("audio_features_failed")
    return reasons


def _qc_warning_codes(
    *,
    difficulty_name: str,
    level_is_numeric: bool,
    validate_warnings: int,
    alignment: dict[str, Any],
    alignment_status: str,
) -> list[str]:
    codes: list[str] = []
    if not level_is_numeric:
        codes.append("non_numeric_level")
    if difficulty_name in {"easy", "basic"}:
        codes.append("low_reference_difficulty")
    elif difficulty_name == "advanced":
        codes.append("medium_reference_difficulty")
    if validate_warnings:
        codes.append("validate_warnings")
    hit_rate = alignment.get("onset_hit_rate_50ms")
    if alignment_status == "processed" and hit_rate is not None and hit_rate < 0.1:
        codes.append("low_onset_hit_rate")
    return codes


def _qc_tags(*, level_is_numeric: bool) -> list[str]:
    tags: list[str] = []
    if not level_is_numeric:
        tags.append("level_unknown")
    return tags


def _qc_warnings(*, level_is_numeric: bool) -> list[str]:
    warnings: list[str] = []
    if not level_is_numeric:
        warnings.append("non_numeric_level")
    return warnings


def _training_weight(difficulty_name: str, validate_warnings: int, warning_codes: list[str]) -> float:
    weight = DIFFICULTY_WEIGHTS.get(difficulty_name, 1.0)
    if validate_warnings > 0:
        weight *= 0.9
    if "low_onset_hit_rate" in warning_codes:
        weight *= 0.8
    return round(weight, 6)


def _build_preprocess_summary(
    *,
    raw_sample_count: int,
    songs: list[dict[str, Any]],
    audio_failure_count: int,
    parse_failure_count: int,
    alignment_failure_count: int,
) -> dict[str, Any]:
    difficulties = [difficulty for song in songs for difficulty in song.get("difficulties", [])]
    filter_reason_counts = Counter(
        reason for difficulty in difficulties for reason in difficulty.get("filter_reasons", [])
    )
    warning_code_counts = Counter(
        code for difficulty in difficulties for code in difficulty.get("warning_codes", [])
    )
    return {
        "raw_sample_count": raw_sample_count,
        "dataset_usable_song_count": sum(1 for song in songs if song.get("dataset_usable")),
        "song_with_audio_count": sum(1 for song in songs if song.get("audio_path")),
        "difficulty_count": len(difficulties),
        "usable_difficulty_count": sum(1 for difficulty in difficulties if difficulty.get("usable_for_training")),
        "filtered_difficulty_count": sum(1 for difficulty in difficulties if not difficulty.get("usable_for_training")),
        "filter_reason_counts": dict(filter_reason_counts),
        "warning_code_counts": dict(warning_code_counts),
        "difficulty_name_counts": dict(Counter(d.get("difficulty_name") for d in difficulties)),
        "level_distribution": dict(Counter(str(d.get("level_raw")) for d in difficulties)),
        "note_count_distribution": _note_count_distribution(difficulties),
        "audio_failure_count": audio_failure_count,
        "parse_failure_count": parse_failure_count,
        "alignment_failure_count": alignment_failure_count,
        "top_filter_reasons": filter_reason_counts.most_common(10),
        "top_warning_codes": warning_code_counts.most_common(10),
        "conversion_failed_but_dataset_usable": [
            song.get("song_id")
            for song in songs
            if song.get("conversion_status") == "failed" and song.get("dataset_usable")
        ],
        "samples_with_level_unknown": [
            f"{song.get('song_id')}:difficulty_{difficulty.get('difficulty_index')}"
            for song in songs
            for difficulty in song.get("difficulties", [])
            if difficulty.get("level_raw") == "?"
        ],
        "samples_with_missing_audio": [
            song.get("song_id")
            for song in songs
            if not song.get("audio_path")
        ],
        "samples_with_parse_failure": [
            song.get("song_id")
            for song in songs
            if song.get("processing_errors")
        ],
    }


def _note_count_distribution(difficulties: list[dict[str, Any]]) -> dict[str, int]:
    bins = Counter()
    for difficulty in difficulties:
        count = int(difficulty.get("note_count") or 0)
        if count < 20:
            bins["0-19"] += 1
        elif count < 100:
            bins["20-99"] += 1
        elif count < 300:
            bins["100-299"] += 1
        elif count < 600:
            bins["300-599"] += 1
        else:
            bins["600+"] += 1
    return dict(bins)


def _discover_sample_dirs(root: Path) -> list[Path]:
    if root.is_file():
        return [root.parent] if root.name in {"maidata.txt", "convert_report.json"} else []
    if not root.exists():
        return []
    dirs = {
        path.parent
        for pattern in ("maidata.txt", "convert_report.json")
        for path in root.rglob(pattern)
        if path.is_file()
    }
    return sorted(dirs)


def _read_convert_report(path: Path, errors: list[Any], base_path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        errors.append(
            {
                "stage": "read_convert_report",
                "path": _display_path(path, base_path),
                "message": str(exc),
                "exception_type": type(exc).__name__,
            }
        )
        return None
    return data if isinstance(data, dict) else None


def _raw_song_id(sample_dir: Path, report: dict[str, Any] | None) -> str:
    if report is not None and report.get("song_id"):
        return _safe_path_segment(str(report["song_id"]))
    return _safe_path_segment(sample_dir.name)


def _conversion_status(report: dict[str, Any] | None) -> str | None:
    if report is None:
        return None
    status = report.get("status")
    return str(status) if status is not None else None


def _find_first_existing(directory: Path, filenames: tuple[str, ...]) -> Path | None:
    for filename in filenames:
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    return None


def _parse_level(value: str | None) -> tuple[float | None, bool]:
    if value is None:
        return None, False
    text = str(value).strip()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(\+?)", text)
    if match is None:
        return None, False
    level = float(match.group(1))
    if match.group(2):
        level += 0.7
    return level, True


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if not 0.999 <= total <= 1.001:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    if min(train_ratio, val_ratio, test_ratio) < 0:
        raise ValueError("split ratios must be non-negative")


def _songs_with_usable_difficulties(songs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for song in songs:
        difficulties = [
            difficulty
            for difficulty in song.get("difficulties", [])
            if difficulty.get("usable_for_training")
        ]
        if difficulties:
            copied = dict(song)
            copied["difficulties"] = difficulties
            result.append(copied)
    return result


def _difficulty_groups(songs: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups = []
    for song in songs:
        for difficulty in song.get("difficulties", []):
            if not difficulty.get("usable_for_training"):
                continue
            copied = dict(song)
            copied["difficulties"] = [difficulty]
            groups.append([copied])
    return groups


def _partition_groups(
    groups: list[list[dict[str, Any]]],
    *,
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]], list[list[dict[str, Any]]]]:
    count = len(groups)
    train_count = int(round(count * train_ratio))
    val_count = int(round(count * val_ratio))
    if train_count + val_count > count:
        val_count = max(0, count - train_count)
    return (
        groups[:train_count],
        groups[train_count : train_count + val_count],
        groups[train_count + val_count :],
    )


def _subset_training_manifest(manifest: dict[str, Any], groups: list[list[dict[str, Any]]]) -> dict[str, Any]:
    songs = [song for group in groups for song in group]
    subset = {
        key: value
        for key, value in manifest.items()
        if key not in {"songs", "song_count", "difficulty_count", "usable_difficulty_count", "summary"}
    }
    subset["songs"] = songs
    subset["song_count"] = len(songs)
    subset["difficulty_count"] = sum(len(song.get("difficulties", [])) for song in songs)
    subset["usable_difficulty_count"] = subset["difficulty_count"]
    subset["summary"] = _build_preprocess_summary(
        raw_sample_count=len(songs),
        songs=songs,
        audio_failure_count=0,
        parse_failure_count=0,
        alignment_failure_count=0,
    )
    return subset


def _split_counts(manifest: dict[str, Any]) -> dict[str, int]:
    return {
        "song_count": int(manifest.get("song_count", 0)),
        "difficulty_count": int(manifest.get("difficulty_count", 0)),
    }


def _level_distribution(manifest: dict[str, Any]) -> dict[str, int]:
    return dict(
        Counter(
            str(difficulty.get("level_raw"))
            for song in manifest.get("songs", [])
            for difficulty in song.get("difficulties", [])
        )
    )


def _difficulty_name_distribution(manifest: dict[str, Any]) -> dict[str, int]:
    return dict(
        Counter(
            str(difficulty.get("difficulty_name"))
            for song in manifest.get("songs", [])
            for difficulty in song.get("difficulties", [])
        )
    )


def _error_dict(
    stage: str,
    exc: Exception,
    *,
    song_id: str,
    path: str | None = None,
    difficulty_index: int | None = None,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "message": str(exc),
        "song_id": song_id,
        "path": path,
        "difficulty_index": difficulty_index,
        "exception_type": type(exc).__name__,
    }


def _base_for_output(output_path: str | Path | None) -> Path:
    if output_path is None:
        return Path.cwd().resolve()
    return Path(output_path).resolve().parent


def _resolve_display_path(path: str | Path, base_path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (base_path / candidate).resolve()


def _display_optional(path: Path | None, base_path: Path) -> str | None:
    return _display_path(path, base_path) if path is not None else None


def _display_path(path: Path, base_path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(base_path.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _safe_path_segment(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("._") or "unknown"


def _write_json(payload: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
