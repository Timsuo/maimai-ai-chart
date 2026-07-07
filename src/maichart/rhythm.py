"""Rule-based rhythm skeleton baseline for V2."""

from __future__ import annotations

import bisect
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from maichart.alignment import load_alignment_report_json
from maichart.audio import AudioFeatureFrame, AudioFeatureSet, load_audio_features_json
from maichart.labels import FrameLabelSet, load_frame_labels_json

RHYTHM_PROFILE_SCHEMA = "maichart-rhythm-profile-v1"
RHYTHM_SKELETON_SCHEMA = "maichart-rhythm-skeleton-v1"
RHYTHM_EVALUATION_SCHEMA = "maichart-rhythm-skeleton-evaluation-v1"
EVENT_TYPES = ("tap", "break", "hold_start", "slide_start", "touch")


@dataclass(slots=True)
class RhythmLevelBandProfile:
    """Aggregated rhythm behavior for a target level band."""

    min_level: float
    max_level: float
    target_note_density_per_sec: float
    target_selected_frame_ratio: float
    event_type_distribution: dict[str, float]
    onset_hit_rate_50ms: float | None = None
    sample_count: int = 0


@dataclass(slots=True)
class RhythmProfile:
    """Dataset-derived rhythm profile."""

    schema: str
    level_bands: list[RhythmLevelBandProfile]
    global_stats: dict[str, Any]


@dataclass(slots=True)
class RhythmSkeletonFrame:
    """One candidate rhythm-skeleton frame."""

    frame_index: int
    time_sec: float
    beat: str | None
    tick: int | None
    selected: bool
    event_type: str | None
    score: float
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RhythmSkeleton:
    """Rule-based rhythm skeleton output."""

    schema: str
    song_id: str | None
    target_level: float
    division: int
    duration_sec: float
    frames: list[RhythmSkeletonFrame]
    summary: dict[str, Any]


@dataclass(slots=True)
class RhythmSkeletonEvaluation:
    """Evaluation of a generated skeleton against reference frame labels."""

    schema: str
    predicted_selected_frames: int
    reference_note_frames: int
    precision: float
    recall: float
    f1: float
    note_count_error: int
    density_error: float
    event_type_distribution_difference: dict[str, float]
    onset_supported_rate: float | None = None


def build_rhythm_profile_from_dataset_manifest(
    manifest_path: str | Path,
    *,
    level_band_size: float = 2.0,
) -> RhythmProfile:
    """Build a rhythm profile from manifest frame labels and alignment reports."""

    manifest_file = Path(manifest_path).resolve()
    manifest_base = manifest_file.parent
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    buckets: dict[tuple[float, float], list[dict[str, Any]]] = {}

    for song in manifest.get("songs", []):
        for difficulty in song.get("difficulties", []):
            level = _parse_level(difficulty.get("level"))
            frame_labels_path = difficulty.get("frame_labels_path")
            alignment_report_path = difficulty.get("alignment_report_path")
            if level is None or not frame_labels_path:
                continue
            try:
                labels = load_frame_labels_json(_resolve_manifest_path(frame_labels_path, manifest_base))
                alignment = (
                    load_alignment_report_json(_resolve_manifest_path(alignment_report_path, manifest_base))
                    if alignment_report_path
                    else None
                )
            except Exception:
                continue
            stats = _profile_stats_for_labels(labels, alignment)
            band_min = math.floor(level / level_band_size) * level_band_size
            band_max = band_min + level_band_size
            buckets.setdefault((band_min, band_max), []).append(stats)

    bands = [
        _aggregate_level_band(band_min, band_max, samples)
        for (band_min, band_max), samples in sorted(buckets.items())
    ]
    all_samples = [sample for samples in buckets.values() for sample in samples]
    return RhythmProfile(
        schema=RHYTHM_PROFILE_SCHEMA,
        level_bands=bands,
        global_stats={
            "sample_count": len(all_samples),
            "level_band_size": float(level_band_size),
            "mean_note_density_per_sec": _mean([sample["density"] for sample in all_samples]),
            "mean_selected_frame_ratio": _mean([sample["selected_ratio"] for sample in all_samples]),
        },
    )


def generate_rhythm_skeleton(
    audio_features: AudioFeatureSet,
    *,
    target_level: float,
    profile: RhythmProfile | None = None,
    division: int = 16,
    max_density_per_sec: float | None = None,
) -> RhythmSkeleton:
    """Generate a rule-based rhythm skeleton from audio features."""

    return _generate_rhythm_skeleton_from_scored(
        audio_features,
        scored=_score_audio_frames(audio_features),
        target_level=target_level,
        profile=profile,
        division=division,
        max_density_per_sec=max_density_per_sec,
    )


def _generate_rhythm_skeleton_from_scored(
    audio_features: AudioFeatureSet,
    *,
    scored: list[dict[str, Any]],
    target_level: float,
    profile: RhythmProfile | None,
    division: int,
    max_density_per_sec: float | None,
) -> RhythmSkeleton:
    target_density = _target_density(target_level, profile)
    if max_density_per_sec is not None:
        target_density = min(target_density, max_density_per_sec)
    else:
        target_density = min(target_density, 6.0)
    duration = max(float(audio_features.duration_sec), _last_audio_time(audio_features), 0.001)
    target_count = max(1, int(round(duration * target_density))) if audio_features.feature_frames else 0
    min_interval = _minimum_interval_sec(target_level)
    selected_indices = _select_scored_frames(scored, target_count, min_interval)
    distribution = _target_event_distribution(target_level, profile)
    event_types = _assign_event_types(scored, selected_indices, distribution)

    frames: list[RhythmSkeletonFrame] = []
    for item in scored:
        selected = item["frame_index"] in selected_indices
        if not selected:
            continue
        event_type = event_types.get(item["frame_index"]) if selected else None
        reasons = list(item["reasons"])
        reasons.append("selected_top_score")
        frames.append(
            RhythmSkeletonFrame(
                frame_index=int(item["frame_index"]),
                time_sec=float(item["time_sec"]),
                beat=_estimated_beat(item["time_sec"], audio_features.tempo_bpm),
                tick=None,
                selected=selected,
                event_type=event_type,
                score=float(item["score"]),
                reasons=reasons,
            )
        )

    event_counts = {event_type: 0 for event_type in EVENT_TYPES}
    for event_type in event_types.values():
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
    selected_count = sum(1 for frame in frames if frame.selected)
    return RhythmSkeleton(
        schema=RHYTHM_SKELETON_SCHEMA,
        song_id=_song_id_from_audio_path(audio_features.audio_path),
        target_level=float(target_level),
        division=int(division),
        duration_sec=duration,
        frames=frames,
        summary={
            "selected_frame_count": selected_count,
            "candidate_frame_count": len(scored),
            "estimated_note_density_per_sec": selected_count / duration if duration else 0.0,
            "event_type_counts": event_counts,
            "target_note_density_per_sec": target_density,
            "min_interval_sec": min_interval,
        },
    )


def evaluate_rhythm_skeleton(
    skeleton: RhythmSkeleton,
    reference_labels: FrameLabelSet,
    *,
    tolerance_frames: int = 1,
) -> RhythmSkeletonEvaluation:
    """Evaluate a rhythm skeleton against human frame-label note frames."""

    predicted = [frame.frame_index for frame in skeleton.frames if frame.selected]
    reference = [
        frame.frame_index
        for frame in reference_labels.frames
        if frame.labels.has_note
    ]
    matches = _match_frames(predicted, reference, tolerance_frames=tolerance_frames)
    precision = matches / len(predicted) if predicted else 0.0
    recall = matches / len(reference) if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    predicted_density = len(predicted) / skeleton.duration_sec if skeleton.duration_sec else 0.0
    reference_duration = _reference_duration(reference_labels)
    reference_density = len(reference) / reference_duration if reference_duration else 0.0
    return RhythmSkeletonEvaluation(
        schema=RHYTHM_EVALUATION_SCHEMA,
        predicted_selected_frames=len(predicted),
        reference_note_frames=len(reference),
        precision=precision,
        recall=recall,
        f1=f1,
        note_count_error=len(predicted) - len(reference),
        density_error=predicted_density - reference_density,
        event_type_distribution_difference=_event_distribution_difference(skeleton, reference_labels),
        onset_supported_rate=_selected_onset_supported_rate(skeleton),
    )


def rhythm_profile_to_dict(profile: RhythmProfile) -> dict[str, Any]:
    """Convert a rhythm profile to JSON-compatible primitives."""

    return asdict(profile)


def rhythm_profile_to_json(profile: RhythmProfile, *, indent: int = 2) -> str:
    """Serialize a rhythm profile to JSON."""

    return json.dumps(rhythm_profile_to_dict(profile), ensure_ascii=False, indent=indent)


def rhythm_profile_from_dict(data: dict[str, Any]) -> RhythmProfile:
    """Load a rhythm profile from JSON-compatible primitives."""

    return RhythmProfile(
        schema=str(data.get("schema", RHYTHM_PROFILE_SCHEMA)),
        level_bands=[
            RhythmLevelBandProfile(**band)
            for band in data.get("level_bands", [])
        ],
        global_stats=dict(data.get("global_stats", {})),
    )


def rhythm_profile_from_json(payload: str) -> RhythmProfile:
    """Deserialize a rhythm profile from JSON text."""

    data = json.loads(payload)
    if not isinstance(data, dict):
        raise TypeError("Rhythm profile JSON must decode to an object.")
    return rhythm_profile_from_dict(data)


def save_rhythm_profile_json(profile: RhythmProfile, path: str | Path) -> None:
    """Write a rhythm profile JSON file."""

    _write_json(path, rhythm_profile_to_json(profile))


def load_rhythm_profile_json(path: str | Path) -> RhythmProfile:
    """Read a rhythm profile JSON file."""

    return rhythm_profile_from_json(Path(path).read_text(encoding="utf-8"))


def rhythm_skeleton_to_dict(skeleton: RhythmSkeleton) -> dict[str, Any]:
    """Convert a rhythm skeleton to JSON-compatible primitives."""

    return asdict(skeleton)


def rhythm_skeleton_to_json(skeleton: RhythmSkeleton, *, indent: int = 2) -> str:
    """Serialize a rhythm skeleton to JSON."""

    return json.dumps(rhythm_skeleton_to_dict(skeleton), ensure_ascii=False, indent=indent)


def rhythm_skeleton_from_dict(data: dict[str, Any]) -> RhythmSkeleton:
    """Load a rhythm skeleton from JSON-compatible primitives."""

    return RhythmSkeleton(
        schema=str(data.get("schema", RHYTHM_SKELETON_SCHEMA)),
        song_id=data.get("song_id"),
        target_level=float(data.get("target_level", 0.0)),
        division=int(data.get("division", 16)),
        duration_sec=float(data.get("duration_sec", 0.0)),
        frames=[RhythmSkeletonFrame(**frame) for frame in data.get("frames", [])],
        summary=dict(data.get("summary", {})),
    )


def rhythm_skeleton_from_json(payload: str) -> RhythmSkeleton:
    """Deserialize a rhythm skeleton from JSON text."""

    data = json.loads(payload)
    if not isinstance(data, dict):
        raise TypeError("Rhythm skeleton JSON must decode to an object.")
    return rhythm_skeleton_from_dict(data)


def save_rhythm_skeleton_json(skeleton: RhythmSkeleton, path: str | Path) -> None:
    """Write a rhythm skeleton JSON file."""

    _write_json(path, rhythm_skeleton_to_json(skeleton))


def load_rhythm_skeleton_json(path: str | Path) -> RhythmSkeleton:
    """Read a rhythm skeleton JSON file."""

    return rhythm_skeleton_from_json(Path(path).read_text(encoding="utf-8"))


def rhythm_evaluation_to_dict(evaluation: RhythmSkeletonEvaluation) -> dict[str, Any]:
    """Convert rhythm evaluation to JSON-compatible primitives."""

    return asdict(evaluation)


def rhythm_evaluation_to_json(evaluation: RhythmSkeletonEvaluation, *, indent: int = 2) -> str:
    """Serialize rhythm evaluation to JSON."""

    return json.dumps(rhythm_evaluation_to_dict(evaluation), ensure_ascii=False, indent=indent)


def rhythm_evaluation_from_dict(data: dict[str, Any]) -> RhythmSkeletonEvaluation:
    """Load rhythm evaluation from JSON-compatible primitives."""

    return RhythmSkeletonEvaluation(
        schema=str(data.get("schema", RHYTHM_EVALUATION_SCHEMA)),
        predicted_selected_frames=int(data.get("predicted_selected_frames", 0)),
        reference_note_frames=int(data.get("reference_note_frames", 0)),
        precision=float(data.get("precision", 0.0)),
        recall=float(data.get("recall", 0.0)),
        f1=float(data.get("f1", 0.0)),
        note_count_error=int(data.get("note_count_error", 0)),
        density_error=float(data.get("density_error", 0.0)),
        event_type_distribution_difference=dict(data.get("event_type_distribution_difference", {})),
        onset_supported_rate=data.get("onset_supported_rate"),
    )


def save_rhythm_evaluation_json(evaluation: RhythmSkeletonEvaluation, path: str | Path) -> None:
    """Write rhythm skeleton evaluation JSON."""

    _write_json(path, rhythm_evaluation_to_json(evaluation))


def load_rhythm_evaluation_json(path: str | Path) -> RhythmSkeletonEvaluation:
    """Read rhythm skeleton evaluation JSON."""

    return rhythm_evaluation_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def build_rhythm_skeletons_for_dataset_manifest(
    manifest_path: str | Path,
    *,
    out_dir: str | Path,
    profile: RhythmProfile | None = None,
    level_source: str = "difficulty",
    default_level: float = 10.0,
    division: int = 16,
    force: bool = False,
    output_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Generate rhythm skeletons for a dataset manifest."""

    manifest_file = Path(manifest_path).resolve()
    manifest_base = manifest_file.parent
    output_path = Path(output_manifest_path).resolve() if output_manifest_path else manifest_file
    skeleton_dir = Path(out_dir).resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    errors = manifest.setdefault("errors", [])
    summary = {"processed": 0, "skipped": 0, "failed": 0}

    for song in manifest.get("songs", []):
        song_id = str(song.get("song_id") or "unknown")
        audio_features_path = song.get("audio_features_path")
        difficulties = list(song.get("difficulties", []))
        if not audio_features_path:
            summary["skipped"] += len(difficulties)
            continue
        pending: list[tuple[dict[str, Any], float, Path]] = []
        for difficulty in difficulties:
            difficulty_index = difficulty.get("index")
            target_level = (
                _parse_level(difficulty.get("level"))
                if level_source == "difficulty"
                else None
            )
            target_level = target_level if target_level is not None else default_level
            skeleton_path = skeleton_dir / song_id / f"difficulty_{difficulty_index}.rhythm_skeleton.json"
            difficulty["rhythm_skeleton_path"] = _display_path(skeleton_path, manifest_base)
            if skeleton_path.exists() and not force:
                summary["skipped"] += 1
                difficulty["rhythm_skeleton_status"] = "skipped"
                difficulty["rhythm_skeleton_reason"] = "rhythm skeleton already exists"
                continue
            pending.append((difficulty, target_level, skeleton_path))
        if not pending:
            continue
        try:
            audio_features = load_audio_features_json(_resolve_manifest_path(audio_features_path, manifest_base))
            scored_audio_frames = _score_audio_frames(audio_features)
        except Exception as exc:
            summary["failed"] += len(pending)
            errors.append(
                {
                    "stage": "rhythm_skeleton",
                    "message": str(exc),
                    "song_id": song_id,
                    "exception_type": type(exc).__name__,
                }
            )
            continue

        for difficulty, target_level, skeleton_path in pending:
            difficulty_index = difficulty.get("index")
            try:
                skeleton = _generate_rhythm_skeleton_from_scored(
                    audio_features,
                    scored=scored_audio_frames,
                    target_level=target_level,
                    profile=profile,
                    division=division,
                    max_density_per_sec=None,
                )
                skeleton.song_id = song_id
                _write_json(skeleton_path, rhythm_skeleton_to_json(skeleton, indent=None))
                summary["processed"] += 1
                difficulty["rhythm_skeleton_status"] = "processed"
                difficulty.pop("rhythm_skeleton_reason", None)
            except Exception as exc:  # noqa: BLE001 - keep batch generation non-fatal.
                summary["failed"] += 1
                difficulty["rhythm_skeleton_status"] = "failed"
                difficulty["rhythm_skeleton_reason"] = str(exc)
                errors.append(
                    {
                        "stage": "rhythm_skeleton",
                        "message": str(exc),
                        "song_id": song_id,
                        "difficulty_index": difficulty_index,
                        "exception_type": type(exc).__name__,
                    }
                )

    manifest["rhythm_skeleton_summary"] = summary
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"manifest": manifest, **summary, "error_count": len(errors)}


def _profile_stats_for_labels(labels: FrameLabelSet, alignment) -> dict[str, Any]:
    note_count = sum(frame.labels.note_count for frame in labels.frames)
    frames_with_notes = sum(1 for frame in labels.frames if frame.labels.has_note)
    duration = _reference_duration(labels)
    return {
        "density": note_count / duration if duration else 0.0,
        "selected_ratio": frames_with_notes / len(labels.frames) if labels.frames else 0.0,
        "event_counts": _reference_event_counts(labels),
        "onset_hit_rate_50ms": (
            alignment.summary.onset_hit_rate_50ms
            if alignment is not None
            else None
        ),
    }


def _aggregate_level_band(
    band_min: float,
    band_max: float,
    samples: list[dict[str, Any]],
) -> RhythmLevelBandProfile:
    event_counts = {event_type: 0 for event_type in EVENT_TYPES}
    for sample in samples:
        for event_type, count in sample["event_counts"].items():
            event_counts[event_type] = event_counts.get(event_type, 0) + count
    total_events = sum(event_counts.values())
    return RhythmLevelBandProfile(
        min_level=float(band_min),
        max_level=float(band_max),
        target_note_density_per_sec=_mean([sample["density"] for sample in samples]),
        target_selected_frame_ratio=_mean([sample["selected_ratio"] for sample in samples]),
        event_type_distribution={
            event_type: (event_counts[event_type] / total_events if total_events else 0.0)
            for event_type in EVENT_TYPES
        },
        onset_hit_rate_50ms=_mean_or_none(
            [
                sample["onset_hit_rate_50ms"]
                for sample in samples
                if sample["onset_hit_rate_50ms"] is not None
            ]
        ),
        sample_count=len(samples),
    )


def _score_audio_frames(audio_features: AudioFeatureSet) -> list[dict[str, Any]]:
    frames = audio_features.feature_frames
    onset_values = [frame.onset_strength for frame in frames]
    percussive_values = [frame.percussive_rms for frame in frames]
    rms_values = [frame.rms for frame in frames]
    onset_range = _normalization_range(onset_values)
    percussive_range = _normalization_range(percussive_values)
    rms_range = _normalization_range(rms_values)
    beat_times = [beat.time_sec for beat in audio_features.beats]
    onset_times = [onset.time_sec for onset in audio_features.onsets]
    scored = []
    for frame in frames:
        onset_norm = _normalize_with_range(frame.onset_strength, onset_range)
        perc_norm = _normalize_with_range(frame.percussive_rms, percussive_range)
        rms_norm = _normalize_with_range(frame.rms, rms_range)
        beat_bonus = _proximity_bonus(frame.time_sec, beat_times, tolerance=0.08)
        near_onset = _proximity_bonus(frame.time_sec, onset_times, tolerance=0.07)
        score = (
            0.45 * onset_norm
            + 0.25 * perc_norm
            + 0.15 * rms_norm
            + 0.15 * beat_bonus
        )
        if near_onset > 0:
            score += 0.08 * near_onset
        reasons = []
        if near_onset > 0.5:
            reasons.append("near_onset")
        if beat_bonus > 0.5:
            reasons.append("beat_aligned")
        if onset_norm > 0.7:
            reasons.append("strong_onset")
        if perc_norm > 0.7:
            reasons.append("strong_percussive_rms")
        if rms_norm > 0.7:
            reasons.append("high_rms")
        scored.append(
            {
                "frame_index": int(frame.frame_index),
                "time_sec": float(frame.time_sec),
                "score": min(1.0, max(0.0, score)),
                "reasons": reasons,
                "onset_norm": onset_norm,
                "percussive_norm": perc_norm,
                "rms_norm": rms_norm,
                "beat_bonus": beat_bonus,
                "near_onset": near_onset,
            }
        )
    return scored


def _select_scored_frames(
    scored: list[dict[str, Any]],
    target_count: int,
    min_interval_sec: float,
) -> set[int]:
    selected: list[dict[str, Any]] = []
    selected_times: list[float] = []
    for item in sorted(scored, key=lambda value: value["score"], reverse=True):
        if len(selected) >= target_count:
            break
        if item["score"] <= 0.02:
            continue
        insert_at = bisect.bisect_left(selected_times, item["time_sec"])
        previous_too_close = (
            insert_at > 0
            and item["time_sec"] - selected_times[insert_at - 1] < min_interval_sec
        )
        next_too_close = (
            insert_at < len(selected_times)
            and selected_times[insert_at] - item["time_sec"] < min_interval_sec
        )
        if previous_too_close or next_too_close:
            continue
        selected.append(item)
        selected_times.insert(insert_at, item["time_sec"])
    return {item["frame_index"] for item in selected}


def _assign_event_types(
    scored: list[dict[str, Any]],
    selected_indices: set[int],
    distribution: dict[str, float],
) -> dict[int, str]:
    selected = [item for item in scored if item["frame_index"] in selected_indices]
    event_types: dict[int, str] = {}
    remaining = {item["frame_index"] for item in selected}
    total = len(selected)
    quotas = {
        event_type: int(round(total * distribution.get(event_type, 0.0)))
        for event_type in EVENT_TYPES
        if event_type != "tap"
    }
    for event_type, quota in quotas.items():
        ranked = sorted(
            [item for item in selected if item["frame_index"] in remaining],
            key=lambda item: _event_suitability(item, event_type),
            reverse=True,
        )
        for item in ranked[: max(0, quota)]:
            if _event_suitability(item, event_type) <= 0:
                continue
            event_types[item["frame_index"]] = event_type
            remaining.remove(item["frame_index"])
    for frame_index in remaining:
        event_types[frame_index] = "tap"
    return event_types


def _event_suitability(item: dict[str, Any], event_type: str) -> float:
    if event_type == "break":
        return item["onset_norm"] * 0.5 + item["percussive_norm"] * 0.3 + item["beat_bonus"] * 0.2
    if event_type == "hold_start":
        return item["rms_norm"] * 0.5 + (1.0 - item["percussive_norm"]) * 0.2 + item["near_onset"] * 0.3
    if event_type == "slide_start":
        return item["rms_norm"] * 0.35 + item["percussive_norm"] * 0.25 + item["onset_norm"] * 0.4
    if event_type == "touch":
        return item["onset_norm"] * 0.4 + item["beat_bonus"] * 0.2 + item["score"] * 0.4
    return item["score"]


def _target_density(target_level: float, profile: RhythmProfile | None) -> float:
    band = _profile_band_for_level(profile, target_level)
    if band is not None and band.target_note_density_per_sec > 0:
        return band.target_note_density_per_sec
    if target_level < 4:
        return 0.8
    if target_level < 7:
        return 1.4
    if target_level < 10:
        return 2.2
    if target_level < 12:
        return 3.0
    if target_level < 14:
        return 4.0
    return 5.2


def _target_event_distribution(target_level: float, profile: RhythmProfile | None) -> dict[str, float]:
    band = _profile_band_for_level(profile, target_level)
    if band is not None and sum(band.event_type_distribution.values()) > 0:
        return dict(band.event_type_distribution)
    if target_level < 7:
        return {"tap": 0.82, "break": 0.03, "hold_start": 0.08, "slide_start": 0.05, "touch": 0.02}
    if target_level < 12:
        return {"tap": 0.72, "break": 0.05, "hold_start": 0.10, "slide_start": 0.10, "touch": 0.03}
    return {"tap": 0.66, "break": 0.06, "hold_start": 0.10, "slide_start": 0.15, "touch": 0.03}


def _profile_band_for_level(profile: RhythmProfile | None, level: float) -> RhythmLevelBandProfile | None:
    if profile is None:
        return None
    for band in profile.level_bands:
        if band.min_level <= level < band.max_level:
            return band
    return None


def _minimum_interval_sec(target_level: float) -> float:
    if target_level < 4:
        return 0.34
    if target_level < 7:
        return 0.24
    if target_level < 10:
        return 0.16
    if target_level < 13:
        return 0.10
    return 0.075


def _reference_event_counts(labels: FrameLabelSet) -> dict[str, int]:
    counts = {event_type: 0 for event_type in EVENT_TYPES}
    for frame in labels.frames:
        counts["tap"] += frame.labels.tap_count
        counts["break"] += frame.labels.break_count
        counts["hold_start"] += frame.labels.hold_start_count
        counts["slide_start"] += frame.labels.slide_start_count
        counts["touch"] += frame.labels.touch_count + frame.labels.touch_hold_start_count
    return counts


def _match_frames(predicted: list[int], reference: list[int], *, tolerance_frames: int) -> int:
    unmatched = set(reference)
    matches = 0
    for frame in predicted:
        candidates = [candidate for candidate in range(frame - tolerance_frames, frame + tolerance_frames + 1) if candidate in unmatched]
        if not candidates:
            continue
        chosen = min(candidates, key=lambda candidate: abs(candidate - frame))
        unmatched.remove(chosen)
        matches += 1
    return matches


def _event_distribution_difference(
    skeleton: RhythmSkeleton,
    reference_labels: FrameLabelSet,
) -> dict[str, float]:
    pred_counts = {event_type: 0 for event_type in EVENT_TYPES}
    for frame in skeleton.frames:
        if frame.selected and frame.event_type is not None:
            pred_counts[frame.event_type] = pred_counts.get(frame.event_type, 0) + 1
    ref_counts = _reference_event_counts(reference_labels)
    pred_total = sum(pred_counts.values())
    ref_total = sum(ref_counts.values())
    return {
        event_type: (
            (pred_counts.get(event_type, 0) / pred_total if pred_total else 0.0)
            - (ref_counts.get(event_type, 0) / ref_total if ref_total else 0.0)
        )
        for event_type in EVENT_TYPES
    }


def _selected_onset_supported_rate(skeleton: RhythmSkeleton) -> float | None:
    selected = [frame for frame in skeleton.frames if frame.selected]
    if not selected:
        return None
    supported = sum(1 for frame in selected if "near_onset" in frame.reasons or "strong_onset" in frame.reasons)
    return supported / len(selected)


def _reference_duration(labels: FrameLabelSet) -> float:
    times = [frame.time_sec for frame in labels.frames if frame.time_sec is not None]
    if len(times) >= 2:
        return max(times) - min(times) + _median_step(times)
    return 1.0


def _median_step(times: list[float]) -> float:
    sorted_times = sorted(times)
    steps = [b - a for a, b in zip(sorted_times, sorted_times[1:]) if b > a]
    if not steps:
        return 1.0
    steps.sort()
    return steps[len(steps) // 2]


def _last_audio_time(audio_features: AudioFeatureSet) -> float:
    if not audio_features.feature_frames:
        return 0.0
    return max(frame.time_sec for frame in audio_features.feature_frames)


def _normalization_range(values: list[float]) -> tuple[float, float] | None:
    if not values:
        return None
    maximum = max(values)
    minimum = min(values)
    if maximum <= minimum:
        return None
    return minimum, maximum


def _normalize_with_range(value: float, value_range: tuple[float, float] | None) -> float:
    if value_range is None:
        return 0.0
    minimum, maximum = value_range
    return (float(value) - minimum) / (maximum - minimum)


def _proximity_bonus(time_sec: float, event_times: list[float], *, tolerance: float) -> float:
    if not event_times:
        return 0.0
    index = bisect.bisect_left(event_times, time_sec)
    candidates = []
    if index < len(event_times):
        candidates.append(event_times[index])
    if index > 0:
        candidates.append(event_times[index - 1])
    distance = min(abs(time_sec - candidate) for candidate in candidates)
    if distance > tolerance:
        return 0.0
    return 1.0 - distance / tolerance


def _estimated_beat(time_sec: float, tempo_bpm: float) -> str | None:
    if tempo_bpm <= 0:
        return None
    value = time_sec * tempo_bpm / 60.0
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _song_id_from_audio_path(audio_path: str) -> str | None:
    path = Path(audio_path)
    if path.parent.name:
        return path.parent.name
    return path.stem or None


def _parse_level(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("+", ".7")
    try:
        return float(text)
    except ValueError:
        return None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_or_none(values: list[float]) -> float | None:
    return _mean(values) if values else None


def _write_json(path: str | Path, payload: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(payload + ("\n" if not payload.endswith("\n") else ""), encoding="utf-8")


def _resolve_manifest_path(path: str | Path, base_path: Path) -> Path:
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
