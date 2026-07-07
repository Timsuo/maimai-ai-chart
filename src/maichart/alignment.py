"""Chart-audio alignment reports for V2 analysis."""

from __future__ import annotations

import bisect
import json
import statistics
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

from maichart.audio import AudioFeatureFrame, AudioFeatureSet, load_audio_features_json
from maichart.ir import ChartIR
from maichart.labels import FrameLabel, FrameLabelSet, load_frame_labels_json
from maichart.serialization import load_chart_json

ALIGNMENT_REPORT_SCHEMA = "maichart-alignment-report-v1"
NOTE_TYPE_KEYS = ("tap", "break", "hold_start", "slide_start", "touch")


@dataclass(slots=True)
class NearestOnset:
    """Nearest audio onset for one chart frame."""

    time_sec: float
    delta_ms: float
    strength: float


@dataclass(slots=True)
class AlignmentAudioSample:
    """Nearest sampled audio features for one chart frame."""

    onset_strength: float
    rms: float
    percussive_rms: float
    harmonic_rms: float


@dataclass(slots=True)
class AlignmentFrame:
    """One chart frame with sampled audio and nearest onset information."""

    frame_index: int
    time_sec: float | None
    beat: str
    has_note: bool
    note_types: list[str] = field(default_factory=list)
    nearest_onset: NearestOnset | None = None
    audio: AlignmentAudioSample | None = None


@dataclass(slots=True)
class NoteTypeAlignmentStats:
    """Alignment stats for one note category."""

    count: int = 0
    onset_hit_rate_50ms: float = 0.0
    mean_delta_ms: float | None = None
    median_delta_ms: float | None = None
    mean_onset_strength: float | None = None


@dataclass(slots=True)
class DensityPoint:
    """One note-density window."""

    index: int
    start_time_sec: float | None = None
    end_time_sec: float | None = None
    start_beat: str | None = None
    end_beat: str | None = None
    note_count: int = 0


@dataclass(slots=True)
class AlignmentSummary:
    """Top-level alignment summary."""

    note_count: int
    frames_with_notes: int
    nearest_onset_mean_delta_ms: float | None
    nearest_onset_median_delta_ms: float | None
    onset_hit_rate_25ms: float
    onset_hit_rate_50ms: float
    onset_hit_rate_100ms: float


@dataclass(slots=True)
class AlignmentDensity:
    """Density curves in time and beat windows."""

    notes_per_second: list[DensityPoint] = field(default_factory=list)
    notes_per_4beat_window: list[DensityPoint] = field(default_factory=list)
    notes_per_16beat_window: list[DensityPoint] = field(default_factory=list)


@dataclass(slots=True)
class AlignmentReport:
    """Chart-audio alignment report JSON payload."""

    schema: str
    song_id: str | None
    difficulty: int | None
    onset_tolerance_ms: float
    summary: AlignmentSummary
    by_note_type: dict[str, NoteTypeAlignmentStats] = field(default_factory=dict)
    density: AlignmentDensity = field(default_factory=AlignmentDensity)
    frames: list[AlignmentFrame] = field(default_factory=list)


def build_alignment_report(
    chart_ir: ChartIR,
    frame_labels: FrameLabelSet,
    audio_features: AudioFeatureSet,
    onset_tolerance_ms: float = 50.0,
) -> AlignmentReport:
    """Build a chart-audio alignment report for one difficulty."""

    onset_times = [onset.time_sec for onset in audio_features.onsets]
    audio_frame_times = [frame.time_sec for frame in audio_features.feature_frames]
    note_frames = [frame for frame in frame_labels.frames if frame.labels.has_note]
    frame_deltas: list[float] = []
    frame_onset_strengths: list[float] = []
    by_type_deltas: dict[str, list[float]] = {key: [] for key in NOTE_TYPE_KEYS}
    by_type_strengths: dict[str, list[float]] = {key: [] for key in NOTE_TYPE_KEYS}
    by_type_counts: dict[str, int] = {key: 0 for key in NOTE_TYPE_KEYS}
    report_frames: list[AlignmentFrame] = []

    for frame in frame_labels.frames:
        nearest_onset = _nearest_onset(frame, audio_features, onset_times)
        audio_sample = _nearest_audio_sample(frame, audio_features, audio_frame_times)
        if frame.labels.has_note and nearest_onset is not None:
            frame_deltas.append(abs(nearest_onset.delta_ms))
            frame_onset_strengths.append(nearest_onset.strength)
            _record_note_type_stats(
                frame,
                nearest_onset,
                by_type_deltas=by_type_deltas,
                by_type_strengths=by_type_strengths,
                by_type_counts=by_type_counts,
            )
        elif frame.labels.has_note:
            _record_note_type_counts_only(frame, by_type_counts)

        report_frames.append(
            AlignmentFrame(
                frame_index=frame.frame_index,
                time_sec=frame.time_sec,
                beat=frame.beat,
                has_note=frame.labels.has_note,
                note_types=list(frame.labels.note_types),
                nearest_onset=nearest_onset if frame.labels.has_note else None,
                audio=audio_sample,
            )
        )

    return AlignmentReport(
        schema=ALIGNMENT_REPORT_SCHEMA,
        song_id=frame_labels.song_id or chart_ir.metadata.title,
        difficulty=frame_labels.difficulty or chart_ir.difficulty.index,
        onset_tolerance_ms=float(onset_tolerance_ms),
        summary=AlignmentSummary(
            note_count=sum(frame.labels.note_count for frame in frame_labels.frames),
            frames_with_notes=len(note_frames),
            nearest_onset_mean_delta_ms=_mean_or_none(frame_deltas),
            nearest_onset_median_delta_ms=_median_or_none(frame_deltas),
            onset_hit_rate_25ms=_hit_rate(frame_deltas, 25.0),
            onset_hit_rate_50ms=_hit_rate(frame_deltas, 50.0),
            onset_hit_rate_100ms=_hit_rate(frame_deltas, 100.0),
        ),
        by_note_type={
            key: _note_type_stats(
                count=by_type_counts[key],
                deltas=by_type_deltas[key],
                strengths=by_type_strengths[key],
            )
            for key in NOTE_TYPE_KEYS
        },
        density=_build_density(frame_labels),
        frames=report_frames,
    )


def alignment_report_to_dict(report: AlignmentReport) -> dict[str, Any]:
    """Convert an alignment report to JSON-compatible primitives."""

    return asdict(report)


def alignment_report_to_json(report: AlignmentReport, *, indent: int = 2) -> str:
    """Serialize an alignment report to JSON."""

    return json.dumps(alignment_report_to_dict(report), ensure_ascii=False, indent=indent)


def alignment_report_from_dict(data: dict[str, Any]) -> AlignmentReport:
    """Load an alignment report from JSON-compatible primitives."""

    density_data = data.get("density") or {}
    summary_data = data.get("summary") or {}
    return AlignmentReport(
        schema=str(data.get("schema", ALIGNMENT_REPORT_SCHEMA)),
        song_id=data.get("song_id"),
        difficulty=data.get("difficulty"),
        onset_tolerance_ms=float(data.get("onset_tolerance_ms", 50.0)),
        summary=AlignmentSummary(**summary_data),
        by_note_type={
            key: NoteTypeAlignmentStats(**value)
            for key, value in (data.get("by_note_type") or {}).items()
        },
        density=AlignmentDensity(
            notes_per_second=[
                DensityPoint(**point)
                for point in density_data.get("notes_per_second", [])
            ],
            notes_per_4beat_window=[
                DensityPoint(**point)
                for point in density_data.get("notes_per_4beat_window", [])
            ],
            notes_per_16beat_window=[
                DensityPoint(**point)
                for point in density_data.get("notes_per_16beat_window", [])
            ],
        ),
        frames=[
            AlignmentFrame(
                frame_index=int(frame.get("frame_index", 0)),
                time_sec=frame.get("time_sec"),
                beat=str(frame.get("beat", "0")),
                has_note=bool(frame.get("has_note", False)),
                note_types=list(frame.get("note_types", [])),
                nearest_onset=(
                    NearestOnset(**frame["nearest_onset"])
                    if frame.get("nearest_onset") is not None
                    else None
                ),
                audio=(
                    AlignmentAudioSample(**frame["audio"])
                    if frame.get("audio") is not None
                    else None
                ),
            )
            for frame in data.get("frames", [])
        ],
    )


def alignment_report_from_json(payload: str) -> AlignmentReport:
    """Deserialize an alignment report from JSON text."""

    data = json.loads(payload)
    if not isinstance(data, dict):
        raise TypeError("Alignment report JSON must decode to an object.")
    return alignment_report_from_dict(data)


def save_alignment_report_json(report: AlignmentReport, path: str | Path) -> None:
    """Write an alignment report JSON file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(alignment_report_to_json(report), encoding="utf-8")


def load_alignment_report_json(path: str | Path) -> AlignmentReport:
    """Read an alignment report JSON file."""

    return alignment_report_from_json(Path(path).read_text(encoding="utf-8"))


def build_alignment_reports_for_dataset_manifest(
    manifest_path: str | Path,
    *,
    out_dir: str | Path,
    onset_tolerance_ms: float = 50.0,
    force: bool = False,
    output_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build alignment reports for manifest difficulties with labels and audio."""

    manifest_file = Path(manifest_path).resolve()
    manifest_base = manifest_file.parent
    output_path = Path(output_manifest_path).resolve() if output_manifest_path else manifest_file
    report_dir = Path(out_dir).resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    errors = manifest.setdefault("errors", [])
    summary = {"processed": 0, "skipped": 0, "failed": 0}

    for song in manifest.get("songs", []):
        song_id = str(song.get("song_id") or "unknown")
        audio_features_path = song.get("audio_features_path")
        if not audio_features_path:
            skipped = len(song.get("difficulties", []))
            summary["skipped"] += skipped
            for difficulty in song.get("difficulties", []):
                difficulty["alignment_status"] = "skipped"
                difficulty["alignment_reason"] = "missing audio_features_path"
            continue

        for difficulty in song.get("difficulties", []):
            difficulty_index = difficulty.get("index")
            frame_labels_path = difficulty.get("frame_labels_path")
            chart_ir_path = difficulty.get("chart_ir_path")
            report_path = report_dir / song_id / f"difficulty_{difficulty_index}.alignment_report.json"
            difficulty["alignment_report_path"] = _display_path(report_path, manifest_base)

            if not frame_labels_path:
                summary["skipped"] += 1
                difficulty["alignment_status"] = "skipped"
                difficulty["alignment_reason"] = "missing frame_labels_path"
                continue
            if not chart_ir_path:
                summary["skipped"] += 1
                difficulty["alignment_status"] = "skipped"
                difficulty["alignment_reason"] = "missing chart_ir_path"
                continue
            if report_path.exists() and not force:
                summary["skipped"] += 1
                difficulty["alignment_status"] = "skipped"
                difficulty["alignment_reason"] = "alignment report already exists"
                continue

            try:
                report = build_alignment_report(
                    load_chart_json(_resolve_manifest_path(chart_ir_path, manifest_base)),
                    load_frame_labels_json(_resolve_manifest_path(frame_labels_path, manifest_base)),
                    load_audio_features_json(_resolve_manifest_path(audio_features_path, manifest_base)),
                    onset_tolerance_ms=onset_tolerance_ms,
                )
                save_alignment_report_json(report, report_path)
                summary["processed"] += 1
                difficulty["alignment_status"] = "processed"
                difficulty.pop("alignment_reason", None)
            except Exception as exc:  # noqa: BLE001 - batch analysis should continue.
                summary["failed"] += 1
                difficulty["alignment_status"] = "failed"
                difficulty["alignment_reason"] = str(exc)
                errors.append(
                    {
                        "stage": "alignment",
                        "message": str(exc),
                        "song_id": song_id,
                        "difficulty_index": difficulty_index,
                        "exception_type": type(exc).__name__,
                    }
                )

    manifest["alignment_summary"] = summary
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "manifest": manifest,
        **summary,
        "error_count": len(errors),
    }


def _nearest_onset(
    frame: FrameLabel,
    audio_features: AudioFeatureSet,
    onset_times: list[float],
) -> NearestOnset | None:
    if frame.time_sec is None or not onset_times:
        return None
    index = bisect.bisect_left(onset_times, frame.time_sec)
    candidates = []
    if index < len(onset_times):
        candidates.append(index)
    if index > 0:
        candidates.append(index - 1)
    best_index = min(
        candidates,
        key=lambda candidate: abs(onset_times[candidate] - float(frame.time_sec)),
    )
    onset = audio_features.onsets[best_index]
    return NearestOnset(
        time_sec=float(onset.time_sec),
        delta_ms=(float(onset.time_sec) - float(frame.time_sec)) * 1000.0,
        strength=float(onset.strength),
    )


def _nearest_audio_sample(
    frame: FrameLabel,
    audio_features: AudioFeatureSet,
    audio_frame_times: list[float],
) -> AlignmentAudioSample | None:
    if frame.time_sec is None or not audio_frame_times:
        return None
    index = bisect.bisect_left(audio_frame_times, frame.time_sec)
    candidates = []
    if index < len(audio_frame_times):
        candidates.append(index)
    if index > 0:
        candidates.append(index - 1)
    best_index = min(
        candidates,
        key=lambda candidate: abs(audio_frame_times[candidate] - float(frame.time_sec)),
    )
    sample = audio_features.feature_frames[best_index]
    return AlignmentAudioSample(
        onset_strength=float(sample.onset_strength),
        rms=float(sample.rms),
        percussive_rms=float(sample.percussive_rms),
        harmonic_rms=float(sample.harmonic_rms),
    )


def _record_note_type_stats(
    frame: FrameLabel,
    onset: NearestOnset,
    *,
    by_type_deltas: dict[str, list[float]],
    by_type_strengths: dict[str, list[float]],
    by_type_counts: dict[str, int],
) -> None:
    counts = _note_type_counts(frame)
    for key, count in counts.items():
        by_type_counts[key] += count
        by_type_deltas[key].extend([abs(onset.delta_ms)] * count)
        by_type_strengths[key].extend([onset.strength] * count)


def _record_note_type_counts_only(
    frame: FrameLabel,
    by_type_counts: dict[str, int],
) -> None:
    for key, count in _note_type_counts(frame).items():
        by_type_counts[key] += count


def _note_type_counts(frame: FrameLabel) -> dict[str, int]:
    labels = frame.labels
    return {
        "tap": labels.tap_count,
        "break": labels.break_count,
        "hold_start": labels.hold_start_count,
        "slide_start": labels.slide_start_count,
        "touch": labels.touch_count + labels.touch_hold_start_count,
    }


def _note_type_stats(
    *,
    count: int,
    deltas: list[float],
    strengths: list[float],
) -> NoteTypeAlignmentStats:
    return NoteTypeAlignmentStats(
        count=count,
        onset_hit_rate_50ms=_hit_rate(deltas, 50.0),
        mean_delta_ms=_mean_or_none(deltas),
        median_delta_ms=_median_or_none(deltas),
        mean_onset_strength=_mean_or_none(strengths),
    )


def _build_density(frame_labels: FrameLabelSet) -> AlignmentDensity:
    note_frames = [
        frame
        for frame in frame_labels.frames
        if frame.labels.note_count > 0
    ]
    return AlignmentDensity(
        notes_per_second=_time_density(note_frames),
        notes_per_4beat_window=_beat_density(note_frames, window_beats=4),
        notes_per_16beat_window=_beat_density(note_frames, window_beats=16),
    )


def _time_density(note_frames: list[FrameLabel]) -> list[DensityPoint]:
    bins: dict[int, int] = {}
    for frame in note_frames:
        if frame.time_sec is None:
            continue
        index = int(float(frame.time_sec) // 1)
        bins[index] = bins.get(index, 0) + frame.labels.note_count
    return [
        DensityPoint(
            index=index,
            start_time_sec=float(index),
            end_time_sec=float(index + 1),
            note_count=bins[index],
        )
        for index in sorted(bins)
    ]


def _beat_density(note_frames: list[FrameLabel], *, window_beats: int) -> list[DensityPoint]:
    bins: dict[int, int] = {}
    for frame in note_frames:
        beat = _parse_beat(frame.beat)
        index = int(beat // window_beats)
        bins[index] = bins.get(index, 0) + frame.labels.note_count
    return [
        DensityPoint(
            index=index,
            start_beat=_format_beat(Fraction(index * window_beats, 1)),
            end_beat=_format_beat(Fraction((index + 1) * window_beats, 1)),
            note_count=bins[index],
        )
        for index in sorted(bins)
    ]


def _hit_rate(deltas: list[float], tolerance_ms: float) -> float:
    if not deltas:
        return 0.0
    return sum(1 for delta in deltas if abs(delta) <= tolerance_ms) / len(deltas)


def _mean_or_none(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _median_or_none(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _parse_beat(value: str) -> Fraction:
    return Fraction(value)


def _format_beat(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


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
