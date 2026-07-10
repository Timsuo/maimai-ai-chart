"""Dataset for the V2.5 frame-level Transformer baseline."""

from __future__ import annotations

import bisect
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from maichart.training.frame_label_codec import DEFAULT_NOTE_TYPES, FrameLabelCodec

AUDIO_FEATURE_KEYS = (
    "onset_strength",
    "rms",
    "percussive_rms",
    "harmonic_rms",
    "spectral_centroid",
    "spectral_bandwidth",
    "zero_crossing_rate",
)
GRID_FEATURE_KEYS = (
    "beat_phase_sin",
    "beat_phase_cos",
    "bar_phase_sin",
    "bar_phase_cos",
    "song_progress",
    "local_bpm_norm",
    "difficulty_norm",
    "level_norm",
)
FEATURE_SETS = ("audio7", "audio7_plus_grid")


class TrainingDataError(RuntimeError):
    """Raised when a training sample is missing required cached data."""


@dataclass(slots=True)
class TrainingSampleRef:
    song_id: str
    difficulty_index: int
    level: float | None
    audio_features_path: Path
    frame_labels_path: Path
    alignment_report_path: Path
    chart_ir_path: Path
    manifest_song: dict[str, Any]
    manifest_difficulty: dict[str, Any]


class MaichartV25Dataset(Dataset):
    """Load manifest/cache data into frame-level tensors."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        cache_dir: str | Path = "cache",
        codec: FrameLabelCodec | None = None,
        include_unusable: bool = False,
        load_chart_ir: bool = False,
        feature_set: str = "audio7",
    ) -> None:
        self.feature_set = _normalize_feature_set(feature_set)
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest_base = self.manifest_path.parent
        self.cache_dir = Path(cache_dir).resolve()
        self.codec = codec or FrameLabelCodec(DEFAULT_NOTE_TYPES)
        self.load_chart_ir = load_chart_ir
        self.manifest = self._load_json(self.manifest_path, "training manifest")
        self.samples = self._collect_samples(include_unusable=include_unusable)
        if not self.samples:
            raise TrainingDataError(
                f"No usable training difficulties found in {self.manifest_path}."
            )

    @property
    def input_dim(self) -> int:
        return self.feature_dim

    @property
    def feature_dim(self) -> int:
        if self.feature_set == "audio7":
            return len(AUDIO_FEATURE_KEYS)
        return len(AUDIO_FEATURE_KEYS) + len(GRID_FEATURE_KEYS)

    @property
    def num_note_types(self) -> int:
        return self.codec.num_note_types

    @property
    def num_start_pattern_types(self) -> int:
        return self.codec.num_start_pattern_types

    @property
    def start_pattern_vocab(self) -> tuple[str, ...]:
        return self.codec.start_pattern_vocab

    @property
    def num_chord_size_start_classes(self) -> int:
        return self.codec.num_chord_size_start_classes

    @property
    def note_type_vocab(self) -> tuple[str, ...]:
        return self.codec.note_types

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        ref = self.samples[index]
        audio_features = self._load_json(ref.audio_features_path, "audio_features.json")
        frame_labels = self._load_json(ref.frame_labels_path, "frame_labels.json")
        alignment_report = self._load_json(
            ref.alignment_report_path,
            "alignment_report.json",
        )
        chart_ir = (
            self._load_json(ref.chart_ir_path, "chart_ir.json")
            if self.load_chart_ir
            else None
        )

        frames = frame_labels.get("frames")
        if not isinstance(frames, list):
            raise TrainingDataError(
                f"{ref.frame_labels_path} is missing list field 'frames'."
            )
        feature_frames = audio_features.get("feature_frames")
        if not isinstance(feature_frames, list) or not feature_frames:
            raise TrainingDataError(
                f"{ref.audio_features_path} is missing non-empty list field "
                "'feature_frames'."
            )

        x = _align_audio_to_label_frames(
            feature_frames,
            frames,
            ref.audio_features_path,
            feature_set=self.feature_set,
            audio_features=audio_features,
            ref=ref,
        )
        y = self.codec.encode_frames(frames)
        if x.size(0) != y["note_presence"].size(0):
            raise TrainingDataError(
                f"Feature/label length mismatch for {ref.song_id} difficulty "
                f"{ref.difficulty_index}: x={x.size(0)}, y={y['note_presence'].size(0)}."
            )

        return {
            "x": x,
            "y": y,
            "meta": {
                "song_id": ref.song_id,
                "difficulty_index": ref.difficulty_index,
                "level": ref.level,
                "audio_features_path": str(ref.audio_features_path),
                "frame_labels_path": str(ref.frame_labels_path),
                "alignment_report_path": str(ref.alignment_report_path),
                "chart_ir_path": str(ref.chart_ir_path),
                "alignment_summary": alignment_report.get("summary"),
                "chart_ir": chart_ir,
                "feature_set": self.feature_set,
            },
        }

    def _collect_samples(self, *, include_unusable: bool) -> list[TrainingSampleRef]:
        songs = self.manifest.get("songs")
        if not isinstance(songs, list):
            raise TrainingDataError(
                f"{self.manifest_path} is missing list field 'songs'."
            )

        samples: list[TrainingSampleRef] = []
        for song in songs:
            if not isinstance(song, dict):
                continue
            song_id = str(song.get("song_id") or "unknown")
            audio_features_path = _song_audio_features_path(
                song,
                song_id=song_id,
                cache_dir=self.cache_dir,
                manifest_base=self.manifest_base,
            )
            for difficulty in song.get("difficulties") or []:
                if not isinstance(difficulty, dict):
                    continue
                if not include_unusable and difficulty.get("usable_for_training") is False:
                    continue
                difficulty_index = _difficulty_index(difficulty)
                ref = TrainingSampleRef(
                    song_id=song_id,
                    difficulty_index=difficulty_index,
                    level=_level_or_none(difficulty.get("level")),
                    audio_features_path=audio_features_path,
                    frame_labels_path=_difficulty_path(
                        difficulty,
                        "frame_labels_path",
                        self.cache_dir
                        / "frame_labels"
                        / song_id
                        / f"difficulty_{difficulty_index}.frame_labels.json",
                        self.manifest_base,
                    ),
                    alignment_report_path=_difficulty_path(
                        difficulty,
                        "alignment_report_path",
                        self.cache_dir
                        / "alignment_reports"
                        / song_id
                        / f"difficulty_{difficulty_index}.alignment_report.json",
                        self.manifest_base,
                    ),
                    chart_ir_path=_difficulty_path(
                        difficulty,
                        "chart_ir_path",
                        self.cache_dir
                        / "chart_ir"
                        / song_id
                        / f"difficulty_{difficulty_index}.chart_ir.json",
                        self.manifest_base,
                    ),
                    manifest_song=song,
                    manifest_difficulty=difficulty,
                )
                _check_paths(ref)
                samples.append(ref)
        return samples

    @staticmethod
    def _load_json(path: Path, label: str) -> dict[str, Any]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise TrainingDataError(f"Missing {label}: {path}") from exc
        except json.JSONDecodeError as exc:
            raise TrainingDataError(f"Invalid JSON in {label}: {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise TrainingDataError(f"{label} must decode to a JSON object: {path}")
        return data


def _align_audio_to_label_frames(
    audio_frames: list[dict[str, Any]],
    label_frames: list[dict[str, Any]],
    audio_path: Path,
    *,
    feature_set: str = "audio7",
    audio_features: dict[str, Any] | None = None,
    ref: TrainingSampleRef | None = None,
) -> torch.Tensor:
    feature_set = _normalize_feature_set(feature_set)
    audio_times = [float(frame.get("time_sec") or 0.0) for frame in audio_frames]
    vectors: list[list[float]] = []
    grid_context = _grid_context(audio_features or {}, label_frames, ref)
    for frame in label_frames:
        time_sec = frame.get("time_sec")
        if time_sec is None:
            vector = [0.0] * len(AUDIO_FEATURE_KEYS)
        else:
            audio_frame = audio_frames[_nearest_index(audio_times, float(time_sec))]
            try:
                vector = [_feature_value(audio_frame, key) for key in AUDIO_FEATURE_KEYS]
            except KeyError as exc:
                raise TrainingDataError(
                    f"{audio_path} feature frame is missing required key {exc.args[0]!r}."
                ) from exc
        if feature_set == "audio7_plus_grid":
            vector.extend(_grid_feature_values(frame, grid_context))
        vectors.append(vector)
    if not vectors:
        raise TrainingDataError("frame_labels.json contains no frames.")
    return torch.tensor(vectors, dtype=torch.float32)


def _normalize_feature_set(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in FEATURE_SETS:
        raise TrainingDataError(
            f"Unsupported feature_set={value!r}; expected one of: {', '.join(FEATURE_SETS)}."
        )
    return normalized


def _grid_context(
    audio_features: dict[str, Any],
    label_frames: list[dict[str, Any]],
    ref: TrainingSampleRef | None,
) -> dict[str, float]:
    grid = audio_features.get("grid") if isinstance(audio_features.get("grid"), dict) else {}
    if not grid:
        for frame in label_frames:
            candidate = frame.get("grid")
            if isinstance(candidate, dict):
                grid = candidate
                break
    difficulty = float(ref.difficulty_index if ref is not None else _frame_difficulty(label_frames))
    level = ref.level if ref is not None else None
    return {
        "total_frames": float(len(label_frames)),
        "ticks_per_beat": _float_or_none(grid.get("ticks_per_beat")) or _ticks_per_beat(label_frames),
        "local_bpm_norm": _default_bpm(audio_features, ref) / 240.0,
        "difficulty_norm": difficulty / 5.0,
        "level_norm": (float(level) / 15.0) if level is not None else (difficulty / 5.0),
    }


def _grid_feature_values(frame: dict[str, Any], context: dict[str, float]) -> list[float]:
    beat = _beat_value(frame, context["ticks_per_beat"])
    beat_phase = beat % 1.0
    # Frame labels do not carry a meter map yet, so bar phase is a 4/4 fallback.
    bar_phase = (beat % 4.0) / 4.0
    frame_index = _float_or_none(frame.get("frame_index")) or 0.0
    song_progress = frame_index / max(context["total_frames"] - 1.0, 1.0)
    song_progress = min(1.0, max(0.0, song_progress))
    return [
        math.sin(2.0 * math.pi * beat_phase),
        math.cos(2.0 * math.pi * beat_phase),
        math.sin(2.0 * math.pi * bar_phase),
        math.cos(2.0 * math.pi * bar_phase),
        song_progress,
        context["local_bpm_norm"],
        context["difficulty_norm"],
        context["level_norm"],
    ]


def _beat_value(frame: dict[str, Any], ticks_per_beat: float) -> float:
    beat = _float_or_none(frame.get("beat"))
    if beat is not None:
        return beat
    tick = _float_or_none(frame.get("tick"))
    if tick is not None and ticks_per_beat > 0:
        return tick / ticks_per_beat
    return 0.0


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, str) and "/" in value:
            return float(Fraction(value))
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _ticks_per_beat(label_frames: list[dict[str, Any]]) -> float:
    previous_tick = None
    previous_beat = None
    for frame in label_frames:
        grid = frame.get("grid")
        if isinstance(grid, dict):
            value = _float_or_none(grid.get("ticks_per_beat"))
            if value is not None and value > 0:
                return value
        tick = _float_or_none(frame.get("tick"))
        beat = _float_or_none(frame.get("beat"))
        if tick is not None and beat is not None:
            if previous_tick is not None and previous_beat is not None:
                tick_delta = tick - previous_tick
                beat_delta = beat - previous_beat
                if tick_delta > 0 and beat_delta > 0:
                    return tick_delta / beat_delta
            previous_tick = tick
            previous_beat = beat
    return 1920.0


def _frame_difficulty(label_frames: list[dict[str, Any]]) -> float:
    for frame in label_frames:
        value = _float_or_none(frame.get("difficulty"))
        if value is not None:
            return value
    return 0.0


def _default_bpm(audio_features: dict[str, Any], ref: TrainingSampleRef | None) -> float:
    candidates: list[Any] = [
        audio_features.get("tempo_bpm"),
        audio_features.get("bpm"),
        audio_features.get("estimated_bpm"),
    ]
    if ref is not None:
        audio = ref.manifest_song.get("audio")
        if isinstance(audio, dict):
            candidates.extend([audio.get("estimated_bpm"), audio.get("metadata_bpm")])
        candidates.extend(
            [
                ref.manifest_song.get("estimated_bpm"),
                ref.manifest_song.get("bpm"),
                ref.manifest_difficulty.get("bpm"),
            ]
        )
    for value in candidates:
        bpm = _float_or_none(value)
        if bpm is not None and bpm > 0:
            return bpm
    return 120.0


def _nearest_index(values: list[float], target: float) -> int:
    index = bisect.bisect_left(values, target)
    candidates: list[int] = []
    if index < len(values):
        candidates.append(index)
    if index > 0:
        candidates.append(index - 1)
    return min(candidates, key=lambda candidate: abs(values[candidate] - target))


def _feature_value(frame: dict[str, Any], key: str) -> float:
    if key not in frame:
        raise KeyError(key)
    value = float(frame[key] or 0.0)
    if key in {"spectral_centroid", "spectral_bandwidth"}:
        return value / 10000.0
    return value


def _song_audio_features_path(
    song: dict[str, Any],
    *,
    song_id: str,
    cache_dir: Path,
    manifest_base: Path,
) -> Path:
    audio = song.get("audio") if isinstance(song.get("audio"), dict) else {}
    path = audio.get("audio_features_path") or song.get("audio_features_path")
    fallback = cache_dir / "audio_features" / f"{song_id}.audio_features.json"
    return _resolve_path(path or fallback, manifest_base)


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


def _difficulty_index(difficulty: dict[str, Any]) -> int:
    value = difficulty.get("difficulty_index", difficulty.get("index"))
    if value is None:
        raise TrainingDataError("Difficulty entry is missing 'difficulty_index'.")
    return int(value)


def _level_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _check_paths(ref: TrainingSampleRef) -> None:
    paths = {
        "audio_features.json": ref.audio_features_path,
        "frame_labels.json": ref.frame_labels_path,
        "alignment_report.json": ref.alignment_report_path,
        "chart_ir.json": ref.chart_ir_path,
    }
    for label, path in paths.items():
        if not path.is_file():
            raise TrainingDataError(
                f"Missing {label} for {ref.song_id} difficulty "
                f"{ref.difficulty_index}: {path}"
            )
