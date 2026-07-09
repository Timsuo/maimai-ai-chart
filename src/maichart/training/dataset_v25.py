"""Dataset for the V2.5 frame-level Transformer baseline."""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
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
    ) -> None:
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
        return len(AUDIO_FEATURE_KEYS)

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

        x = _align_audio_to_label_frames(feature_frames, frames, ref.audio_features_path)
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
) -> torch.Tensor:
    audio_times = [float(frame.get("time_sec") or 0.0) for frame in audio_frames]
    vectors: list[list[float]] = []
    for frame in label_frames:
        time_sec = frame.get("time_sec")
        if time_sec is None:
            vectors.append([0.0] * len(AUDIO_FEATURE_KEYS))
            continue
        audio_frame = audio_frames[_nearest_index(audio_times, float(time_sec))]
        try:
            vectors.append([_feature_value(audio_frame, key) for key in AUDIO_FEATURE_KEYS])
        except KeyError as exc:
            raise TrainingDataError(
                f"{audio_path} feature frame is missing required key {exc.args[0]!r}."
            ) from exc
    if not vectors:
        raise TrainingDataError("frame_labels.json contains no frames.")
    return torch.tensor(vectors, dtype=torch.float32)


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
