"""V2 dataset manifest and ChartIR cache builder."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from maichart.builder import build_chart_ir_by_difficulty_index
from maichart.maidata import RawMaidataChart, parse_maidata_file
from maichart.serialization import save_chart_json
from maichart.stats import DifficultyStats, compute_raw_maidata_stats
from maichart.validation import validate_raw_maidata_chart

DATASET_MANIFEST_SCHEMA = "maichart-dataset-manifest-v1"
AUDIO_FILENAMES = ("track.mp3", "track.wav")
BACKGROUND_FILENAMES = ("bg.png",)


@dataclass(slots=True)
class DatasetBuildError:
    """A non-fatal dataset build error."""

    stage: str
    message: str
    maidata_path: str | None = None
    song_id: str | None = None
    difficulty_index: int | None = None
    exception_type: str | None = None


@dataclass(slots=True)
class DatasetDifficultyEntry:
    """Manifest entry for one successfully parsed chart difficulty."""

    index: int
    level: str | None
    designer: str | None
    chart_ir_path: str
    timing_points: int
    note_count: int
    type_counts: dict[str, int] = field(default_factory=dict)
    parse_coverage: float = 1.0
    unknown_token_count: int = 0
    validate_errors: int = 0
    validate_warnings: int = 0
    duration_kind_counts: dict[str, int] = field(default_factory=dict)
    slide_pattern_counts: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class DatasetSongEntry:
    """Manifest entry for one source song directory."""

    song_id: str
    title: str | None
    artist: str | None
    maidata_path: str
    audio_path: str | None
    background_path: str | None
    has_audio: bool
    difficulties: list[DatasetDifficultyEntry] = field(default_factory=list)


@dataclass(slots=True)
class DatasetManifest:
    """Dataset manifest produced by the V2 cache builder."""

    schema: str
    source_root: str
    cache_dir: str
    song_count: int
    difficulty_count: int
    songs: list[DatasetSongEntry] = field(default_factory=list)
    errors: list[DatasetBuildError] = field(default_factory=list)


def discover_maidata_files(source_root: str | Path) -> list[Path]:
    """Recursively find ``maidata.txt`` files under a source directory."""

    root = Path(source_root)
    if root.is_file():
        return [root] if root.name.lower() == "maidata.txt" else []
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("maidata.txt") if path.is_file())


def build_dataset_manifest(
    source_root: str | Path,
    *,
    cache_dir: str | Path,
    force: bool = False,
    encoding: str | None = None,
    path_base: str | Path | None = None,
) -> DatasetManifest:
    """Build a dataset manifest and ChartIR cache files from local samples."""

    source_root_path = Path(source_root).resolve()
    cache_dir_path = Path(cache_dir).resolve()
    base_path = Path(path_base).resolve() if path_base is not None else Path.cwd().resolve()
    songs: list[DatasetSongEntry] = []
    errors: list[DatasetBuildError] = []

    for maidata_path in discover_maidata_files(source_root_path):
        song_id = infer_song_id(maidata_path)
        try:
            chart = parse_maidata_file(maidata_path, encoding=encoding)
        except Exception as exc:  # noqa: BLE001 - manifest build must continue.
            errors.append(
                _build_error(
                    "parse_maidata",
                    exc,
                    maidata_path=maidata_path,
                    song_id=song_id,
                    base_path=base_path,
                )
            )
            continue

        audio_path = _find_first_existing(maidata_path.parent, AUDIO_FILENAMES)
        background_path = _find_first_existing(maidata_path.parent, BACKGROUND_FILENAMES)
        song = DatasetSongEntry(
            song_id=song_id,
            title=chart.title,
            artist=chart.artist,
            maidata_path=_display_path(maidata_path, base_path),
            audio_path=_display_path(audio_path, base_path) if audio_path else None,
            background_path=_display_path(background_path, base_path) if background_path else None,
            has_audio=audio_path is not None,
        )

        if not chart.difficulties:
            errors.append(
                DatasetBuildError(
                    stage="parse_maidata",
                    message="No difficulty blocks found.",
                    maidata_path=song.maidata_path,
                    song_id=song_id,
                )
            )

        for difficulty in chart.difficulties:
            try:
                song.difficulties.append(
                    _build_difficulty_entry(
                        chart,
                        song_id=song_id,
                        difficulty_index=difficulty.index,
                        cache_dir=cache_dir_path,
                        force=force,
                        base_path=base_path,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - one bad difficulty should not kill the build.
                errors.append(
                    _build_error(
                        "difficulty",
                        exc,
                        maidata_path=maidata_path,
                        song_id=song_id,
                        difficulty_index=difficulty.index,
                        base_path=base_path,
                    )
                )

        songs.append(song)

    return DatasetManifest(
        schema=DATASET_MANIFEST_SCHEMA,
        source_root=_display_path(source_root_path, base_path),
        cache_dir=_display_path(cache_dir_path, base_path),
        song_count=len(songs),
        difficulty_count=sum(len(song.difficulties) for song in songs),
        songs=songs,
        errors=errors,
    )


def save_dataset_manifest(manifest: DatasetManifest, path: str | Path) -> None:
    """Write a dataset manifest JSON file."""

    Path(path).write_text(dataset_manifest_to_json(manifest), encoding="utf-8")


def dataset_manifest_to_dict(manifest: DatasetManifest) -> dict[str, Any]:
    """Convert a dataset manifest to JSON-compatible primitives."""

    return asdict(manifest)


def dataset_manifest_to_json(manifest: DatasetManifest, *, indent: int = 2) -> str:
    """Serialize a dataset manifest to JSON."""

    return json.dumps(dataset_manifest_to_dict(manifest), ensure_ascii=False, indent=indent)


def infer_song_id(maidata_path: str | Path) -> str:
    """Infer a stable song id from the parent directory name."""

    name = Path(maidata_path).parent.name.strip()
    return _safe_path_segment(name or "unknown")


def _build_difficulty_entry(
    chart: RawMaidataChart,
    *,
    song_id: str,
    difficulty_index: int,
    cache_dir: Path,
    force: bool,
    base_path: Path,
) -> DatasetDifficultyEntry:
    stats = compute_raw_maidata_stats(chart, difficulty_index=difficulty_index)
    if not stats.difficulties:
        raise ValueError(f"No stats produced for difficulty {difficulty_index}.")
    difficulty_stats = stats.difficulties[0]
    report = validate_raw_maidata_chart(chart, difficulty_index=difficulty_index)
    chart_ir_path = cache_dir / song_id / f"difficulty_{difficulty_index}.chart_ir.json"

    if force or not chart_ir_path.exists():
        chart_ir_path.parent.mkdir(parents=True, exist_ok=True)
        save_chart_json(build_chart_ir_by_difficulty_index(chart, difficulty_index), chart_ir_path)

    return _difficulty_entry_from_stats(
        difficulty_stats,
        chart_ir_path=_display_path(chart_ir_path, base_path),
        validate_errors=report.errors,
        validate_warnings=report.warnings,
    )


def _difficulty_entry_from_stats(
    stats: DifficultyStats,
    *,
    chart_ir_path: str,
    validate_errors: int,
    validate_warnings: int,
) -> DatasetDifficultyEntry:
    return DatasetDifficultyEntry(
        index=stats.difficulty_index,
        level=stats.level,
        designer=stats.designer,
        chart_ir_path=chart_ir_path,
        timing_points=stats.timing_points,
        note_count=stats.note_count,
        type_counts=stats.type_counts,
        parse_coverage=stats.parse_coverage,
        unknown_token_count=stats.unknown_token_count,
        validate_errors=validate_errors,
        validate_warnings=validate_warnings,
        duration_kind_counts=stats.duration_kind_counts,
        slide_pattern_counts=stats.slide_pattern_counts,
    )


def _find_first_existing(directory: Path, filenames: tuple[str, ...]) -> Path | None:
    for filename in filenames:
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    return None


def _build_error(
    stage: str,
    exc: Exception,
    *,
    maidata_path: Path,
    song_id: str,
    base_path: Path,
    difficulty_index: int | None = None,
) -> DatasetBuildError:
    return DatasetBuildError(
        stage=stage,
        message=str(exc),
        maidata_path=_display_path(maidata_path, base_path),
        song_id=song_id,
        difficulty_index=difficulty_index,
        exception_type=type(exc).__name__,
    )


def _display_path(path: Path, base_path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(base_path).as_posix()
    except ValueError:
        return str(resolved)


def _safe_path_segment(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("._") or "unknown"
