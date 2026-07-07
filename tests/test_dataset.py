import json
from pathlib import Path
from unittest import SkipTest

from maichart import (
    build_dataset_manifest,
    dataset_manifest_to_dict,
    discover_maidata_files,
    load_chart_json,
)
from maichart.cli import main


def _write_maidata(directory: Path, *, title: str = "Dataset Song") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "maidata.txt"
    path.write_text(
        f"&title={title}\n"
        "&artist=Tester\n"
        "&wholebpm=120\n"
        "&lv_1=1\n"
        "&des_1=Basic Designer\n"
        "&inote_1={4}1,2h[4:1],3-6[4:1],E\n",
        encoding="utf-8",
    )
    return path


def test_discover_scans_directory_with_one_maidata(tmp_path) -> None:
    maidata_path = _write_maidata(tmp_path / "sample_a")

    assert discover_maidata_files(tmp_path) == [maidata_path]


def test_build_manifest_for_multiple_sample_directories(tmp_path) -> None:
    _write_maidata(tmp_path / "417", title="Song 417")
    _write_maidata(tmp_path / "513", title="Song 513")

    manifest = build_dataset_manifest(
        tmp_path,
        cache_dir=tmp_path / "cache",
        path_base=tmp_path,
    )
    data = dataset_manifest_to_dict(manifest)

    assert data["song_count"] == 2
    assert data["difficulty_count"] == 2
    assert [song["song_id"] for song in data["songs"]] == ["417", "513"]
    assert data["errors"] == []


def test_missing_audio_is_recorded_as_has_audio_false(tmp_path) -> None:
    _write_maidata(tmp_path / "no_audio")

    manifest = build_dataset_manifest(
        tmp_path,
        cache_dir=tmp_path / "cache",
        path_base=tmp_path,
    )

    assert manifest.songs[0].has_audio is False
    assert manifest.songs[0].audio_path is None


def test_present_audio_is_recorded_as_has_audio_true(tmp_path) -> None:
    sample_dir = tmp_path / "with_audio"
    _write_maidata(sample_dir)
    (sample_dir / "track.mp3").write_bytes(b"fake mp3")
    (sample_dir / "bg.png").write_bytes(b"fake png")

    manifest = build_dataset_manifest(
        tmp_path,
        cache_dir=tmp_path / "cache",
        path_base=tmp_path,
    )
    song = manifest.songs[0]

    assert song.has_audio is True
    assert song.audio_path == "with_audio/track.mp3"
    assert song.background_path == "with_audio/bg.png"


def test_successful_build_generates_chart_ir_cache_and_stats(tmp_path) -> None:
    _write_maidata(tmp_path / "cache_song")

    manifest = build_dataset_manifest(
        tmp_path,
        cache_dir=tmp_path / "cache",
        path_base=tmp_path,
    )
    difficulty = manifest.songs[0].difficulties[0]
    chart_ir_path = tmp_path / difficulty.chart_ir_path
    ir = load_chart_json(chart_ir_path)

    assert chart_ir_path.is_file()
    assert ir.metadata.title == "Dataset Song"
    assert difficulty.timing_points == 4
    assert difficulty.note_count == 3
    assert difficulty.type_counts == {"tap": 1, "hold": 1, "slide": 1}
    assert difficulty.parse_coverage == 1.0
    assert difficulty.unknown_token_count == 0
    assert difficulty.validate_errors == 0
    assert difficulty.duration_kind_counts == {"grid_fraction": 2}
    assert difficulty.slide_pattern_counts == {"-": 1}


def test_parse_failure_is_recorded_without_stopping_build(tmp_path) -> None:
    _write_maidata(tmp_path / "good")
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    (bad_dir / "maidata.txt").write_bytes(b"\xff\xfe\x00")

    manifest = build_dataset_manifest(
        tmp_path,
        cache_dir=tmp_path / "cache",
        encoding="utf-8",
        path_base=tmp_path,
    )

    assert manifest.song_count == 1
    assert manifest.difficulty_count == 1
    assert len(manifest.errors) == 1
    assert manifest.errors[0].stage == "parse_maidata"
    assert manifest.errors[0].song_id == "bad"


def test_force_rebuilds_existing_chart_ir_cache(tmp_path) -> None:
    _write_maidata(tmp_path / "force_song")
    cache_dir = tmp_path / "cache"
    manifest = build_dataset_manifest(tmp_path, cache_dir=cache_dir, path_base=tmp_path)
    chart_ir_path = tmp_path / manifest.songs[0].difficulties[0].chart_ir_path
    chart_ir_path.write_text("stale cache", encoding="utf-8")

    build_dataset_manifest(tmp_path, cache_dir=cache_dir, path_base=tmp_path)
    assert chart_ir_path.read_text(encoding="utf-8") == "stale cache"

    build_dataset_manifest(tmp_path, cache_dir=cache_dir, path_base=tmp_path, force=True)
    assert json.loads(chart_ir_path.read_text(encoding="utf-8"))["schema_version"] == 1


def test_dataset_cli_smoke(tmp_path) -> None:
    _write_maidata(tmp_path / "cli_song")
    output = tmp_path / "manifest.json"
    cache_dir = tmp_path / "chart_ir_cache"

    assert main(["dataset", "build", str(tmp_path), "-o", str(output), "--cache-dir", str(cache_dir)]) == 0

    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["schema"] == "maichart-dataset-manifest-v1"
    assert data["song_count"] == 1
    assert data["difficulty_count"] == 1
    assert (tmp_path / data["songs"][0]["difficulties"][0]["chart_ir_path"]).is_file()


def test_417_and_513_dataset_smoke_if_available(tmp_path) -> None:
    candidates = []
    for sample_id in ("417", "513"):
        matches = sorted(Path("sample_runs").glob(f"*{sample_id}*/maidata.txt"))
        if matches:
            candidates.append(matches[-1])
    if not candidates:
        raise SkipTest("417/513 sample maidata.txt files are not available in sample_runs.")

    for maidata_path in candidates:
        manifest = build_dataset_manifest(
            maidata_path,
            cache_dir=tmp_path / "cache",
            path_base=tmp_path,
        )
        assert manifest.song_count == 1
        assert manifest.difficulty_count >= 1
        assert manifest.errors == []
