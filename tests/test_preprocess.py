import json
from pathlib import Path
from unittest import SkipTest

import pytest

from maichart import (
    AudioFeatureFrame,
    AudioFeatureSet,
    AudioOnset,
    build_dataset_splits,
    build_raw_sample_manifest,
    build_training_manifest,
    raw_sample_manifest_to_dict,
    training_manifest_to_dict,
)
from maichart.cli import main


def _write_sample(
    directory: Path,
    *,
    song_id: str | None = None,
    status: str = "success",
    level_1: str = "2",
    level_4: str = "12",
    inote_1: str | None = None,
    inote_4: str | None = None,
    audio: bool = True,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if inote_1 is None:
        inote_1 = "{4}" + ",".join(["1"] * 25)
    if inote_4 is None:
        inote_4 = "{4}" + ",".join(["1"] * 30)
    fields = [
        "&title=Preprocess Song",
        "&artist=Tester",
        "&wholebpm=120",
        f"&lv_1={level_1}",
        "&des_1=Easy Designer",
        f"&inote_1={inote_1}",
        f"&lv_4={level_4}",
        "&des_4=Expert Designer",
        f"&inote_4={inote_4}",
    ]
    (directory / "maidata.txt").write_text("\n".join(fields) + "\n", encoding="utf-8")
    if audio:
        (directory / "track.wav").write_bytes(b"fake audio")
    report = {"status": status}
    if song_id is not None:
        report["song_id"] = song_id
    (directory / "convert_report.json").write_text(json.dumps(report), encoding="utf-8")


def _patch_audio(monkeypatch) -> None:
    def fake_analyze(audio_path, division=16, sample_rate=22050, **kwargs):
        return AudioFeatureSet(
            schema="maichart-audio-features-v1",
            audio_path=str(audio_path),
            sample_rate=sample_rate,
            duration_sec=60.0,
            tempo_bpm=120.0,
            onsets=[
                AudioOnset(index=index, time_sec=index * 0.125, strength=1.0)
                for index in range(80)
            ],
            feature_frames=[
                AudioFeatureFrame(
                    frame_index=index,
                    time_sec=index * 0.125,
                    onset_strength=1.0,
                    rms=0.1,
                    percussive_rms=0.1,
                    harmonic_rms=0.0,
                    spectral_centroid=1000.0,
                    spectral_bandwidth=500.0,
                    zero_crossing_rate=0.1,
                )
                for index in range(80)
            ],
            division=division,
        )

    monkeypatch.setattr("maichart.preprocess.analyze_audio_file", fake_analyze)


def test_raw_scan_single_sample_reads_convert_report(tmp_path) -> None:
    _write_sample(tmp_path / "song", song_id="music001", status="failed")

    manifest = build_raw_sample_manifest(tmp_path, output_path=tmp_path / "raw.json")
    data = raw_sample_manifest_to_dict(manifest)

    assert data["schema"] == "maichart-raw-sample-manifest-v1"
    assert data["sample_count"] == 1
    sample = data["samples"][0]
    assert sample["song_id"] == "music001"
    assert sample["conversion_status"] == "failed"
    assert sample["dataset_usable"] is True
    assert sample["dataset_usable_reasons"] == ["maidata_exists", "audio_exists"]
    assert (tmp_path / "raw.json").is_file()


def test_raw_scan_multiple_samples_and_missing_files(tmp_path) -> None:
    _write_sample(tmp_path / "a", song_id="dup")
    _write_sample(tmp_path / "b", song_id="dup")
    report_only = tmp_path / "report_only"
    report_only.mkdir()
    (report_only / "convert_report.json").write_text('{"song_id": "report_only"}', encoding="utf-8")

    manifest = build_raw_sample_manifest(tmp_path)
    data = raw_sample_manifest_to_dict(manifest)

    assert data["sample_count"] == 3
    assert data["warnings"]
    assert [sample["song_id"] for sample in data["samples"][:2]] == ["dup__dup1", "dup__dup2"]
    assert data["samples"][2]["dataset_usable"] is False
    assert "missing_maidata" in data["samples"][2]["errors"]


def test_training_manifest_builds_caches_and_weights(tmp_path, monkeypatch) -> None:
    _patch_audio(monkeypatch)
    _write_sample(tmp_path / "song")

    manifest = build_training_manifest(
        tmp_path,
        cache_dir=tmp_path / "cache",
        output_path=tmp_path / "training.json",
        force=True,
    )
    data = training_manifest_to_dict(manifest)
    song = data["songs"][0]
    difficulties = {entry["difficulty_index"]: entry for entry in song["difficulties"]}

    assert data["difficulty_count"] == 2
    assert data["usable_difficulty_count"] == 2
    assert song["audio"]["audio_features_path"] == "cache/audio_features/song.audio_features.json"
    assert (tmp_path / difficulties[4]["chart_ir_path"]).is_file()
    assert (tmp_path / difficulties[4]["frame_labels_path"]).is_file()
    assert (tmp_path / difficulties[4]["alignment_report_path"]).is_file()
    assert difficulties[1]["training_weight"] < difficulties[4]["training_weight"]
    assert difficulties[4]["usable_for_training"] is True
    assert data["summary"]["filter_reason_counts"] == {}


def test_training_filters_empty_inote_unknown_level_low_notes_and_validation_errors(tmp_path, monkeypatch) -> None:
    _patch_audio(monkeypatch)
    _write_sample(
        tmp_path / "filters",
        level_1="?",
        inote_1="",
        level_4="12",
        inote_4="{4}1h[-1:1]",
    )

    manifest = build_training_manifest(
        tmp_path,
        cache_dir=tmp_path / "cache",
        output_path=tmp_path / "training.json",
        force=True,
    )
    data = training_manifest_to_dict(manifest)
    by_index = {
        difficulty["difficulty_index"]: difficulty
        for difficulty in data["songs"][0]["difficulties"]
    }

    assert by_index[1]["usable_for_training"] is False
    assert {"missing_chart", "note_count_too_low"} <= set(by_index[1]["filter_reasons"])
    assert "non_numeric_level" not in by_index[1]["filter_reasons"]
    assert by_index[1]["qc_tags"] == ["level_unknown"]
    assert by_index[1]["warnings"] == ["non_numeric_level"]
    assert "non_numeric_level" in by_index[1]["warning_codes"]
    assert by_index[4]["usable_for_training"] is False
    assert {"note_count_too_low", "validate_errors"} <= set(by_index[4]["filter_reasons"])
    assert data["summary"]["filter_reason_counts"]["note_count_too_low"] == 2
    assert "non_numeric_level" not in data["summary"]["filter_reason_counts"]
    assert data["summary"]["warning_code_counts"]["non_numeric_level"] == 1


def test_training_unknown_level_is_usable_when_other_qc_passes(tmp_path, monkeypatch) -> None:
    _patch_audio(monkeypatch)
    _write_sample(
        tmp_path / "unknown_level",
        level_1="?",
        inote_1="{4}" + ",".join(["1"] * 25),
        level_4="12",
    )

    manifest = build_training_manifest(
        tmp_path,
        cache_dir=tmp_path / "cache",
        output_path=tmp_path / "training.json",
        force=True,
    )
    data = training_manifest_to_dict(manifest)
    by_index = {
        difficulty["difficulty_index"]: difficulty
        for difficulty in data["songs"][0]["difficulties"]
    }

    assert by_index[1]["level_raw"] == "?"
    assert by_index[1]["level"] is None
    assert by_index[1]["usable_for_training"] is True
    assert by_index[1]["filter_reasons"] == []
    assert by_index[1]["qc_tags"] == ["level_unknown"]
    assert by_index[1]["warnings"] == ["non_numeric_level"]
    assert "non_numeric_level" in by_index[1]["warning_codes"]


def test_training_manifest_includes_remaster_difficulty_six(tmp_path, monkeypatch) -> None:
    _patch_audio(monkeypatch)
    sample_dir = tmp_path / "remaster"
    _write_sample(sample_dir)
    maidata = (sample_dir / "maidata.txt").read_text(encoding="utf-8")
    maidata += "\n&lv_6=?\n&des_6=Remaster Designer\n&inote_6={4}" + ",".join(["1"] * 25) + "\n"
    (sample_dir / "maidata.txt").write_text(maidata, encoding="utf-8")

    manifest = build_training_manifest(
        tmp_path,
        cache_dir=tmp_path / "cache",
        output_path=tmp_path / "training.json",
        force=True,
    )
    data = training_manifest_to_dict(manifest)
    by_index = {
        difficulty["difficulty_index"]: difficulty
        for difficulty in data["songs"][0]["difficulties"]
    }

    assert by_index[6]["difficulty_name"] == "remaster"
    assert by_index[6]["level_raw"] == "?"
    assert by_index[6]["level"] is None
    assert by_index[6]["usable_for_training"] is True
    assert by_index[6]["qc_tags"] == ["level_unknown"]


def test_training_single_parse_failure_does_not_stop_other_samples(tmp_path, monkeypatch) -> None:
    _patch_audio(monkeypatch)
    _write_sample(tmp_path / "good")
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "maidata.txt").write_bytes(b"\xff\xfe\x00")
    (bad / "track.wav").write_bytes(b"fake audio")

    manifest = build_training_manifest(
        tmp_path,
        cache_dir=tmp_path / "cache",
        output_path=tmp_path / "training.json",
        encoding="utf-8",
    )
    data = training_manifest_to_dict(manifest)

    assert data["song_count"] == 2
    assert data["difficulty_count"] == 2
    assert data["summary"]["parse_failure_count"] == 1
    assert any(error["stage"] == "parse_maidata" for error in data["errors"])


def test_split_is_reproducible_by_song_and_only_usable(tmp_path, monkeypatch) -> None:
    _patch_audio(monkeypatch)
    for index in range(10):
        _write_sample(tmp_path / f"song_{index:02d}")
    manifest = build_training_manifest(
        tmp_path,
        cache_dir=tmp_path / "cache",
        output_path=tmp_path / "training.json",
        force=True,
    )
    data = training_manifest_to_dict(manifest)
    data["songs"][0]["difficulties"][0]["usable_for_training"] = False
    Path(tmp_path / "training.json").write_text(json.dumps(data), encoding="utf-8")

    first = build_dataset_splits(
        tmp_path / "training.json",
        output_dir=tmp_path / "splits_a",
        seed=7,
    )
    second = build_dataset_splits(
        tmp_path / "training.json",
        output_dir=tmp_path / "splits_b",
        seed=7,
    )

    assert first.train == second.train
    train = json.loads((tmp_path / "splits_a" / "train_manifest.json").read_text(encoding="utf-8"))
    val = json.loads((tmp_path / "splits_a" / "val_manifest.json").read_text(encoding="utf-8"))
    test = json.loads((tmp_path / "splits_a" / "test_manifest.json").read_text(encoding="utf-8"))
    split_song_ids = [
        {song["song_id"] for song in split["songs"]}
        for split in (train, val, test)
    ]

    assert split_song_ids[0].isdisjoint(split_song_ids[1])
    assert split_song_ids[0].isdisjoint(split_song_ids[2])
    assert all(
        difficulty["usable_for_training"]
        for split in (train, val, test)
        for song in split["songs"]
        for difficulty in song["difficulties"]
    )
    summary = json.loads((tmp_path / "splits_a" / "split_summary.json").read_text(encoding="utf-8"))
    assert summary["level_distribution"]


def test_preprocess_cli_scan_build_split(tmp_path, monkeypatch) -> None:
    _patch_audio(monkeypatch)
    _write_sample(tmp_path / "song")

    assert main(["preprocess", "scan", str(tmp_path), "-o", str(tmp_path / "raw.json")]) == 0
    assert main([
        "preprocess",
        "build",
        str(tmp_path),
        "--cache-dir",
        str(tmp_path / "cache"),
        "-o",
        str(tmp_path / "training.json"),
        "--limit",
        "1",
    ]) == 0
    assert main([
        "preprocess",
        "split",
        str(tmp_path / "training.json"),
        "--out-dir",
        str(tmp_path / "splits"),
        "--seed",
        "42",
        "--split-by-song",
    ]) == 0

    assert json.loads((tmp_path / "raw.json").read_text(encoding="utf-8"))["sample_count"] == 1
    assert (tmp_path / "splits" / "split_summary.json").is_file()


def test_real_raw_chart_data_smoke_if_available(tmp_path) -> None:
    raw_root = Path("raw_chart_data/output")
    if not raw_root.exists():
        raise SkipTest("raw_chart_data/output is not available.")

    pytest.importorskip("librosa")
    scan = build_raw_sample_manifest(raw_root, output_path=tmp_path / "raw.json")
    if scan.sample_count == 0:
        raise SkipTest("No raw samples are available.")

    manifest = build_training_manifest(
        raw_root,
        cache_dir=tmp_path / "cache",
        output_path=tmp_path / "training.json",
        limit=1,
        sample_rate=8000,
    )
    data = training_manifest_to_dict(manifest)

    assert data["song_count"] == 1
    assert data["difficulty_count"] >= 1
    assert (tmp_path / "cache").exists()
