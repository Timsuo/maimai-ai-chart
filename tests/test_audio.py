import importlib
import json
from pathlib import Path
from unittest import SkipTest

import pytest

from maichart import (
    AudioFeatureDependencyError,
    analyze_audio_file,
    audio_features_to_dict,
    build_audio_features_for_dataset_manifest,
    load_audio_features_json,
    save_audio_features_json,
)
from maichart.cli import main


def _audio_deps():
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("librosa")
    return np, sf


def _write_synthetic_wav(path: Path) -> None:
    np, sf = _audio_deps()
    sample_rate = 8000
    duration = 0.75
    times = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    audio = 0.08 * np.sin(2.0 * np.pi * 440.0 * times)
    for onset_time in (0.15, 0.35, 0.55):
        start = int(onset_time * sample_rate)
        audio[start : start + 32] += np.hanning(32) * 0.8
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate)


def test_missing_librosa_has_clear_error(monkeypatch, tmp_path) -> None:
    original = importlib.import_module

    def fake_import(name, package=None):
        if name == "librosa":
            raise ImportError("missing librosa")
        return original(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    with pytest.raises(AudioFeatureDependencyError) as excinfo:
        analyze_audio_file(tmp_path / "missing.wav")

    assert "pip install -e \".[audio]\"" in str(excinfo.value)


def test_synthetic_wav_extracts_features(tmp_path) -> None:
    audio_path = tmp_path / "synthetic.wav"
    _write_synthetic_wav(audio_path)

    features = analyze_audio_file(
        audio_path,
        sample_rate=8000,
        hop_length=1024,
        max_feature_frames=64,
    )

    assert features.schema == "maichart-audio-features-v1"
    assert features.sample_rate == 8000
    assert 0.7 <= features.duration_sec <= 0.8
    assert features.tempo_bpm >= 0.0
    assert isinstance(features.beats, list)
    assert isinstance(features.onsets, list)
    assert features.feature_frames
    assert features.feature_frames[0].rms >= 0.0


def test_audio_features_json_round_trip(tmp_path) -> None:
    audio_path = tmp_path / "synthetic.wav"
    output_path = tmp_path / "features.json"
    _write_synthetic_wav(audio_path)

    features = analyze_audio_file(audio_path, sample_rate=8000, hop_length=1024, max_feature_frames=64)
    save_audio_features_json(features, output_path)
    loaded = load_audio_features_json(output_path)

    assert audio_features_to_dict(loaded) == audio_features_to_dict(features)


def test_audio_features_json_has_no_numpy_serialization_failure(tmp_path) -> None:
    audio_path = tmp_path / "synthetic.wav"
    _write_synthetic_wav(audio_path)

    payload = json.dumps(
        audio_features_to_dict(
            analyze_audio_file(audio_path, sample_rate=8000, hop_length=1024, max_feature_frames=64)
        )
    )

    assert "maichart-audio-features-v1" in payload


def test_onset_and_beat_fields_exist(tmp_path) -> None:
    audio_path = tmp_path / "synthetic.wav"
    _write_synthetic_wav(audio_path)

    data = audio_features_to_dict(
        analyze_audio_file(audio_path, sample_rate=8000, hop_length=1024, max_feature_frames=64)
    )

    assert "beats" in data
    assert "onsets" in data
    if data["onsets"]:
        assert {"index", "time_sec", "strength"} <= set(data["onsets"][0])


def test_manifest_without_audio_is_skipped_without_audio_dependencies(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "maichart-dataset-manifest-v1",
                "source_root": ".",
                "cache_dir": "chart_ir",
                "song_count": 1,
                "difficulty_count": 0,
                "songs": [
                    {
                        "song_id": "no_audio",
                        "title": "No Audio",
                        "artist": None,
                        "maidata_path": "no_audio/maidata.txt",
                        "audio_path": None,
                        "background_path": None,
                        "has_audio": False,
                        "difficulties": [],
                    }
                ],
                "errors": [],
            }
        ),
        encoding="utf-8",
    )

    summary = build_audio_features_for_dataset_manifest(
        manifest_path,
        out_dir=tmp_path / "audio_features",
    )
    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert summary["processed"] == 0
    assert summary["skipped"] == 1
    assert data["songs"][0]["audio_features_status"] == "skipped"


def test_manifest_with_audio_generates_audio_features_path(tmp_path) -> None:
    audio_path = tmp_path / "song" / "track.wav"
    _write_synthetic_wav(audio_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "maichart-dataset-manifest-v1",
                "source_root": ".",
                "cache_dir": "chart_ir",
                "song_count": 1,
                "difficulty_count": 0,
                "songs": [
                    {
                        "song_id": "song",
                        "title": "Song",
                        "artist": None,
                        "maidata_path": "song/maidata.txt",
                        "audio_path": "song/track.wav",
                        "background_path": None,
                        "has_audio": True,
                        "difficulties": [],
                    }
                ],
                "errors": [],
            }
        ),
        encoding="utf-8",
    )

    summary = build_audio_features_for_dataset_manifest(
        manifest_path,
        out_dir=tmp_path / "audio_features",
        force=True,
    )
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_path = tmp_path / data["songs"][0]["audio_features_path"]

    assert summary["processed"] == 1
    assert feature_path.is_file()
    assert data["songs"][0]["audio_features_status"] == "processed"


def test_cli_single_audio_analyze_smoke(tmp_path) -> None:
    audio_path = tmp_path / "synthetic.wav"
    output_path = tmp_path / "audio_features.json"
    _write_synthetic_wav(audio_path)

    assert main([
        "audio",
        "analyze",
        str(audio_path),
        "-o",
        str(output_path),
        "--sample-rate",
        "8000",
    ]) == 0

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["schema"] == "maichart-audio-features-v1"
    assert data["feature_frames"]


def test_cli_manifest_audio_features_smoke(tmp_path) -> None:
    audio_path = tmp_path / "song" / "track.wav"
    _write_synthetic_wav(audio_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "maichart-dataset-manifest-v1",
                "source_root": ".",
                "cache_dir": "chart_ir",
                "song_count": 1,
                "difficulty_count": 0,
                "songs": [
                    {
                        "song_id": "song",
                        "title": "Song",
                        "artist": None,
                        "maidata_path": "song/maidata.txt",
                        "audio_path": "song/track.wav",
                        "background_path": None,
                        "has_audio": True,
                        "difficulties": [],
                    }
                ],
                "errors": [],
            }
        ),
        encoding="utf-8",
    )

    assert main([
        "dataset",
        "audio-features",
        str(manifest_path),
        "--out-dir",
        str(tmp_path / "audio_features"),
        "--sample-rate",
        "8000",
        "--force",
    ]) == 0

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["audio_feature_summary"]["processed"] == 1
    assert (tmp_path / data["songs"][0]["audio_features_path"]).is_file()


def test_real_sample_audio_smoke_if_available(tmp_path) -> None:
    pytest.importorskip("librosa")
    candidates = sorted(Path("sample_runs").glob("*/track.mp3")) + sorted(
        Path("sample_runs").glob("*/track.wav")
    )
    if not candidates:
        raise SkipTest("No real sample audio is available in sample_runs.")

    features = analyze_audio_file(
        candidates[0],
        sample_rate=8000,
        hop_length=2048,
        max_feature_frames=128,
        max_duration_sec=3.0,
    )

    assert features.duration_sec > 0.0
    assert features.feature_frames
