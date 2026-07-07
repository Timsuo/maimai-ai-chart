import importlib.util
import json
import sys
from pathlib import Path


def _load_tool_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "build_small_batch_manifest.py"
    spec = importlib.util.spec_from_file_location("build_small_batch_manifest", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_small_batch_manifest_from_training_manifest(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    manifest_path = tmp_path / "manifests" / "training_manifest.json"
    cache_dir = tmp_path / "cache"
    manifest_path.parent.mkdir()

    songs = []
    sample_specs = [
        ("music001", 1, 4, 35, {"tap": 32, "hold": 2, "slide": 1}, []),
        ("music002", 2, 7, 80, {"tap": 46, "hold": 26, "slide": 8}, ["low_reference_difficulty"]),
        ("music003", 3, 9, 160, {"tap": 100, "hold": 20, "slide": 40}, []),
        ("music004", 4, 12, 260, {"tap": 130, "hold": 30, "slide": 100}, []),
        ("music005", 5, 13, 420, {"tap": 210, "hold": 60, "slide": 150}, ["low_onset_hit_rate"]),
        ("music006", 4, 11, 210, {"tap": 80, "hold": 90, "slide": 40}, []),
        ("music007", 5, 14, 520, {"tap": 320, "hold": 80, "slide": 120}, []),
        ("music008", 2, 6, 55, {"tap": 48, "hold": 4, "slide": 3}, []),
    ]
    for index, (song_id, difficulty, level, note_count, type_counts, warnings) in enumerate(sample_specs):
        duration = 60.0 + index
        audio_path = cache_dir / "audio_features" / f"{song_id}.audio_features.json"
        chart_path = cache_dir / "chart_ir" / song_id / f"difficulty_{difficulty}.chart_ir.json"
        labels_path = cache_dir / "frame_labels" / song_id / f"difficulty_{difficulty}.frame_labels.json"
        report_path = cache_dir / "alignment_reports" / song_id / f"difficulty_{difficulty}.alignment_report.json"
        _write_json(audio_path, {"duration_sec": duration, "tempo_bpm": 120 + index})
        _write_json(chart_path, {"schema_version": "test"})
        _write_json(labels_path, _frame_labels(song_id, difficulty, duration, long_intro=index == 3))
        _write_json(report_path, {"summary": {"note_count": note_count}})
        songs.append(
            {
                "song_id": song_id,
                "title": f"Song {index}",
                "audio": {
                    "duration_sec": duration,
                    "estimated_bpm": 120 + index,
                    "audio_features_path": audio_path.relative_to(tmp_path).as_posix(),
                },
                "difficulties": [
                    {
                        "difficulty_index": difficulty,
                        "difficulty_name": "master" if difficulty == 5 else "expert",
                        "level": float(level),
                        "chart_ir_path": chart_path.relative_to(tmp_path).as_posix(),
                        "frame_labels_path": labels_path.relative_to(tmp_path).as_posix(),
                        "alignment_report_path": report_path.relative_to(tmp_path).as_posix(),
                        "note_count": note_count,
                        "type_counts": type_counts,
                        "warning_codes": warnings,
                        "usable_for_training": True,
                    }
                ],
            }
        )

    _write_json(
        manifest_path,
        {
            "schema": "maichart-training-manifest-v1",
            "song_count": len(songs),
            "difficulty_count": len(songs),
            "usable_difficulty_count": len(songs),
            "songs": songs,
        },
    )

    tool = _load_tool_module()
    output_path = tmp_path / "manifests" / "small_batch_v1.json"
    output = tool.build_small_batch_manifest(
        manifest_path,
        cache_dir=cache_dir,
        output_path=output_path,
        target_count=24,
        seed=7,
    )

    assert output_path.is_file()
    assert output["actual_sample_count"] == len(sample_specs)
    assert output["metadata"]["candidate_count"] == len(sample_specs)
    assert output["metadata"]["usable_candidate_count"] == len(sample_specs)
    assert output["metadata"]["warnings"]
    for sample in output["samples"]:
        assert sample["sample_id"]
        assert sample["music_id"]
        assert sample["difficulty"]
        assert sample["tags"]
        assert sample["selection_reason"]
        assert sample["paths"]["chart_ir"]
    all_tags = {tag for sample in output["samples"] for tag in sample["tags"]}
    assert {"low_difficulty", "medium_difficulty", "high_difficulty"} <= all_tags
    assert {"slide_heavy", "hold_heavy"} & all_tags


def test_small_batch_skips_entries_missing_difficulty_index(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    manifest_path = tmp_path / "manifests" / "training_manifest.json"
    cache_dir = tmp_path / "cache"
    manifest_path.parent.mkdir()
    song_id = "music001"
    audio_path = cache_dir / "audio_features" / f"{song_id}.audio_features.json"
    _write_json(audio_path, {"duration_sec": 60.0, "tempo_bpm": 120})
    _write_json(
        manifest_path,
        {
            "schema": "maichart-training-manifest-v1",
            "song_count": 1,
            "difficulty_count": 1,
            "usable_difficulty_count": 1,
            "songs": [
                {
                    "song_id": song_id,
                    "title": "Missing Index",
                    "audio": {
                        "duration_sec": 60.0,
                        "estimated_bpm": 120,
                        "audio_features_path": audio_path.relative_to(tmp_path).as_posix(),
                    },
                    "difficulties": [
                        {
                            "difficulty_name": "master",
                            "level": 13.0,
                            "note_count": 100,
                            "type_counts": {"tap": 90, "hold": 5, "slide": 5},
                            "usable_for_training": True,
                        }
                    ],
                }
            ],
        },
    )

    tool = _load_tool_module()
    output = tool.build_small_batch_manifest(
        manifest_path,
        cache_dir=cache_dir,
        output_path=tmp_path / "manifests" / "small_batch.json",
        target_count=24,
        seed=7,
    )

    assert output["actual_sample_count"] == 0
    assert output["metadata"]["missing_fields"]["difficulty_index"] == 1
    assert "difficulty_unknown" not in json.dumps(output)
    assert any("missing difficulty_index" in warning for warning in output["metadata"]["warnings"])


def _frame_labels(song_id: str, difficulty: int, duration: float, *, long_intro: bool) -> dict:
    frames = []
    first_note = 12 if long_intro else 1
    for index in range(int(duration)):
        note_count = 0
        if index >= first_note and index % 4 == 0:
            note_count = 1
        if index > duration * 0.65 and index % 2 == 0:
            note_count += 2
        frames.append(
            {
                "frame_index": index,
                "time_sec": float(index),
                "labels": {"has_note": note_count > 0, "note_count": note_count},
            }
        )
    return {
        "schema": "maichart-frame-labels-v1",
        "song_id": song_id,
        "difficulty": difficulty,
        "frames": frames,
    }


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
