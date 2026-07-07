import json
from pathlib import Path
from unittest import SkipTest

from maichart import (
    AudioBeat,
    AudioFeatureFrame,
    AudioFeatureSet,
    AudioOnset,
    ChartIR,
    FrameGrid,
    FrameLabel,
    FrameLabelSet,
    FrameLabels,
    build_alignment_report,
    build_rhythm_profile_from_dataset_manifest,
    evaluate_rhythm_skeleton,
    generate_rhythm_skeleton,
    load_rhythm_evaluation_json,
    load_rhythm_profile_json,
    load_rhythm_skeleton_json,
    rhythm_profile_to_dict,
    rhythm_skeleton_to_dict,
    RhythmSkeleton,
    RhythmSkeletonFrame,
    save_alignment_report_json,
    save_audio_features_json,
    save_chart_json,
    save_frame_labels_json,
    save_rhythm_profile_json,
    save_rhythm_skeleton_json,
)
from maichart.cli import main


def _audio_features() -> AudioFeatureSet:
    frames = []
    for index in range(40):
        strong = index in {2, 6, 10, 14, 18, 22, 26, 30, 34, 38}
        frames.append(
            AudioFeatureFrame(
                frame_index=index,
                time_sec=index * 0.1,
                onset_strength=1.0 if strong else 0.15 + (index % 3) * 0.05,
                rms=0.7 if strong else 0.25,
                percussive_rms=0.9 if strong else 0.1,
                harmonic_rms=0.25,
                spectral_centroid=1000.0,
                spectral_bandwidth=500.0,
                zero_crossing_rate=0.1,
            )
        )
    return AudioFeatureSet(
        schema="maichart-audio-features-v1",
        audio_path="sample_runs/song/track.wav",
        sample_rate=8000,
        duration_sec=4.0,
        tempo_bpm=120.0,
        beats=[AudioBeat(index=i, time_sec=i * 0.5) for i in range(8)],
        onsets=[
            AudioOnset(index=i, time_sec=time_sec, strength=1.0)
            for i, time_sec in enumerate([0.2, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8])
        ],
        feature_frames=frames,
    )


def _frame_labels() -> FrameLabelSet:
    frames = []
    for index in range(40):
        has_note = index in {2, 6, 10, 14, 18}
        frames.append(
            FrameLabel(
                frame_index=index,
                beat=str(index / 4).rstrip("0").rstrip("."),
                tick=index * 480,
                time_sec=index * 0.1,
                labels=FrameLabels(
                    has_note=has_note,
                    note_count=1 if has_note else 0,
                    tap_count=1 if index in {2, 6, 10} else 0,
                    break_count=1 if index == 6 else 0,
                    hold_start_count=1 if index == 14 else 0,
                    slide_start_count=1 if index == 18 else 0,
                    note_types=["tap"] if has_note else [],
                ),
            )
        )
    return FrameLabelSet(
        schema="maichart-frame-labels-v1",
        song_id="song",
        difficulty=5,
        grid=FrameGrid(division=16, ticks_per_beat=1920, ticks_per_frame=480),
        frames=frames,
    )


def test_rhythm_profile_json_round_trip(tmp_path) -> None:
    profile = _build_fake_profile(tmp_path)
    path = tmp_path / "profile.json"

    save_rhythm_profile_json(profile, path)
    loaded = load_rhythm_profile_json(path)

    assert rhythm_profile_to_dict(loaded) == rhythm_profile_to_dict(profile)


def test_build_profile_from_fake_dataset_manifest(tmp_path) -> None:
    profile = _build_fake_profile(tmp_path)

    assert profile.schema == "maichart-rhythm-profile-v1"
    assert profile.global_stats["sample_count"] == 1
    assert profile.level_bands[0].sample_count == 1
    assert profile.level_bands[0].target_note_density_per_sec > 0
    assert profile.level_bands[0].event_type_distribution["tap"] > 0


def test_generate_skeleton_from_synthetic_audio_features() -> None:
    skeleton = generate_rhythm_skeleton(_audio_features(), target_level=12.0)

    assert skeleton.schema == "maichart-rhythm-skeleton-v1"
    assert skeleton.summary["selected_frame_count"] > 0
    assert any(frame.selected for frame in skeleton.frames)
    assert all(frame.event_type is not None for frame in skeleton.frames if frame.selected)


def test_higher_target_level_selects_more_frames() -> None:
    low = generate_rhythm_skeleton(_audio_features(), target_level=5.0)
    high = generate_rhythm_skeleton(_audio_features(), target_level=13.0)

    assert high.summary["selected_frame_count"] > low.summary["selected_frame_count"]


def test_skeleton_does_not_select_overdense_frames() -> None:
    skeleton = generate_rhythm_skeleton(_audio_features(), target_level=8.0)
    selected_times = [frame.time_sec for frame in skeleton.frames if frame.selected]

    assert all(b - a >= 0.16 for a, b in zip(selected_times, selected_times[1:]))


def test_event_type_counts_match_selected_frames() -> None:
    skeleton = generate_rhythm_skeleton(_audio_features(), target_level=12.0)

    selected_count = sum(1 for frame in skeleton.frames if frame.selected)
    assert sum(skeleton.summary["event_type_counts"].values()) == selected_count


def test_rhythm_skeleton_json_round_trip(tmp_path) -> None:
    skeleton = generate_rhythm_skeleton(_audio_features(), target_level=12.0)
    path = tmp_path / "skeleton.json"

    save_rhythm_skeleton_json(skeleton, path)
    loaded = load_rhythm_skeleton_json(path)

    assert rhythm_skeleton_to_dict(loaded) == rhythm_skeleton_to_dict(skeleton)


def test_evaluation_precision_recall_f1() -> None:
    skeleton = RhythmSkeleton(
        schema="maichart-rhythm-skeleton-v1",
        song_id="song",
        target_level=12.0,
        division=16,
        duration_sec=4.0,
        frames=[
            RhythmSkeletonFrame(index, index * 0.1, None, None, True, "tap", 1.0, [])
            for index in (2, 6, 9, 30)
        ],
        summary={"selected_frame_count": 4, "event_type_counts": {"tap": 4}},
    )

    evaluation = evaluate_rhythm_skeleton(skeleton, _frame_labels(), tolerance_frames=1)

    assert evaluation.predicted_selected_frames == 4
    assert evaluation.reference_note_frames == 5
    assert evaluation.precision == 0.75
    assert evaluation.recall == 0.6
    assert round(evaluation.f1, 6) == round(2 * 0.75 * 0.6 / (0.75 + 0.6), 6)
    assert evaluation.note_count_error == -1


def test_cli_rhythm_profile_smoke(tmp_path) -> None:
    manifest_path = _write_fake_dataset(tmp_path)
    output = tmp_path / "profile.json"

    assert main(["rhythm", "profile", str(manifest_path), "-o", str(output)]) == 0

    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["schema"] == "maichart-rhythm-profile-v1"
    assert data["level_bands"]


def test_cli_rhythm_generate_smoke(tmp_path) -> None:
    audio_path = tmp_path / "audio.json"
    output = tmp_path / "skeleton.json"
    save_audio_features_json(_audio_features(), audio_path)

    assert main(["rhythm", "generate", str(audio_path), "--level", "12.5", "-o", str(output)]) == 0

    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["schema"] == "maichart-rhythm-skeleton-v1"
    assert data["summary"]["selected_frame_count"] > 0


def test_cli_rhythm_evaluate_smoke(tmp_path) -> None:
    skeleton_path = tmp_path / "skeleton.json"
    labels_path = tmp_path / "labels.json"
    output = tmp_path / "eval.json"
    save_rhythm_skeleton_json(generate_rhythm_skeleton(_audio_features(), target_level=12.0), skeleton_path)
    save_frame_labels_json(_frame_labels(), labels_path)

    assert main(["rhythm", "evaluate", str(skeleton_path), str(labels_path), "-o", str(output)]) == 0

    loaded = load_rhythm_evaluation_json(output)
    assert loaded.schema == "maichart-rhythm-skeleton-evaluation-v1"


def test_dataset_rhythm_batch_smoke(tmp_path) -> None:
    manifest_path = _write_fake_dataset(tmp_path)
    profile = build_rhythm_profile_from_dataset_manifest(manifest_path)
    profile_path = tmp_path / "profile.json"
    save_rhythm_profile_json(profile, profile_path)

    assert main([
        "dataset",
        "rhythm",
        str(manifest_path),
        "--out-dir",
        str(tmp_path / "skeletons"),
        "--profile",
        str(profile_path),
        "--level-source",
        "difficulty",
    ]) == 0

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    difficulty = data["songs"][0]["difficulties"][0]
    assert data["rhythm_skeleton_summary"] == {"processed": 1, "skipped": 0, "failed": 0}
    assert (tmp_path / difficulty["rhythm_skeleton_path"]).is_file()


def test_real_sample_rhythm_smoke_if_available(tmp_path) -> None:
    manifest_path = Path("sample_runs/v2_dataset_manifest.json")
    audio_path = Path("sample_runs/v2_audio_features/maichart_513.audio_features.json")
    labels_path = Path("sample_runs/v2_frame_labels/maichart_513/difficulty_5.frame_labels.json")
    if not (manifest_path.exists() and audio_path.exists() and labels_path.exists()):
        raise SkipTest("Real manifest, audio features, or frame labels are not available.")

    profile_path = tmp_path / "profile.json"
    skeleton_path = tmp_path / "skeleton.json"
    eval_path = tmp_path / "eval.json"

    assert main(["rhythm", "profile", str(manifest_path), "-o", str(profile_path)]) == 0
    assert main([
        "rhythm",
        "generate",
        str(audio_path),
        "--level",
        "12.0",
        "--profile",
        str(profile_path),
        "-o",
        str(skeleton_path),
    ]) == 0
    assert main(["rhythm", "evaluate", str(skeleton_path), str(labels_path), "-o", str(eval_path)]) == 0

    evaluation = load_rhythm_evaluation_json(eval_path)
    assert evaluation.predicted_selected_frames > 0
    assert evaluation.reference_note_frames > 0


def _build_fake_profile(tmp_path: Path):
    return build_rhythm_profile_from_dataset_manifest(_write_fake_dataset(tmp_path))


def _write_fake_dataset(tmp_path: Path) -> Path:
    chart_path = tmp_path / "chart_ir" / "song" / "difficulty_5.chart_ir.json"
    labels_path = tmp_path / "labels" / "song" / "difficulty_5.frame_labels.json"
    audio_path = tmp_path / "audio" / "song.audio_features.json"
    alignment_path = tmp_path / "alignment" / "song" / "difficulty_5.alignment_report.json"
    manifest_path = tmp_path / "manifest.json"
    chart_path.parent.mkdir(parents=True)
    save_chart_json(ChartIR(), chart_path)
    labels = _frame_labels()
    audio = _audio_features()
    save_frame_labels_json(labels, labels_path)
    save_audio_features_json(audio, audio_path)
    save_alignment_report_json(build_alignment_report(ChartIR(), labels, audio), alignment_path)
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "maichart-dataset-manifest-v1",
                "source_root": ".",
                "cache_dir": "chart_ir",
                "song_count": 1,
                "difficulty_count": 1,
                "songs": [
                    {
                        "song_id": "song",
                        "title": "Song",
                        "artist": None,
                        "maidata_path": "song/maidata.txt",
                        "audio_path": "song/track.wav",
                        "audio_features_path": "audio/song.audio_features.json",
                        "background_path": None,
                        "has_audio": True,
                        "difficulties": [
                            {
                                "index": 5,
                                "level": "12.0",
                                "chart_ir_path": "chart_ir/song/difficulty_5.chart_ir.json",
                                "frame_labels_path": "labels/song/difficulty_5.frame_labels.json",
                                "alignment_report_path": "alignment/song/difficulty_5.alignment_report.json",
                            }
                        ],
                    }
                ],
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    return manifest_path
