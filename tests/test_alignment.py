import json
from pathlib import Path
from unittest import SkipTest

from maichart import (
    AudioBeat,
    AudioFeatureFrame,
    AudioFeatureSet,
    AudioOnset,
    ChartIR,
    ChartMetadata,
    DifficultyMetadata,
    FrameGrid,
    FrameLabel,
    FrameLabelSet,
    FrameLabels,
    alignment_report_to_dict,
    build_alignment_report,
    load_alignment_report_json,
    save_alignment_report_json,
    save_audio_features_json,
    save_chart_json,
    save_frame_labels_json,
)
from maichart.cli import main


def _chart() -> ChartIR:
    return ChartIR(
        metadata=ChartMetadata(title="Alignment Song"),
        difficulty=DifficultyMetadata(index=5, level="12"),
    )


def _frame_labels() -> FrameLabelSet:
    return FrameLabelSet(
        schema="maichart-frame-labels-v1",
        song_id="song",
        difficulty=5,
        grid=FrameGrid(division=16, ticks_per_beat=1920, ticks_per_frame=480),
        frames=[
            FrameLabel(
                frame_index=0,
                beat="0",
                tick=0,
                time_sec=1.0,
                labels=FrameLabels(
                    has_note=True,
                    note_count=2,
                    tap_count=2,
                    break_count=1,
                    note_types=["tap"],
                    positions=["1", "2"],
                ),
            ),
            FrameLabel(
                frame_index=1,
                beat="1/4",
                tick=480,
                time_sec=2.0,
                labels=FrameLabels(
                    has_note=True,
                    note_count=1,
                    hold_start_count=1,
                    note_types=["hold"],
                    positions=["3"],
                ),
            ),
            FrameLabel(
                frame_index=2,
                beat="1/2",
                tick=960,
                time_sec=3.0,
                labels=FrameLabels(
                    has_note=True,
                    note_count=3,
                    slide_start_count=1,
                    touch_count=1,
                    touch_hold_start_count=1,
                    note_types=["slide", "touch", "touch_hold"],
                    positions=["4", "A1", "C"],
                ),
            ),
            FrameLabel(
                frame_index=3,
                beat="3/4",
                tick=1440,
                time_sec=4.0,
                labels=FrameLabels(),
            ),
        ],
    )


def _audio_features() -> AudioFeatureSet:
    return AudioFeatureSet(
        schema="maichart-audio-features-v1",
        audio_path="track.wav",
        sample_rate=22050,
        duration_sec=5.0,
        tempo_bpm=120.0,
        beats=[AudioBeat(index=0, time_sec=1.0)],
        onsets=[
            AudioOnset(index=0, time_sec=1.02, strength=0.9),
            AudioOnset(index=1, time_sec=2.06, strength=0.7),
            AudioOnset(index=2, time_sec=3.2, strength=0.3),
        ],
        feature_frames=[
            AudioFeatureFrame(
                frame_index=0,
                time_sec=0.95,
                onset_strength=0.8,
                rms=0.2,
                percussive_rms=0.15,
                harmonic_rms=0.05,
                spectral_centroid=1000.0,
                spectral_bandwidth=500.0,
                zero_crossing_rate=0.1,
            ),
            AudioFeatureFrame(
                frame_index=1,
                time_sec=3.1,
                onset_strength=0.4,
                rms=0.1,
                percussive_rms=0.06,
                harmonic_rms=0.04,
                spectral_centroid=900.0,
                spectral_bandwidth=450.0,
                zero_crossing_rate=0.08,
            ),
        ],
    )


def test_simple_labels_and_audio_generate_report() -> None:
    report = build_alignment_report(_chart(), _frame_labels(), _audio_features())

    assert report.schema == "maichart-alignment-report-v1"
    assert report.song_id == "song"
    assert report.difficulty == 5
    assert report.summary.note_count == 6
    assert report.summary.frames_with_notes == 3
    assert len(report.frames) == 4


def test_nearest_onset_delta_is_correct() -> None:
    report = build_alignment_report(_chart(), _frame_labels(), _audio_features())

    assert round(report.frames[0].nearest_onset.delta_ms, 6) == 20.0
    assert round(report.frames[1].nearest_onset.delta_ms, 6) == 60.0
    assert round(report.frames[2].nearest_onset.delta_ms, 6) == 200.0


def test_onset_hit_rates_are_correct() -> None:
    report = build_alignment_report(_chart(), _frame_labels(), _audio_features())

    assert report.summary.onset_hit_rate_25ms == 1 / 3
    assert report.summary.onset_hit_rate_50ms == 1 / 3
    assert report.summary.onset_hit_rate_100ms == 2 / 3


def test_by_note_type_stats_are_correct() -> None:
    report = build_alignment_report(_chart(), _frame_labels(), _audio_features())

    assert report.by_note_type["tap"].count == 2
    assert report.by_note_type["break"].count == 1
    assert report.by_note_type["hold_start"].count == 1
    assert report.by_note_type["slide_start"].count == 1
    assert report.by_note_type["touch"].count == 2
    assert report.by_note_type["tap"].onset_hit_rate_50ms == 1.0
    assert report.by_note_type["hold_start"].onset_hit_rate_50ms == 0.0


def test_empty_frame_does_not_affect_stats() -> None:
    report = build_alignment_report(_chart(), _frame_labels(), _audio_features())

    assert report.frames[3].has_note is False
    assert report.frames[3].nearest_onset is None
    assert report.summary.frames_with_notes == 3


def test_no_onsets_does_not_crash() -> None:
    audio = _audio_features()
    audio.onsets = []

    report = build_alignment_report(_chart(), _frame_labels(), audio)

    assert report.summary.nearest_onset_mean_delta_ms is None
    assert report.summary.onset_hit_rate_50ms == 0.0
    assert report.frames[0].nearest_onset is None


def test_audio_frame_count_can_differ_from_chart_frame_count() -> None:
    report = build_alignment_report(_chart(), _frame_labels(), _audio_features())

    assert len(_audio_features().feature_frames) == 2
    assert len(report.frames) == 4
    assert all(frame.audio is not None for frame in report.frames)


def test_density_curves_exist_and_are_reasonable() -> None:
    report = build_alignment_report(_chart(), _frame_labels(), _audio_features())

    assert [point.note_count for point in report.density.notes_per_second] == [2, 1, 3]
    assert report.density.notes_per_4beat_window[0].note_count == 6
    assert report.density.notes_per_16beat_window[0].note_count == 6


def test_alignment_report_json_round_trip(tmp_path) -> None:
    path = tmp_path / "alignment_report.json"
    report = build_alignment_report(_chart(), _frame_labels(), _audio_features())

    save_alignment_report_json(report, path)
    loaded = load_alignment_report_json(path)

    assert alignment_report_to_dict(loaded) == alignment_report_to_dict(report)


def test_align_cli_single_file_smoke(tmp_path) -> None:
    chart_path = tmp_path / "chart.json"
    labels_path = tmp_path / "labels.json"
    audio_path = tmp_path / "audio.json"
    report_path = tmp_path / "alignment.json"
    save_chart_json(_chart(), chart_path)
    save_frame_labels_json(_frame_labels(), labels_path)
    save_audio_features_json(_audio_features(), audio_path)

    assert main([
        "align",
        str(chart_path),
        str(labels_path),
        str(audio_path),
        "-o",
        str(report_path),
    ]) == 0

    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["schema"] == "maichart-alignment-report-v1"
    assert data["summary"]["note_count"] == 6


def test_dataset_align_cli_batch_smoke(tmp_path) -> None:
    chart_path = tmp_path / "cache" / "song" / "difficulty_5.chart_ir.json"
    labels_path = tmp_path / "labels" / "song" / "difficulty_5.frame_labels.json"
    audio_path = tmp_path / "audio" / "song.audio_features.json"
    manifest_path = tmp_path / "manifest.json"
    chart_path.parent.mkdir(parents=True)
    save_chart_json(_chart(), chart_path)
    save_frame_labels_json(_frame_labels(), labels_path)
    save_audio_features_json(_audio_features(), audio_path)
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "maichart-dataset-manifest-v1",
                "source_root": ".",
                "cache_dir": "cache",
                "song_count": 1,
                "difficulty_count": 2,
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
                                "chart_ir_path": "cache/song/difficulty_5.chart_ir.json",
                                "frame_labels_path": "labels/song/difficulty_5.frame_labels.json",
                            },
                            {
                                "index": 4,
                                "chart_ir_path": "cache/song/difficulty_4.chart_ir.json",
                            },
                        ],
                    }
                ],
                "errors": [],
            }
        ),
        encoding="utf-8",
    )

    assert main([
        "dataset",
        "align",
        str(manifest_path),
        "--out-dir",
        str(tmp_path / "alignment"),
    ]) == 0

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    difficulties = data["songs"][0]["difficulties"]
    assert data["alignment_summary"] == {"processed": 1, "skipped": 1, "failed": 0}
    assert "alignment_report_path" in difficulties[0]
    assert (tmp_path / difficulties[0]["alignment_report_path"]).is_file()
    assert difficulties[1]["alignment_status"] == "skipped"


def test_real_sample_alignment_smoke_if_available(tmp_path) -> None:
    chart_path = Path("sample_runs/v2_chart_ir_cache/maichart_513/difficulty_5.chart_ir.json")
    labels_path = Path("sample_runs/v2_frame_labels/maichart_513/difficulty_5.frame_labels.json")
    audio_path = Path("sample_runs/v2_audio_features/maichart_513.audio_features.json")
    if not (chart_path.exists() and labels_path.exists() and audio_path.exists()):
        raise SkipTest("Real 513 ChartIR, frame labels, and audio features are not available.")

    assert main([
        "align",
        str(chart_path),
        str(labels_path),
        str(audio_path),
        "-o",
        str(tmp_path / "alignment_report.json"),
    ]) == 0

    report = load_alignment_report_json(tmp_path / "alignment_report.json")
    assert report.summary.note_count > 0
    assert report.summary.frames_with_notes > 0
