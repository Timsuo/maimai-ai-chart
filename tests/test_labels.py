import json
from pathlib import Path
from unittest import SkipTest

from maichart import (
    BpmEvent,
    ChartIR,
    ChartMetadata,
    DifficultyMetadata,
    Note,
    TimingData,
    build_dataset_manifest,
    build_frame_labels_from_chart_ir,
    frame_labels_to_dict,
    load_chart_json,
    load_frame_labels_json,
    save_chart_json,
    save_dataset_manifest,
    save_frame_labels_json,
)
from maichart.cli import main


def _chart(notes: list[Note]) -> ChartIR:
    return ChartIR(
        metadata=ChartMetadata(title="Labels"),
        difficulty=DifficultyMetadata(index=5, level="12"),
        timing=TimingData(bpms=[BpmEvent(bpm=120.0, beat=0.0, tick=0, time_sec=0.0)]),
        notes=notes,
    )


def test_empty_chart_ir_generates_empty_frame_labels() -> None:
    labels = build_frame_labels_from_chart_ir(_chart([]), song_id="empty")

    assert labels.schema == "maichart-frame-labels-v1"
    assert labels.song_id == "empty"
    assert labels.grid.ticks_per_frame == 480
    assert len(labels.frames) == 1
    assert labels.frames[0].labels.has_note is False


def test_single_tap_maps_to_correct_frame() -> None:
    labels = build_frame_labels_from_chart_ir(
        _chart([Note(note_type="tap", position="1", tick=480, beat=0.25, time_sec=0.125)])
    )

    assert labels.frames[1].tick == 480
    assert labels.frames[1].labels.has_note is True
    assert labels.frames[1].labels.note_count == 1
    assert labels.frames[1].labels.tap_count == 1
    assert labels.frames[1].labels.positions == ["1"]


def test_break_tap_counts_as_tap_and_break() -> None:
    labels = build_frame_labels_from_chart_ir(
        _chart([Note(note_type="tap", position="2", tick=0, modifiers={"break": True})])
    )

    frame_labels = labels.frames[0].labels
    assert frame_labels.tap_count == 1
    assert frame_labels.break_count == 1


def test_hold_start_and_active_frames() -> None:
    labels = build_frame_labels_from_chart_ir(
        _chart(
            [
                Note(
                    note_type="hold",
                    position="3",
                    tick=0,
                    duration_ticks=960,
                    duration={"raw": "[8:1]", "kind": "grid_fraction"},
                )
            ]
        )
    )

    assert labels.frames[0].labels.hold_start_count == 1
    assert labels.frames[0].labels.hold_active_count == 1
    assert labels.frames[1].labels.hold_active_count == 1
    assert labels.frames[2].labels.hold_active_count == 0
    assert labels.frames[0].labels.duration_kinds == ["grid_fraction"]


def test_zero_duration_hold_does_not_crash() -> None:
    labels = build_frame_labels_from_chart_ir(
        _chart([Note(note_type="hold", position="4", tick=0, duration_ticks=0)])
    )

    assert labels.frames[0].labels.hold_start_count == 1
    assert labels.frames[0].labels.hold_active_count == 0


def test_slide_start_and_active_frames() -> None:
    labels = build_frame_labels_from_chart_ir(
        _chart(
            [
                Note(
                    note_type="slide",
                    position="1",
                    end_position="4",
                    tick=0,
                    duration_ticks=960,
                    segments=[{"pattern": "-", "duration": {"kind": "grid_fraction"}}],
                    modifiers={"slide_pattern": "-"},
                )
            ]
        )
    )

    assert labels.frames[0].labels.slide_start_count == 1
    assert labels.frames[0].labels.slide_active_count == 1
    assert labels.frames[1].labels.slide_active_count == 1
    assert labels.frames[0].labels.slide_patterns == ["-"]


def test_chained_slide_records_segment_patterns() -> None:
    labels = build_frame_labels_from_chart_ir(
        _chart(
            [
                Note(
                    note_type="slide",
                    position="1",
                    tick=0,
                    duration_ticks=1440,
                    segments=[
                        {"pattern": "-", "duration": {"kind": "grid_fraction"}},
                        {"pattern": "<", "duration": {"kind": "grid_fraction"}},
                    ],
                    modifiers={"slide_pattern": "-"},
                )
            ]
        )
    )

    assert labels.frames[0].labels.slide_start_count == 1
    assert labels.frames[0].labels.slide_patterns == ["-", "<"]
    assert labels.frames[0].labels.duration_kinds == ["grid_fraction"]


def test_touch_and_touch_hold_counts() -> None:
    labels = build_frame_labels_from_chart_ir(
        _chart(
            [
                Note(note_type="touch", position="A1", tick=0),
                Note(note_type="touch_hold", position="C", tick=0, duration_ticks=480),
            ]
        )
    )

    frame_labels = labels.frames[0].labels
    assert frame_labels.touch_count == 1
    assert frame_labels.touch_hold_start_count == 1
    assert frame_labels.hold_active_count == 1
    assert frame_labels.positions == ["A1", "C"]


def test_seconds_duration_does_not_crash() -> None:
    labels = build_frame_labels_from_chart_ir(
        _chart(
            [
                Note(
                    note_type="hold",
                    position="5",
                    tick=0,
                    duration_sec=0.5,
                    duration={"raw": "[#0.5]", "kind": "seconds"},
                )
            ]
        )
    )

    assert labels.frames[0].labels.hold_start_count == 1
    assert labels.frames[0].labels.duration_kinds == ["seconds"]


def test_timing_pair_duration_does_not_crash() -> None:
    labels = build_frame_labels_from_chart_ir(
        _chart(
            [
                Note(
                    note_type="slide",
                    position="1",
                    tick=0,
                    duration_sec=1.0,
                    duration={"raw": "[0.1##1.0]", "kind": "timing_pair"},
                    segments=[{"pattern": ">", "duration": {"kind": "timing_pair"}}],
                )
            ]
        )
    )

    assert labels.frames[0].labels.slide_start_count == 1
    assert labels.frames[0].labels.duration_kinds == ["timing_pair"]


def test_frame_labels_json_round_trip_preserves_fields(tmp_path) -> None:
    path = tmp_path / "labels.json"
    labels = build_frame_labels_from_chart_ir(
        _chart([Note(note_type="tap", position="1", tick=0)]),
        song_id="round_trip",
    )
    save_path = path

    save_frame_labels_json(labels, save_path)
    loaded = load_frame_labels_json(save_path)

    assert frame_labels_to_dict(loaded) == frame_labels_to_dict(labels)


def test_cli_single_file_generates_labels(tmp_path) -> None:
    chart_path = tmp_path / "song" / "difficulty_5.chart_ir.json"
    labels_path = tmp_path / "labels.json"
    chart_path.parent.mkdir(parents=True)
    save_chart_json(_chart([Note(note_type="tap", position="1", tick=0)]), chart_path)

    assert main(["labels", "build", str(chart_path), "-o", str(labels_path), "--division", "16"]) == 0

    data = json.loads(labels_path.read_text(encoding="utf-8"))
    assert data["schema"] == "maichart-frame-labels-v1"
    assert data["song_id"] == "song"
    assert data["frames"][0]["labels"]["tap_count"] == 1


def test_manifest_batch_generates_labels_and_updates_manifest(tmp_path, monkeypatch) -> None:
    sample_dir = tmp_path / "raw" / "513"
    sample_dir.mkdir(parents=True)
    (sample_dir / "maidata.txt").write_text(
        "&title=Batch\n"
        "&artist=Tester\n"
        "&wholebpm=120\n"
        "&lv_5=12\n"
        "&inote_5={16}1,2h[8:1],E\n",
        encoding="utf-8",
    )
    manifest = build_dataset_manifest(
        tmp_path / "raw",
        cache_dir=tmp_path / "chart_ir_cache",
        path_base=tmp_path,
    )
    manifest_path = tmp_path / "manifest.json"
    save_dataset_manifest(manifest, manifest_path)

    monkeypatch.chdir(tmp_path)

    assert main(["dataset", "labels", str(manifest_path), "--out-dir", "frame_labels"]) == 0

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    difficulty = data["songs"][0]["difficulties"][0]
    assert "frame_labels_path" in difficulty
    assert (tmp_path / difficulty["frame_labels_path"]).is_file()
    assert difficulty["frame_labels_path"].startswith("frame_labels/")


def test_417_and_513_frame_label_smoke_if_available(tmp_path) -> None:
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
            cache_dir=tmp_path / "chart_ir_cache",
            path_base=tmp_path,
        )
        difficulty = manifest.songs[0].difficulties[0]
        chart = load_chart_json(tmp_path / difficulty.chart_ir_path)
        labels = build_frame_labels_from_chart_ir(chart, song_id=manifest.songs[0].song_id)
        assert labels.frames
        assert any(frame.labels.has_note for frame in labels.frames)
