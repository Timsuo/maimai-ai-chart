import csv
import importlib.util
import json
from pathlib import Path

from maichart import (
    SlideEvent,
    SlideSegment,
    build_chart_ir_by_difficulty_index,
    build_frame_labels_from_chart_ir,
    build_frame_labels_from_event_ir,
    chart_ir_to_event_ir,
    frame_labels_to_dict,
    parse_maidata_metadata,
    save_chart_json,
)


def test_tap_generates_v2_start_fields() -> None:
    labels = _event_labels_from_inote("{8}1,E")
    frame = labels["frames"][0]["labels"]

    assert labels["schema"] == "maichart-frame-labels-v2"
    assert frame["has_note"] is True
    assert frame["note_count"] == 1
    assert frame["note_start_count"] == 1
    assert frame["tap_count"] == 1
    assert frame["positions"] == ["1"]


def test_break_tap_generates_break_count() -> None:
    labels = _event_labels_from_inote("{8}1b,E")
    frame = labels["frames"][0]["labels"]

    assert frame["tap_count"] == 1
    assert frame["break_count"] == 1


def test_hold_generates_start_and_active_frames() -> None:
    labels = _event_labels_from_inote("{8}2h[8:1],E")

    assert labels["frames"][0]["labels"]["hold_start_count"] == 1
    assert labels["frames"][0]["labels"]["hold_active_count"] == 1
    assert labels["frames"][1]["labels"]["hold_active_count"] == 1
    assert labels["frames"][2]["labels"]["hold_active_count"] == 0


def test_zero_duration_hold_does_not_crash() -> None:
    labels = _event_labels_from_inote("{8}4h[1:0],E")
    frame = labels["frames"][0]["labels"]

    assert frame["hold_start_count"] == 1
    assert frame["hold_active_count"] == 0


def test_simple_slide_legacy_policy_generates_head_and_motion() -> None:
    labels = _event_labels_from_inote("{8}1-4[8:1],E", slide_launch_policy="legacy")
    frame = labels["frames"][0]["labels"]

    assert frame["slide_start_count"] == 1
    assert frame["slide_head_count"] == 1
    assert frame["slide_motion_start_count"] == 1
    assert frame["slide_motion_active_count"] == 1
    assert labels["frames"][1]["labels"]["slide_motion_active_count"] == 1
    assert frame["slide_launch_offset_kinds"] == ["legacy_zero"]


def test_simple_slide_unknown_policy_records_unknown_without_motion() -> None:
    labels = _event_labels_from_inote("{8}1-4[8:1],E", slide_launch_policy="unknown")
    frame = labels["frames"][0]["labels"]

    assert frame["slide_start_count"] == 1
    assert frame["slide_head_count"] == 1
    assert frame["slide_motion_start_count"] == 0
    assert frame["slide_motion_active_count"] == 0
    assert frame["slide_unknown_launch_offset_count"] == 1
    assert frame["slide_launch_offset_kinds"] == ["unknown"]


def test_chained_slide_records_multiple_segment_patterns() -> None:
    labels = _event_labels_from_inote("{8}1-4[8:1]*<6[8:1],E")
    frame = labels["frames"][0]["labels"]

    assert frame["slide_patterns"] == ["-", "<"]
    assert frame["slide_travel_duration_kinds"] == ["compound"]


def test_final_slide_endpoint_does_not_affect_start_position_label() -> None:
    chart = _event_ir_from_inote("{8}1-4[8:1]*-6[8:1],E")
    slide = next(event for event in chart.events if isinstance(event, SlideEvent))
    assert slide.end_position == "6"

    labels = build_frame_labels_from_event_ir(chart)

    assert labels["frames"][0]["labels"]["positions"] == ["1"]


def test_touch_and_touch_hold_generate_counts() -> None:
    labels = _event_labels_from_inote("{8}A1/Ch[8:1],E")
    frame = labels["frames"][0]["labels"]

    assert frame["touch_count"] == 1
    assert frame["touch_hold_start_count"] == 1
    assert frame["hold_active_count"] == 1
    assert frame["positions"] == ["A1", "C"]


def test_legacy_policy_simple_sample_aligns_with_v1_labels() -> None:
    chart_ir = _chart_ir_from_inote("{8}1/8,2h[8:1],1-4[8:1],A1,E")
    event_ir = chart_ir_to_event_ir(chart_ir)

    v1 = frame_labels_to_dict(build_frame_labels_from_chart_ir(chart_ir))
    v2 = build_frame_labels_from_event_ir(event_ir, slide_launch_policy="legacy")

    for frame_index, v1_frame in enumerate(v1["frames"]):
        v1_labels = v1_frame["labels"]
        v2_labels = v2["frames"][frame_index]["labels"]
        for key in (
            "has_note",
            "note_count",
            "tap_count",
            "break_count",
            "hold_start_count",
            "hold_active_count",
            "slide_start_count",
            "slide_active_count",
            "touch_count",
            "touch_hold_start_count",
            "note_types",
            "positions",
            "slide_patterns",
            "duration_kinds",
        ):
            assert v2_labels[key] == v1_labels[key]


def test_default_one_beat_policy_offsets_slide_motion() -> None:
    labels = _event_labels_from_inote(
        "{8}1-4[8:1],E",
        slide_launch_policy="default_one_beat",
    )

    assert labels["frames"][0]["labels"]["slide_head_count"] == 1
    assert labels["frames"][0]["labels"]["slide_motion_active_count"] == 0
    assert labels["frames"][4]["labels"]["slide_motion_start_count"] == 1
    assert labels["frames"][4]["labels"]["slide_motion_active_count"] == 1
    assert labels["frames"][4]["labels"]["slide_launch_offset_kinds"] == [
        "default_one_beat"
    ]


def test_explicit_slide_launch_offset_is_respected() -> None:
    event_ir = _event_ir_from_inote("{8}1-4[8:1],E")
    slide = next(event for event in event_ir.events if isinstance(event, SlideEvent))
    event_ir.events = [
        SlideEvent(
            head_tick=slide.head_tick,
            start_position=slide.start_position,
            launch_offset_ticks=480,
            travel_duration_ticks=slide.travel_duration_ticks,
            segments=[
                SlideSegment(
                    start_position=segment.start_position,
                    path_type=segment.path_type,
                    path_args=segment.path_args,
                    end_position=segment.end_position,
                    travel_duration_ticks=segment.travel_duration_ticks,
                    duration_raw=segment.duration_raw,
                    duration_kind=segment.duration_kind,
                    raw_notation=segment.raw_notation,
                    path_parts=segment.path_parts,
                )
                for segment in slide.segments
            ],
            end_position=slide.end_position,
        )
    ]

    labels = build_frame_labels_from_event_ir(event_ir, slide_launch_policy="unknown")

    assert labels["frames"][0]["labels"]["slide_head_count"] == 1
    assert labels["frames"][0]["labels"]["slide_motion_active_count"] == 0
    assert labels["frames"][1]["labels"]["slide_motion_start_count"] == 1
    assert labels["frames"][1]["labels"]["slide_launch_offset_kinds"] == ["explicit"]


def test_event_frame_labels_cache_builder_writes_side_path_cache(tmp_path) -> None:
    builder = _load_cache_builder()
    chart = _chart_ir_from_inote("{8}1/8,2h[8:1],1-4[8:1],A1,E")
    chart_path = tmp_path / "cache_qc_fixed" / "chart_ir" / "song_a" / "difficulty_5.chart_ir.json"
    chart_path.parent.mkdir(parents=True)
    save_chart_json(chart, chart_path)

    legacy_labels = frame_labels_to_dict(build_frame_labels_from_chart_ir(chart))
    legacy_path = tmp_path / "cache_qc_fixed" / "frame_labels" / "song_a" / "difficulty_5.frame_labels.json"
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_text(json.dumps(legacy_labels, ensure_ascii=False), encoding="utf-8")

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "maichart-training-manifest-v1",
                "songs": [
                    {
                        "song_id": "song_a",
                        "difficulties": [
                            {
                                "difficulty_index": 5,
                                "usable_for_training": True,
                                "chart_ir_path": str(chart_path),
                                "frame_labels_path": "legacy/frame_labels.json",
                            }
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = builder.build_event_frame_labels_cache(
        manifest_path=manifest_path,
        cache_dir=tmp_path / "cache_qc_fixed",
        out_cache_dir=tmp_path / "cache_event_v2",
        out_manifest_path=tmp_path / "manifest_event_v2.json",
        division=16,
        slide_launch_policy="legacy",
    )

    out_manifest = json.loads((tmp_path / "manifest_event_v2.json").read_text(encoding="utf-8"))
    difficulty = out_manifest["songs"][0]["difficulties"][0]
    labels_path = tmp_path / difficulty["event_frame_labels_path"]
    event_ir_path = (
        tmp_path
        / "cache_event_v2"
        / "event_ir"
        / "song_a"
        / "difficulty_5.event_ir.json"
    )
    v2 = json.loads(labels_path.read_text(encoding="utf-8"))

    assert event_ir_path.is_file()
    assert labels_path.is_file()
    assert difficulty["frame_labels_path"] == "legacy/frame_labels.json"
    assert difficulty["event_frame_labels_schema"] == "maichart-frame-labels-v2"
    assert difficulty["event_frame_labels_policy"] == "legacy"
    assert result["summary"]["written_event_ir_count"] == 1
    assert result["summary"]["written_frame_labels_v2_count"] == 1

    for frame_index, v1_frame in enumerate(legacy_labels["frames"]):
        v1_labels = v1_frame["labels"]
        v2_labels = v2["frames"][frame_index]["labels"]
        for key in _legacy_alignment_keys():
            assert v2_labels[key] == v1_labels[key]


def test_event_frame_labels_cache_builder_unknown_policy_summary(tmp_path) -> None:
    builder = _load_cache_builder()
    chart_path = tmp_path / "cache_qc_fixed" / "chart_ir" / "song_slide" / "difficulty_5.chart_ir.json"
    chart_path.parent.mkdir(parents=True)
    save_chart_json(_chart_ir_from_inote("{8}1-4[8:1],E"), chart_path)
    manifest_path = _write_manifest(
        tmp_path,
        [
            {
                "song_id": "song_slide",
                "difficulty_index": 5,
                "chart_ir_path": str(chart_path),
            }
        ],
    )

    result = builder.build_event_frame_labels_cache(
        manifest_path=manifest_path,
        cache_dir=tmp_path / "cache_qc_fixed",
        out_cache_dir=tmp_path / "cache_event_v2",
        out_manifest_path=tmp_path / "manifest_event_v2.json",
        division=16,
        slide_launch_policy="unknown",
    )

    summary = result["summary"]
    assert summary["slide_launch_policy"] == "unknown"
    assert summary["total_slide_heads"] == 1
    assert summary["total_slide_unknown_launch_offset_count"] == 1
    assert summary["unknown_launch_offset_per_slide_head"] == 1.0

    summary_csv = tmp_path / "cache_event_v2" / "event_frame_labels_build_summary.csv"
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["slide_launch_policy"] == "unknown"


def test_event_frame_labels_cache_builder_records_errors_without_stopping(tmp_path) -> None:
    builder = _load_cache_builder()
    good_chart_path = (
        tmp_path / "cache_qc_fixed" / "chart_ir" / "good_song" / "difficulty_5.chart_ir.json"
    )
    good_chart_path.parent.mkdir(parents=True)
    save_chart_json(_chart_ir_from_inote("{8}1,E"), good_chart_path)
    manifest_path = _write_manifest(
        tmp_path,
        [
            {
                "song_id": "good_song",
                "difficulty_index": 5,
                "chart_ir_path": str(good_chart_path),
            },
            {
                "song_id": "bad_song",
                "difficulty_index": 5,
                "chart_ir_path": str(tmp_path / "missing.chart_ir.json"),
            },
        ],
    )

    result = builder.build_event_frame_labels_cache(
        manifest_path=manifest_path,
        cache_dir=tmp_path / "cache_qc_fixed",
        out_cache_dir=tmp_path / "cache_event_v2",
        out_manifest_path=tmp_path / "manifest_event_v2.json",
        division=16,
        slide_launch_policy="legacy",
    )

    out_manifest = json.loads((tmp_path / "manifest_event_v2.json").read_text(encoding="utf-8"))
    good = out_manifest["songs"][0]["difficulties"][0]
    bad = out_manifest["songs"][1]["difficulties"][0]
    assert result["summary"]["sample_count"] == 2
    assert result["summary"]["written_frame_labels_v2_count"] == 1
    assert result["summary"]["error_count"] == 1
    assert "event_frame_labels_path" in good
    assert "event_frame_labels_path" not in bad

    errors_path = tmp_path / "cache_event_v2" / "event_frame_labels_build_errors.csv"
    with errors_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["song_id"] == "bad_song"
    assert rows[0]["error_type"] == "FileNotFoundError"


def _event_labels_from_inote(inote: str, *, slide_launch_policy: str = "legacy") -> dict:
    return build_frame_labels_from_event_ir(
        _event_ir_from_inote(inote),
        slide_launch_policy=slide_launch_policy,
    )


def _event_ir_from_inote(inote: str):
    return chart_ir_to_event_ir(_chart_ir_from_inote(inote))


def _chart_ir_from_inote(inote: str):
    raw = (
        "&title=Event Labels\n"
        "&artist=Tester\n"
        "&wholebpm=120\n"
        "&lv_5=13\n"
        f"&inote_5={inote}\n"
    )
    return build_chart_ir_by_difficulty_index(parse_maidata_metadata(raw), 5)


def _load_cache_builder():
    path = Path(__file__).resolve().parents[1] / "tools" / "build_event_frame_labels_cache.py"
    spec = importlib.util.spec_from_file_location("build_event_frame_labels_cache", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_manifest(tmp_path: Path, samples: list[dict]) -> Path:
    songs = []
    for sample in samples:
        songs.append(
            {
                "song_id": sample["song_id"],
                "difficulties": [
                    {
                        "difficulty_index": sample["difficulty_index"],
                        "usable_for_training": True,
                        "chart_ir_path": sample["chart_ir_path"],
                        "frame_labels_path": f"legacy/{sample['song_id']}.frame_labels.json",
                    }
                ],
            }
        )
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema": "maichart-training-manifest-v1",
                "songs": songs,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def _legacy_alignment_keys() -> tuple[str, ...]:
    return (
        "has_note",
        "note_count",
        "tap_count",
        "break_count",
        "hold_start_count",
        "hold_active_count",
        "slide_start_count",
        "slide_active_count",
        "touch_count",
        "touch_hold_start_count",
        "note_types",
        "positions",
        "slide_patterns",
        "duration_kinds",
    )
