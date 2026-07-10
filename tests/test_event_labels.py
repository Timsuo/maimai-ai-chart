from maichart import (
    SlideEvent,
    SlideSegment,
    build_chart_ir_by_difficulty_index,
    build_frame_labels_from_chart_ir,
    build_frame_labels_from_event_ir,
    chart_ir_to_event_ir,
    frame_labels_to_dict,
    parse_maidata_metadata,
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
