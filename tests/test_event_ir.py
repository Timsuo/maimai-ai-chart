from maichart import (
    ChartEventIR,
    ChartMetadata,
    DifficultyMetadata,
    HoldEvent,
    MeterEvent,
    SlideEvent,
    SlideSegment,
    TapEvent,
    TimingData,
    TimingEvent,
    TouchEvent,
    TouchHoldEvent,
    UnknownChartToken,
    build_chart_ir_by_difficulty_index,
    chart_ir_to_event_ir,
    event_ir_from_dict,
    event_ir_to_dict,
    parse_maidata_metadata,
)
from maichart.ir import BpmEvent, ChartIR, Note


def test_event_ir_dict_round_trip() -> None:
    event_ir = ChartEventIR(
        schema_version=1,
        metadata=ChartMetadata(title="Event"),
        difficulty=DifficultyMetadata(index=5, level="13"),
        timing_events=[TimingEvent(tick=0, bpm=180.0, raw_notation="(180)")],
        meter_events=[
            MeterEvent(tick=0, numerator=4, denominator=4, ticks_per_measure=7680)
        ],
        events=[
            TapEvent(tick=0, position="1", is_break=True, raw_notation="1b"),
            HoldEvent(
                head_tick=480,
                position="2",
                duration_ticks=960,
                duration_raw="[8:1]",
                duration_kind="grid_fraction",
            ),
            SlideEvent(
                head_tick=960,
                start_position="1",
                launch_offset_ticks=None,
                travel_duration_ticks=960,
                segments=[
                    SlideSegment(
                        start_position="1",
                        path_type="-",
                        path_args=["4"],
                        end_position="4",
                        travel_duration_ticks=960,
                    )
                ],
                end_position="4",
                raw_notation="1-4[8:1]",
            ),
            TouchEvent(tick=1440, area="C", position="C", firework=True),
            TouchHoldEvent(
                head_tick=1920,
                area="E",
                position="E3",
                duration_ticks=3840,
            ),
        ],
        unknown_tokens=[
            UnknownChartToken(tick=2400, raw_token="Z9", reason="unsupported")
        ],
        raw="&inote_5=...",
    )

    loaded = event_ir_from_dict(event_ir_to_dict(event_ir))

    assert loaded == event_ir
    assert event_ir_to_dict(loaded)["events"][2]["event_type"] == "slide"


def test_tap_conversion() -> None:
    event_ir = _event_ir_from_inote("{8}1,E")
    event = _single_event(event_ir, TapEvent)

    assert event.tick == 0
    assert event.position == "1"
    assert event.is_break is False
    assert event.is_ex is False
    assert event.raw_notation == "1"


def test_break_tap_conversion() -> None:
    event_ir = _event_ir_from_inote("{8}1b,E")
    event = _single_event(event_ir, TapEvent)

    assert event.position == "1"
    assert event.is_break is True
    assert event.raw_notation == "1b"


def test_hold_conversion() -> None:
    event_ir = _event_ir_from_inote("{8}2h[4:1],E")
    event = _single_event(event_ir, HoldEvent)

    assert event.head_tick == 0
    assert event.position == "2"
    assert event.duration_ticks == 1920
    assert event.duration_raw == "[4:1]"
    assert event.duration_kind == "grid_fraction"
    assert event.raw_notation == "2h[4:1]"


def test_zero_duration_hold_conversion() -> None:
    event_ir = _event_ir_from_inote("{8}4h[1:0],E")
    event = _single_event(event_ir, HoldEvent)

    assert event.duration_ticks == 0
    assert event.duration_raw == "[1:0]"


def test_simple_slide_conversion() -> None:
    event_ir = _event_ir_from_inote("{8}1-4[8:1],E")
    event = _single_event(event_ir, SlideEvent)

    assert event.head_tick == 0
    assert event.start_position == "1"
    assert event.launch_offset_ticks is None
    assert event.travel_duration_ticks == 960
    assert event.end_position == "4"
    assert event.raw_notation == "1-4[8:1]"
    assert event.duration_raw == "[8:1]"
    assert event.duration_kind == "grid_fraction"
    assert event.segments == [
        SlideSegment(
            start_position="1",
            path_type="-",
            path_args=["4"],
            end_position="4",
            travel_duration_ticks=960,
            duration_raw="[8:1]",
            duration_kind="grid_fraction",
            raw_notation="1-4[8:1]",
        )
    ]


def test_chained_slide_conversion_preserves_segments_and_final_endpoint() -> None:
    event_ir = _event_ir_from_inote("{8}1-4[8:1]*-6[8:1],E")
    event = _single_event(event_ir, SlideEvent)

    assert len(event.segments) == 2
    assert event.segments[0].start_position == "1"
    assert event.segments[0].end_position == "4"
    assert event.segments[1].start_position == "4"
    assert event.segments[1].end_position == "6"
    assert event.end_position == "6"
    assert event.travel_duration_ticks == 1920


def test_compound_slide_conversion_preserves_path_parts() -> None:
    event_ir = _event_ir_from_inote("{8}1<7v5<6[4:3]*>3v5<6[4:3],E")
    event = _single_event(event_ir, SlideEvent)

    assert len(event.segments) == 2
    assert event.segments[0].path_type == "compound"
    assert event.segments[0].path_args == ["7", "5", "6"]
    assert event.segments[0].path_parts == [
        {"pattern": "<", "path_args": [7], "end_position": 7},
        {"pattern": "v", "path_args": [5], "end_position": 5},
        {"pattern": "<", "path_args": [6], "end_position": 6},
    ]
    assert event.segments[1].start_position == "6"
    assert event.end_position == "6"


def test_touch_conversion() -> None:
    event_ir = _event_ir_from_inote("{8}Cf,E")
    event = _single_event(event_ir, TouchEvent)

    assert event.tick == 0
    assert event.area == "C"
    assert event.position == "C"
    assert event.firework is True
    assert event.raw_notation == "Cf"


def test_touch_hold_conversion() -> None:
    event_ir = _event_ir_from_inote("{8}E3h[2:1],E")
    event = _single_event(event_ir, TouchHoldEvent)

    assert event.head_tick == 0
    assert event.area == "E"
    assert event.position == "E3"
    assert event.duration_ticks == 3840
    assert event.duration_raw == "[2:1]"
    assert event.duration_kind == "grid_fraction"


def test_bpm_event_conversion() -> None:
    chart = ChartIR(
        timing=TimingData(
            bpms=[BpmEvent(tick=960, beat=0.5, time_sec=0.25, bpm=180.0, raw="(180)")]
        )
    )

    event_ir = chart_ir_to_event_ir(chart)

    assert event_ir.timing_events == [
        TimingEvent(
            tick=960,
            bpm=180.0,
            beat=0.5,
            time_sec=0.25,
            raw_notation="(180)",
        )
    ]
    assert event_ir.meter_events == []


def test_slide_final_endpoint_uses_last_segment_not_note_endpoint() -> None:
    chart = ChartIR(
        notes=[
            Note(
                note_type="slide",
                tick=0,
                position="1",
                end_position="4",
                duration_ticks=1920,
                segments=[
                    {"start_position": 1, "pattern": "-", "path_args": [4], "end_position": 4},
                    {"start_position": 4, "pattern": "-", "path_args": [6], "end_position": 6},
                ],
            )
        ]
    )

    event = _single_event(chart_ir_to_event_ir(chart), SlideEvent)

    assert event.end_position == "6"


def test_slide_launch_offset_is_unknown_when_chart_ir_has_no_explicit_value() -> None:
    event_ir = _event_ir_from_inote("{8}1-4[8:1],E")
    event = _single_event(event_ir, SlideEvent)

    assert event.launch_offset_ticks is None


def test_slide_raw_notation_is_preserved() -> None:
    event_ir = _event_ir_from_inote("{8}1-4[8:1]*-6[8:1],E")
    event = _single_event(event_ir, SlideEvent)

    assert event.raw_notation == "1-4[8:1]*-6[8:1]"


def _event_ir_from_inote(inote: str) -> ChartEventIR:
    raw = (
        "&title=Event Test\n"
        "&artist=Tester\n"
        "&wholebpm=120\n"
        "&lv_5=13\n"
        f"&inote_5={inote}\n"
    )
    chart = parse_maidata_metadata(raw)
    return chart_ir_to_event_ir(build_chart_ir_by_difficulty_index(chart, 5))


def _single_event(event_ir: ChartEventIR, event_type: type) -> object:
    events = [event for event in event_ir.events if isinstance(event, event_type)]
    assert len(events) == 1
    return events[0]
