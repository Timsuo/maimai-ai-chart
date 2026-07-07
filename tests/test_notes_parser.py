from fractions import Fraction
import json
from pathlib import Path
from unittest import SkipTest

from maichart import (
    RawTimingPoint,
    parse_basic_hold_token,
    parse_basic_slide_token,
    parse_basic_tap_break_token,
    parse_basic_touch_token,
    SLIDE_PATTERN_CATALOG,
    parse_maidata_metadata,
    parse_timing_point_notes,
    parse_timing_points_notes,
    parsed_note_to_chart_note,
    parsed_timing_points_notes_to_json,
    tokenize_difficulty_timing,
    tokenize_inote_timing,
)


def make_point(note_text: str, index: int = 0) -> RawTimingPoint:
    return RawTimingPoint(
        index=index,
        raw=note_text,
        note_text=note_text,
        beat=Fraction(index, 2),
        tick=index * 960,
        time_sec=index * 0.25,
        division=8,
        bpm=120.0,
    )


def test_single_normal_tap() -> None:
    parsed = parse_timing_point_notes(make_point("1"))

    assert len(parsed.notes) == 1
    assert parsed.notes[0].note_type == "tap"
    assert parsed.notes[0].position == 1
    assert parsed.notes[0].is_break is False
    assert parsed.unknown_tokens == []


def test_single_break_tap() -> None:
    parsed = parse_timing_point_notes(make_point("1b"))

    assert len(parsed.notes) == 1
    assert parsed.notes[0].position == 1
    assert parsed.notes[0].is_break is True


def test_simultaneous_normal_taps_share_group_id() -> None:
    parsed = parse_timing_point_notes(make_point("1/8"))

    assert [note.position for note in parsed.notes] == [1, 8]
    assert parsed.notes[0].group_id == "timing-0"
    assert parsed.notes[1].group_id == "timing-0"


def test_simultaneous_normal_and_break() -> None:
    parsed = parse_timing_point_notes(make_point("1b/8"))

    assert [(note.position, note.is_break) for note in parsed.notes] == [
        (1, True),
        (8, False),
    ]
    assert len({note.group_id for note in parsed.notes}) == 1


def test_simultaneous_double_break() -> None:
    parsed = parse_timing_point_notes(make_point("1b/8b"))

    assert [(note.position, note.is_break) for note in parsed.notes] == [
        (1, True),
        (8, True),
    ]
    assert parsed.notes[0].group_id == parsed.notes[1].group_id


def test_empty_note_text() -> None:
    parsed = parse_timing_point_notes(make_point(""))

    assert parsed.notes == []
    assert parsed.unknown_tokens == []


def test_basic_hold() -> None:
    parsed = parse_timing_point_notes(make_point("1h[4:1]"))

    assert len(parsed.notes) == 1
    assert parsed.notes[0].note_type == "hold"
    assert parsed.notes[0].position == 1
    assert parsed.notes[0].duration_beats == Fraction(1, 1)
    assert parsed.notes[0].duration_ticks == 1920
    assert parsed.notes[0].duration_sec == 0.5
    assert parsed.unknown_tokens == []


def test_basic_hold_fractional_duration() -> None:
    parsed = parse_timing_point_notes(make_point("2h[8:3]"))

    assert parsed.notes[0].note_type == "hold"
    assert parsed.notes[0].position == 2
    assert parsed.notes[0].duration_beats == Fraction(3, 2)
    assert parsed.notes[0].duration_ticks == 2880


def test_zero_duration_hold_is_parsed() -> None:
    parsed = parse_timing_point_notes(make_point("4h[1:0]"))

    assert parsed.notes[0].note_type == "hold"
    assert parsed.notes[0].position == 4
    assert parsed.notes[0].duration_beats == Fraction(0, 1)
    assert parsed.notes[0].duration_ticks == 0


def test_basic_dash_slide() -> None:
    parsed = parse_timing_point_notes(make_point("1-4[8:1]"))

    assert len(parsed.notes) == 1
    assert parsed.notes[0].note_type == "slide"
    assert parsed.notes[0].position == 1
    assert parsed.notes[0].end_position == 4
    assert parsed.notes[0].slide_pattern == "-"
    assert parsed.notes[0].duration_beats == Fraction(1, 2)
    assert parsed.notes[0].duration_ticks == 960
    assert parsed.notes[0].duration_sec == 0.25
    assert parsed.notes[0].segments[0]["trajectory"] == {
        "trajectory_id": "1-4",
        "pattern": "-",
        "start_position": 1,
        "end_position": 4,
        "path_args": [4],
        "raw": "1-4",
    }
    assert parsed.unknown_tokens == []


def test_basic_arrow_slide() -> None:
    parsed = parse_timing_point_notes(make_point("1>4[8:1]"))

    assert parsed.notes[0].note_type == "slide"
    assert parsed.notes[0].position == 1
    assert parsed.notes[0].end_position == 4
    assert parsed.notes[0].slide_pattern == ">"


def test_basic_curved_slide() -> None:
    parsed = parse_timing_point_notes(make_point("2q6[32:3]"))

    assert parsed.notes[0].note_type == "slide"
    assert parsed.notes[0].position == 2
    assert parsed.notes[0].end_position == 6
    assert parsed.notes[0].slide_pattern == "q"
    assert parsed.notes[0].duration_beats == Fraction(3, 8)
    assert parsed.notes[0].duration_ticks == 720


def test_slide_with_non_integer_duration_tick_does_not_crash() -> None:
    parsed = parse_timing_point_notes(make_point("2q6[352:49]"))

    assert parsed.notes[0].note_type == "slide"
    assert parsed.notes[0].position == 2
    assert parsed.notes[0].end_position == 6
    assert parsed.notes[0].slide_pattern == "q"
    assert parsed.notes[0].duration_beats == Fraction(49, 88)
    assert parsed.notes[0].duration_ticks is None
    assert parsed.notes[0].duration_sec is not None


def test_pp_slide_pattern() -> None:
    parsed = parse_timing_point_notes(make_point("7pp1[8:1]"))

    assert parsed.notes[0].note_type == "slide"
    assert parsed.notes[0].position == 7
    assert parsed.notes[0].end_position == 1
    assert parsed.notes[0].slide_pattern == "pp"


def test_break_slide_head() -> None:
    parsed = parse_timing_point_notes(make_point("3b-1[8:1]"))

    assert parsed.notes[0].note_type == "slide"
    assert parsed.notes[0].position == 3
    assert parsed.notes[0].end_position == 1
    assert parsed.notes[0].is_break is True
    assert parsed.notes[0].slide_pattern == "-"


def test_v_slide_pattern_with_two_end_positions() -> None:
    parsed = parse_timing_point_notes(make_point("2V84[8:1]"))

    assert parsed.notes[0].note_type == "slide"
    assert parsed.notes[0].position == 2
    assert parsed.notes[0].end_position == 4
    assert parsed.notes[0].slide_pattern == "V"
    assert parsed.notes[0].segments[0]["trajectory"] == {
        "trajectory_id": "2V84",
        "pattern": "V",
        "start_position": 2,
        "end_position": 4,
        "path_args": [8, 4],
        "raw": "2V84",
    }


def test_chained_slide_is_parsed_as_one_note() -> None:
    parsed = parse_timing_point_notes(make_point("1-4[8:1]*-6[8:1]"))

    assert len(parsed.notes) == 1
    assert parsed.notes[0].note_type == "slide"
    assert parsed.notes[0].position == 1
    assert parsed.notes[0].slide_segments[0]["end_position"] == 4
    assert parsed.notes[0].slide_segments[1]["end_position"] == 6
    assert parsed.notes[0].segments[1]["trajectory"]["start_position"] == 4
    assert parsed.notes[0].segments[1]["trajectory"]["trajectory_id"] == "4-6"
    assert parsed.notes[0].duration_beats == Fraction(1, 1)
    assert parsed.unknown_tokens == []


def test_chained_pp_qq_slide_is_parsed() -> None:
    parsed = parse_timing_point_notes(make_point("4pp4[2:1]*qq4[2:1]"))

    assert parsed.notes[0].slide_segments[0]["pattern"] == "pp"
    assert parsed.notes[0].slide_segments[1]["pattern"] == "qq"


def test_compound_slide_path_is_preserved_as_known_parts() -> None:
    parsed = parse_timing_point_notes(make_point("1<7v5<6[4:3]*>3v5<6[4:3]"))

    assert parsed.unknown_tokens == []
    assert parsed.notes[0].note_type == "slide"
    assert parsed.notes[0].slide_pattern == "compound"
    assert parsed.notes[0].slide_segments[0]["path_text"] == "<7v5<6"
    assert parsed.notes[0].slide_segments[1]["path_text"] == ">3v5<6"
    assert parsed.notes[0].segments[0]["trajectory"]["pattern"] == "compound"
    assert parsed.notes[0].segments[0]["path_parts"] == [
        {"pattern": "<", "path_args": [7], "end_position": 7},
        {"pattern": "v", "path_args": [5], "end_position": 5},
        {"pattern": "<", "path_args": [6], "end_position": 6},
    ]


def test_slide_pattern_catalog_contains_v1_supported_patterns() -> None:
    assert set(SLIDE_PATTERN_CATALOG) >= {
        "-",
        "<",
        ">",
        "p",
        "q",
        "pp",
        "qq",
        "s",
        "z",
        "v",
        "V",
        "w",
    }
    assert SLIDE_PATTERN_CATALOG["V"].path_arg_count == 2


def test_basic_touch_tap() -> None:
    parsed = parse_timing_point_notes(make_point("A1"))

    assert parsed.unknown_tokens == []
    assert parsed.notes[0].note_type == "touch"
    assert parsed.notes[0].position == "A1"
    assert parsed.notes[0].modifiers["touch_area"] == "A"
    assert parsed.notes[0].modifiers["firework"] is False


def test_center_touch_and_firework_touch() -> None:
    parsed = parse_timing_point_notes(make_point("C/Cf/E2f"))

    assert [note.position for note in parsed.notes] == ["C", "C", "E2"]
    assert [note.modifiers["firework"] for note in parsed.notes] == [
        False,
        True,
        True,
    ]
    assert parsed.unknown_tokens == []


def test_touch_hold() -> None:
    parsed = parse_timing_point_notes(make_point("Chf[1:1]/E3h[2:1]"))

    assert [note.note_type for note in parsed.notes] == ["touch_hold", "touch_hold"]
    assert parsed.notes[0].position == "C"
    assert parsed.notes[0].modifiers["firework"] is True
    assert parsed.notes[0].duration_beats == Fraction(4, 1)
    assert parsed.notes[1].position == "E3"
    assert parsed.notes[1].duration_beats == Fraction(2, 1)


def test_mixed_known_tokens() -> None:
    parsed = parse_timing_point_notes(make_point("1/8/2h[4:1]/1-4[8:1]"))

    assert [(note.note_type, note.position) for note in parsed.notes] == [
        ("tap", 1),
        ("tap", 8),
        ("hold", 2),
        ("slide", 1),
    ]
    assert parsed.unknown_tokens == []
    assert parsed.notes[0].group_id == parsed.notes[1].group_id
    assert parsed.notes[1].group_id == parsed.notes[2].group_id
    assert parsed.notes[2].group_id == parsed.notes[3].group_id


def test_top_level_slash_splitter_ignores_square_brackets() -> None:
    parsed = parse_timing_point_notes(make_point("1/2h[4/1]/8b"))

    assert [(note.position, note.is_break) for note in parsed.notes] == [
        (1, False),
        (8, True),
    ]
    assert parsed.unknown_tokens[0].raw == "2h[4/1]"


def test_malformed_or_modified_holds_remain_unknown() -> None:
    assert parse_basic_hold_token("1h[4:1]") is not None
    assert parse_basic_hold_token("1hx[4:1]") is not None
    assert parse_basic_hold_token("1hb[4:1]") is not None
    assert parse_basic_hold_token("1hbx[4:1]") is not None
    assert parse_basic_hold_token("1h[4:1]x") is None
    assert parse_basic_hold_token("1bh[4:1]") is None
    assert parse_basic_hold_token("1h[4/1]") is None


def test_malformed_or_compound_slides_remain_unknown() -> None:
    assert parse_basic_slide_token("1-4[8:1]") is not None
    assert parse_basic_slide_token("1>4[8:1]") is not None
    assert parse_basic_slide_token("1x>4[8:1]") is not None
    assert parse_basic_slide_token("1bx-4[8:1]b") is not None
    assert parse_basic_slide_token("2q6[32:3]") is not None
    assert parse_basic_slide_token("7pp1[8:1]") is not None
    assert parse_basic_slide_token("3b-1[8:1]") is not None
    assert parse_basic_slide_token("2V84[8:1]") is not None
    assert parse_basic_slide_token("1-4[8:1]*-6[8:1]") is not None
    assert parse_basic_slide_token("1-4[8:1]x") is not None
    assert parse_basic_slide_token("1-4-7[8:1]") is not None
    assert parse_basic_slide_token("1-4[8/1]") is None
    assert parse_basic_slide_token("1<2[0.6383##2.0455]") is not None
    assert parse_basic_slide_token("3b-1[8:1]*-7[8:1]") is not None


def test_seconds_and_timing_pair_tokens_are_parsed() -> None:
    parsed = parse_timing_point_notes(
        make_point("8h[#0.8057]/1<2[0.6383##2.0455]")
    )

    assert parsed.unknown_tokens == []
    assert [note.note_type for note in parsed.notes] == ["hold", "slide"]
    assert parsed.notes[0].duration is not None
    assert parsed.notes[0].duration.kind == "seconds"
    assert parsed.notes[1].duration is not None
    assert parsed.notes[1].duration.kind == "timing_pair"


def test_token_parser_requires_full_match() -> None:
    assert parse_basic_tap_break_token("1") is not None
    assert parse_basic_tap_break_token("1b") is not None
    assert parse_basic_tap_break_token("1x") is not None
    assert parse_basic_tap_break_token("1bx") is not None
    assert parse_basic_tap_break_token("1>4[8:1]") is None
    assert parse_basic_tap_break_token("2h[4:1]") is None
    assert parse_basic_tap_break_token("A1") is None
    assert parse_basic_touch_token("A1") is not None
    assert parse_basic_touch_token("Cf") is not None
    assert parse_basic_touch_token("Ch[1:1]") is not None


def test_timing_tokenizer_integration() -> None:
    points = tokenize_inote_timing("{8}1,2,1b/8,2h[4:1],1>4[8:1],Z9,E")
    parsed_points = parse_timing_points_notes(points)

    assert parsed_points[0].notes[0].position == 1
    assert parsed_points[1].notes[0].position == 2
    assert [(note.position, note.is_break) for note in parsed_points[2].notes] == [
        (1, True),
        (8, False),
    ]
    assert parsed_points[3].notes[0].note_type == "hold"
    assert parsed_points[3].notes[0].duration_beats == Fraction(1, 1)
    assert parsed_points[4].notes[0].note_type == "slide"
    assert parsed_points[4].notes[0].end_position == 4
    assert parsed_points[5].unknown_tokens[0].raw == "Z9"
    assert parsed_points[6].notes == []


def test_chart_ir_note_conversion() -> None:
    parsed = parse_timing_point_notes(make_point("1b"))

    chart_note = parsed_note_to_chart_note(parsed.notes[0])

    assert chart_note.note_type == "tap"
    assert chart_note.position == "1"
    assert chart_note.modifiers["break"] is True
    assert chart_note.beat == 0.0


def test_chart_ir_touch_conversion() -> None:
    parsed = parse_timing_point_notes(make_point("Cf"))

    chart_note = parsed_note_to_chart_note(parsed.notes[0])

    assert chart_note.note_type == "touch"
    assert chart_note.position == "C"
    assert chart_note.modifiers["touch_area"] == "C"
    assert chart_note.modifiers["firework"] is True


def test_chart_ir_hold_conversion() -> None:
    parsed = parse_timing_point_notes(make_point("2h[8:3]"))

    chart_note = parsed_note_to_chart_note(parsed.notes[0])

    assert chart_note.note_type == "hold"
    assert chart_note.position == "2"
    assert chart_note.duration_beats == 1.5
    assert chart_note.duration_ticks == 2880
    assert chart_note.duration_sec == 0.75


def test_chart_ir_slide_conversion() -> None:
    parsed = parse_timing_point_notes(make_point("1-4[8:1]"))

    chart_note = parsed_note_to_chart_note(parsed.notes[0])

    assert chart_note.note_type == "slide"
    assert chart_note.position == "1"
    assert chart_note.end_position == "4"
    assert chart_note.path == ["1", "-", "4"]
    assert chart_note.segments[0]["trajectory"]["trajectory_id"] == "1-4"
    assert chart_note.segments[0]["duration"]["raw"] == "[8:1]"
    assert chart_note.modifiers["slide_pattern"] == "-"
    assert chart_note.duration_beats == 0.5
    assert chart_note.duration_ticks == 960


def test_notes_json_contains_inspection_fields() -> None:
    parsed = parse_timing_point_notes(make_point("1b/8/2h[4:1]/1>4[8:1]/Z9"))

    data = json.loads(parsed_timing_points_notes_to_json([parsed]))

    assert data[0]["timing_index"] == 0
    assert data[0]["notes"][0]["is_break"] is True
    assert data[0]["notes"][1]["position"] == 8
    assert data[0]["notes"][2]["note_type"] == "hold"
    assert data[0]["notes"][2]["duration_beats"] == "1"
    assert data[0]["notes"][3]["note_type"] == "slide"
    assert data[0]["notes"][3]["end_position"] == 4
    assert data[0]["notes"][3]["slide_pattern"] == ">"
    assert data[0]["notes"][3]["segments"][0]["trajectory"]["pattern"] == ">"
    assert data[0]["unknown_tokens"][0]["raw"] == "Z9"


def test_417_sample_smoke_if_available() -> None:
    candidates = sorted(Path("sample_runs").glob("maichart_417_*/maidata.txt"))
    if not candidates:
        raise SkipTest("417 sample maidata.txt is not available in sample_runs.")

    chart = parse_maidata_metadata(candidates[-1].read_text(encoding="utf-8"))
    difficulty = next(item for item in chart.difficulties if item.index == 5)
    points = tokenize_difficulty_timing(difficulty, initial_bpm=float(chart.wholebpm or 120))
    parsed_points = parse_timing_points_notes(points)

    note_count = sum(len(point.notes) for point in parsed_points)
    hold_count = sum(
        1
        for point in parsed_points
        for note in point.notes
        if note.note_type == "hold"
    )
    slide_count = sum(
        1
        for point in parsed_points
        for note in point.notes
        if note.note_type == "slide"
    )
    unknown_count = sum(len(point.unknown_tokens) for point in parsed_points)

    assert note_count > 0
    assert hold_count > 0
    assert slide_count > 0
    assert unknown_count >= 0
