from fractions import Fraction
import json

from maichart import (
    build_chart_ir_by_difficulty_index,
    chart_from_json,
    chart_to_json,
    compute_raw_maidata_stats,
    parse_duration_expr,
    parse_maidata_metadata,
    parse_timing_point_notes,
    validate_raw_maidata_chart,
)
from test_notes_parser import make_point


def test_grid_duration_expr() -> None:
    duration = parse_duration_expr("[8:1]")

    assert duration.kind == "grid_fraction"
    assert duration.beats == Fraction(1, 2)
    assert duration.ticks == 960


def test_larger_grid_duration_expr() -> None:
    duration = parse_duration_expr("[4:3]")

    assert duration.kind == "grid_fraction"
    assert duration.beats == Fraction(3, 1)
    assert duration.ticks == 5760


def test_non_integer_tick_grid_duration_expr() -> None:
    duration = parse_duration_expr("[352:49]")

    assert duration.kind == "grid_fraction"
    assert duration.beats == Fraction(49, 88)
    assert duration.ticks is None


def test_seconds_duration_expr() -> None:
    duration = parse_duration_expr("[#0.8057]")

    assert duration.kind == "seconds"
    assert duration.seconds == 0.8057
    assert duration.beats is None


def test_timing_pair_duration_expr() -> None:
    duration = parse_duration_expr("[0.6383##2.0455]")

    assert duration.kind == "timing_pair"
    assert duration.values == [0.6383, 2.0455]
    assert duration.seconds == 2.0455


def test_seconds_hold_is_parsed() -> None:
    parsed = parse_timing_point_notes(make_point("8h[#0.8057]"))

    assert parsed.unknown_tokens == []
    assert parsed.notes[0].note_type == "hold"
    assert parsed.notes[0].duration is not None
    assert parsed.notes[0].duration.kind == "seconds"
    assert parsed.notes[0].duration.seconds == 0.8057
    assert parsed.notes[0].duration_beats == Fraction(8057, 5000)


def test_timing_pair_slide_is_parsed() -> None:
    parsed = parse_timing_point_notes(make_point("1<2[0.6383##2.0455]"))

    assert parsed.unknown_tokens == []
    assert parsed.notes[0].note_type == "slide"
    assert parsed.notes[0].slide_pattern == "<"
    assert parsed.notes[0].duration is not None
    assert parsed.notes[0].duration.kind == "timing_pair"
    assert parsed.notes[0].duration.values == [0.6383, 2.0455]


def test_duration_raw_survives_chart_ir_json_round_trip() -> None:
    chart = parse_maidata_metadata(
        "&title=Duration\n"
        "&artist=Tester\n"
        "&wholebpm=120\n"
        "&lv_1=1\n"
        "&inote_1={8}8h[#0.8057],1<2[0.6383##2.0455],E\n"
    )
    ir = build_chart_ir_by_difficulty_index(chart, 1)
    loaded = chart_from_json(chart_to_json(ir))

    assert loaded.notes[0].duration is not None
    assert loaded.notes[0].duration["raw"] == "[#0.8057]"
    assert loaded.notes[0].duration["kind"] == "seconds"
    assert loaded.notes[1].duration is not None
    assert loaded.notes[1].duration["raw"] == "[0.6383##2.0455]"
    assert loaded.notes[1].duration["kind"] == "timing_pair"


def test_stats_duration_kind_distribution() -> None:
    chart = parse_maidata_metadata(
        "&title=Duration Stats\n"
        "&artist=Tester\n"
        "&wholebpm=120\n"
        "&lv_1=1\n"
        "&inote_1={8}1h[8:1],8h[#0.8057],1<2[0.6383##2.0455],E\n"
    )

    stats = compute_raw_maidata_stats(chart, difficulty_index=1)

    assert stats.difficulties[0].duration_kind_counts == {
        "grid_fraction": 1,
        "seconds": 1,
        "timing_pair": 1,
    }
    assert stats.difficulties[0].unknown_token_count == 0


def test_validate_seconds_and_timing_pair_have_no_errors() -> None:
    chart = parse_maidata_metadata(
        "&title=Duration Validate\n"
        "&artist=Tester\n"
        "&wholebpm=120\n"
        "&lv_1=1\n"
        "&inote_1={8}8h[#0.8057],1<2[0.6383##2.0455],E\n"
    )

    report = validate_raw_maidata_chart(chart, difficulty_index=1)

    assert report.ok is True
    assert report.errors == 0
    assert not any(issue.code == "unknown-token" for issue in report.issues)


def test_validate_invalid_duration_block_is_error() -> None:
    chart = parse_maidata_metadata(
        "&title=Bad Duration\n"
        "&artist=Tester\n"
        "&wholebpm=120\n"
        "&lv_1=1\n"
        "&inote_1={8}1h[bad],E\n"
    )

    report = validate_raw_maidata_chart(chart, difficulty_index=1)

    assert report.ok is False
    assert any(issue.code == "invalid-duration" for issue in report.issues)


def test_notes_json_contains_duration_expr() -> None:
    parsed = parse_timing_point_notes(make_point("8h[#0.8057]"))
    data = json.loads(json.dumps(parsed.notes[0].duration.raw))

    assert data == "[#0.8057]"
