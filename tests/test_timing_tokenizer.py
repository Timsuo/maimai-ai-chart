from fractions import Fraction
import json
from pathlib import Path

from maichart import (
    RawDifficultyBlock,
    parse_maidata_metadata,
    timing_points_to_json,
    tokenize_difficulty_timing,
    tokenize_inote_timing,
)


def test_basic_grid_timing() -> None:
    points = tokenize_inote_timing("{8}1,2,,3,E")

    assert len(points) == 5
    assert points[0].raw == "{8}1"
    assert points[0].note_text == "1"
    assert points[0].beat == Fraction(0, 1)
    assert points[0].tick == 0
    assert points[0].division == 8
    assert points[0].directives[0].directive_type == "grid"
    assert points[1].note_text == "2"
    assert points[1].beat == Fraction(1, 2)
    assert points[1].tick == 960
    assert points[2].raw == ""
    assert points[2].note_text == ""
    assert points[3].beat == Fraction(3, 2)
    assert points[4].directives[0].directive_type == "end"


def test_bpm_and_grid_at_same_point_keep_notes_opaque() -> None:
    points = tokenize_inote_timing("(180){16}1/8,2h[4:1],1>4[8:1],E")

    assert [directive.directive_type for directive in points[0].directives] == ["bpm", "grid"]
    assert points[0].bpm == 180.0
    assert points[0].division == 16
    assert points[0].note_text == "1/8"
    assert points[1].note_text == "2h[4:1]"
    assert points[1].beat == Fraction(1, 4)
    assert points[1].tick == 480
    assert points[1].time_sec == 1 / 12
    assert points[2].note_text == "1>4[8:1]"


def test_empty_timing_points_are_preserved() -> None:
    points = tokenize_inote_timing("1,,,E")

    assert [point.raw for point in points] == ["1", "", "", "E"]
    assert [point.note_text for point in points] == ["1", "", "", ""]
    assert [point.beat for point in points] == [
        Fraction(0, 1),
        Fraction(1, 1),
        Fraction(2, 1),
        Fraction(3, 1),
    ]


def test_beat_and_tick_values_are_exact_and_stable() -> None:
    points = tokenize_inote_timing("{24}1,2,3")

    assert [point.beat for point in points] == [
        Fraction(0, 1),
        Fraction(1, 6),
        Fraction(1, 3),
    ]
    assert [point.tick for point in points] == [0, 320, 640]


def test_multiline_inote_strings_are_supported() -> None:
    points = tokenize_inote_timing("\n{4}\n1,\n2,\nE")

    assert points[0].note_text == "1"
    assert points[1].raw == "\n2"
    assert points[1].note_text == "2"
    assert points[2].directives[0].directive_type == "end"


def test_unknown_note_syntax_remains_opaque() -> None:
    points = tokenize_inote_timing("{8}???/@weird,A1$not-yet,E")

    assert points[0].note_text == "???/@weird"
    assert points[1].note_text == "A1$not-yet"


def test_top_level_comma_splitter_ignores_square_brackets() -> None:
    points = tokenize_inote_timing("1h[4,1],2,E")

    assert len(points) == 3
    assert points[0].note_text == "1h[4,1]"
    assert points[1].note_text == "2"


def test_integration_with_raw_difficulty_block() -> None:
    difficulty = RawDifficultyBlock(index=2, inote="{16}1,2,E")

    points = tokenize_difficulty_timing(difficulty, initial_bpm=150.0)

    assert points[0].division == 16
    assert points[0].bpm == 150.0
    assert points[1].beat == Fraction(1, 4)


def test_realistic_fixture_timing() -> None:
    fixture = Path("tests/fixtures/maidata_realistic.txt").read_text(encoding="utf-8")
    chart = parse_maidata_metadata(fixture)
    difficulty = next(item for item in chart.difficulties if item.index == 4)

    points = tokenize_difficulty_timing(difficulty, initial_bpm=float(chart.wholebpm or 120))

    assert points[0].note_text == "1"
    assert [directive.directive_type for directive in points[0].directives] == ["grid", "bpm"]
    assert points[0].division == 4
    assert points[0].bpm == 174.0
    assert points[4].note_text == "5h[2:1]"
    assert points[5].note_text == "6-2[4:1]"


def test_timing_points_json_contains_inspection_fields() -> None:
    points = tokenize_inote_timing("{8}1,2,E")

    data = json.loads(timing_points_to_json(points))

    assert data[1]["beat"] == "1/2"
    assert data[1]["tick"] == 960
    assert data[1]["note_text"] == "2"
    assert data[0]["directives"][0]["value"] == 8
