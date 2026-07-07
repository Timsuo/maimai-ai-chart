import json
from pathlib import Path
from unittest import SkipTest

from maichart import (
    build_chart_ir_by_difficulty_index,
    chart_from_json,
    chart_to_json,
    compute_raw_maidata_stats,
    parse_maidata_file,
    parse_maidata_metadata,
    read_text_with_encoding,
    validate_raw_maidata_chart,
)
from maichart.cli import main
from maichart.notes import parse_timing_points_notes
from maichart.timing import tokenize_difficulty_timing


def test_read_utf8_file(tmp_path) -> None:
    path = tmp_path / "maidata.txt"
    path.write_text("&title=UTF8\n&artist=Tester\n", encoding="utf-8")

    text, encoding = read_text_with_encoding(path)

    assert text.startswith("&title=UTF8")
    assert encoding in {"utf-8-sig", "utf-8"}


def test_read_utf8_bom_file(tmp_path) -> None:
    path = tmp_path / "maidata.txt"
    path.write_text("&title=BOM\n&artist=Tester\n", encoding="utf-8-sig")

    chart = parse_maidata_file(path)

    assert chart.title == "BOM"


def test_read_explicit_cp932_encoding(tmp_path) -> None:
    path = tmp_path / "maidata.txt"
    path.write_bytes("&title=海底譚\n&artist=n-buna\n".encode("cp932"))

    text, encoding = read_text_with_encoding(path, encoding="cp932")
    chart = parse_maidata_file(path, encoding="cp932")

    assert encoding == "cp932"
    assert "&title=海底譚" in text
    assert chart.title == "海底譚"


def test_chart_ir_preserves_unknown_tokens() -> None:
    raw = (
        "&title=Unknown\n"
        "&artist=Tester\n"
        "&wholebpm=120\n"
        "&lv_1=1\n"
        "&inote_1={8}1,Z9,???bad,E\n"
    )

    ir = build_chart_ir_by_difficulty_index(parse_maidata_metadata(raw), 1)
    data = json.loads(chart_to_json(ir))

    assert [token.raw_token for token in ir.unknown_tokens] == ["Z9", "???bad"]
    assert data["unknown_tokens"][0]["raw_token"] == "Z9"
    assert data["unknown_tokens"][1]["timing_index"] == 2


def test_old_chart_ir_json_without_unknown_tokens_still_loads() -> None:
    payload = json.dumps(
        {
            "schema_version": 1,
            "metadata": {"title": "Old"},
            "difficulty": {"index": 1},
            "timing": {"bpms": [], "grids": [], "hspeeds": []},
            "notes": [],
            "raw": None,
        }
    )

    ir = chart_from_json(payload)

    assert ir.metadata.title == "Old"
    assert ir.unknown_tokens == []


def test_stats_parse_coverage_with_unknown_token() -> None:
    chart = parse_maidata_metadata(
        "&title=Coverage\n"
        "&artist=Tester\n"
        "&wholebpm=120\n"
        "&lv_1=1\n"
        "&inote_1={8}1,Z9,E\n"
    )

    stats = compute_raw_maidata_stats(chart)
    difficulty = stats.difficulties[0]

    assert difficulty.parsed_token_count == 1
    assert difficulty.unknown_token_count == 1
    assert difficulty.total_token_count == 2
    assert difficulty.parse_coverage == 0.5


def test_validate_unknown_token_warning_by_default() -> None:
    chart = parse_maidata_metadata(
        "&title=Validate\n"
        "&artist=Tester\n"
        "&wholebpm=120\n"
        "&lv_1=1\n"
        "&inote_1={8}1,Z9,E\n"
    )

    report = validate_raw_maidata_chart(chart)

    assert report.ok is True
    assert report.errors == 0
    assert report.warnings == 1
    assert report.issues[0].code == "unknown-token"


def test_validate_unknown_token_error_in_strict_mode() -> None:
    chart = parse_maidata_metadata(
        "&title=Validate\n"
        "&artist=Tester\n"
        "&wholebpm=120\n"
        "&lv_1=1\n"
        "&inote_1={8}1,Z9,E\n"
    )

    report = validate_raw_maidata_chart(chart, strict=True)

    assert report.ok is False
    assert report.errors == 1
    assert report.warnings == 0
    assert report.issues[0].severity == "error"


def test_cli_encoding_and_strict_validate(tmp_path) -> None:
    path = tmp_path / "maidata.txt"
    stats_out = tmp_path / "stats.json"
    validate_out = tmp_path / "validate.json"
    path.write_bytes(
        (
            "&title=海底譚\n"
            "&artist=n-buna\n"
            "&wholebpm=120\n"
            "&lv_1=1\n"
            "&inote_1={8}1,Z9,E\n"
        ).encode("cp932")
    )

    assert main(["stats", str(path), "--encoding", "cp932", "-o", str(stats_out)]) == 0
    assert (
        main(
            [
                "validate",
                str(path),
                "--encoding",
                "cp932",
                "--strict",
                "-o",
                str(validate_out),
            ]
        )
        == 1
    )

    stats = json.loads(stats_out.read_text(encoding="utf-8"))
    report = json.loads(validate_out.read_text(encoding="utf-8"))

    assert stats["title"] == "海底譚"
    assert stats["difficulties"][0]["parse_coverage"] == 0.5
    assert report["errors"] == 1


def test_417_parse_coverage_if_available() -> None:
    candidates = sorted(Path("sample_runs").glob("maichart_417_*/maidata.txt"))
    if not candidates:
        raise SkipTest("417 sample maidata.txt is not available in sample_runs.")

    chart = parse_maidata_file(candidates[-1])
    stats = compute_raw_maidata_stats(chart, difficulty_index=5)
    report = validate_raw_maidata_chart(chart, difficulty_index=5)
    difficulty = stats.difficulties[0]

    assert difficulty.parse_coverage == 1.0
    assert difficulty.unknown_token_count == 0
    assert report.ok is True


def test_stats_validate_and_ir_with_compound_slide_tokens() -> None:
    chart = parse_maidata_metadata(
        "&title=Complex\n"
        "&artist=Tester\n"
        "&wholebpm=120\n"
        "&lv_5=13\n"
        "&inote_5={8}2q6[352:49],8h[#0.8057],"
        "1<2[0.6383##2.0455],3b-1[8:1]*-7[8:1],E\n"
    )

    points = tokenize_difficulty_timing(chart.difficulties[0], initial_bpm=120.0)
    parsed = parse_timing_points_notes(points)
    ir = build_chart_ir_by_difficulty_index(chart, 5)
    stats = compute_raw_maidata_stats(chart, difficulty_index=5)
    report = validate_raw_maidata_chart(chart, difficulty_index=5)

    assert parsed[0].notes[0].duration_beats.numerator == 49
    assert parsed[0].notes[0].duration_beats.denominator == 88
    assert parsed[0].notes[0].duration_ticks is None
    assert len(ir.unknown_tokens) == 0
    assert stats.difficulties[0].unknown_token_count == 0
    assert stats.difficulties[0].parse_coverage == 1.0
    assert stats.difficulties[0].duration_kind_counts == {
        "grid_fraction": 1,
        "seconds": 1,
        "timing_pair": 1,
        "compound": 1,
    }
    assert report.ok is True
    assert {issue.code for issue in report.issues} == {"non-integer-duration-tick"}


def test_513_sample_smoke_if_available() -> None:
    candidates = sorted(Path("sample_runs").glob("maichart_513/**/maidata.txt"))
    if not candidates:
        raise SkipTest("513 sample maidata.txt is not available in sample_runs.")

    chart = parse_maidata_file(candidates[-1])
    for difficulty_index in (2, 3, 4, 5):
        difficulty = next(item for item in chart.difficulties if item.index == difficulty_index)
        points = tokenize_difficulty_timing(difficulty, initial_bpm=float(chart.wholebpm or 120))
        parsed_points = parse_timing_points_notes(points)
        ir = build_chart_ir_by_difficulty_index(chart, difficulty_index)
        stats = compute_raw_maidata_stats(chart, difficulty_index=difficulty_index)
        report = validate_raw_maidata_chart(chart, difficulty_index=difficulty_index)
        raw_unknown = {token.raw_token for token in ir.unknown_tokens}

        assert points
        assert parsed_points
        assert ir.notes or ir.unknown_tokens
        assert stats.difficulties[0].total_token_count > 0
        assert report.errors == 0
        assert raw_unknown == set()

    difficulty4 = compute_raw_maidata_stats(chart, difficulty_index=4).difficulties[0]
    assert difficulty4.note_count > 0
