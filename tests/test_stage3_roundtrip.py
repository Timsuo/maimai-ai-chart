import json
from pathlib import Path
from unittest import SkipTest

from maichart import (
    build_chart_ir_by_difficulty_index,
    chart_from_json,
    chart_to_json,
    compute_raw_maidata_stats,
    export_chart_ir_to_maidata,
    parse_maidata_metadata,
    validate_raw_maidata_chart,
)
from maichart.cli import main
from maichart.timing import tokenize_difficulty_timing


def test_build_chart_ir_for_difficulty() -> None:
    raw = (
        "&title=Round Trip\n"
        "&artist=Tester\n"
        "&first=0\n"
        "&wholebpm=180\n"
        "&lv_5=13\n"
        "&des_5=Codex\n"
        "&inote_5=(180){8}1,2h[4:1],1-4[8:1],E\n"
    )

    chart = parse_maidata_metadata(raw)
    ir = build_chart_ir_by_difficulty_index(chart, 5)

    assert ir.metadata.title == "Round Trip"
    assert ir.difficulty.index == 5
    assert ir.timing.bpms[0].bpm == 180.0
    assert [note.note_type for note in ir.notes] == ["tap", "hold", "slide"]
    assert ir.raw == chart.difficulties[0].raw_inote


def test_export_chart_ir_round_trip_key_fields() -> None:
    raw = (
        "&title=Round Trip\n"
        "&artist=Tester\n"
        "&first=0\n"
        "&wholebpm=180\n"
        "&lv_5=13\n"
        "&des_5=Codex\n"
        "&inote_5=(180){8}1,2h[4:1],1-4[8:1],E\n"
    )

    original = parse_maidata_metadata(raw)
    ir = build_chart_ir_by_difficulty_index(original, 5)
    exported = export_chart_ir_to_maidata(chart_from_json(chart_to_json(ir)))
    parsed = parse_maidata_metadata(exported)

    assert parsed.title == "Round Trip"
    assert parsed.artist == "Tester"
    assert parsed.wholebpm == "180"
    assert parsed.levels[5] == "13"
    assert parsed.designers[5] == "Codex"
    assert parsed.difficulties[0].inote == original.difficulties[0].inote


def test_validate_and_stats_for_synthetic_chart() -> None:
    chart = parse_maidata_metadata(
        "&title=Stats\n"
        "&artist=Tester\n"
        "&wholebpm=120\n"
        "&lv_1=1\n"
        "&inote_1={8}1,Z9,E\n"
    )

    report = validate_raw_maidata_chart(chart)
    stats = compute_raw_maidata_stats(chart)

    assert report.ok is True
    assert report.warnings == 1
    assert report.issues[0].code == "unknown-token"
    assert stats.difficulties[0].note_count == 1
    assert stats.difficulties[0].unknown_token_count == 1


def test_cli_parse_ir_export_round_trip(tmp_path) -> None:
    source = tmp_path / "maidata.txt"
    ir_path = tmp_path / "chart.json"
    exported_path = tmp_path / "exported_maidata.txt"
    source.write_text(
        "&title=CLI Round Trip\n"
        "&artist=Tester\n"
        "&first=0\n"
        "&wholebpm=150\n"
        "&lv_4=12\n"
        "&des_4=CLI\n"
        "&inote_4=(150){8}1/8,2h[4:1],1>4[8:1],E\n",
        encoding="utf-8",
    )

    assert main(["parse", str(source), "--difficulty", "4", "--ir", "-o", str(ir_path)]) == 0
    assert main(["export", str(ir_path), "-o", str(exported_path)]) == 0

    parsed = parse_maidata_metadata(exported_path.read_text(encoding="utf-8"))

    assert parsed.title == "CLI Round Trip"
    assert parsed.wholebpm == "150"
    assert parsed.difficulties[0].inote == "(150){8}1/8,2h[4:1],1>4[8:1],E\n"


def test_cli_validate_and_stats(tmp_path) -> None:
    source = tmp_path / "maidata.txt"
    validate_out = tmp_path / "validate.json"
    stats_out = tmp_path / "stats.json"
    source.write_text(
        "&title=CLI Stats\n"
        "&artist=Tester\n"
        "&wholebpm=120\n"
        "&lv_1=1\n"
        "&inote_1={8}1,2,E\n",
        encoding="utf-8",
    )

    assert main(["validate", str(source), "--difficulty", "1", "-o", str(validate_out)]) == 0
    assert main(["stats", str(source), "--difficulty", "1", "-o", str(stats_out)]) == 0

    validate_data = json.loads(validate_out.read_text(encoding="utf-8"))
    stats_data = json.loads(stats_out.read_text(encoding="utf-8"))

    assert validate_data["ok"] is True
    assert stats_data["difficulties"][0]["note_count"] == 2


def test_417_round_trip_if_available() -> None:
    candidates = sorted(Path("sample_runs").glob("maichart_417_*/maidata.txt"))
    if not candidates:
        raise SkipTest("417 sample maidata.txt is not available in sample_runs.")

    original = parse_maidata_metadata(candidates[-1].read_text(encoding="utf-8"))
    ir = build_chart_ir_by_difficulty_index(original, 5)
    exported = export_chart_ir_to_maidata(ir)
    parsed = parse_maidata_metadata(exported)

    original_difficulty = next(item for item in original.difficulties if item.index == 5)
    parsed_difficulty = next(item for item in parsed.difficulties if item.index == 5)
    original_points = tokenize_difficulty_timing(
        original_difficulty,
        initial_bpm=float(original.wholebpm or 120),
    )
    parsed_points = tokenize_difficulty_timing(
        parsed_difficulty,
        initial_bpm=float(parsed.wholebpm or 120),
    )

    assert parsed.title == original.title
    assert parsed.wholebpm == original.wholebpm
    assert parsed.levels[5] == original.levels[5]
    assert len(parsed_points) == len(original_points)
    assert parsed_points[-1].tick == original_points[-1].tick
