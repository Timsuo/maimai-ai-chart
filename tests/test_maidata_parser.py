import json
from pathlib import Path

from maichart import parse_maidata_metadata, raw_maidata_to_json


def test_parse_top_level_metadata_fields() -> None:
    text = (
        "&title=Test Title\n"
        "&artist=Test Artist\n"
        "&wholebpm=180\n"
        "&first=1.25\n"
        "&lv_1=3\n"
        "&des_1=Alice\n"
        "&inote_1=1,2,3,\n"
    )

    chart = parse_maidata_metadata(text)

    assert chart.title == "Test Title"
    assert chart.artist == "Test Artist"
    assert chart.wholebpm == "180"
    assert chart.first == "1.25"
    assert chart.levels == {1: "3"}
    assert chart.designers == {1: "Alice"}


def test_preserves_multiline_inote_block() -> None:
    inote = "\n{4}\n(180)\n1,2,\n3,\n"
    text = (
        "&title=Multiline\n"
        "&lv_1=12+\n"
        "&des_1=Bob\n"
        f"&inote_1={inote}"
        "&lv_2=13\n"
    )

    chart = parse_maidata_metadata(text)

    assert chart.difficulties[0].inote == inote
    assert chart.difficulties[0].raw_inote == f"&inote_1={inote}"
    assert chart.levels[2] == "13"


def test_collects_raw_difficulty_blocks_without_parsing_notes() -> None:
    text = (
        "&lv_3=14\n"
        "&des_3=Carol\n"
        "&inote_3=\n"
        "{8}\n"
        "1h[2:1],3-7[4:1],\n"
    )

    chart = parse_maidata_metadata(text)

    assert len(chart.difficulties) == 1
    difficulty = chart.difficulties[0]
    assert difficulty.index == 3
    assert difficulty.level == "14"
    assert difficulty.designer == "Carol"
    assert "1h[2:1],3-7[4:1]," in (difficulty.inote or "")


def test_collects_remaster_difficulty_six() -> None:
    text = (
        "&lv_6=?\n"
        "&des_6=Remaster Designer\n"
        "&inote_6=\n"
        "{8}\n"
        "1,2,3,4,\n"
    )

    chart = parse_maidata_metadata(text)

    assert len(chart.difficulties) == 1
    difficulty = chart.difficulties[0]
    assert difficulty.index == 6
    assert difficulty.level == "?"
    assert difficulty.designer == "Remaster Designer"
    assert difficulty.inote == "\n{8}\n1,2,3,4,\n"


def test_raw_maidata_json_output() -> None:
    chart = parse_maidata_metadata("&title=JSON Song\n&inote_1=1,\n")

    payload = raw_maidata_to_json(chart)
    data = json.loads(payload)

    assert data["title"] == "JSON Song"
    assert data["difficulties"][0]["index"] == 1
    assert data["difficulties"][0]["inote"] == "1,\n"


def test_realistic_fixture_structure() -> None:
    fixture = Path("tests/fixtures/maidata_realistic.txt").read_text(encoding="utf-8")

    chart = parse_maidata_metadata(fixture)

    assert chart.title == "Future Rhythm"
    assert chart.levels[5] == "14+"
    assert len(chart.difficulties) == 5
    assert chart.difficulties[3].index == 4
    assert chart.difficulties[3].designer == "Realistic Designer"
    assert chart.difficulties[3].inote == "\n{4}\n(174)\n1,2,3,4,\n5h[2:1],6-2[4:1],\n"
    assert chart.difficulties[4].inote == "\n{8}\n1,8,1,8,\n"
