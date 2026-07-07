from maichart import (
    BpmEvent,
    ChartIR,
    ChartMetadata,
    DifficultyMetadata,
    GridEvent,
    HSpeedEvent,
    Note,
    TimingData,
    chart_from_json,
    chart_to_json,
    load_chart_json,
    save_chart_json,
)


def make_basic_chart() -> ChartIR:
    return ChartIR(
        metadata=ChartMetadata(
            title="Sample Song",
            artist="Sample Artist",
            audio_file="sample.ogg",
            background_file="sample.png",
            offset=-0.125,
            raw="&title=Sample Song",
        ),
        difficulty=DifficultyMetadata(
            index=3,
            name="Expert",
            level="12+",
            designer="Chart Designer",
            raw="&des=Chart Designer",
        ),
        timing=TimingData(
            bpms=[BpmEvent(bpm=180.0, beat=0.0, tick=0, time_sec=0.0, raw="(180)")],
            grids=[GridEvent(division=4, beat=0.0, tick=0, raw="{4}")],
            hspeeds=[HSpeedEvent(speed=1.0, beat=0.0, tick=0, raw="<1.0>")],
        ),
        notes=[
            Note(note_type="tap", position="1", beat=0.0, tick=0, time_sec=0.0, raw="1"),
            Note(
                note_type="hold",
                position="2",
                beat=1.0,
                tick=480,
                duration_beats=1.0,
                duration_ticks=480,
                raw="2h[1:1]",
            ),
            Note(
                note_type="slide",
                position="3",
                end_position="7",
                beat=2.0,
                tick=960,
                duration_beats=2.0,
                duration_ticks=960,
                path=["3", "-", "7"],
                segments=[
                    {
                        "trajectory": {
                            "trajectory_id": "3-7",
                            "pattern": "-",
                            "start_position": 3,
                            "end_position": 7,
                            "path_args": [7],
                            "raw": "3-7",
                        },
                        "duration": {"raw": "[2:1]", "kind": "grid_fraction"},
                    }
                ],
                raw="3-7[2:1]",
            ),
        ],
        raw="raw chart text",
    )


def test_construct_basic_chart_ir() -> None:
    chart = make_basic_chart()

    assert chart.schema_version == 1
    assert chart.metadata.title == "Sample Song"
    assert chart.difficulty.name == "Expert"
    assert chart.timing.bpms[0].bpm == 180.0
    assert chart.notes[0].note_type == "tap"


def test_serialize_chart_ir_to_json() -> None:
    chart = make_basic_chart()

    payload = chart_to_json(chart)

    assert '"schema_version": 1' in payload
    assert '"title": "Sample Song"' in payload
    assert '"note_type": "slide"' in payload
    assert '"raw": "raw chart text"' in payload


def test_load_chart_ir_from_json() -> None:
    chart = chart_from_json(chart_to_json(make_basic_chart()))

    assert chart.metadata.audio_file == "sample.ogg"
    assert chart.difficulty.index == 3
    assert chart.timing.grids[0].division == 4
    assert chart.notes[2].path == ["3", "-", "7"]


def test_save_and_load_chart_json_file(tmp_path) -> None:
    path = tmp_path / "chart.json"
    original = make_basic_chart()

    save_chart_json(original, path)
    loaded = load_chart_json(path)

    assert loaded == original


def test_tap_hold_and_slide_notes_can_be_represented() -> None:
    chart = make_basic_chart()
    note_types = {note.note_type for note in chart.notes}

    assert {"tap", "hold", "slide"}.issubset(note_types)
    assert chart.notes[1].duration_ticks == 480
    assert chart.notes[2].end_position == "7"
    assert chart.notes[2].segments[0]["trajectory"]["pattern"] == "-"


def test_old_chart_ir_json_without_segments_still_loads() -> None:
    payload = (
        '{"schema_version":1,"metadata":{},"difficulty":{},"timing":{},'
        '"notes":[{"note_type":"slide","position":"1","end_position":"4"}],'
        '"unknown_tokens":[]}'
    )

    chart = chart_from_json(payload)

    assert chart.notes[0].segments == []
