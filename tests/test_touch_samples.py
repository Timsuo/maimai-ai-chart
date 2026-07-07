from pathlib import Path
from unittest import SkipTest

from maichart import (
    build_chart_ir_by_difficulty_index,
    compute_raw_maidata_stats,
    parse_maidata_file,
    validate_raw_maidata_chart,
)


TOUCH_SAMPLE_IDS = (
    11821,
    11898,
    11956,
    11051,
    11106,
    11152,
    11184,
    11206,
    11222,
    11228,
    11448,
    11451,
    11818,
    11820,
)


def test_touch_sample_smoke_if_available() -> None:
    sample_paths = {
        sample_id: sorted(Path("sample_runs").glob(f"maichart_{sample_id}/**/maidata.txt"))
        for sample_id in TOUCH_SAMPLE_IDS
    }
    available = {
        sample_id: paths[-1]
        for sample_id, paths in sample_paths.items()
        if paths
    }
    if not available:
        raise SkipTest("Touch sample maidata.txt files are not available in sample_runs.")

    for sample_id, path in available.items():
        chart = parse_maidata_file(path)
        parsed_touch_notes = 0
        unknown_tokens = 0

        for difficulty in chart.difficulties:
            ir = build_chart_ir_by_difficulty_index(chart, difficulty.index)
            stats = compute_raw_maidata_stats(chart, difficulty_index=difficulty.index)
            report = validate_raw_maidata_chart(chart, difficulty_index=difficulty.index)

            parsed_touch_notes += sum(
                1
                for note in ir.notes
                if note.note_type in {"touch", "touch_hold"}
            )
            unknown_tokens += len(ir.unknown_tokens)

            assert stats.difficulties[0].parse_coverage == 1.0
            assert report.errors == 0, f"{sample_id} difficulty {difficulty.index}"

        assert parsed_touch_notes > 0, sample_id
        assert unknown_tokens == 0, sample_id
