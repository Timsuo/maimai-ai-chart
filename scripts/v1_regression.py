"""Run the V1 parser regression pipeline against local sample_runs data."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from maichart.builder import build_chart_ir_by_difficulty_index
from maichart.maidata import parse_maidata_file
from maichart.notes import parse_timing_points_notes
from maichart.serialization import chart_to_json
from maichart.stats import compute_raw_maidata_stats
from maichart.timing import tokenize_difficulty_timing
from maichart.validation import validate_raw_maidata_chart

CORE_SAMPLE_IDS = (
    "417",
    "513",
    "786",
    "799",
    "803",
    "812",
    "820",
    "825",
    "833",
    "834",
    "432",
    "471",
    "496",
    "773",
    "777",
    "835",
)

NEW_NOTE_SAMPLE_IDS = (
    "11821",
    "11898",
    "11956",
    "11051",
    "11106",
    "11152",
    "11184",
    "11206",
    "11222",
    "11228",
    "11448",
    "11451",
    "11818",
    "11820",
)

SAMPLE_SETS = {
    "core": CORE_SAMPLE_IDS,
    "new-notes": NEW_NOTE_SAMPLE_IDS,
    "all": CORE_SAMPLE_IDS + NEW_NOTE_SAMPLE_IDS,
}


def run_regression(sample_root: Path, sample_ids: tuple[str, ...]) -> dict[str, Any]:
    """Run metadata, timing, notes, IR, stats, validate, and IR JSON checks."""

    report: dict[str, Any] = {
        "requested_sample_ids": list(sample_ids),
        "sample_count": 0,
        "difficulty_count": 0,
        "skipped_samples": [],
        "pipeline_failures": 0,
        "pipeline_failure_details": [],
        "validate_errors": 0,
        "validate_warnings": 0,
        "validate_issue_details": [],
        "unknown_token_count": 0,
        "unknown_token_top": {},
        "min_parse_coverage": 1.0,
        "coverage_below_one": [],
        "note_type_distribution": {},
        "touch_note_count": 0,
        "duration_kind_distribution": {},
        "slide_pattern_distribution": {},
    }

    unknown_counter: Counter[str] = Counter()
    note_type_counter: Counter[str] = Counter()
    duration_counter: Counter[str] = Counter()
    slide_counter: Counter[str] = Counter()
    min_coverage = 1.0

    for sample_id in sample_ids:
        path = _find_sample_path(sample_root, sample_id)
        if path is None:
            report["skipped_samples"].append(sample_id)
            continue

        report["sample_count"] += 1
        try:
            chart = parse_maidata_file(path)
        except Exception as exc:
            report["pipeline_failures"] += 1
            report["pipeline_failure_details"].append(
                _failure(sample_id, None, "metadata parse", exc)
            )
            continue

        for difficulty in chart.difficulties:
            difficulty_index = difficulty.index
            report["difficulty_count"] += 1
            try:
                initial_bpm = float(chart.wholebpm or 120)
                points = tokenize_difficulty_timing(difficulty, initial_bpm=initial_bpm)
                parsed_points = parse_timing_points_notes(points)
                ir = build_chart_ir_by_difficulty_index(chart, difficulty_index)
                chart_to_json(ir)
                stats = compute_raw_maidata_stats(
                    chart,
                    difficulty_index=difficulty_index,
                )
                validation = validate_raw_maidata_chart(
                    chart,
                    difficulty_index=difficulty_index,
                )
            except Exception as exc:
                report["pipeline_failures"] += 1
                report["pipeline_failure_details"].append(
                    _failure(sample_id, difficulty_index, "pipeline", exc)
                )
                continue

            difficulty_stats = stats.difficulties[0]
            coverage = difficulty_stats.parse_coverage
            min_coverage = min(min_coverage, coverage)
            if coverage < 1.0:
                report["coverage_below_one"].append(
                    {
                        "sample_id": sample_id,
                        "difficulty_index": difficulty_index,
                        "parse_coverage": coverage,
                    }
                )

            report["unknown_token_count"] += len(ir.unknown_tokens)
            unknown_counter.update(token.raw_token or token.raw or "" for token in ir.unknown_tokens)
            note_type_counter.update(note.note_type for note in ir.notes)
            report["touch_note_count"] += sum(
                1
                for note in ir.notes
                if note.note_type in {"touch", "touch_hold"}
            )
            duration_counter.update(difficulty_stats.duration_kind_counts)
            slide_counter.update(difficulty_stats.slide_pattern_counts)
            report["validate_errors"] += validation.errors
            report["validate_warnings"] += validation.warnings
            for issue in validation.issues:
                if issue.severity == "error":
                    report["validate_issue_details"].append(
                        {
                            "sample_id": sample_id,
                            "difficulty_index": difficulty_index,
                            "severity": issue.severity,
                            "code": issue.code,
                            "message": issue.message,
                            "timing_index": issue.timing_index,
                            "raw": issue.raw,
                        }
                    )

            # Keep parsed_points live in the pipeline so note parsing is actually exercised.
            if points and parsed_points is None:
                raise AssertionError("unreachable parser state")

    report["min_parse_coverage"] = min_coverage
    report["unknown_token_top"] = dict(unknown_counter.most_common(20))
    report["note_type_distribution"] = dict(note_type_counter)
    report["duration_kind_distribution"] = dict(duration_counter)
    report["slide_pattern_distribution"] = dict(slide_counter)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-root",
        default=str(ROOT / "sample_runs"),
        help="directory containing maichart_SAMPLE_ID*/maidata.txt files",
    )
    parser.add_argument(
        "--sample-set",
        choices=sorted(SAMPLE_SETS),
        default="core",
        help="named sample set to run",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        help="explicit sample IDs; overrides --sample-set",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=str(ROOT / "sample_runs" / "v1_regression_report.json"),
        help="output JSON report path",
    )
    args = parser.parse_args(argv)

    sample_ids = tuple(args.sample_ids) if args.sample_ids else SAMPLE_SETS[args.sample_set]
    report = run_regression(Path(args.sample_root), sample_ids)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["pipeline_failures"] or report["validate_errors"] else 0


def _find_sample_path(sample_root: Path, sample_id: str) -> Path | None:
    direct = sample_root / f"maichart_{sample_id}" / "maidata.txt"
    if direct.exists():
        return direct
    matches = sorted(sample_root.glob(f"maichart_{sample_id}_*/maidata.txt"))
    return matches[-1] if matches else None


def _failure(
    sample_id: str,
    difficulty_index: int | None,
    stage: str,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "difficulty_index": difficulty_index,
        "stage": stage,
        "error": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
    }


if __name__ == "__main__":
    raise SystemExit(main())
