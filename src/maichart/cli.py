"""Command-line entry point for maichart."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from maichart.audio import (
    AudioFeatureDependencyError,
    analyze_audio_file,
    build_audio_features_for_dataset_manifest,
    load_audio_features_json,
    save_audio_features_json,
)
from maichart.alignment import (
    build_alignment_report,
    build_alignment_reports_for_dataset_manifest,
    save_alignment_report_json,
)
from maichart.builder import build_chart_ir_by_difficulty_index
from maichart.dataset import (
    build_dataset_manifest,
    save_dataset_manifest,
)
from maichart.exporter import export_chart_ir_to_maidata
from maichart.labels import (
    build_frame_labels_for_dataset_manifest,
    build_frame_labels_from_chart_ir,
    load_frame_labels_json,
    save_frame_labels_json,
)
from maichart.maidata import parse_maidata_file, raw_maidata_to_json
from maichart.notes import parse_timing_points_notes, parsed_timing_points_notes_to_json
from maichart.preprocess import (
    build_dataset_splits,
    build_raw_sample_manifest,
    build_training_manifest,
)
from maichart.serialization import chart_to_json, load_chart_json
from maichart.rhythm import (
    build_rhythm_profile_from_dataset_manifest,
    build_rhythm_skeletons_for_dataset_manifest,
    evaluate_rhythm_skeleton,
    generate_rhythm_skeleton,
    load_rhythm_profile_json,
    load_rhythm_skeleton_json,
    save_rhythm_evaluation_json,
    save_rhythm_profile_json,
    save_rhythm_skeleton_json,
)
from maichart.stats import chart_stats_to_json, compute_raw_maidata_stats
from maichart.timing import timing_points_to_json, tokenize_difficulty_timing
from maichart.validation import validate_raw_maidata_chart, validation_report_to_json


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level CLI parser."""

    parser = argparse.ArgumentParser(
        prog="maichart",
        description="V1 tools for Maidata/Simai-like maimai fan chart files.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_cmd = subparsers.add_parser("parse", help="parse Maidata text into raw metadata JSON")
    parse_cmd.add_argument("input", help="input Maidata/Simai-like chart file")
    parse_cmd.add_argument("-o", "--output", help="output raw metadata JSON path")
    parse_cmd.add_argument("--encoding", help="input text encoding, or auto-detect if omitted")
    parse_cmd.add_argument(
        "--difficulty",
        type=int,
        choices=range(1, 6),
        metavar="1-5",
        help="difficulty index to inspect or convert",
    )
    parse_cmd.add_argument(
        "--timing",
        action="store_true",
        help="output raw timing points for the selected difficulty",
    )
    parse_cmd.add_argument(
        "--notes",
        action="store_true",
        help="output parsed basic tap/break/hold/slide notes for the selected difficulty",
    )
    parse_cmd.add_argument(
        "--ir",
        action="store_true",
        help="output ChartIR JSON for the selected difficulty",
    )
    parse_cmd.set_defaults(handler=_parse_command)

    export_cmd = subparsers.add_parser("export", help="export ChartIR JSON to Maidata text")
    export_cmd.add_argument("input", help="input ChartIR JSON file")
    export_cmd.add_argument("-o", "--output", help="output Maidata text path")
    export_cmd.set_defaults(handler=_export_command)

    validate_cmd = subparsers.add_parser("validate", help="validate a Maidata file")
    validate_cmd.add_argument("input", help="input Maidata/Simai-like chart file")
    validate_cmd.add_argument("--difficulty", type=int, choices=range(1, 6), metavar="1-5")
    validate_cmd.add_argument("--encoding", help="input text encoding, or auto-detect if omitted")
    validate_cmd.add_argument(
        "--strict",
        action="store_true",
        help="treat unknown note tokens as validation errors",
    )
    validate_cmd.add_argument("-o", "--output", help="output validation JSON path")
    validate_cmd.set_defaults(handler=_validate_command)

    stats_cmd = subparsers.add_parser("stats", help="compute Maidata parser statistics")
    stats_cmd.add_argument("input", help="input Maidata/Simai-like chart file")
    stats_cmd.add_argument("--difficulty", type=int, choices=range(1, 6), metavar="1-5")
    stats_cmd.add_argument("--encoding", help="input text encoding, or auto-detect if omitted")
    stats_cmd.add_argument("-o", "--output", help="output stats JSON path")
    stats_cmd.set_defaults(handler=_stats_command)

    dataset_cmd = subparsers.add_parser("dataset", help="V2 dataset utilities")
    dataset_subparsers = dataset_cmd.add_subparsers(dest="dataset_command", required=True)
    dataset_build_cmd = dataset_subparsers.add_parser(
        "build",
        help="build a dataset manifest and ChartIR cache",
    )
    dataset_build_cmd.add_argument("source_root", help="directory containing sample maidata.txt files")
    dataset_build_cmd.add_argument(
        "-o",
        "--output",
        required=True,
        help="output dataset manifest JSON path",
    )
    dataset_build_cmd.add_argument(
        "--cache-dir",
        required=True,
        help="directory for generated ChartIR JSON cache files",
    )
    dataset_build_cmd.add_argument(
        "--encoding",
        help="input text encoding, or auto-detect if omitted",
    )
    dataset_build_cmd.add_argument(
        "--force",
        action="store_true",
        help="rebuild ChartIR cache files even when they already exist",
    )
    dataset_build_cmd.set_defaults(handler=_dataset_build_command)

    dataset_labels_cmd = dataset_subparsers.add_parser(
        "labels",
        help="build frame labels for every ChartIR cache in a dataset manifest",
    )
    dataset_labels_cmd.add_argument("manifest", help="dataset manifest JSON path")
    dataset_labels_cmd.add_argument(
        "--out-dir",
        required=True,
        help="directory for generated frame-label JSON files",
    )
    dataset_labels_cmd.add_argument(
        "--division",
        type=int,
        default=16,
        help="beat grid division per 4/4 measure; default: 16",
    )
    dataset_labels_cmd.add_argument(
        "-o",
        "--output",
        help="optional output manifest path; defaults to updating the input manifest",
    )
    dataset_labels_cmd.set_defaults(handler=_dataset_labels_command)

    dataset_audio_cmd = dataset_subparsers.add_parser(
        "audio-features",
        help="build audio features for every song in a dataset manifest",
    )
    dataset_audio_cmd.add_argument("manifest", help="dataset manifest JSON path")
    dataset_audio_cmd.add_argument(
        "--out-dir",
        required=True,
        help="directory for generated audio feature JSON files",
    )
    dataset_audio_cmd.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="analysis sample rate; default: 22050",
    )
    dataset_audio_cmd.add_argument(
        "--division",
        type=int,
        default=16,
        help="reserved chart/audio division value; default: 16",
    )
    dataset_audio_cmd.add_argument(
        "--force",
        action="store_true",
        help="rebuild audio feature files even when they already exist",
    )
    dataset_audio_cmd.add_argument(
        "-o",
        "--output",
        help="optional output manifest path; defaults to updating the input manifest",
    )
    dataset_audio_cmd.set_defaults(handler=_dataset_audio_features_command)

    dataset_align_cmd = dataset_subparsers.add_parser(
        "align",
        help="build chart-audio alignment reports for a dataset manifest",
    )
    dataset_align_cmd.add_argument("manifest", help="dataset manifest JSON path")
    dataset_align_cmd.add_argument(
        "--out-dir",
        required=True,
        help="directory for generated alignment report JSON files",
    )
    dataset_align_cmd.add_argument(
        "--onset-tolerance-ms",
        type=float,
        default=50.0,
        help="primary onset tolerance in milliseconds; default: 50",
    )
    dataset_align_cmd.add_argument(
        "--force",
        action="store_true",
        help="rebuild alignment reports even when they already exist",
    )
    dataset_align_cmd.add_argument(
        "-o",
        "--output",
        help="optional output manifest path; defaults to updating the input manifest",
    )
    dataset_align_cmd.set_defaults(handler=_dataset_align_command)

    dataset_rhythm_cmd = dataset_subparsers.add_parser(
        "rhythm",
        help="generate rhythm skeletons for a dataset manifest",
    )
    dataset_rhythm_cmd.add_argument("manifest", help="dataset manifest JSON path")
    dataset_rhythm_cmd.add_argument(
        "--out-dir",
        required=True,
        help="directory for generated rhythm skeleton JSON files",
    )
    dataset_rhythm_cmd.add_argument("--profile", help="optional rhythm profile JSON path")
    dataset_rhythm_cmd.add_argument(
        "--level-source",
        choices=("difficulty", "default"),
        default="difficulty",
        help="target level source; default: difficulty",
    )
    dataset_rhythm_cmd.add_argument(
        "--default-level",
        type=float,
        default=10.0,
        help="fallback target level; default: 10.0",
    )
    dataset_rhythm_cmd.add_argument(
        "--division",
        type=int,
        default=16,
        help="rhythm skeleton division value; default: 16",
    )
    dataset_rhythm_cmd.add_argument(
        "--force",
        action="store_true",
        help="rebuild rhythm skeletons even when they already exist",
    )
    dataset_rhythm_cmd.add_argument(
        "-o",
        "--output",
        help="optional output manifest path; defaults to updating the input manifest",
    )
    dataset_rhythm_cmd.set_defaults(handler=_dataset_rhythm_command)

    preprocess_cmd = subparsers.add_parser("preprocess", help="V2 dataset QC and training manifest utilities")
    preprocess_subparsers = preprocess_cmd.add_subparsers(dest="preprocess_command", required=True)

    preprocess_scan_cmd = preprocess_subparsers.add_parser(
        "scan",
        help="scan raw sample directories without heavy analysis",
    )
    preprocess_scan_cmd.add_argument("raw_root", help="raw dataset root")
    preprocess_scan_cmd.add_argument("-o", "--output", required=True, help="output raw manifest JSON path")
    preprocess_scan_cmd.set_defaults(handler=_preprocess_scan_command)

    preprocess_build_cmd = preprocess_subparsers.add_parser(
        "build",
        help="build ChartIR/frame/audio/alignment caches and a training manifest",
    )
    preprocess_build_cmd.add_argument("raw_root", help="raw dataset root")
    preprocess_build_cmd.add_argument("--cache-dir", required=True, help="cache directory")
    preprocess_build_cmd.add_argument("-o", "--output", required=True, help="output training manifest JSON path")
    preprocess_build_cmd.add_argument("--division", type=int, default=16, help="frame division; default: 16")
    preprocess_build_cmd.add_argument("--force", action="store_true", help="rebuild readable existing caches")
    preprocess_build_cmd.add_argument("--encoding", help="input text encoding, or auto-detect if omitted")
    preprocess_build_cmd.add_argument("--limit", type=int, help="limit samples for debugging")
    preprocess_build_cmd.add_argument("--skip-audio", action="store_true", help="skip audio feature extraction")
    preprocess_build_cmd.add_argument("--skip-alignment", action="store_true", help="skip alignment report generation")
    preprocess_build_cmd.add_argument("--sample-rate", type=int, default=22050, help="audio sample rate; default: 22050")
    preprocess_build_cmd.add_argument("--workers", type=int, default=1, help="reserved; current implementation is serial")
    preprocess_build_cmd.set_defaults(handler=_preprocess_build_command)

    preprocess_split_cmd = preprocess_subparsers.add_parser(
        "split",
        help="build train/val/test split manifests",
    )
    preprocess_split_cmd.add_argument("training_manifest", help="training manifest JSON path")
    preprocess_split_cmd.add_argument("--out-dir", required=True, help="output split directory")
    preprocess_split_cmd.add_argument("--train-ratio", type=float, default=0.8)
    preprocess_split_cmd.add_argument("--val-ratio", type=float, default=0.1)
    preprocess_split_cmd.add_argument("--test-ratio", type=float, default=0.1)
    preprocess_split_cmd.add_argument("--seed", type=int, default=42)
    preprocess_split_cmd.add_argument("--split-by-song", action="store_true", default=True)
    preprocess_split_cmd.add_argument("--split-by-difficulty", dest="split_by_song", action="store_false")
    preprocess_split_cmd.set_defaults(handler=_preprocess_split_command)

    labels_cmd = subparsers.add_parser("labels", help="frame-label utilities")
    labels_subparsers = labels_cmd.add_subparsers(dest="labels_command", required=True)
    labels_build_cmd = labels_subparsers.add_parser(
        "build",
        help="build frame labels from one ChartIR JSON file",
    )
    labels_build_cmd.add_argument("input", help="input ChartIR JSON file")
    labels_build_cmd.add_argument("-o", "--output", required=True, help="output frame labels JSON path")
    labels_build_cmd.add_argument(
        "--division",
        type=int,
        default=16,
        help="beat grid division per 4/4 measure; default: 16",
    )
    labels_build_cmd.add_argument(
        "--song-id",
        help="song id to store in the frame labels; defaults to the ChartIR parent directory",
    )
    labels_build_cmd.set_defaults(handler=_labels_build_command)

    audio_cmd = subparsers.add_parser("audio", help="audio feature utilities")
    audio_subparsers = audio_cmd.add_subparsers(dest="audio_command", required=True)
    audio_analyze_cmd = audio_subparsers.add_parser(
        "analyze",
        help="analyze one audio file into audio_features JSON",
    )
    audio_analyze_cmd.add_argument("input", help="input audio file")
    audio_analyze_cmd.add_argument("-o", "--output", required=True, help="output audio features JSON path")
    audio_analyze_cmd.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="analysis sample rate; default: 22050",
    )
    audio_analyze_cmd.add_argument(
        "--division",
        type=int,
        default=16,
        help="reserved chart/audio division value; default: 16",
    )
    audio_analyze_cmd.set_defaults(handler=_audio_analyze_command)

    align_cmd = subparsers.add_parser(
        "align",
        help="build one chart-audio alignment report",
    )
    align_cmd.add_argument("chart_ir", help="input ChartIR JSON file")
    align_cmd.add_argument("frame_labels", help="input frame labels JSON file")
    align_cmd.add_argument("audio_features", help="input audio features JSON file")
    align_cmd.add_argument("-o", "--output", required=True, help="output alignment report JSON path")
    align_cmd.add_argument(
        "--onset-tolerance-ms",
        type=float,
        default=50.0,
        help="primary onset tolerance in milliseconds; default: 50",
    )
    align_cmd.set_defaults(handler=_align_command)

    rhythm_cmd = subparsers.add_parser("rhythm", help="rhythm skeleton utilities")
    rhythm_subparsers = rhythm_cmd.add_subparsers(dest="rhythm_command", required=True)
    rhythm_profile_cmd = rhythm_subparsers.add_parser(
        "profile",
        help="build a dataset rhythm profile",
    )
    rhythm_profile_cmd.add_argument("manifest", help="dataset manifest JSON path")
    rhythm_profile_cmd.add_argument("-o", "--output", required=True, help="output rhythm profile JSON path")
    rhythm_profile_cmd.add_argument(
        "--level-band-size",
        type=float,
        default=2.0,
        help="level band size for profile aggregation; default: 2.0",
    )
    rhythm_profile_cmd.set_defaults(handler=_rhythm_profile_command)

    rhythm_generate_cmd = rhythm_subparsers.add_parser(
        "generate",
        help="generate a rhythm skeleton from audio features",
    )
    rhythm_generate_cmd.add_argument("audio_features", help="input audio features JSON path")
    rhythm_generate_cmd.add_argument("--level", type=float, required=True, help="target difficulty level")
    rhythm_generate_cmd.add_argument("--profile", help="optional rhythm profile JSON path")
    rhythm_generate_cmd.add_argument("-o", "--output", required=True, help="output rhythm skeleton JSON path")
    rhythm_generate_cmd.add_argument("--division", type=int, default=16, help="rhythm skeleton division value")
    rhythm_generate_cmd.add_argument("--max-density-per-sec", type=float, help="optional density cap")
    rhythm_generate_cmd.set_defaults(handler=_rhythm_generate_command)

    rhythm_evaluate_cmd = rhythm_subparsers.add_parser(
        "evaluate",
        help="evaluate a rhythm skeleton against frame labels",
    )
    rhythm_evaluate_cmd.add_argument("rhythm_skeleton", help="input rhythm skeleton JSON path")
    rhythm_evaluate_cmd.add_argument("frame_labels", help="reference frame labels JSON path")
    rhythm_evaluate_cmd.add_argument("-o", "--output", required=True, help="output evaluation JSON path")
    rhythm_evaluate_cmd.add_argument(
        "--tolerance-frames",
        type=int,
        default=1,
        help="matching tolerance in frame indices; default: 1",
    )
    rhythm_evaluate_cmd.set_defaults(handler=_rhythm_evaluate_command)

    return parser


def _parse_command(args: argparse.Namespace) -> int:
    """Parse Maidata top-level metadata and raw difficulty blocks."""

    chart = parse_maidata_file(args.input, encoding=args.encoding)
    selected_modes = [args.timing, args.notes, args.ir]
    if sum(1 for selected in selected_modes if selected) > 1:
        raise SystemExit("Use only one inspection mode: --timing, --notes, or --ir.")

    if args.timing or args.notes or args.ir:
        if args.difficulty is None:
            raise SystemExit("--timing/--notes/--ir requires --difficulty 1-5")
        difficulty = next(
            (
                candidate
                for candidate in chart.difficulties
                if candidate.index == args.difficulty
            ),
            None,
        )
        if difficulty is None:
            raise SystemExit(f"No difficulty {args.difficulty} found in {args.input!r}.")
        initial_bpm = _parse_initial_bpm(chart.wholebpm)
        if args.ir:
            ir = build_chart_ir_by_difficulty_index(chart, args.difficulty)
            payload = chart_to_json(ir)
        else:
            points = tokenize_difficulty_timing(difficulty, initial_bpm=initial_bpm)
            if args.notes:
                payload = parsed_timing_points_notes_to_json(parse_timing_points_notes(points))
            else:
                payload = timing_points_to_json(points)
    else:
        payload = raw_maidata_to_json(chart)

    _write_or_print(payload, args.output)

    return 0


def _export_command(args: argparse.Namespace) -> int:
    chart = load_chart_json(args.input)
    _write_or_print(export_chart_ir_to_maidata(chart), args.output, final_newline=False)
    return 0


def _validate_command(args: argparse.Namespace) -> int:
    chart = parse_maidata_file(args.input, encoding=args.encoding)
    report = validate_raw_maidata_chart(
        chart,
        difficulty_index=args.difficulty,
        strict=args.strict,
    )
    _write_or_print(validation_report_to_json(report), args.output)
    return 0 if report.ok else 1


def _stats_command(args: argparse.Namespace) -> int:
    chart = parse_maidata_file(args.input, encoding=args.encoding)
    stats = compute_raw_maidata_stats(chart, difficulty_index=args.difficulty)
    _write_or_print(chart_stats_to_json(stats), args.output)
    return 0


def _dataset_build_command(args: argparse.Namespace) -> int:
    output_path = Path(args.output)
    manifest = build_dataset_manifest(
        args.source_root,
        cache_dir=args.cache_dir,
        force=args.force,
        encoding=args.encoding,
        path_base=output_path.parent,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_dataset_manifest(manifest, output_path)
    return 0


def _dataset_labels_command(args: argparse.Namespace) -> int:
    build_frame_labels_for_dataset_manifest(
        args.manifest,
        out_dir=args.out_dir,
        division=args.division,
        output_manifest_path=args.output,
    )
    return 0


def _dataset_audio_features_command(args: argparse.Namespace) -> int:
    try:
        build_audio_features_for_dataset_manifest(
            args.manifest,
            out_dir=args.out_dir,
            division=args.division,
            sample_rate=args.sample_rate,
            force=args.force,
            output_manifest_path=args.output,
        )
    except AudioFeatureDependencyError as exc:
        raise SystemExit(str(exc)) from exc
    return 0


def _dataset_align_command(args: argparse.Namespace) -> int:
    build_alignment_reports_for_dataset_manifest(
        args.manifest,
        out_dir=args.out_dir,
        onset_tolerance_ms=args.onset_tolerance_ms,
        force=args.force,
        output_manifest_path=args.output,
    )
    return 0


def _dataset_rhythm_command(args: argparse.Namespace) -> int:
    profile = load_rhythm_profile_json(args.profile) if args.profile else None
    build_rhythm_skeletons_for_dataset_manifest(
        args.manifest,
        out_dir=args.out_dir,
        profile=profile,
        level_source=args.level_source,
        default_level=args.default_level,
        division=args.division,
        force=args.force,
        output_manifest_path=args.output,
    )
    return 0


def _preprocess_scan_command(args: argparse.Namespace) -> int:
    build_raw_sample_manifest(args.raw_root, output_path=args.output)
    return 0


def _preprocess_build_command(args: argparse.Namespace) -> int:
    if args.workers != 1:
        print("[preprocess] --workers is reserved; running serially.")
    build_training_manifest(
        args.raw_root,
        cache_dir=args.cache_dir,
        output_path=args.output,
        division=args.division,
        force=args.force,
        encoding=args.encoding,
        limit=args.limit,
        skip_audio=args.skip_audio,
        skip_alignment=args.skip_alignment,
        sample_rate=args.sample_rate,
    )
    return 0


def _preprocess_split_command(args: argparse.Namespace) -> int:
    build_dataset_splits(
        args.training_manifest,
        output_dir=args.out_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        split_by_song=args.split_by_song,
    )
    return 0


def _labels_build_command(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    chart = load_chart_json(input_path)
    labels = build_frame_labels_from_chart_ir(
        chart,
        division=args.division,
        song_id=args.song_id or input_path.parent.name,
    )
    save_frame_labels_json(labels, args.output)
    return 0


def _audio_analyze_command(args: argparse.Namespace) -> int:
    try:
        features = analyze_audio_file(
            args.input,
            division=args.division,
            sample_rate=args.sample_rate,
        )
    except AudioFeatureDependencyError as exc:
        raise SystemExit(str(exc)) from exc
    save_audio_features_json(features, args.output)
    return 0


def _align_command(args: argparse.Namespace) -> int:
    report = build_alignment_report(
        load_chart_json(args.chart_ir),
        load_frame_labels_json(args.frame_labels),
        load_audio_features_json(args.audio_features),
        onset_tolerance_ms=args.onset_tolerance_ms,
    )
    save_alignment_report_json(report, args.output)
    return 0


def _rhythm_profile_command(args: argparse.Namespace) -> int:
    profile = build_rhythm_profile_from_dataset_manifest(
        args.manifest,
        level_band_size=args.level_band_size,
    )
    save_rhythm_profile_json(profile, args.output)
    return 0


def _rhythm_generate_command(args: argparse.Namespace) -> int:
    profile = load_rhythm_profile_json(args.profile) if args.profile else None
    skeleton = generate_rhythm_skeleton(
        load_audio_features_json(args.audio_features),
        target_level=args.level,
        profile=profile,
        division=args.division,
        max_density_per_sec=args.max_density_per_sec,
    )
    save_rhythm_skeleton_json(skeleton, args.output)
    return 0


def _rhythm_evaluate_command(args: argparse.Namespace) -> int:
    evaluation = evaluate_rhythm_skeleton(
        load_rhythm_skeleton_json(args.rhythm_skeleton),
        load_frame_labels_json(args.frame_labels),
        tolerance_frames=args.tolerance_frames,
    )
    save_rhythm_evaluation_json(evaluation, args.output)
    return 0


def _parse_initial_bpm(value: str | None) -> float:
    if value is None:
        return 120.0
    try:
        return float(value)
    except ValueError:
        return 120.0


def _write_or_print(
    payload: str,
    output: str | None,
    *,
    final_newline: bool = True,
) -> None:
    if output:
        suffix = "\n" if final_newline and not payload.endswith("\n") else ""
        Path(output).write_text(payload + suffix, encoding="utf-8")
    else:
        print(payload, end="" if payload.endswith("\n") else "\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
