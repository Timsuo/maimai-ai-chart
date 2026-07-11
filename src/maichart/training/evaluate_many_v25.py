"""Batch-evaluate a V2.5 checkpoint over usable training-manifest samples."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import torch

from maichart.training.collate import collate_v25
from maichart.training.dataset_v25 import FEATURE_SETS, MaichartV25Dataset, TrainingDataError
from maichart.training.evaluate_v25 import (
    _load_checkpoint,
    _load_model_state_dict,
    _model_from_checkpoint,
    _resolve_checkpoint_feature_set,
    _with_flat_detail_aliases,
    compute_v25_pattern_details,
    compute_v25_metrics,
)
from maichart.training.utils_v25 import resolve_device

CSV_FIELDS = (
    "sample_index",
    "song_id",
    "difficulty_index",
    "level",
    "note_presence_f1",
    "button_micro_f1",
    "button_best_f1",
    "button_best_threshold",
    "note_type_accuracy",
    "density_mae",
    "note_start_f1",
    "note_start_f1@0.5",
    "note_start_best_f1",
    "note_start_best_threshold",
    "note_start_precision@0.5",
    "note_start_recall@0.5",
    "note_start_pred_positive_rate@0.5",
    "note_start_target_positive_rate",
    "pattern_start_accuracy",
    "pattern_start_macro_f1",
    "chord_size_accuracy",
    "chord_size_macro_f1",
)

SUMMARY_FIELDS = (
    "note_presence_f1",
    "button_micro_f1",
    "button_best_f1",
    "button_best_threshold",
    "note_type_accuracy",
    "density_mae",
    "note_start_f1",
    "note_start_f1@0.5",
    "note_start_best_f1",
    "note_start_best_threshold",
    "note_start_precision@0.5",
    "note_start_recall@0.5",
    "note_start_pred_positive_rate@0.5",
    "note_start_target_positive_rate",
    "pattern_start_accuracy",
    "pattern_start_macro_f1",
    "chord_size_accuracy",
    "chord_size_macro_f1",
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        device = resolve_device(args.device)
        checkpoint = _load_checkpoint(Path(args.checkpoint), device)
        feature_set = _resolve_checkpoint_feature_set(checkpoint, args.feature_set)
        dataset = MaichartV25Dataset(
            args.manifest,
            cache_dir=args.cache_dir,
            feature_set=feature_set,
            cache_samples=args.cache_samples,
        )
        model = _model_from_checkpoint(checkpoint, dataset).to(device)
        _load_model_state_dict(model, checkpoint)
        model.eval()
    except TrainingDataError as exc:
        parser.error(str(exc))
    except FileNotFoundError as exc:
        parser.error(f"Missing checkpoint: {exc.filename}")

    rows = evaluate_many(
        dataset,
        model,
        device,
        max_samples=args.max_samples,
    )
    if not rows:
        parser.error("No usable samples were evaluated.")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(output_csv, rows)
    print(f"Wrote {len(rows)} sample metrics to {output_csv}", flush=True)
    if args.details_json:
        details = evaluate_many_details(
            dataset,
            model,
            device,
            max_samples=args.max_samples,
        )
        details_path = Path(args.details_json)
        details_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(details_path, details)
        print(f"Wrote pattern/chord details to {details_path}", flush=True)
    print_summary(rows)
    return 0


def evaluate_many(
    dataset: MaichartV25Dataset,
    model: torch.nn.Module,
    device: torch.device,
    *,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    limit = len(dataset) if max_samples is None else min(max(0, max_samples), len(dataset))
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for sample_index in range(limit):
            sample = dataset[sample_index]
            batch = collate_v25([sample])
            outputs = model(
                batch["x"].to(device),
                padding_mask=batch["padding_mask"].to(device),
            )
            metrics = compute_v25_metrics(outputs, batch["y"], batch["loss_mask"])
            rows.append(_row_from_metrics(sample_index, sample, metrics))
    return rows


def evaluate_many_details(
    dataset: MaichartV25Dataset,
    model: torch.nn.Module,
    device: torch.device,
    *,
    max_samples: int | None = None,
) -> dict[str, Any]:
    limit = len(dataset) if max_samples is None else min(max(0, max_samples), len(dataset))
    aggregate: dict[str, Any] | None = None
    with torch.no_grad():
        for sample_index in range(limit):
            sample = dataset[sample_index]
            batch = collate_v25([sample])
            outputs = model(
                batch["x"].to(device),
                padding_mask=batch["padding_mask"].to(device),
            )
            details = compute_v25_pattern_details(outputs, batch["y"], batch["loss_mask"])
            aggregate = details if aggregate is None else _merge_details(aggregate, details)
    return aggregate or {}


def print_summary(rows: list[dict[str, Any]]) -> None:
    print("Metric summary:", flush=True)
    print("metric,mean,median,min,max", flush=True)
    for field in SUMMARY_FIELDS:
        values = [float(row[field]) for row in rows]
        print(
            f"{field},"
            f"{statistics.fmean(values):.6f},"
            f"{statistics.median(values):.6f},"
            f"{min(values):.6f},"
            f"{max(values):.6f}",
            flush=True,
        )


def _row_from_metrics(
    sample_index: int,
    sample: dict[str, Any],
    metrics: dict[str, float],
) -> dict[str, Any]:
    meta = sample.get("meta", {})
    return {
        "sample_index": sample_index,
        "song_id": meta.get("song_id"),
        "difficulty_index": meta.get("difficulty_index"),
        "level": meta.get("level"),
        "note_presence_f1": metrics["note_presence_f1"],
        "button_micro_f1": metrics["button_micro_f1"],
        "button_best_f1": metrics["button_best_f1"],
        "button_best_threshold": metrics["button_best_threshold"],
        "note_type_accuracy": metrics["note_type_accuracy"],
        "density_mae": metrics["density_mae"],
        "note_start_f1": metrics["note_start_f1"],
        "note_start_f1@0.5": metrics["note_start_f1@0.5"],
        "note_start_best_f1": metrics["note_start_best_f1"],
        "note_start_best_threshold": metrics["note_start_best_threshold"],
        "note_start_precision@0.5": metrics["note_start_precision@0.5"],
        "note_start_recall@0.5": metrics["note_start_recall@0.5"],
        "note_start_pred_positive_rate@0.5": metrics["note_start_pred_positive_rate@0.5"],
        "note_start_target_positive_rate": metrics["note_start_target_positive_rate"],
        "pattern_start_accuracy": metrics["pattern_start_accuracy"],
        "pattern_start_macro_f1": metrics["pattern_start_macro_f1"],
        "chord_size_accuracy": metrics["chord_size_accuracy"],
        "chord_size_macro_f1": metrics["chord_size_macro_f1"],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _merge_details(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return _with_flat_detail_aliases({
        "pattern_start": _merge_one_detail(left["pattern_start"], right["pattern_start"]),
        "chord_size": _merge_one_detail(left["chord_size"], right["chord_size"]),
    })


def _merge_one_detail(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    confusion = [
        [int(a) + int(b) for a, b in zip(left_row, right_row, strict=False)]
        for left_row, right_row in zip(left["confusion_matrix"], right["confusion_matrix"], strict=False)
    ]
    per_class = _per_class_from_confusion(confusion, left["per_class"])
    total = sum(sum(row) for row in confusion)
    correct = sum(confusion[index][index] for index in range(len(confusion)))
    macro_values = [item["f1"] for item in per_class if item["target_count"] > 0]
    return {
        "total_count": total,
        "accuracy": correct / total if total > 0 else 0.0,
        "macro_f1": sum(macro_values) / len(macro_values) if macro_values else 0.0,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def _per_class_from_confusion(
    confusion: list[list[int]],
    class_template: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for class_index, template in enumerate(class_template):
        tp = int(confusion[class_index][class_index])
        target_count = int(sum(confusion[class_index]))
        pred_count = int(sum(row[class_index] for row in confusion))
        fp = pred_count - tp
        fn = target_count - tp
        precision = tp / pred_count if pred_count > 0 else 0.0
        recall = tp / target_count if target_count > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        details.append(
            {
                "class_id": template["class_id"],
                "class_name": template["class_name"],
                "target_count": target_count,
                "pred_count": pred_count,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return details


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Training manifest JSON path.")
    parser.add_argument("--cache-dir", default="cache", help="Cache root directory.")
    parser.add_argument("--feature-set", choices=FEATURE_SETS, help="Feature set override; defaults to checkpoint feature_set, then audio7.")
    parser.add_argument("--cache-samples", action="store_true", help="Cache decoded dataset samples in memory after first load.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--output-csv", required=True, help="CSV path for per-sample metrics.")
    parser.add_argument("--details-json", help="Optional JSON path for pattern/chord per-class details.")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
