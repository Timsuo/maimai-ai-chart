"""Visualize a maichart training/evaluation run.

The tool reads run artifacts when they are present, writes stable PNG figures,
and emits summary.json / summary.md with warnings instead of failing on missing
optional files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

LOSS_FIELDS = ("loss", "loss_note", "loss_buttons", "loss_type", "loss_density")
LOSS_COMPONENT_FIELDS = ("loss_note", "loss_buttons", "loss_type", "loss_density")
SAMPLE_METRIC_FIELDS = (
    "note_presence_f1",
    "button_micro_f1",
    "button_best_f1",
    "note_type_accuracy",
)
DIFFICULTY_METRIC_FIELDS = (
    "note_presence_f1",
    "button_best_f1",
    "note_type_accuracy",
    "density_mae",
)
NOTE_TYPE_ORDER = ("tap", "hold", "slide", "touch", "break", "empty")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run_dir = Path(args.run_dir)
    run_dir_exists = run_dir.exists()
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    train_rows = _read_csv_if_exists(run_dir / "train_log.csv", warnings, label="train_log.csv")
    val_rows = _read_csv_if_exists(run_dir / "val_log.csv", warnings, label="val_log.csv")
    test_rows = _read_csv_if_exists(run_dir / "test_metrics.csv", warnings, label="test_metrics.csv")

    if not run_dir_exists:
        warnings.append(f"Run directory does not exist: {run_dir}")
    if not val_rows:
        warnings.append("val_log.csv is missing; overfitting cannot be judged from this run.")
    if args.manifest and not Path(args.manifest).exists():
        warnings.append(f"Manifest does not exist: {args.manifest}")
    if args.cache_dir and not Path(args.cache_dir).exists():
        warnings.append(f"Cache directory does not exist: {args.cache_dir}")
    if test_rows and len(test_rows) < 10:
        warnings.append("Test set has fewer than 10 samples; metrics should be treated as smoke-test reference only.")

    generated_files: list[str] = []
    generated_files.extend(_write_stage_one_figures(output_dir, train_rows, val_rows, test_rows, warnings))
    generated_files.extend(_write_prediction_figures(run_dir / "predictions", output_dir, warnings))
    generated_files.extend(_write_threshold_sweep(run_dir, output_dir, test_rows, warnings))

    summary = build_summary(
        train_rows=train_rows,
        test_rows=test_rows,
        warnings=warnings,
        generated_files=generated_files,
        run_dir=run_dir,
        manifest=Path(args.manifest) if args.manifest else None,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
    )
    _write_json(output_dir / "summary.json", summary)
    _write_summary_md(output_dir / "summary.md", summary)
    print(f"Wrote {len(generated_files)} PNG files plus summary.json and summary.md to {output_dir}")
    return 0


def build_summary(
    *,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    warnings: list[str],
    generated_files: list[str],
    run_dir: Path,
    manifest: Path | None,
    cache_dir: Path | None,
) -> dict[str, Any]:
    best_train = _best_row(train_rows, "loss")
    final_train = train_rows[-1] if train_rows else None
    total_loss_reduction = _relative_reduction_percent(train_rows, "loss")
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "manifest": str(manifest) if manifest else None,
        "cache_dir": str(cache_dir) if cache_dir else None,
        "best_epoch": _get(best_train, "epoch"),
        "best_train_loss": _get(best_train, "loss"),
        "final_train_loss": _get(final_train, "loss"),
        "total_loss_reduction_percent": total_loss_reduction,
        "mean_note_presence_f1": _mean_field(test_rows, "note_presence_f1"),
        "mean_button_micro_f1": _mean_field(test_rows, "button_micro_f1"),
        "mean_button_best_f1": _mean_field(test_rows, "button_best_f1"),
        "mean_note_type_accuracy": _mean_field(test_rows, "note_type_accuracy"),
        "mean_density_mae": _mean_field(test_rows, "density_mae"),
        "worst_samples_by_button_f1": _worst_samples(test_rows, "button_best_f1", reverse=False),
        "worst_samples_by_density_mae": _worst_samples(test_rows, "density_mae", reverse=True),
        "warnings": _dedupe(warnings),
        "generated_files": generated_files,
    }
    return summary


def _write_stage_one_figures(
    output_dir: Path,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    warnings: list[str],
) -> list[str]:
    generated: list[str] = []
    generated.append(_plot_total_loss(output_dir, train_rows, val_rows, warnings))
    generated.append(_plot_loss_components(output_dir, train_rows, val_rows, warnings))
    generated.append(_plot_relative_loss(output_dir, train_rows, warnings))
    generated.append(_plot_metrics_by_sample(output_dir, test_rows, warnings))
    generated.append(_plot_density_mae_by_sample(output_dir, test_rows, warnings))
    generated.append(_plot_metrics_by_difficulty(output_dir, test_rows, warnings))
    return generated


def _plot_total_loss(
    output_dir: Path,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    warnings: list[str],
) -> str:
    path = output_dir / "01_training_total_loss.png"
    if not _has_field(train_rows, "loss"):
        warnings.append("train_log.csv does not contain a usable loss column.")
        return _placeholder(path, "Training total loss", "Missing train loss data")
    plt = _plt()
    plt.figure(figsize=(9, 5))
    plt.plot(_epochs(train_rows), _values(train_rows, "loss"), marker="o", linewidth=1.6, label="train loss")
    if _has_field(val_rows, "loss"):
        plt.plot(_epochs(val_rows), _values(val_rows, "loss"), marker="o", linewidth=1.6, label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training total loss")
    plt.grid(alpha=0.25)
    plt.legend()
    _save_current(path)
    return path.name


def _plot_loss_components(
    output_dir: Path,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    warnings: list[str],
) -> str:
    path = output_dir / "02_training_loss_components.png"
    fields = [field for field in LOSS_COMPONENT_FIELDS if _has_field(train_rows, field)]
    if not fields:
        warnings.append("train_log.csv does not contain usable loss component columns.")
        return _placeholder(path, "Training loss components", "Missing loss component data")
    plt = _plt()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for axis, field in zip(axes.flatten(), LOSS_COMPONENT_FIELDS):
        if _has_field(train_rows, field):
            axis.plot(_epochs(train_rows), _values(train_rows, field), linewidth=1.5, label=f"train {field}")
        if _has_field(val_rows, field):
            axis.plot(_epochs(val_rows), _values(val_rows, field), linewidth=1.5, label=f"val {field}")
        axis.set_title(field)
        axis.set_xlabel("epoch")
        axis.set_ylabel("loss")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


def _plot_relative_loss(output_dir: Path, train_rows: list[dict[str, Any]], warnings: list[str]) -> str:
    path = output_dir / "03_relative_loss_reduction.png"
    reductions = {
        field: _relative_reduction_percent(train_rows, field)
        for field in LOSS_FIELDS
        if _relative_reduction_percent(train_rows, field) is not None
    }
    if not reductions:
        warnings.append("Cannot compute relative loss reduction from train_log.csv.")
        return _placeholder(path, "Relative loss reduction", "Missing first/final loss values")
    plt = _plt()
    plt.figure(figsize=(10, 5))
    labels = list(reductions)
    values = [float(reductions[label]) for label in labels]
    colors = ["#4C78A8" if value >= 0 else "#E45756" for value in values]
    plt.bar(labels, values, color=colors)
    plt.axhline(0.0, color="#333333", linewidth=0.8)
    plt.ylabel("relative reduction (%)")
    plt.title("Loss reduction from first to final epoch")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.25)
    _save_current(path)
    return path.name


def _plot_metrics_by_sample(output_dir: Path, test_rows: list[dict[str, Any]], warnings: list[str]) -> str:
    path = output_dir / "04_test_metrics_by_sample.png"
    if not test_rows or not any(_has_field(test_rows, field) for field in SAMPLE_METRIC_FIELDS):
        warnings.append("test_metrics.csv does not contain usable per-sample F1/accuracy metrics.")
        return _placeholder(path, "Test metrics by sample", "Missing test metrics")
    plt = _plt()
    labels = _sample_labels(test_rows)
    x = list(range(len(test_rows)))
    width = 0.18
    plt.figure(figsize=(max(10, len(test_rows) * 0.7), 5))
    for offset, field in enumerate(SAMPLE_METRIC_FIELDS):
        values = _values(test_rows, field, default=math.nan)
        shifted = [index + (offset - 1.5) * width for index in x]
        plt.bar(shifted, values, width=width, label=field)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylim(0.0, 1.05)
    plt.ylabel("score")
    plt.title("Test metrics by sample")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    _save_current(path)
    return path.name


def _plot_density_mae_by_sample(output_dir: Path, test_rows: list[dict[str, Any]], warnings: list[str]) -> str:
    path = output_dir / "05_density_mae_by_sample.png"
    if not _has_field(test_rows, "density_mae"):
        warnings.append("test_metrics.csv does not contain a usable density_mae column.")
        return _placeholder(path, "Density MAE by sample", "Missing density_mae")
    plt = _plt()
    labels = _sample_labels(test_rows)
    plt.figure(figsize=(max(10, len(test_rows) * 0.65), 5))
    plt.bar(labels, _values(test_rows, "density_mae"), color="#F58518")
    plt.ylabel("density MAE")
    plt.title("Density MAE by sample")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.25)
    _save_current(path)
    return path.name


def _plot_metrics_by_difficulty(output_dir: Path, test_rows: list[dict[str, Any]], warnings: list[str]) -> str:
    path = output_dir / "06_metrics_by_difficulty.png"
    if not test_rows or not _has_field(test_rows, "difficulty_index"):
        warnings.append("test_metrics.csv does not contain difficulty_index for grouped metrics.")
        return _placeholder(path, "Metrics by difficulty", "Missing difficulty_index")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in test_rows:
        key = str(row.get("difficulty_index", "unknown"))
        grouped.setdefault(key, []).append(row)
    labels = sorted(grouped, key=_sort_key)
    if not labels:
        return _placeholder(path, "Metrics by difficulty", "Missing grouped metrics")
    plt = _plt()
    x = list(range(len(labels)))
    width = 0.18
    plt.figure(figsize=(max(10, len(labels) * 1.2), 5))
    for offset, field in enumerate(DIFFICULTY_METRIC_FIELDS):
        values = [_mean_field(grouped[label], field) for label in labels]
        shifted = [index + (offset - 1.5) * width for index in x]
        plt.bar(shifted, [math.nan if value is None else value for value in values], width=width, label=field)
    plt.xticks(x, labels)
    plt.ylabel("mean value")
    plt.title("Metrics by difficulty")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    _save_current(path)
    return path.name


def _write_prediction_figures(predictions_dir: Path, output_dir: Path, warnings: list[str]) -> list[str]:
    if not predictions_dir.exists():
        warnings.append("predictions/ is missing; prediction timeline/distribution figures were skipped.")
        return []
    generated: list[str] = []
    records = [_load_prediction_record(path, warnings) for path in sorted(predictions_dir.rglob("*.json"))]
    records = [record for record in records if record is not None]
    if not records:
        warnings.append("predictions/ contains no recognized per-frame prediction JSON files.")
        return generated
    for record in records:
        sample_name = _safe_name(record["sample_name"])
        density_path = output_dir / f"{sample_name}_density_timeline.png"
        button_path = output_dir / f"{sample_name}_button_distribution.png"
        type_path = output_dir / f"{sample_name}_note_type_distribution.png"
        _plot_density_timeline(density_path, record)
        _plot_button_distribution(button_path, record)
        _plot_note_type_distribution(type_path, record)
        generated.extend([density_path.name, button_path.name, type_path.name])
    return generated


def _load_prediction_record(path: Path, warnings: list[str]) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append(f"Could not read prediction JSON {path}: {exc}")
        return None
    target = data.get("target") or data.get("ground_truth") or data.get("truth")
    if not isinstance(target, dict):
        return None
    density_true = _numeric_list(target.get("density") or target.get("density_target"))
    density_pred = _numeric_list(data.get("density_prediction") or data.get("predicted_density"))
    buttons_true = _matrix(target.get("buttons") or target.get("button_target"))
    button_prob = _matrix(data.get("button_probability") or data.get("button_prediction"))
    type_true = _int_list(target.get("note_type") or target.get("note_type_target"))
    type_pred = _int_list(data.get("note_type_prediction") or data.get("predicted_note_type"))
    if not any([density_true and density_pred, buttons_true and button_prob, type_true and type_pred]):
        return None
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    sample_name = _sample_name_from_meta(meta, fallback=path.stem)
    return {
        "sample_name": sample_name,
        "frames": _numeric_list(data.get("frame_indices")) or list(range(max(len(density_true), len(density_pred)))),
        "density_true": density_true,
        "density_pred": density_pred,
        "buttons_true": buttons_true,
        "button_prob": button_prob,
        "type_true": type_true,
        "type_pred": type_pred,
        "note_type_vocab": data.get("note_type_vocab") if isinstance(data.get("note_type_vocab"), list) else [],
    }


def _plot_density_timeline(path: Path, record: dict[str, Any]) -> None:
    true_values = record["density_true"]
    pred_values = record["density_pred"]
    if not true_values or not pred_values:
        _placeholder(path, "Density timeline", "Missing density arrays")
        return
    count = min(len(true_values), len(pred_values), len(record["frames"]))
    plt = _plt()
    plt.figure(figsize=(12, 4))
    plt.plot(record["frames"][:count], true_values[:count], linewidth=1.2, label="ground truth density")
    plt.plot(record["frames"][:count], pred_values[:count], linewidth=1.2, label="predicted density")
    plt.xlabel("frame")
    plt.ylabel("note count / density")
    plt.title(record["sample_name"])
    plt.grid(alpha=0.25)
    plt.legend()
    _save_current(path)


def _plot_button_distribution(path: Path, record: dict[str, Any]) -> None:
    true_matrix = record["buttons_true"]
    pred_matrix = record["button_prob"]
    if not true_matrix or not pred_matrix:
        _placeholder(path, "Button distribution", "Missing button arrays")
        return
    true_counts = _column_counts(true_matrix, threshold=0.5)
    pred_counts = _column_counts(pred_matrix, threshold=0.5)
    labels = [str(index) for index in range(1, max(len(true_counts), len(pred_counts)) + 1)]
    true_counts = _pad(true_counts, len(labels))
    pred_counts = _pad(pred_counts, len(labels))
    plt = _plt()
    x = list(range(len(labels)))
    plt.figure(figsize=(8, 4))
    plt.bar([index - 0.2 for index in x], true_counts, width=0.4, label="ground truth count")
    plt.bar([index + 0.2 for index in x], pred_counts, width=0.4, label="predicted count")
    plt.xticks(x, labels)
    plt.xlabel("button")
    plt.ylabel("count")
    plt.title(record["sample_name"])
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    _save_current(path)


def _plot_note_type_distribution(path: Path, record: dict[str, Any]) -> None:
    if not record["type_true"] or not record["type_pred"]:
        _placeholder(path, "Note type distribution", "Missing note type arrays")
        return
    true_counts = _note_type_counts(record["type_true"], record["note_type_vocab"])
    pred_counts = _note_type_counts(record["type_pred"], record["note_type_vocab"])
    plt = _plt()
    x = list(range(len(NOTE_TYPE_ORDER)))
    plt.figure(figsize=(9, 4))
    plt.bar([index - 0.2 for index in x], [true_counts[label] for label in NOTE_TYPE_ORDER], width=0.4, label="ground truth count")
    plt.bar([index + 0.2 for index in x], [pred_counts[label] for label in NOTE_TYPE_ORDER], width=0.4, label="predicted count")
    plt.xticks(x, NOTE_TYPE_ORDER)
    plt.ylabel("count")
    plt.title(record["sample_name"])
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    _save_current(path)


def _write_threshold_sweep(
    run_dir: Path,
    output_dir: Path,
    test_rows: list[dict[str, Any]],
    warnings: list[str],
) -> list[str]:
    path = output_dir / "threshold_sweep.png"
    series = _threshold_series_from_rows(test_rows)
    if not series:
        for candidate in sorted(run_dir.rglob("*threshold*.csv")):
            series = _threshold_series_from_rows(_read_csv(candidate))
            if series:
                break
    if not series:
        for candidate in sorted(run_dir.rglob("*metrics*.json")):
            series = _threshold_series_from_json(candidate)
            if series:
                break
    if not series:
        warnings.append("No threshold sweep data found; threshold_sweep.png was skipped.")
        return []
    plt = _plt()
    plt.figure(figsize=(8, 5))
    for label, points in series.items():
        points = sorted(points)
        plt.plot([point[0] for point in points], [point[1] for point in points], marker="o", label=label)
    plt.xlabel("threshold")
    plt.ylabel("F1")
    plt.title("Threshold sweep")
    plt.ylim(0.0, 1.05)
    plt.grid(alpha=0.25)
    plt.legend()
    _save_current(path)
    return [path.name]


def _threshold_series_from_rows(rows: list[dict[str, Any]]) -> dict[str, list[tuple[float, float]]]:
    if not rows:
        return {}
    buckets: dict[str, dict[float, list[float]]] = {}
    for row in rows:
        for key, value in row.items():
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                continue
            match = re.search(r"(button|note).*f1@([0-9.]+)", str(key))
            if not match:
                continue
            label = f"{match.group(1)} F1"
            threshold = float(match.group(2).rstrip("."))
            buckets.setdefault(label, {}).setdefault(threshold, []).append(float(value))
    return {
        label: [(threshold, sum(values) / len(values)) for threshold, values in sorted(points.items())]
        for label, points in buckets.items()
    }


def _threshold_series_from_json(path: Path) -> dict[str, list[tuple[float, float]]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return _threshold_series_from_rows([data])


def _read_csv_if_exists(path: Path, warnings: list[str], *, label: str) -> list[dict[str, Any]]:
    if not path.exists():
        warnings.append(f"{label} is missing.")
        return []
    return _read_csv(path)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [{key: _convert(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def _convert(value: str | None) -> Any:
    if value is None:
        return None
    text = value.strip()
    if text == "":
        return None
    try:
        number = float(text)
    except ValueError:
        return text
    if math.isfinite(number) and number.is_integer():
        return int(number)
    return number


def _values(rows: list[dict[str, Any]], field: str, *, default: float | None = None) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(field, default)
        values.append(float(value) if isinstance(value, (int, float)) else math.nan)
    return values


def _epochs(rows: list[dict[str, Any]]) -> list[float]:
    if _has_field(rows, "epoch"):
        return _values(rows, "epoch")
    return [float(index + 1) for index in range(len(rows))]


def _has_field(rows: list[dict[str, Any]], field: str) -> bool:
    return any(isinstance(row.get(field), (int, float)) for row in rows)


def _relative_reduction_percent(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [value for value in _values(rows, field) if math.isfinite(value)]
    if len(values) < 2 or values[0] == 0:
        return None
    return (values[0] - values[-1]) / abs(values[0]) * 100.0


def _best_row(rows: list[dict[str, Any]], field: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if isinstance(row.get(field), (int, float))]
    if not candidates:
        return None
    return min(candidates, key=lambda row: float(row[field]))


def _mean_field(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if isinstance(row.get(field), (int, float))]
    return sum(values) / len(values) if values else None


def _worst_samples(rows: list[dict[str, Any]], field: str, *, reverse: bool) -> list[dict[str, Any]]:
    candidates = [row for row in rows if isinstance(row.get(field), (int, float))]
    ordered = sorted(candidates, key=lambda row: float(row[field]), reverse=reverse)[:5]
    return [
        {
            "sample_index": row.get("sample_index"),
            "song_id": row.get("song_id"),
            "difficulty_index": row.get("difficulty_index"),
            field: row.get(field),
        }
        for row in ordered
    ]


def _get(row: dict[str, Any] | None, field: str) -> Any:
    return row.get(field) if row else None


def _sample_labels(rows: list[dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    for index, row in enumerate(rows):
        sample_index = row.get("sample_index", index)
        song_id = row.get("song_id")
        difficulty = row.get("difficulty_index")
        if song_id is not None and difficulty is not None:
            labels.append(f"{sample_index}:{song_id}:d{difficulty}")
        else:
            labels.append(str(sample_index))
    return labels


def _sample_name_from_meta(meta: dict[str, Any], *, fallback: str) -> str:
    song_id = meta.get("song_id")
    difficulty = meta.get("difficulty_index")
    if song_id is not None and difficulty is not None:
        return f"sample_{song_id}_d{difficulty}"
    return f"sample_{fallback}"


def _safe_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return safe or "sample"


def _sort_key(value: str) -> tuple[int, Any]:
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


def _numeric_list(values: Any) -> list[float]:
    if not isinstance(values, list):
        return []
    result: list[float] = []
    for value in values:
        if isinstance(value, (int, float)):
            result.append(float(value))
    return result


def _int_list(values: Any) -> list[int]:
    return [int(value) for value in _numeric_list(values)]


def _matrix(values: Any) -> list[list[float]]:
    if not isinstance(values, list):
        return []
    result: list[list[float]] = []
    for row in values:
        if isinstance(row, list):
            result.append([float(value) for value in row if isinstance(value, (int, float))])
    return [row for row in result if row]


def _column_counts(matrix: list[list[float]], *, threshold: float) -> list[int]:
    width = max((len(row) for row in matrix), default=0)
    counts = [0 for _ in range(width)]
    for row in matrix:
        for index, value in enumerate(row):
            if value >= threshold:
                counts[index] += 1
    return counts


def _note_type_counts(values: list[int], vocab: list[Any]) -> dict[str, int]:
    counts = {label: 0 for label in NOTE_TYPE_ORDER}
    for value in values:
        label = str(vocab[value]) if 0 <= value < len(vocab) else str(value)
        normalized = _normalize_note_type(label)
        if normalized in counts:
            counts[normalized] += 1
    return counts


def _normalize_note_type(label: str) -> str:
    lowered = label.lower()
    if "empty" in lowered or "none" in lowered or lowered == "0":
        return "empty"
    if "break" in lowered:
        return "break"
    if "touch" in lowered:
        return "touch"
    if "slide" in lowered:
        return "slide"
    if "hold" in lowered:
        return "hold"
    if "tap" in lowered:
        return "tap"
    return lowered


def _pad(values: list[int], length: int) -> list[int]:
    return values + [0] * max(0, length - len(values))


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _placeholder(path: Path, title: str, message: str) -> str:
    plt = _plt()
    plt.figure(figsize=(8, 4))
    plt.title(title)
    plt.text(0.5, 0.5, message, ha="center", va="center", fontsize=13, wrap=True)
    plt.axis("off")
    _save_current(path)
    return path.name


def _save_current(path: Path) -> None:
    plt = _plt()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Run visualization summary",
        "",
        f"- run_dir: `{summary['run_dir']}`",
        f"- best_epoch: `{summary['best_epoch']}`",
        f"- best_train_loss: `{_fmt(summary['best_train_loss'])}`",
        f"- final_train_loss: `{_fmt(summary['final_train_loss'])}`",
        f"- total_loss_reduction_percent: `{_fmt(summary['total_loss_reduction_percent'])}`",
        f"- mean_note_presence_f1: `{_fmt(summary['mean_note_presence_f1'])}`",
        f"- mean_button_micro_f1: `{_fmt(summary['mean_button_micro_f1'])}`",
        f"- mean_button_best_f1: `{_fmt(summary['mean_button_best_f1'])}`",
        f"- mean_note_type_accuracy: `{_fmt(summary['mean_note_type_accuracy'])}`",
        f"- mean_density_mae: `{_fmt(summary['mean_density_mae'])}`",
        "",
        "## Worst samples by button F1",
        "",
    ]
    lines.extend(_sample_lines(summary["worst_samples_by_button_f1"], "button_best_f1"))
    lines.extend(["", "## Worst samples by density MAE", ""])
    lines.extend(_sample_lines(summary["worst_samples_by_density_mae"], "density_mae"))
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {warning}" for warning in summary["warnings"]] or ["- none"])
    lines.extend(["", "## Generated files", ""])
    lines.extend([f"- `{name}`" for name in summary["generated_files"]] or ["- none"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sample_lines(samples: list[dict[str, Any]], field: str) -> list[str]:
    if not samples:
        return ["- none"]
    return [
        "- sample_index={sample_index}, song_id={song_id}, difficulty_index={difficulty_index}, {field}={value}".format(
            field=field,
            value=_fmt(sample.get(field)),
            **sample,
        )
        for sample in samples
    ]


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Directory containing run artifacts.")
    parser.add_argument("--manifest", help="Manifest used by the run; recorded and checked for warnings.")
    parser.add_argument("--cache-dir", help="Cache directory used by the run; recorded and checked for warnings.")
    parser.add_argument("--output-dir", help="Directory where figures and summary files are written.")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
