"""Evaluate a V2.5 frame-level Transformer checkpoint on one sample."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch

from maichart.models.transformer_v25 import MaichartTransformerV25
from maichart.training.collate import collate_v25
from maichart.training.dataset_v25 import MaichartV25Dataset, TrainingDataError
from maichart.training.frame_label_codec import CHORD_SIZE_START_CLASSES, START_PATTERN_TYPES
from maichart.training.utils_v25 import resolve_device, select_sample_index


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        dataset = MaichartV25Dataset(args.manifest, cache_dir=args.cache_dir)
        sample_index = select_sample_index(
            dataset,
            sample_index=args.sample_index,
            song_id=args.song_id,
            difficulty_index=args.difficulty_index,
        )
        sample = dataset[sample_index]
        device = resolve_device(args.device)
        checkpoint = _load_checkpoint(Path(args.checkpoint), device)
        model = _model_from_checkpoint(checkpoint, dataset).to(device)
        _load_model_state_dict(model, checkpoint)
        model.eval()
    except TrainingDataError as exc:
        parser.error(str(exc))
    except FileNotFoundError as exc:
        parser.error(f"Missing checkpoint: {exc.filename}")

    batch = collate_v25([sample])
    with torch.no_grad():
        outputs = model(batch["x"].to(device), padding_mask=batch["padding_mask"].to(device))

    predictions = decode_predictions(outputs, batch["loss_mask"])
    metrics = compute_v25_metrics(outputs, batch["y"], batch["loss_mask"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "metrics.json",
        {
            **metrics,
            "sample_index": sample_index,
            "song_id": sample["meta"].get("song_id"),
            "difficulty_index": sample["meta"].get("difficulty_index"),
            "level": sample["meta"].get("level"),
            "checkpoint": str(Path(args.checkpoint)),
        },
    )
    _write_json(
        output_dir / "prediction_sample.json",
        {
            "meta": sample["meta"],
            "note_type_vocab": list(dataset.note_type_vocab),
            "target": target_sample(batch["y"], batch["loss_mask"]),
            **predictions,
        },
    )
    if args.details_json:
        details_path = Path(args.details_json)
        details_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(
            details_path,
            compute_v25_pattern_details(outputs, batch["y"], batch["loss_mask"]),
        )
    _write_plots(output_dir, batch["y"], predictions)
    print(f"Wrote evaluation outputs to {output_dir}", flush=True)
    return 0


def compute_v25_metrics(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    loss_mask: torch.Tensor,
) -> dict[str, float]:
    mask = loss_mask.to(dtype=torch.bool, device=outputs["note_presence_logits"].device)
    note_target = targets["note_presence"].to(mask.device).squeeze(-1)[mask].bool()
    note_prob = torch.sigmoid(outputs["note_presence_logits"]).squeeze(-1)[mask]
    note_pred = note_prob >= 0.5

    button_target = targets["buttons"].to(mask.device)[mask].bool()
    button_prob = torch.sigmoid(outputs["button_logits"])[mask]
    button_pred = button_prob >= 0.5

    type_target = targets["note_type"].to(mask.device)[mask]
    type_pred = outputs["type_logits"].argmax(dim=-1)[mask]

    density_target = targets["density"].to(mask.device).squeeze(-1)[mask]
    density_pred = outputs["density_pred"].squeeze(-1)[mask]

    note_p, note_r, note_f1 = _precision_recall_f1(note_pred, note_target)
    note_start_metrics = _note_start_metrics(outputs, targets, mask)
    pattern_metrics = _masked_multiclass_metrics(
        outputs.get("pattern_start_logits"),
        targets.get("pattern_start"),
        mask,
        num_classes=len(START_PATTERN_TYPES),
        prefix="pattern_start",
    )
    chord_metrics = _masked_multiclass_metrics(
        outputs.get("chord_size_logits"),
        targets.get("chord_size_start"),
        mask,
        num_classes=len(CHORD_SIZE_START_CLASSES),
        prefix="chord_size",
    )
    button_micro_p, button_micro_r, button_micro_f1 = _precision_recall_f1(
        button_pred.reshape(-1),
        button_target.reshape(-1),
    )
    button_macro_f1 = _macro_button_f1(button_pred, button_target)
    button_f1_at_005 = _micro_f1_at_threshold(button_prob, button_target, 0.05)
    button_f1_at_010 = _micro_f1_at_threshold(button_prob, button_target, 0.10)
    button_f1_at_020 = _micro_f1_at_threshold(button_prob, button_target, 0.20)
    button_best_threshold, button_best_f1 = _best_f1_threshold(
        button_prob.reshape(-1),
        button_target.reshape(-1),
    )
    note_best_threshold, note_best_f1 = _best_f1_threshold(note_prob, note_target)
    note_type_accuracy = (
        (type_pred == type_target).float().mean().item() if type_target.numel() else 0.0
    )
    density_mae = (
        torch.abs(density_pred - density_target).mean().item()
        if density_target.numel()
        else 0.0
    )

    return {
        "note_presence_precision": float(note_p),
        "note_presence_recall": float(note_r),
        "note_presence_f1": float(note_f1),
        "button_micro_precision": float(button_micro_p),
        "button_micro_recall": float(button_micro_r),
        "button_micro_f1": float(button_micro_f1),
        "button_macro_f1": float(button_macro_f1),
        "button_micro_f1@0.05": float(button_f1_at_005),
        "button_micro_f1@0.10": float(button_f1_at_010),
        "button_micro_f1@0.20": float(button_f1_at_020),
        "button_best_f1": float(button_best_f1),
        "button_best_threshold": float(button_best_threshold),
        "note_type_accuracy": float(note_type_accuracy),
        "note_best_f1": float(note_best_f1),
        "note_best_threshold": float(note_best_threshold),
        "density_mae": float(density_mae),
        **note_start_metrics,
        **pattern_metrics,
        **chord_metrics,
    }


def decode_predictions(
    outputs: dict[str, torch.Tensor],
    loss_mask: torch.Tensor,
) -> dict[str, Any]:
    mask = loss_mask[0].to(dtype=torch.bool, device=outputs["note_presence_logits"].device)
    note_prob = torch.sigmoid(outputs["note_presence_logits"][0, :, 0])[mask]
    button_prob = torch.sigmoid(outputs["button_logits"][0])[mask]
    type_pred = outputs["type_logits"][0].argmax(dim=-1)[mask]
    density_pred = outputs["density_pred"][0, :, 0][mask]
    predictions = {
        "frame_indices": list(range(int(mask.sum().item()))),
        "note_presence_probability": _tolist(note_prob),
        "button_probability": _tolist(button_prob),
        "note_type_prediction": _tolist(type_pred),
        "density_prediction": _tolist(density_pred),
    }
    if "note_start_logits" in outputs:
        predictions["note_start_probability"] = _tolist(
            torch.sigmoid(outputs["note_start_logits"][0, :, 0])[mask]
        )
    if "pattern_start_logits" in outputs:
        predictions["pattern_start_prediction"] = _tolist(
            outputs["pattern_start_logits"][0].argmax(dim=-1)[mask]
        )
    if "chord_size_logits" in outputs:
        predictions["chord_size_prediction"] = _tolist(
            outputs["chord_size_logits"][0].argmax(dim=-1)[mask]
        )
    return predictions


def target_sample(targets: dict[str, torch.Tensor], loss_mask: torch.Tensor) -> dict[str, Any]:
    mask = loss_mask[0].to(dtype=torch.bool)
    return {
        "note_presence": _tolist(targets["note_presence"][0, :, 0][mask]),
        "buttons": _tolist(targets["buttons"][0][mask]),
        "note_type": _tolist(targets["note_type"][0][mask]),
        "density": _tolist(targets["density"][0, :, 0][mask]),
        "note_start": _tolist(targets["note_start"][0, :, 0][mask]),
        "pattern_start": _tolist(targets["pattern_start"][0][mask]),
        "chord_size_start": _tolist(targets["chord_size_start"][0][mask]),
    }


def _precision_recall_f1(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float, float]:
    pred = pred.bool()
    target = target.bool()
    tp = torch.logical_and(pred, target).sum().item()
    fp = torch.logical_and(pred, ~target).sum().item()
    fn = torch.logical_and(~pred, target).sum().item()
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
    return precision, recall, f1


def _macro_button_f1(pred: torch.Tensor, target: torch.Tensor) -> float:
    if pred.numel() == 0:
        return 0.0
    f1s = [_precision_recall_f1(pred[:, index], target[:, index])[2] for index in range(pred.size(1))]
    return float(sum(f1s) / len(f1s))


def _note_start_metrics(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    mask: torch.Tensor,
) -> dict[str, float]:
    zero_metrics = {
        "note_start_precision": 0.0,
        "note_start_recall": 0.0,
        "note_start_f1": 0.0,
        "note_start_precision@0.5": 0.0,
        "note_start_recall@0.5": 0.0,
        "note_start_f1@0.5": 0.0,
        "note_start_best_f1": 0.0,
        "note_start_best_threshold": 0.5,
        "note_start_pred_positive_rate@0.5": 0.0,
        "note_start_target_positive_rate": 0.0,
    }
    if "note_start_logits" not in outputs or "note_start" not in targets:
        return zero_metrics
    target = targets["note_start"].to(mask.device).squeeze(-1)[mask].bool()
    probability = torch.sigmoid(outputs["note_start_logits"]).squeeze(-1)[mask]
    if target.numel() == 0:
        return zero_metrics
    pred = probability >= 0.5
    precision, recall, f1 = _precision_recall_f1(pred, target)
    best_threshold, best_f1 = _best_f1_threshold(probability, target)
    pred_positive_rate = pred.float().mean().item() if pred.numel() else 0.0
    target_positive_rate = target.float().mean().item() if target.numel() else 0.0
    return {
        "note_start_precision": float(precision),
        "note_start_recall": float(recall),
        "note_start_f1": float(f1),
        "note_start_precision@0.5": float(precision),
        "note_start_recall@0.5": float(recall),
        "note_start_f1@0.5": float(f1),
        "note_start_best_f1": float(best_f1),
        "note_start_best_threshold": float(best_threshold),
        "note_start_pred_positive_rate@0.5": float(pred_positive_rate),
        "note_start_target_positive_rate": float(target_positive_rate),
    }


def _masked_multiclass_metrics(
    logits: torch.Tensor | None,
    target: torch.Tensor | None,
    mask: torch.Tensor,
    *,
    num_classes: int,
    prefix: str,
) -> dict[str, float]:
    if logits is None or target is None:
        return {f"{prefix}_accuracy": 0.0, f"{prefix}_macro_f1": 0.0}
    target = target.to(mask.device)
    valid = mask & (target != -100)
    if not valid.any():
        return {f"{prefix}_accuracy": 0.0, f"{prefix}_macro_f1": 0.0}
    pred = logits.argmax(dim=-1).to(mask.device)[valid]
    target = target[valid]
    accuracy = (pred == target).float().mean().item()
    macro_f1 = _macro_multiclass_f1(pred, target, num_classes)
    return {f"{prefix}_accuracy": float(accuracy), f"{prefix}_macro_f1": float(macro_f1)}


def _macro_multiclass_f1(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    f1s: list[float] = []
    for class_index in range(num_classes):
        class_target = target == class_index
        if not class_target.any():
            continue
        class_pred = pred == class_index
        f1s.append(_precision_recall_f1(class_pred, class_target)[2])
    return float(sum(f1s) / len(f1s)) if f1s else 0.0


def compute_v25_pattern_details(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    loss_mask: torch.Tensor,
) -> dict[str, Any]:
    mask = loss_mask.to(dtype=torch.bool, device=outputs["note_presence_logits"].device)
    details = {
        "pattern_start": _masked_multiclass_details(
            outputs.get("pattern_start_logits"),
            targets.get("pattern_start"),
            mask,
            class_names=START_PATTERN_TYPES,
        ),
        "chord_size": _masked_multiclass_details(
            outputs.get("chord_size_logits"),
            targets.get("chord_size_start"),
            mask,
            class_names=CHORD_SIZE_START_CLASSES,
        ),
    }
    return _with_flat_detail_aliases(details)


def _masked_multiclass_details(
    logits: torch.Tensor | None,
    target: torch.Tensor | None,
    mask: torch.Tensor,
    *,
    class_names: tuple[str, ...],
) -> dict[str, Any]:
    num_classes = len(class_names)
    empty_confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    if logits is None or target is None:
        return {
            "total_count": 0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "per_class": _empty_per_class_details(class_names),
            "confusion_matrix": empty_confusion,
        }
    target = target.to(mask.device)
    valid = mask & (target != -100)
    if not valid.any():
        return {
            "total_count": 0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "per_class": _empty_per_class_details(class_names),
            "confusion_matrix": empty_confusion,
        }

    pred = logits.argmax(dim=-1).to(mask.device)[valid].long()
    target = target[valid].long()
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long, device=target.device)
    for true_class, pred_class in zip(target.reshape(-1), pred.reshape(-1), strict=False):
        if 0 <= int(true_class.item()) < num_classes and 0 <= int(pred_class.item()) < num_classes:
            confusion[true_class, pred_class] += 1
    per_class = _per_class_details_from_confusion(confusion, class_names)
    accuracy = (pred == target).float().mean().item()
    macro_values = [item["f1"] for item in per_class if item["target_count"] > 0]
    macro_f1 = sum(macro_values) / len(macro_values) if macro_values else 0.0
    return {
        "total_count": int(target.numel()),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "confusion_matrix": confusion.detach().cpu().tolist(),
    }


def _empty_per_class_details(class_names: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        {
            "class_id": class_index,
            "class_name": class_name,
            "target_count": 0,
            "pred_count": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
        for class_index, class_name in enumerate(class_names)
    ]


def _per_class_details_from_confusion(
    confusion: torch.Tensor,
    class_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for class_index, class_name in enumerate(class_names):
        tp = int(confusion[class_index, class_index].item())
        target_count = int(confusion[class_index, :].sum().item())
        pred_count = int(confusion[:, class_index].sum().item())
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
                "class_id": class_index,
                "class_name": class_name,
                "target_count": target_count,
                "pred_count": pred_count,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )
    return details


def _with_flat_detail_aliases(details: dict[str, Any]) -> dict[str, Any]:
    pattern = details["pattern_start"]
    chord = details["chord_size"]
    details.update(
        {
            "pattern_start_per_class_precision": _per_class_metric_map(pattern, "precision"),
            "pattern_start_per_class_recall": _per_class_metric_map(pattern, "recall"),
            "pattern_start_per_class_f1": _per_class_metric_map(pattern, "f1"),
            "pattern_start_confusion_matrix": pattern["confusion_matrix"],
            "chord_size_per_class_precision": _per_class_metric_map(chord, "precision"),
            "chord_size_per_class_recall": _per_class_metric_map(chord, "recall"),
            "chord_size_per_class_f1": _per_class_metric_map(chord, "f1"),
            "chord_size_confusion_matrix": chord["confusion_matrix"],
        }
    )
    return details


def _per_class_metric_map(detail: dict[str, Any], metric: str) -> dict[str, float]:
    return {
        str(item["class_name"]): float(item[metric])
        for item in detail["per_class"]
    }


def _micro_f1_at_threshold(
    probability: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
) -> float:
    return _precision_recall_f1(
        probability.reshape(-1) >= threshold,
        target.reshape(-1).bool(),
    )[2]


def _best_f1_threshold(
    probability: torch.Tensor,
    target: torch.Tensor,
) -> tuple[float, float]:
    if probability.numel() == 0:
        return 0.5, 0.0
    thresholds = torch.linspace(0.0, 1.0, steps=101, device=probability.device)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        f1 = _precision_recall_f1(probability >= threshold, target.bool())[2]
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold.item())
    return best_threshold, best_f1


def _write_plots(output_dir: Path, targets: dict[str, torch.Tensor], predictions: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = predictions["frame_indices"]
    note_target = targets["note_presence"][0, : len(frames), 0].cpu().tolist()
    button_target = targets["buttons"][0, : len(frames)].cpu().transpose(0, 1)
    button_target_flat = targets["buttons"][0, : len(frames)].cpu().reshape(-1)
    density_target = targets["density"][0, : len(frames), 0].cpu().tolist()
    button_pred = torch.tensor(predictions["button_probability"], dtype=torch.float32).transpose(0, 1)
    button_pred_flat = torch.tensor(predictions["button_probability"], dtype=torch.float32).reshape(-1)

    plt.figure(figsize=(12, 4))
    plt.plot(frames, note_target, label="target note_presence", linewidth=1.2)
    plt.plot(frames, predictions["note_presence_probability"], label="predicted probability", linewidth=1.2)
    plt.xlabel("frame")
    plt.ylabel("note")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "note_presence_compare.png", dpi=150)
    plt.close()

    _save_button_heatmap(plt, button_target, output_dir / "button_heatmap_target.png", "target buttons")
    _save_button_heatmap(plt, button_pred, output_dir / "button_heatmap_pred.png", "predicted button probability")
    _save_button_probability_histogram(
        plt,
        positive=button_pred_flat[button_target_flat.bool()],
        negative=button_pred_flat[~button_target_flat.bool()],
        path=output_dir / "button_probability_histogram.png",
    )

    plt.figure(figsize=(12, 4))
    plt.plot(frames, density_target, label="target density", linewidth=1.2)
    plt.plot(frames, predictions["density_prediction"], label="predicted density", linewidth=1.2)
    plt.xlabel("frame")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "density_compare.png", dpi=150)
    plt.close()


def _save_button_heatmap(plt, values: torch.Tensor, path: Path, title: str) -> None:
    plt.figure(figsize=(12, 4))
    plt.imshow(values.cpu().numpy(), aspect="auto", origin="lower", vmin=0.0, vmax=1.0)
    plt.yticks(range(8), [str(index) for index in range(1, 9)])
    plt.xlabel("frame")
    plt.ylabel("button")
    plt.title(title)
    plt.colorbar(label="value")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _save_button_probability_histogram(
    plt,
    *,
    positive: torch.Tensor,
    negative: torch.Tensor,
    path: Path,
) -> None:
    plt.figure(figsize=(8, 4))
    bins = [index / 50.0 for index in range(51)]
    if negative.numel() > 0:
        plt.hist(
            negative.cpu().numpy(),
            bins=bins,
            alpha=0.55,
            label="target negative",
            density=True,
        )
    if positive.numel() > 0:
        plt.hist(
            positive.cpu().numpy(),
            bins=bins,
            alpha=0.55,
            label="target positive",
            density=True,
        )
    plt.xlabel("button probability")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise TrainingDataError(f"Checkpoint does not contain model_state_dict: {path}")
    return checkpoint


def _load_model_state_dict(model: MaichartTransformerV25, checkpoint: dict[str, Any]) -> None:
    result = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if result.missing_keys or result.unexpected_keys:
        print(
            "Loaded checkpoint with non-strict state dict: "
            f"missing={list(result.missing_keys)} "
            f"unexpected={list(result.unexpected_keys)}",
            flush=True,
        )


def _model_from_checkpoint(
    checkpoint: dict[str, Any],
    dataset: MaichartV25Dataset,
) -> MaichartTransformerV25:
    config = _config_to_dict(checkpoint.get("model_config"))
    if not config:
        config = {
            "input_dim": dataset.input_dim,
            "num_note_types": dataset.num_note_types,
        }
    return MaichartTransformerV25(**config)


def _config_to_dict(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, dict):
        return dict(config)
    if is_dataclass(config):
        return asdict(config)
    values = {
        key: getattr(config, key)
        for key in (
            "input_dim",
            "num_note_types",
            "num_start_pattern_types",
            "num_chord_size_classes",
            "d_model",
            "nhead",
            "num_layers",
            "dropout",
            "dim_feedforward",
            "positional_encoding",
            "max_len",
            "density_nonnegative",
        )
        if hasattr(config, key)
    }
    return values


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _tolist(tensor: torch.Tensor) -> list[Any]:
    return tensor.detach().cpu().tolist()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Training manifest JSON path.")
    parser.add_argument("--cache-dir", default="cache", help="Cache root directory.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--output-dir", required=True, help="Directory for metrics and plots.")
    parser.add_argument("--sample-index", type=int)
    parser.add_argument("--song-id")
    parser.add_argument("--difficulty-index", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--details-json", help="Optional JSON path for pattern/chord per-class details.")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
