"""Command-line training entrypoint for the V2.5 baseline."""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from maichart.models.transformer_v25 import MaichartTransformerV25
from maichart.training.collate import collate_v25
from maichart.training.dataset_v25 import FEATURE_SETS, MaichartV25Dataset, TrainingDataError
from maichart.training.evaluate_v25 import compute_v25_metrics
from maichart.training.losses_v25 import compute_v25_losses
from maichart.training.frame_label_codec import (
    CHORD_SIZE_START_CLASSES,
    START_PATTERN_IGNORE_INDEX,
    START_PATTERN_TYPES,
)
from maichart.training.utils_v25 import (
    assert_sample_is_finite,
    print_sample_sanity,
    resolve_device,
    select_sample_index,
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _set_reproducibility(args.seed, deterministic=args.deterministic)

    try:
        dataset = MaichartV25Dataset(
            args.manifest,
            cache_dir=args.cache_dir,
            feature_set=args.feature_set,
        )
        val_dataset = (
            MaichartV25Dataset(
                args.val_manifest,
                cache_dir=args.cache_dir,
                feature_set=args.feature_set,
            )
            if args.val_manifest
            else None
        )
    except TrainingDataError as exc:
        parser.error(str(exc))

    try:
        selected_index = select_sample_index(
            dataset,
            sample_index=args.sample_index,
            song_id=args.song_id,
            difficulty_index=args.difficulty_index,
        )
        sanity_sample = dataset[selected_index]
        assert_sample_is_finite(sanity_sample)
    except TrainingDataError as exc:
        parser.error(str(exc))
    train_dataset = Subset(dataset, [selected_index]) if args.overfit_one_sample else dataset
    print(
        "V2.5 dataset: "
        f"feature_set={dataset.feature_set} "
        f"samples={len(dataset)} train_samples={len(train_dataset)} "
        f"val_samples={len(val_dataset) if val_dataset is not None else 0} "
        f"input_dim={dataset.input_dim} "
        f"frames_selected={sanity_sample['x'].shape[0]} "
        f"note_types={dataset.num_note_types} "
        f"buttons=8",
        flush=True,
    )
    print(
        "Label dims: "
        f"note_presence={tuple(sanity_sample['y']['note_presence'].shape)} "
        f"buttons={tuple(sanity_sample['y']['buttons'].shape)} "
        f"note_type={tuple(sanity_sample['y']['note_type'].shape)} "
        f"density={tuple(sanity_sample['y']['density'].shape)} "
        f"note_start={tuple(sanity_sample['y']['note_start'].shape)} "
        f"pattern_start={tuple(sanity_sample['y']['pattern_start'].shape)} "
        f"chord_size_start={tuple(sanity_sample['y']['chord_size_start'].shape)}",
        flush=True,
    )
    print_sample_sanity(sanity_sample, dataset.note_type_vocab)

    try:
        device = resolve_device(args.device)
    except TrainingDataError as exc:
        parser.error(str(exc))
    print(f"Device: {device}", flush=True)
    print(
        f"Reproducibility: seed={args.seed} deterministic={args.deterministic}",
        flush=True,
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=not args.overfit_one_sample,
        collate_fn=collate_v25,
        generator=loader_generator,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.val_batch_size or args.batch_size,
            shuffle=False,
            collate_fn=collate_v25,
        )
        if val_dataset is not None
        else None
    )
    model = MaichartTransformerV25(
        input_dim=dataset.input_dim,
        num_note_types=dataset.num_note_types,
        num_start_pattern_types=dataset.num_start_pattern_types,
        num_chord_size_classes=dataset.num_chord_size_start_classes,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        dim_feedforward=args.dim_feedforward,
        density_nonnegative=args.density_nonnegative,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_csv = Path(args.log_csv) if args.log_csv else checkpoint_dir / "train_log.csv"
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    _init_log_csv(log_csv)
    best_loss = float("inf")
    best_val_loss = float("inf")
    epochs_without_val_improvement = 0
    last_epoch = 0
    try:
        (
            note_pos_weight,
            button_pos_weight,
            note_start_pos_weight,
            pattern_class_weight,
            chord_class_weight,
        ) = _resolve_training_weights(args, train_dataset)
    except TrainingDataError as exc:
        parser.error(str(exc))
    print(
        f"Loss pos_weight: note={_format_pos_weight(note_pos_weight)} "
        f"buttons={_format_pos_weight(button_pos_weight)} "
        f"note_start={_format_pos_weight(note_start_pos_weight)}",
        flush=True,
    )
    args.resolved_note_start_pos_weight = _serializable_weight(note_start_pos_weight)
    args.resolved_note_pos_weight = _serializable_weight(note_pos_weight)
    args.resolved_button_pos_weight = _serializable_weight(button_pos_weight)
    print(
        f"Loss class_weight: pattern={_format_pos_weight(pattern_class_weight)} "
        f"chord={_format_pos_weight(chord_class_weight)}",
        flush=True,
    )
    print(
        "Run config: "
        f"seed={args.seed} deterministic={args.deterministic} device={device} "
        f"feature_set={args.feature_set} input_dim={dataset.input_dim} "
        f"note_pos_weight={_format_pos_weight(note_pos_weight)} "
        f"button_pos_weight={_format_pos_weight(button_pos_weight)} "
        f"note_start_pos_weight={_format_pos_weight(note_start_pos_weight)} "
        f"pattern_class_weight={_format_pos_weight(pattern_class_weight)} "
        f"chord_class_weight={_format_pos_weight(chord_class_weight)} "
        f"note_start_weight={args.note_start_weight:.4f} "
        f"pattern_start_weight={args.pattern_start_weight:.4f} "
        f"chord_size_weight={args.chord_size_weight:.4f}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        last_epoch = epoch
        try:
            metrics = _train_epoch(
                model,
                loader,
                optimizer,
                device,
                note_pos_weight=note_pos_weight,
                button_pos_weight=button_pos_weight,
                pattern_class_weight=pattern_class_weight,
                chord_class_weight=chord_class_weight,
                note_start_weight=args.note_start_weight,
                note_start_pos_weight=note_start_pos_weight,
                pattern_start_weight=args.pattern_start_weight,
                chord_size_weight=args.chord_size_weight,
            )
        except TrainingDataError as exc:
            parser.error(str(exc))
        print(
            f"epoch={epoch:03d} "
            f"loss={metrics['loss']:.6f} "
            f"note={metrics['loss_note']:.6f} "
            f"buttons={metrics['loss_buttons']:.6f} "
            f"type={metrics['loss_type']:.6f} "
            f"density={metrics['loss_density']:.6f} "
            f"note_start={metrics['loss_note_start']:.6f} "
            f"pattern_start={metrics['loss_pattern_start']:.6f} "
            f"chord_size={metrics['loss_chord_size']:.6f}",
            flush=True,
        )
        val_metrics = None
        should_validate = val_loader is not None and args.val_every > 0 and epoch % args.val_every == 0
        if should_validate:
            try:
                val_metrics = _evaluate_epoch(
                    model,
                    val_loader,
                    device,
                    note_pos_weight=note_pos_weight,
                    button_pos_weight=button_pos_weight,
                    pattern_class_weight=pattern_class_weight,
                    chord_class_weight=chord_class_weight,
                    note_start_weight=args.note_start_weight,
                    note_start_pos_weight=note_start_pos_weight,
                    pattern_start_weight=args.pattern_start_weight,
                    chord_size_weight=args.chord_size_weight,
                )
            except TrainingDataError as exc:
                parser.error(str(exc))
            print(
                f"val epoch={epoch:03d} "
                f"loss={val_metrics['val_loss']:.6f} "
                f"note_f1={val_metrics['val_note_presence_f1']:.6f} "
                f"button_best_f1={val_metrics['val_button_best_f1']:.6f} "
                f"type_acc={val_metrics['val_note_type_accuracy']:.6f} "
                f"density_mae={val_metrics['val_density_mae']:.6f} "
                f"note_start_f1={val_metrics['val_note_start_f1']:.6f} "
                f"pattern_acc={val_metrics['val_pattern_start_accuracy']:.6f} "
                f"chord_acc={val_metrics['val_chord_size_accuracy']:.6f}",
                flush=True,
            )
            if _is_improvement(
                val_metrics["val_loss"],
                best_val_loss,
                min_delta=args.early_stopping_min_delta,
            ):
                best_val_loss = val_metrics["val_loss"]
                epochs_without_val_improvement = 0
                _save_checkpoint(checkpoint_dir / "v25_best_val.pt", model, optimizer, epoch, args)
            else:
                epochs_without_val_improvement += 1

        _append_log_csv(log_csv, epoch, metrics, val_metrics)
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            _save_checkpoint(checkpoint_dir / "v25_best.pt", model, optimizer, epoch, args)
        if args.save_every > 0 and epoch % args.save_every == 0:
            _save_checkpoint(checkpoint_dir / f"v25_epoch_{epoch:03d}.pt", model, optimizer, epoch, args)
        if (
            should_validate
            and args.early_stopping_patience > 0
            and epochs_without_val_improvement >= args.early_stopping_patience
        ):
            print(
                "Early stopping: "
                f"val_loss did not improve for {epochs_without_val_improvement} "
                f"validation check(s). best_val_loss={best_val_loss:.6f}",
                flush=True,
            )
            break

    _save_checkpoint(checkpoint_dir / "v25_last.pt", model, optimizer, last_epoch, args)
    return 0


def _train_epoch(
    model: MaichartTransformerV25,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    note_pos_weight: float | torch.Tensor | None,
    button_pos_weight: float | torch.Tensor | str | None,
    pattern_class_weight: torch.Tensor | None,
    chord_class_weight: torch.Tensor | None,
    note_start_weight: float,
    note_start_pos_weight: float | torch.Tensor | None,
    pattern_start_weight: float,
    chord_size_weight: float,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {
        "loss": 0.0,
        "loss_note": 0.0,
        "loss_buttons": 0.0,
        "loss_type": 0.0,
        "loss_density": 0.0,
        "loss_note_start": 0.0,
        "loss_pattern_start": 0.0,
        "loss_chord_size": 0.0,
    }
    batches = 0
    for batch in loader:
        _assert_batch_is_finite(batch)
        x = batch["x"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        targets = {key: value.to(device) for key, value in batch["y"].items()}
        loss_mask = batch["loss_mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(x, padding_mask=padding_mask)
        losses = compute_v25_losses(
            outputs,
            targets,
            loss_mask,
            note_pos_weight=note_pos_weight,
            button_pos_weight=button_pos_weight,
            pattern_class_weight=pattern_class_weight,
            chord_class_weight=chord_class_weight,
            note_start_weight=note_start_weight,
            note_start_pos_weight=note_start_pos_weight,
            pattern_start_weight=pattern_start_weight,
            chord_size_weight=chord_size_weight,
        )
        losses["loss"].backward()
        optimizer.step()

        batches += 1
        for key in totals:
            totals[key] += float(losses[key].detach().cpu())

    return {key: value / max(1, batches) for key, value in totals.items()}


def _evaluate_epoch(
    model: MaichartTransformerV25,
    loader: DataLoader,
    device: torch.device,
    *,
    note_pos_weight: float | torch.Tensor | None,
    button_pos_weight: float | torch.Tensor | str | None,
    pattern_class_weight: torch.Tensor | None,
    chord_class_weight: torch.Tensor | None,
    note_start_weight: float,
    note_start_pos_weight: float | torch.Tensor | None,
    pattern_start_weight: float,
    chord_size_weight: float,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {
        "val_loss": 0.0,
        "val_note_presence_f1": 0.0,
        "val_button_best_f1": 0.0,
        "val_note_type_accuracy": 0.0,
        "val_density_mae": 0.0,
        "val_loss_note": 0.0,
        "val_loss_buttons": 0.0,
        "val_loss_type": 0.0,
        "val_loss_density": 0.0,
        "val_loss_note_start": 0.0,
        "val_loss_pattern_start": 0.0,
        "val_loss_chord_size": 0.0,
        "val_note_start_f1": 0.0,
        "val_pattern_start_accuracy": 0.0,
        "val_pattern_start_macro_f1": 0.0,
        "val_chord_size_accuracy": 0.0,
        "val_chord_size_macro_f1": 0.0,
    }
    batches = 0
    with torch.no_grad():
        for batch in loader:
            _assert_batch_is_finite(batch)
            x = batch["x"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            targets = {key: value.to(device) for key, value in batch["y"].items()}
            loss_mask = batch["loss_mask"].to(device)
            outputs = model(x, padding_mask=padding_mask)
            losses = compute_v25_losses(
                outputs,
                targets,
                loss_mask,
                note_pos_weight=note_pos_weight,
                button_pos_weight=button_pos_weight,
                pattern_class_weight=pattern_class_weight,
                chord_class_weight=chord_class_weight,
                note_start_weight=note_start_weight,
                note_start_pos_weight=note_start_pos_weight,
                pattern_start_weight=pattern_start_weight,
                chord_size_weight=chord_size_weight,
            )
            metrics = compute_v25_metrics(outputs, targets, loss_mask)
            batches += 1
            totals["val_loss"] += float(losses["loss"].detach().cpu())
            totals["val_note_presence_f1"] += metrics["note_presence_f1"]
            totals["val_button_best_f1"] += metrics["button_best_f1"]
            totals["val_note_type_accuracy"] += metrics["note_type_accuracy"]
            totals["val_density_mae"] += metrics["density_mae"]
            totals["val_loss_note"] += float(losses["loss_note"].detach().cpu())
            totals["val_loss_buttons"] += float(losses["loss_buttons"].detach().cpu())
            totals["val_loss_type"] += float(losses["loss_type"].detach().cpu())
            totals["val_loss_density"] += float(losses["loss_density"].detach().cpu())
            totals["val_loss_note_start"] += float(losses["loss_note_start"].detach().cpu())
            totals["val_loss_pattern_start"] += float(losses["loss_pattern_start"].detach().cpu())
            totals["val_loss_chord_size"] += float(losses["loss_chord_size"].detach().cpu())
            totals["val_note_start_f1"] += metrics["note_start_f1"]
            totals["val_pattern_start_accuracy"] += metrics["pattern_start_accuracy"]
            totals["val_pattern_start_macro_f1"] += metrics["pattern_start_macro_f1"]
            totals["val_chord_size_accuracy"] += metrics["chord_size_accuracy"]
            totals["val_chord_size_macro_f1"] += metrics["chord_size_macro_f1"]
    model.train()
    return {key: value / max(1, batches) for key, value in totals.items()}


def _assert_batch_is_finite(batch: dict) -> None:
    tensors = {"x": batch["x"], **{f"y.{key}": value for key, value in batch["y"].items()}}
    bad = [
        name
        for name, tensor in tensors.items()
        if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all()
    ]
    if bad:
        raise TrainingDataError(f"Batch contains NaN or Inf values in: {', '.join(bad)}.")


def _set_reproducibility(seed: int, *, deterministic: bool) -> None:
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        print("warning: numpy is not installed; numpy.random.seed was not set.", flush=True)
    else:
        np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if not deterministic:
        return
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as exc:  # noqa: BLE001 - deterministic support varies by build.
        print(
            "warning: torch.use_deterministic_algorithms(True) failed: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )


def _resolve_training_weights(
    args: argparse.Namespace,
    train_dataset,
) -> tuple[
    float | torch.Tensor | None,
    float | torch.Tensor | str | None,
    float | torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    note_pos_weight = _parse_pos_weight_arg(args.note_pos_weight, name="--note-pos-weight")
    button_pos_weight = _parse_pos_weight_arg(args.button_pos_weight, name="--button-pos-weight")
    note_start_pos_weight = _parse_pos_weight_arg(
        args.note_start_pos_weight,
        name="--note-start-pos-weight",
    )
    pattern_class_weight_arg = _parse_class_weight_arg(
        args.pattern_class_weight,
        name="--pattern-class-weight",
    )
    chord_class_weight_arg = _parse_class_weight_arg(
        args.chord_class_weight,
        name="--chord-class-weight",
    )
    if (
        note_pos_weight == "auto"
        or button_pos_weight == "auto"
        or note_start_pos_weight == "auto"
        or pattern_class_weight_arg == "sqrt_auto"
        or chord_class_weight_arg == "sqrt_auto"
    ):
        stats = _count_training_targets(train_dataset)
    else:
        stats = None
    if note_pos_weight == "auto":
        note_pos_weight = _neg_pos_ratio(stats["note_positive"], stats["note_total"])
    if note_start_pos_weight == "auto":
        note_start_pos_weight = _neg_pos_ratio(
            stats["note_start_positive"],
            stats["note_start_total"],
        )
    if button_pos_weight == "auto":
        positives = stats["button_positive"]
        totals = stats["button_total"]
        negatives = totals - positives
        button_pos_weight = torch.where(
            positives > 0,
            negatives / positives.clamp_min(1.0),
            torch.ones_like(positives),
        )
    pattern_class_weight = None
    if pattern_class_weight_arg == "sqrt_auto":
        pattern_class_weight = _sqrt_inverse_class_weight(
            stats["pattern_start_counts"],
            cap=args.pattern_class_weight_cap,
        )
    chord_class_weight = None
    if chord_class_weight_arg == "sqrt_auto":
        chord_class_weight = _sqrt_inverse_class_weight(
            stats["chord_size_counts"],
            cap=args.chord_class_weight_cap,
        )
    return (
        note_pos_weight,
        button_pos_weight,
        note_start_pos_weight,
        pattern_class_weight,
        chord_class_weight,
    )


def _parse_pos_weight_arg(value: str | None, *, name: str) -> float | str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized == "none":
        return None
    if normalized == "auto":
        return "auto"
    try:
        parsed = float(value)
    except ValueError as exc:
        raise TrainingDataError(f"{name} must be a number, 'auto', or 'none'.") from exc
    if parsed <= 0:
        raise TrainingDataError(f"{name} must be positive.")
    return parsed


def _parse_class_weight_arg(value: str | None, *, name: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized == "none":
        return None
    if normalized == "sqrt_auto":
        return "sqrt_auto"
    raise TrainingDataError(f"{name} must be 'none' or 'sqrt_auto'.")


def _count_training_targets(train_dataset) -> dict[str, torch.Tensor]:
    note_positive = torch.tensor(0.0)
    note_total = torch.tensor(0.0)
    note_start_positive = torch.tensor(0.0)
    note_start_total = torch.tensor(0.0)
    button_positive = torch.zeros(8, dtype=torch.float32)
    button_total = torch.zeros(8, dtype=torch.float32)
    pattern_start_counts = torch.zeros(len(START_PATTERN_TYPES), dtype=torch.float32)
    chord_size_counts = torch.zeros(len(CHORD_SIZE_START_CLASSES), dtype=torch.float32)
    for index in range(len(train_dataset)):
        sample = train_dataset[index]
        note = sample["y"]["note_presence"].float()
        note_start = sample["y"]["note_start"].float()
        buttons = sample["y"]["buttons"].float()
        pattern_start = sample["y"]["pattern_start"]
        chord_size_start = sample["y"]["chord_size_start"]
        note_positive += note.sum()
        note_total += torch.tensor(float(note.numel()))
        note_start_positive += note_start.sum()
        note_start_total += torch.tensor(float(note_start.numel()))
        button_positive += buttons.sum(dim=0)
        button_total += torch.full((8,), float(buttons.size(0)))
        pattern_start_counts += _count_valid_classes(pattern_start, len(START_PATTERN_TYPES))
        chord_size_counts += _count_valid_classes(chord_size_start, len(CHORD_SIZE_START_CLASSES))
    return {
        "note_positive": note_positive,
        "note_total": note_total,
        "note_start_positive": note_start_positive,
        "note_start_total": note_start_total,
        "button_positive": button_positive,
        "button_total": button_total,
        "pattern_start_counts": pattern_start_counts,
        "chord_size_counts": chord_size_counts,
    }


def _neg_pos_ratio(positive: torch.Tensor, total: torch.Tensor) -> float:
    if float(positive.item()) <= 0:
        return 1.0
    return float(((total - positive) / positive.clamp_min(1.0)).item())


def _count_valid_classes(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    valid = target[target != START_PATTERN_IGNORE_INDEX].long()
    if valid.numel() == 0:
        return torch.zeros(num_classes, dtype=torch.float32)
    return torch.bincount(valid, minlength=num_classes).to(dtype=torch.float32)


def _sqrt_inverse_class_weight(counts: torch.Tensor, *, cap: float) -> torch.Tensor:
    if cap <= 0:
        raise TrainingDataError("class weight cap must be positive.")
    total = counts.sum()
    nonzero = counts > 0
    if float(total.item()) <= 0 or not bool(nonzero.any()):
        return torch.ones_like(counts)
    mean_count = total / max(1, int(nonzero.sum().item()))
    weights = torch.ones_like(counts)
    weights[nonzero] = torch.sqrt(mean_count / counts[nonzero].clamp_min(1.0))
    return weights.clamp(max=float(cap))


def _format_pos_weight(value: float | torch.Tensor | str | None) -> str:
    if value is None:
        return "none"
    if isinstance(value, str):
        return value
    if isinstance(value, torch.Tensor):
        values = [round(float(item), 4) for item in value.detach().cpu().reshape(-1).tolist()]
        return str(values)
    return f"{float(value):.4f}"


def _serializable_weight(value: float | torch.Tensor | str | None):
    if isinstance(value, torch.Tensor):
        values = value.detach().cpu().reshape(-1).tolist()
        return [float(item) for item in values]
    if value is None or isinstance(value, str):
        return value
    return float(value)


def _is_improvement(value: float, best: float, *, min_delta: float) -> bool:
    return value < best - min_delta


def _save_checkpoint(
    path: Path,
    model: MaichartTransformerV25,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": _config_to_dict(model.config),
            "data_config": {
                "feature_set": args.feature_set,
                "input_dim": model.config.input_dim,
            },
            "feature_set": args.feature_set,
            "input_dim": model.config.input_dim,
            "loss_config": {
                "note_start_pos_weight": getattr(args, "resolved_note_start_pos_weight", args.note_start_pos_weight),
                "note_start_weight": args.note_start_weight,
            },
            "seed": args.seed,
            "deterministic": args.deterministic,
            "train_args": vars(args),
            "args": vars(args),
        },
        path,
    )


def _config_to_dict(config) -> dict:
    if isinstance(config, dict):
        return dict(config)
    if is_dataclass(config):
        return asdict(config)
    return dict(config.__dict__)


def _init_log_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_loss_total",
                "loss",
                "loss_note",
                "loss_buttons",
                "loss_type",
                "loss_density",
                "loss_note_start",
                "loss_pattern_start",
                "loss_chord_size",
                "val_loss",
                "val_loss_total",
                "val_loss_note",
                "val_loss_buttons",
                "val_loss_type",
                "val_loss_density",
                "val_loss_note_start",
                "val_loss_pattern_start",
                "val_loss_chord_size",
                "val_note_presence_f1",
                "val_button_best_f1",
                "val_note_type_accuracy",
                "val_density_mae",
                "val_note_start_f1",
                "val_pattern_start_accuracy",
                "val_pattern_start_macro_f1",
                "val_chord_size_accuracy",
                "val_chord_size_macro_f1",
            ],
        )
        writer.writeheader()


def _append_log_csv(
    path: Path,
    epoch: int,
    metrics: dict[str, float],
    val_metrics: dict[str, float] | None,
) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_loss_total",
                "loss",
                "loss_note",
                "loss_buttons",
                "loss_type",
                "loss_density",
                "loss_note_start",
                "loss_pattern_start",
                "loss_chord_size",
                "val_loss",
                "val_loss_total",
                "val_loss_note",
                "val_loss_buttons",
                "val_loss_type",
                "val_loss_density",
                "val_loss_note_start",
                "val_loss_pattern_start",
                "val_loss_chord_size",
                "val_note_presence_f1",
                "val_button_best_f1",
                "val_note_type_accuracy",
                "val_density_mae",
                "val_note_start_f1",
                "val_pattern_start_accuracy",
                "val_pattern_start_macro_f1",
                "val_chord_size_accuracy",
                "val_chord_size_macro_f1",
            ],
        )
        row = {
            "epoch": epoch,
            "train_loss": metrics["loss"],
            "train_loss_total": metrics["loss"],
            **metrics,
        }
        if val_metrics is not None:
            row.update(val_metrics)
            row["val_loss_total"] = val_metrics["val_loss"]
        writer.writerow(row)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Training manifest JSON path.")
    parser.add_argument("--val-manifest", help="Validation training manifest JSON path.")
    parser.add_argument("--cache-dir", default="cache", help="Cache root directory.")
    parser.add_argument("--feature-set", choices=FEATURE_SETS, default="audio7")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--val-batch-size", type=int)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--overfit-one-sample", action="store_true")
    parser.add_argument("--checkpoint-dir", default="checkpoints/v25")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--sample-index", type=int)
    parser.add_argument("--song-id")
    parser.add_argument("--difficulty-index", type=int)
    parser.add_argument("--log-csv")
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--button-pos-weight", default="auto")
    parser.add_argument("--note-pos-weight", default="1.0")
    parser.add_argument("--pattern-class-weight", choices=("none", "sqrt_auto"), default="sqrt_auto")
    parser.add_argument("--pattern-class-weight-cap", type=float, default=10.0)
    parser.add_argument("--chord-class-weight", choices=("none", "sqrt_auto"), default="sqrt_auto")
    parser.add_argument("--chord-class-weight-cap", type=float, default=10.0)
    parser.add_argument("--note-start-weight", type=float, default=1.0)
    parser.add_argument("--note-start-pos-weight", default="1.0")
    parser.add_argument("--pattern-start-weight", type=float, default=0.5)
    parser.add_argument("--chord-size-weight", type=float, default=0.5)
    parser.add_argument("--density-nonnegative", action="store_true")
    parser.add_argument("--seed", type=int, default=8835)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
