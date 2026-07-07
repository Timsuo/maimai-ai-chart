"""Command-line training entrypoint for the V2.5 baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from maichart.models.transformer_v25 import MaichartTransformerV25
from maichart.training.collate import collate_v25
from maichart.training.dataset_v25 import MaichartV25Dataset, TrainingDataError
from maichart.training.losses_v25 import compute_v25_losses


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        dataset = MaichartV25Dataset(args.manifest, cache_dir=args.cache_dir)
    except TrainingDataError as exc:
        parser.error(str(exc))

    train_dataset = Subset(dataset, [0]) if args.overfit_one_sample else dataset
    try:
        first = dataset[0]
    except TrainingDataError as exc:
        parser.error(str(exc))
    print(
        "V2.5 dataset: "
        f"samples={len(dataset)} train_samples={len(train_dataset)} "
        f"input_dim={dataset.input_dim} "
        f"frames_first={first['x'].shape[0]} "
        f"note_types={dataset.num_note_types} "
        f"buttons=8",
        flush=True,
    )
    print(
        "Label dims: "
        f"note_presence={tuple(first['y']['note_presence'].shape)} "
        f"buttons={tuple(first['y']['buttons'].shape)} "
        f"note_type={tuple(first['y']['note_type'].shape)} "
        f"density={tuple(first['y']['density'].shape)}",
        flush=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=not args.overfit_one_sample,
        collate_fn=collate_v25,
    )
    model = MaichartTransformerV25(
        input_dim=dataset.input_dim,
        num_note_types=dataset.num_note_types,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        dim_feedforward=args.dim_feedforward,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        try:
            metrics = _train_epoch(model, loader, optimizer, device)
        except TrainingDataError as exc:
            parser.error(str(exc))
        print(
            f"epoch={epoch:03d} "
            f"loss={metrics['loss']:.6f} "
            f"note={metrics['loss_note']:.6f} "
            f"buttons={metrics['loss_buttons']:.6f} "
            f"type={metrics['loss_type']:.6f} "
            f"density={metrics['loss_density']:.6f}",
            flush=True,
        )
        if args.save_every > 0 and epoch % args.save_every == 0:
            _save_checkpoint(checkpoint_dir / f"v25_epoch_{epoch:03d}.pt", model, optimizer, epoch, args)

    _save_checkpoint(checkpoint_dir / "v25_last.pt", model, optimizer, args.epochs, args)
    return 0


def _train_epoch(
    model: MaichartTransformerV25,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {
        "loss": 0.0,
        "loss_note": 0.0,
        "loss_buttons": 0.0,
        "loss_type": 0.0,
        "loss_density": 0.0,
    }
    batches = 0
    for batch in loader:
        x = batch["x"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        targets = {key: value.to(device) for key, value in batch["y"].items()}
        loss_mask = batch["loss_mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(x, padding_mask=padding_mask)
        losses = compute_v25_losses(outputs, targets, loss_mask)
        losses["loss"].backward()
        optimizer.step()

        batches += 1
        for key in totals:
            totals[key] += float(losses[key].detach().cpu())

    return {key: value / max(1, batches) for key, value in totals.items()}


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
            "model_config": model.config,
            "args": vars(args),
        },
        path,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Training manifest JSON path.")
    parser.add_argument("--cache-dir", default="cache", help="Cache root directory.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--overfit-one-sample", action="store_true")
    parser.add_argument("--checkpoint-dir", default="checkpoints/v25")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
