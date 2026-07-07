from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from maichart.models.transformer_v25 import MaichartTransformerV25
from maichart.training.collate import collate_v25
from maichart.training.dataset_v25 import MaichartV25Dataset
from maichart.training.losses_v25 import compute_v25_losses
from maichart.training.train_v25 import main as train_main


def test_v25_dataset_collate_forward_loss_backward_smoke(tmp_path) -> None:
    dataset = MaichartV25Dataset(
        Path("manifests") / "training_manifest_limit2.json",
        cache_dir="cache",
    )
    sample = dataset[0]

    assert sample["x"].ndim == 2
    assert sample["x"].shape[1] == dataset.input_dim
    assert sample["y"]["note_presence"].shape == (sample["x"].shape[0], 1)
    assert sample["y"]["buttons"].shape == (sample["x"].shape[0], 8)
    assert sample["y"]["note_type"].shape == (sample["x"].shape[0],)
    assert sample["y"]["density"].shape == (sample["x"].shape[0], 1)

    batch = collate_v25([sample, dataset[1]])
    assert batch["x"].ndim == 3
    assert batch["x"].shape[0] == 2
    assert batch["x"].shape[2] == dataset.input_dim
    assert batch["padding_mask"].shape == batch["x"].shape[:2]
    assert batch["loss_mask"].shape == batch["x"].shape[:2]
    assert batch["y"]["buttons"].shape[:2] == batch["x"].shape[:2]

    model = MaichartTransformerV25(
        input_dim=dataset.input_dim,
        num_note_types=dataset.num_note_types,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.0,
    )
    outputs = model(batch["x"], padding_mask=batch["padding_mask"])
    assert outputs["note_presence_logits"].shape == (*batch["x"].shape[:2], 1)
    assert outputs["button_logits"].shape == (*batch["x"].shape[:2], 8)
    assert outputs["type_logits"].shape == (*batch["x"].shape[:2], dataset.num_note_types)
    assert outputs["density_pred"].shape == (*batch["x"].shape[:2], 1)

    losses = compute_v25_losses(outputs, batch["y"], batch["loss_mask"])
    losses["loss"].backward()
    assert losses["loss"].isfinite()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.parameters()
    )


def test_v25_overfit_one_sample_train_entrypoint_starts(tmp_path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    assert train_main(
        [
            "--manifest",
            str(Path("manifests") / "training_manifest_limit2.json"),
            "--cache-dir",
            "cache",
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--lr",
            "1e-4",
            "--overfit-one-sample",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--save-every",
            "0",
            "--d-model",
            "32",
            "--nhead",
            "4",
            "--num-layers",
            "1",
            "--dim-feedforward",
            "64",
            "--dropout",
            "0.0",
        ]
    ) == 0
    assert (checkpoint_dir / "v25_last.pt").is_file()
