from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from maichart.models.transformer_v25 import MaichartTransformerV25
from maichart.training.collate import collate_v25
from maichart.training.dataset_v25 import MaichartV25Dataset, TrainingDataError
from maichart.training.evaluate_v25 import (
    _model_from_checkpoint,
    _resolve_checkpoint_feature_set,
    compute_v25_metrics,
    compute_v25_pattern_details,
)
from maichart.training.frame_label_codec import START_PATTERN_IGNORE_INDEX, START_PATTERN_TO_ID
from maichart.training.losses_v25 import compute_v25_losses
from maichart.training.train_v25 import _resolve_training_weights


def test_v25_minimal_dataset_collate_forward_loss_and_metrics(tmp_path) -> None:
    manifest = _write_minimal_v25_cache(tmp_path)
    dataset = MaichartV25Dataset(manifest, cache_dir=tmp_path / "cache")

    sample = dataset[0]
    assert dataset.feature_dim == 7
    assert dataset.input_dim == 7
    assert sample["meta"]["song_id"] == "song001"
    assert sample["meta"]["difficulty_index"] == 4
    assert sample["x"].shape == (4, dataset.input_dim)
    assert sample["y"]["note_presence"].shape == (4, 1)
    assert sample["y"]["buttons"].shape == (4, 8)
    assert sample["y"]["note_start"].shape == (4, 1)
    assert sample["y"]["pattern_start"].shape == (4,)
    assert sample["y"]["chord_size_start"].shape == (4,)
    assert sample["y"]["pattern_start"].tolist() == [
        START_PATTERN_IGNORE_INDEX,
        START_PATTERN_TO_ID["single_tap"],
        START_PATTERN_TO_ID["single_hold"],
        START_PATTERN_IGNORE_INDEX,
    ]
    assert sample["y"]["chord_size_start"].tolist() == [START_PATTERN_IGNORE_INDEX, 0, 0, START_PATTERN_IGNORE_INDEX]

    batch = collate_v25([sample])
    assert batch["x"].shape == (1, 4, dataset.input_dim)
    assert batch["padding_mask"].shape == (1, 4)
    assert batch["loss_mask"].all()
    assert batch["y"]["note_start"].shape == (1, 4, 1)
    assert batch["y"]["pattern_start"].shape == (1, 4)
    assert batch["y"]["chord_size_start"].shape == (1, 4)

    model = MaichartTransformerV25(
        input_dim=dataset.input_dim,
        num_note_types=dataset.num_note_types,
        num_start_pattern_types=dataset.num_start_pattern_types,
        num_chord_size_classes=dataset.num_chord_size_start_classes,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.0,
    )
    outputs = model(batch["x"], padding_mask=batch["padding_mask"])
    assert outputs["note_presence_logits"].shape == (1, 4, 1)
    assert outputs["button_logits"].shape == (1, 4, 8)
    assert outputs["note_start_logits"].shape == (1, 4, 1)
    assert outputs["pattern_start_logits"].shape == (1, 4, dataset.num_start_pattern_types)
    assert outputs["chord_size_logits"].shape == (1, 4, dataset.num_chord_size_start_classes)

    losses = compute_v25_losses(
        outputs,
        batch["y"],
        batch["loss_mask"],
        note_pos_weight=1.0,
        button_pos_weight="auto",
    )
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    assert any(parameter.grad is not None for parameter in model.parameters())

    metrics = compute_v25_metrics(outputs, batch["y"], batch["loss_mask"])
    assert set(metrics) >= {
        "note_presence_precision",
        "note_presence_recall",
        "note_presence_f1",
        "button_micro_precision",
        "button_micro_recall",
        "button_micro_f1",
        "button_micro_f1@0.05",
        "button_micro_f1@0.10",
        "button_micro_f1@0.20",
        "button_macro_f1",
        "button_best_f1",
        "button_best_threshold",
        "note_type_accuracy",
        "note_best_f1",
        "note_best_threshold",
        "density_mae",
        "note_start_precision",
        "note_start_recall",
        "note_start_f1",
        "note_start_precision@0.5",
        "note_start_recall@0.5",
        "note_start_f1@0.5",
        "note_start_best_f1",
        "note_start_best_threshold",
        "note_start_pred_positive_rate@0.5",
        "note_start_target_positive_rate",
        "pattern_start_accuracy",
        "pattern_start_macro_f1",
        "chord_size_accuracy",
        "chord_size_macro_f1",
    }
    assert all(isinstance(value, float) for value in metrics.values())

    details = compute_v25_pattern_details(outputs, batch["y"], batch["loss_mask"])
    assert details["pattern_start"]["total_count"] == 2
    assert len(details["pattern_start"]["per_class"]) == dataset.num_start_pattern_types
    assert len(details["pattern_start"]["confusion_matrix"]) == dataset.num_start_pattern_types
    assert details["chord_size"]["total_count"] == 2
    assert len(details["chord_size"]["per_class"]) == dataset.num_chord_size_start_classes


def test_v25_audio7_plus_grid_feature_dim_and_values_are_finite(tmp_path) -> None:
    manifest = _write_minimal_v25_cache(tmp_path)
    dataset = MaichartV25Dataset(
        manifest,
        cache_dir=tmp_path / "cache",
        feature_set="audio7_plus_grid",
    )

    sample = dataset[0]
    assert dataset.feature_dim == 15
    assert dataset.input_dim == 15
    assert sample["x"].shape == (4, 15)
    assert torch.isfinite(sample["x"]).all()
    assert sample["meta"]["feature_set"] == "audio7_plus_grid"
    assert sample["x"][0, -3:].tolist() == pytest.approx([120 / 240, 4 / 5, 12 / 15])


def test_v25_note_start_pos_weight_fixed_and_auto(tmp_path) -> None:
    manifest = _write_minimal_v25_cache(tmp_path)
    dataset = MaichartV25Dataset(manifest, cache_dir=tmp_path / "cache")
    args = Namespace(
        note_pos_weight="1.0",
        button_pos_weight="none",
        note_start_pos_weight="2.5",
        pattern_class_weight="none",
        chord_class_weight="none",
        pattern_class_weight_cap=10.0,
        chord_class_weight_cap=10.0,
    )

    weights = _resolve_training_weights(args, dataset)
    assert weights[2] == pytest.approx(2.5)

    args.note_start_pos_weight = "auto"
    weights = _resolve_training_weights(args, dataset)
    assert weights[2] == pytest.approx(1.0)


def test_v25_evaluate_restores_feature_set_and_rejects_input_dim_mismatch(tmp_path) -> None:
    manifest = _write_minimal_v25_cache(tmp_path)
    grid_dataset = MaichartV25Dataset(
        manifest,
        cache_dir=tmp_path / "cache",
        feature_set="audio7_plus_grid",
    )
    checkpoint = {
        "model_state_dict": {},
        "model_config": {
            "input_dim": 15,
            "num_note_types": grid_dataset.num_note_types,
            "d_model": 32,
            "nhead": 4,
            "num_layers": 1,
            "dim_feedforward": 64,
            "dropout": 0.0,
        },
        "data_config": {"feature_set": "audio7_plus_grid", "input_dim": 15},
    }

    assert _resolve_checkpoint_feature_set(checkpoint, None) == "audio7_plus_grid"
    assert _resolve_checkpoint_feature_set(checkpoint, "audio7_plus_grid") == "audio7_plus_grid"
    with pytest.raises(TrainingDataError, match="feature_set mismatch"):
        _resolve_checkpoint_feature_set(checkpoint, "audio7")

    model = _model_from_checkpoint(checkpoint, grid_dataset)
    assert model.config.input_dim == 15

    audio_dataset = MaichartV25Dataset(manifest, cache_dir=tmp_path / "cache")
    with pytest.raises(TrainingDataError, match="Checkpoint input_dim mismatch"):
        _model_from_checkpoint(checkpoint, audio_dataset)


def test_v25_losses_handle_batch_without_note_starts() -> None:
    batch_size = 1
    frames = 3
    num_note_types = 7
    outputs = {
        "note_presence_logits": torch.zeros((batch_size, frames, 1), requires_grad=True),
        "button_logits": torch.zeros((batch_size, frames, 8), requires_grad=True),
        "type_logits": torch.zeros((batch_size, frames, num_note_types), requires_grad=True),
        "density_pred": torch.zeros((batch_size, frames, 1), requires_grad=True),
        "note_start_logits": torch.zeros((batch_size, frames, 1), requires_grad=True),
        "pattern_start_logits": torch.zeros((batch_size, frames, 11), requires_grad=True),
        "chord_size_logits": torch.zeros((batch_size, frames, 3), requires_grad=True),
    }
    targets = {
        "note_presence": torch.zeros((batch_size, frames, 1)),
        "buttons": torch.zeros((batch_size, frames, 8)),
        "note_type": torch.zeros((batch_size, frames), dtype=torch.long),
        "density": torch.zeros((batch_size, frames, 1)),
        "note_start": torch.zeros((batch_size, frames, 1)),
        "pattern_start": torch.full((batch_size, frames), START_PATTERN_IGNORE_INDEX, dtype=torch.long),
        "chord_size_start": torch.full((batch_size, frames), START_PATTERN_IGNORE_INDEX, dtype=torch.long),
    }
    losses = compute_v25_losses(outputs, targets, torch.ones((batch_size, frames), dtype=torch.bool))

    assert torch.isfinite(losses["loss"])
    assert losses["loss_pattern_start"].item() == 0.0
    assert losses["loss_chord_size"].item() == 0.0
    losses["loss"].backward()


def _write_minimal_v25_cache(root: Path) -> Path:
    cache = root / "cache"
    song_id = "song001"
    difficulty = 4
    audio_path = cache / "audio_features" / f"{song_id}.audio_features.json"
    labels_path = cache / "frame_labels" / song_id / f"difficulty_{difficulty}.frame_labels.json"
    align_path = cache / "alignment_reports" / song_id / f"difficulty_{difficulty}.alignment_report.json"
    chart_path = cache / "chart_ir" / song_id / f"difficulty_{difficulty}.chart_ir.json"

    _write_json(
        audio_path,
        {
            "schema": "maichart-audio-features-v1",
            "feature_frames": [
                _audio_frame(0, 0.0),
                _audio_frame(1, 0.1),
                _audio_frame(2, 0.2),
                _audio_frame(3, 0.3),
            ],
        },
    )
    _write_json(
        labels_path,
        {
            "schema": "maichart-frame-labels-v1",
            "song_id": song_id,
            "difficulty": difficulty,
            "frames": [
                _label_frame(0, 0.0, False, [], {}),
                _label_frame(1, 0.1, True, ["1", "3"], {"tap_count": 1}),
                _label_frame(2, 0.2, True, ["2"], {"hold_start_count": 1}),
                _label_frame(3, 0.3, False, [], {}),
            ],
        },
    )
    _write_json(align_path, {"schema": "maichart-alignment-report-v1", "summary": {}})
    _write_json(chart_path, {"schema_version": 1, "notes": []})

    manifest_path = root / "manifest.json"
    _write_json(
        manifest_path,
        {
            "schema": "maichart-training-manifest-v1",
            "songs": [
                {
                    "song_id": song_id,
                    "audio": {"audio_features_path": _rel(audio_path, root)},
                    "difficulties": [
                        {
                            "difficulty_index": difficulty,
                            "level": 12.0,
                            "usable_for_training": True,
                            "frame_labels_path": _rel(labels_path, root),
                            "alignment_report_path": _rel(align_path, root),
                            "chart_ir_path": _rel(chart_path, root),
                        }
                    ],
                }
            ],
        },
    )
    return manifest_path


def _audio_frame(index: int, time_sec: float) -> dict:
    return {
        "frame_index": index,
        "time_sec": time_sec,
        "onset_strength": 0.1 * index,
        "rms": 0.01 * (index + 1),
        "percussive_rms": 0.02 * (index + 1),
        "harmonic_rms": 0.03 * (index + 1),
        "spectral_centroid": 1000.0 + index,
        "spectral_bandwidth": 2000.0 + index,
        "zero_crossing_rate": 0.05,
    }


def _label_frame(
    index: int,
    time_sec: float,
    has_note: bool,
    positions: list[str],
    counts: dict[str, int],
) -> dict:
    labels = {
        "has_note": has_note,
        "note_count": 1 if has_note else 0,
        "tap_count": 0,
        "break_count": 0,
        "hold_start_count": 0,
        "hold_active_count": 0,
        "slide_start_count": 0,
        "slide_active_count": 0,
        "touch_count": 0,
        "touch_hold_start_count": 0,
        "positions": positions,
    }
    labels.update(counts)
    return {
        "frame_index": index,
        "beat": str(index),
        "tick": index * 480,
        "time_sec": time_sec,
        "labels": labels,
    }


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()
