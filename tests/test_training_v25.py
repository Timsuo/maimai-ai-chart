from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from maichart.models.transformer_v25 import MaichartTransformerV25
from maichart.training.collate import collate_v25
from maichart.training.dataset_v25 import MaichartV25Dataset
from maichart.training.evaluate_v25 import compute_v25_metrics
from maichart.training.losses_v25 import compute_v25_losses


def test_v25_minimal_dataset_collate_forward_loss_and_metrics(tmp_path) -> None:
    manifest = _write_minimal_v25_cache(tmp_path)
    dataset = MaichartV25Dataset(manifest, cache_dir=tmp_path / "cache")

    sample = dataset[0]
    assert sample["meta"]["song_id"] == "song001"
    assert sample["meta"]["difficulty_index"] == 4
    assert sample["x"].shape == (4, dataset.input_dim)
    assert sample["y"]["note_presence"].shape == (4, 1)
    assert sample["y"]["buttons"].shape == (4, 8)

    batch = collate_v25([sample])
    assert batch["x"].shape == (1, 4, dataset.input_dim)
    assert batch["padding_mask"].shape == (1, 4)
    assert batch["loss_mask"].all()

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
    assert outputs["note_presence_logits"].shape == (1, 4, 1)
    assert outputs["button_logits"].shape == (1, 4, 8)

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
    }
    assert all(isinstance(value, float) for value in metrics.values())


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
