from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_visualize_run():
    path = Path(__file__).resolve().parents[1] / "tools" / "visualize_run.py"
    spec = importlib.util.spec_from_file_location("visualize_run", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_summary_computes_losses_and_worst_samples() -> None:
    visualize_run = _load_visualize_run()

    summary = visualize_run.build_summary(
        train_rows=[
            {"epoch": 1, "loss": 10.0},
            {"epoch": 2, "loss": 7.5},
            {"epoch": 3, "loss": 5.0},
        ],
        test_rows=[
            {
                "sample_index": 0,
                "song_id": "song_a",
                "difficulty_index": 4,
                "note_presence_f1": 0.5,
                "button_micro_f1": 0.2,
                "button_best_f1": 0.3,
                "note_type_accuracy": 0.6,
                "density_mae": 0.7,
            },
            {
                "sample_index": 1,
                "song_id": "song_b",
                "difficulty_index": 5,
                "note_presence_f1": 0.9,
                "button_micro_f1": 0.4,
                "button_best_f1": 0.1,
                "note_type_accuracy": 0.8,
                "density_mae": 1.2,
            },
        ],
        warnings=["same", "same"],
        generated_files=["01_training_total_loss.png"],
        run_dir=Path("runs/example"),
        manifest=Path("manifest.json"),
        cache_dir=Path("cache"),
    )

    assert summary["best_epoch"] == 3
    assert summary["best_train_loss"] == 5.0
    assert summary["final_train_loss"] == 5.0
    assert summary["total_loss_reduction_percent"] == pytest.approx(50.0)
    assert summary["mean_note_presence_f1"] == pytest.approx(0.7)
    assert summary["mean_density_mae"] == pytest.approx(0.95)
    assert summary["worst_samples_by_button_f1"][0]["sample_index"] == 1
    assert summary["worst_samples_by_density_mae"][0]["sample_index"] == 1
    assert summary["warnings"] == ["same"]


def test_main_writes_stable_files_and_missing_data_warnings(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    visualize_run = _load_visualize_run()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "train_log.csv").write_text(
        "\n".join(
            [
                "epoch,loss,loss_note,loss_buttons,loss_type,loss_density",
                "1,10,3,3,3,1",
                "2,5,1,2,1,1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "test_metrics.csv").write_text(
        "\n".join(
            [
                "sample_index,song_id,difficulty_index,note_presence_f1,button_micro_f1,button_best_f1,note_type_accuracy,density_mae",
                "0,song_a,4,0.8,0.4,0.5,0.6,0.7",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "figures"
    assert visualize_run.main(
        [
            "--run-dir",
            str(run_dir),
            "--manifest",
            str(tmp_path / "missing_manifest.json"),
            "--cache-dir",
            str(tmp_path / "missing_cache"),
            "--output-dir",
            str(output_dir),
        ]
    ) == 0

    expected_pngs = [
        "01_training_total_loss.png",
        "02_training_loss_components.png",
        "03_relative_loss_reduction.png",
        "04_test_metrics_by_sample.png",
        "05_density_mae_by_sample.png",
        "06_metrics_by_difficulty.png",
    ]
    assert all((output_dir / name).is_file() for name in expected_pngs)
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["best_epoch"] == 2
    assert summary["total_loss_reduction_percent"] == pytest.approx(50.0)
    assert any("val_log.csv is missing" in warning for warning in summary["warnings"])
    assert any("fewer than 10 samples" in warning for warning in summary["warnings"])
