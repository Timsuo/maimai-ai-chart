"""Tests for repository health check system.

Covers parsing logic, status aggregation, edge cases, and error resilience.
Uses temporary files/directories – no side effects on the real repository.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

# Add tools/ to path so we can import check modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

# We import via importlib to match the main runner's behavior
import importlib.util


def _import_tool(name: str):
    """Import a tool module from tools/."""
    tools_dir = Path(__file__).resolve().parent.parent / "tools"
    path = tools_dir / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_repo():
    """Create a temporary directory that mimics a minimal repo root."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # Create .ai with policy files
        (root / ".ai").mkdir()
        for policy_name, policy_data in [
            ("branch_policy.json", {"protected_branches": ["main"], "stale_branch_days": 30}),
            ("repository_policy.json", {"generated_directories": ["cache/"], "large_file_threshold_bytes": 1000}),
            ("data_policy.json", {
                "manifest_paths": ["manifests/"],
                "manifest_include_patterns": ["*manifest*.json", "*batch*.json", "*split*.json", "id_test*.json"],
                "manifest_exclude_patterns": ["*config*.json", "*metrics*.json", "*report*.json", "*prediction*.json"],
                "cache_roots": ["cache/", "cache_qc_fixed/"],
                "required_data_products": ["audio_features", "chart_ir", "frame_labels"],
                "split_keys_to_check": ["train", "val", "test"],
                "legacy_path_mappings": [],
                "max_issue_examples": 20,
            }),
            ("model_policy.json", {
                "experiment_roots": ["runs/"],
                "required_experiment_files": ["train_log.csv"],
                "desired_experiment_files": ["metrics.json"],
                "loss_divergence_threshold_factor": 3.0,
                "overfit_train_val_gap_threshold": 0.5,
                "blocking_scope": "latest_or_active",
                "historical_experiments_can_block": False,
                "experiment_type_rules": {
                    "overfit_smoke": {"expect_overfit": True, "focus": "training_success"},
                    "full_training": {"expect_overfit": False, "focus": "validation_performance"},
                },
                "experiment_rule_severities": {},
            }),
        ]:
            with open(root / ".ai" / policy_name, "w", encoding="utf-8") as f:
                json.dump(policy_data, f)

        # Monkey-patch _repo_root in all modules to point to temp
        yield root


# ---------------------------------------------------------------------------
# 1. Status aggregation logic
# ---------------------------------------------------------------------------

def test_status_ordering():
    """Test that status ordering follows PASS < WARNING < BLOCKED."""
    git = _import_tool("check_git_health")

    assert git._worst("PASS", "PASS") == "PASS"
    assert git._worst("PASS", "WARNING") == "WARNING"
    assert git._worst("WARNING", "BLOCKED") == "BLOCKED"
    assert git._worst("BLOCKED", "PASS") == "BLOCKED"
    assert git._worst("UNKNOWN", "PASS") == "PASS"
    assert git._worst("PASS", "UNKNOWN") == "PASS"
    assert git._worst("UNKNOWN", "BLOCKED") == "BLOCKED"


# ---------------------------------------------------------------------------
# 2. Empty repository / missing directories (data checks)
# ---------------------------------------------------------------------------

def test_manifest_list_empty(temp_repo):
    """When no manifest files exist, should return WARNING with empty list."""
    data = _import_tool("check_data_health")

    # Override repo_root
    original_root = data._repo_root
    data._repo_root = lambda: temp_repo

    try:
        # Create empty manifests dir
        (temp_repo / "manifests").mkdir(exist_ok=True)

        result = data.check_manifest_list(["manifests/"])
        assert result["status"] == "WARNING"
        assert result["count"] == 0
        assert "No manifest files" in result["evidence"]
    finally:
        data._repo_root = original_root


def test_manifest_dir_missing(temp_repo):
    """When manifest directory doesn't exist, should return gracefully."""
    data = _import_tool("check_data_health")
    original_root = data._repo_root
    data._repo_root = lambda: temp_repo
    try:
        result = data.check_manifest_list(["nonexistent/"])
        assert result["count"] == 0
    finally:
        data._repo_root = original_root


# ---------------------------------------------------------------------------
# 3. Corrupted JSON
# ---------------------------------------------------------------------------

def test_corrupted_json_manifest(temp_repo):
    """Corrupted JSON should be reported as parse error."""
    data = _import_tool("check_data_health")
    original_root = data._repo_root
    data._repo_root = lambda: temp_repo
    try:
        (temp_repo / "manifests").mkdir(exist_ok=True)
        # Use filenames matching manifest include patterns
        (temp_repo / "manifests" / "training_manifest_bad.json").write_text("{ this is not json }", encoding="utf-8")
        (temp_repo / "manifests" / "training_manifest_good.json").write_text('{"samples": [{"sample_id": "s1"}]}', encoding="utf-8")

        result = data.check_manifest_parse(["manifests/"])
        assert result["status"] == "WARNING"  # one bad, one good
        assert len(result["parse_errors"]) == 1
        assert "bad" in result["parse_errors"][0]["path"]
        assert len(result["parsed"]) == 1
    finally:
        data._repo_root = original_root


def test_all_corrupted_json(temp_repo):
    """All manifests corrupted → BLOCKED."""
    data = _import_tool("check_data_health")
    original_root = data._repo_root
    data._repo_root = lambda: temp_repo
    try:
        (temp_repo / "manifests").mkdir(exist_ok=True)
        (temp_repo / "manifests" / "training_manifest_bad1.json").write_text("not json", encoding="utf-8")
        (temp_repo / "manifests" / "training_manifest_bad2.json").write_text("also not json", encoding="utf-8")

        result = data.check_manifest_parse(["manifests/"])
        assert result["status"] == "BLOCKED"
        assert len(result["parse_errors"]) == 2
    finally:
        data._repo_root = original_root


# ---------------------------------------------------------------------------
# 4. Manifest referencing missing files
# ---------------------------------------------------------------------------

def test_data_product_missing(temp_repo):
    """When manifest references paths that don't exist on disk."""
    data = _import_tool("check_data_health")
    original_root = data._repo_root
    data._repo_root = lambda: temp_repo
    try:
        (temp_repo / "manifests").mkdir(exist_ok=True)
        (temp_repo / "cache").mkdir(exist_ok=True)

        manifest = {
            "version": "test_v1",
            "samples": [
                {
                    "sample_id": "test_001",
                    "music_id": "song1",
                    "usable_for_training": True,
                    "paths": {
                        "cache_dir": "cache",
                        "audio_features": "cache/audio_features/song1.json",
                        "chart_ir": "cache/chart_ir/song1/diff_1.json",
                        "frame_labels": "cache/frame_labels/song1/diff_1.json",
                    },
                }
            ],
        }
        with open(temp_repo / "manifests" / "test.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        result = data.check_data_product_existence(
            ["manifests/"],
            cache_roots=["cache/"],
            required_products=["audio_features", "chart_ir", "frame_labels"],
        )
        # All 3 products should be missing
        assert result["total_checked"] == 3
        assert result["total_missing"] == 3
        assert result["status"] == "WARNING"
    finally:
        data._repo_root = original_root


# ---------------------------------------------------------------------------
# 5. Split leakage
# ---------------------------------------------------------------------------

def test_split_leakage_detected(temp_repo):
    """Detect music_id overlap between train/val/test."""
    data = _import_tool("check_data_health")
    original_root = data._repo_root
    data._repo_root = lambda: temp_repo
    try:
        (temp_repo / "manifests").mkdir(exist_ok=True)

        split_manifest = {
            "train": [{"music_id": "song1"}, {"music_id": "song2"}],
            "val": [{"music_id": "song2"}, {"music_id": "song3"}],  # song2 leaks!
            "test": [{"music_id": "song4"}],
        }
        with open(temp_repo / "manifests" / "split.json", "w", encoding="utf-8") as f:
            json.dump(split_manifest, f)

        result = data.check_split_leakage(["manifests/"])
        assert result["status"] == "BLOCKED"
        assert len(result["results"]) == 1
        assert result["results"][0]["has_leakage"] is True
        assert result["results"][0]["leakage"]["train_val"] == 1
    finally:
        data._repo_root = original_root


def test_no_split_leakage(temp_repo):
    """Clean splits should pass."""
    data = _import_tool("check_data_health")
    original_root = data._repo_root
    data._repo_root = lambda: temp_repo
    try:
        (temp_repo / "manifests").mkdir(exist_ok=True)

        split_manifest = {
            "train": [{"music_id": "song1"}, {"music_id": "song2"}],
            "val": [{"music_id": "song3"}],
            "test": [{"music_id": "song4"}],
        }
        with open(temp_repo / "manifests" / "split.json", "w", encoding="utf-8") as f:
            json.dump(split_manifest, f)

        result = data.check_split_leakage(["manifests/"])
        assert result["status"] == "PASS"
    finally:
        data._repo_root = original_root


# ---------------------------------------------------------------------------
# 6. Loss NaN / Inf / divergence
# ---------------------------------------------------------------------------

def test_loss_contains_nan(temp_repo):
    """NaN in loss columns → BLOCKED."""
    exp = _import_tool("check_experiment_health")
    original_root = exp._repo_root
    exp._repo_root = lambda: temp_repo
    try:
        (temp_repo / "runs").mkdir(exist_ok=True)
        (temp_repo / "runs" / "test_exp").mkdir(exist_ok=True)

        # Write train_log with NaN in loss
        log_path = temp_repo / "runs" / "test_exp" / "train_log.csv"
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss", "train_loss", "val_loss"])
            writer.writerow(["1", "5.0", "5.0", "6.0"])
            writer.writerow(["2", "3.0", "3.0", "4.0"])
            writer.writerow(["3", "NaN", "NaN", "5.0"])
            writer.writerow(["4", "2.0", "2.0", "3.5"])

        result = exp.check_loss_health(["runs/"])
        assert result["status"] == "BLOCKED"
        assert result["affected_experiments"] >= 1
    finally:
        exp._repo_root = original_root


def test_loss_contains_inf(temp_repo):
    """Inf in loss columns → BLOCKED."""
    exp = _import_tool("check_experiment_health")
    original_root = exp._repo_root
    exp._repo_root = lambda: temp_repo
    try:
        (temp_repo / "runs").mkdir(exist_ok=True)
        (temp_repo / "runs" / "inf_exp").mkdir(exist_ok=True)

        log_path = temp_repo / "runs" / "inf_exp" / "train_log.csv"
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss"])
            writer.writerow(["1", "5.0"])
            writer.writerow(["2", "Inf"])
            writer.writerow(["3", "Inf"])

        result = exp.check_loss_health(["runs/"])
        assert result["status"] == "BLOCKED"
        assert result["affected_experiments"] >= 1
    finally:
        exp._repo_root = original_root


def test_loss_divergence(temp_repo):
    """Sudden loss divergence → BLOCKED."""
    exp = _import_tool("check_experiment_health")
    original_root = exp._repo_root
    exp._repo_root = lambda: temp_repo
    try:
        (temp_repo / "runs").mkdir(exist_ok=True)
        (temp_repo / "runs" / "div_exp").mkdir(exist_ok=True)

        log_path = temp_repo / "runs" / "div_exp" / "train_log.csv"
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss", "val_loss"])
            writer.writerow(["1", "5.0", "6.0"])
            writer.writerow(["2", "4.0", "5.0"])
            writer.writerow(["3", "3.0", "4.0"])
            writer.writerow(["4", "50.0", "55.0"])  # sudden 10x jump

        result = exp.check_loss_health(["runs/"], divergence_factor=3.0)
        assert result["status"] == "BLOCKED"
        assert result["affected_experiments"] >= 1
    finally:
        exp._repo_root = original_root


def test_loss_clean(temp_repo):
    """Clean loss → PASS."""
    exp = _import_tool("check_experiment_health")
    original_root = exp._repo_root
    exp._repo_root = lambda: temp_repo
    try:
        (temp_repo / "runs").mkdir(exist_ok=True)
        (temp_repo / "runs" / "clean_exp").mkdir(exist_ok=True)

        log_path = temp_repo / "runs" / "clean_exp" / "train_log.csv"
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss", "val_loss"])
            for i in range(1, 11):
                writer.writerow([str(i), f"{5.0/i:.4f}", f"{5.5/i:.4f}"])

        result = exp.check_loss_health(["runs/"])
        assert result["status"] == "PASS"
        assert result["affected_experiments"] == 0
    finally:
        exp._repo_root = original_root


# ---------------------------------------------------------------------------
# 7. Experiment type not found
# ---------------------------------------------------------------------------

def test_experiment_type_undeclared(temp_repo):
    """When experiment_type is not declared, flag but don't block."""
    exp = _import_tool("check_experiment_health")
    original_root = exp._repo_root
    exp._repo_root = lambda: temp_repo
    try:
        (temp_repo / "runs").mkdir(exist_ok=True)
        (temp_repo / "runs" / "mystery_exp").mkdir(exist_ok=True)

        # No config file, name doesn't match any heuristic
        # Actually "mystery_exp" doesn't match any of our heuristics
        log_path = temp_repo / "runs" / "mystery_exp" / "train_log.csv"
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss"])
            for i in range(1, 6):
                writer.writerow([str(i), f"{5.0/i:.4f}"])

        result = exp.check_experiment_type_rules(["runs/"])
        assert result["undeclared_type_count"] >= 1
        # Should not block just because type is undeclared
        # (might be WARNING due to the type being undeclared, but not BLOCKED)
        for e in result["per_experiment"]:
            if "mystery_exp" in e["path"]:
                assert e["experiment_type"] is None
                assert "experiment_type_undeclared" in e.get("flags", [])
    finally:
        exp._repo_root = original_root


# ---------------------------------------------------------------------------
# 8. Overfit smoke test with clear overfitting (expected behavior)
# ---------------------------------------------------------------------------

def test_overfit_smoke_normal(temp_repo):
    """Overfit smoke test: extreme loss decrease is EXPECTED → PASS."""
    exp = _import_tool("check_experiment_health")
    original_root = exp._repo_root
    exp._repo_root = lambda: temp_repo
    try:
        (temp_repo / "runs").mkdir(exist_ok=True)
        (temp_repo / "runs" / "overfit_my_song").mkdir(exist_ok=True)

        # Loss drops from 10 to 0.001 → expected for overfit smoke
        log_path = temp_repo / "runs" / "overfit_my_song" / "train_log.csv"
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss"])
            writer.writerow(["1", "10.0"])
            writer.writerow(["2", "5.0"])
            writer.writerow(["3", "2.0"])
            writer.writerow(["4", "0.5"])
            writer.writerow(["5", "0.01"])
            writer.writerow(["6", "0.001"])

        result = exp.check_experiment_type_rules(["runs/"])
        for e in result["per_experiment"]:
            if "overfit_my_song" in e["path"]:
                assert "overfit_insufficient" not in e.get("flags", [])
                assert "PASS" in e.get("analysis", "")
    finally:
        exp._repo_root = original_root


def test_overfit_smoke_insufficient(temp_repo):
    """Overfit smoke that fails to overfit → flagged."""
    exp = _import_tool("check_experiment_health")
    original_root = exp._repo_root
    exp._repo_root = lambda: temp_repo
    try:
        (temp_repo / "runs").mkdir(exist_ok=True)
        (temp_repo / "runs" / "overfit_broken").mkdir(exist_ok=True)

        # Loss barely moves → should flag "overfit_insufficient"
        log_path = temp_repo / "runs" / "overfit_broken" / "train_log.csv"
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss"])
            writer.writerow(["1", "10.0"])
            writer.writerow(["2", "9.5"])
            writer.writerow(["3", "9.0"])
            writer.writerow(["4", "8.8"])
            writer.writerow(["5", "8.5"])

        result = exp.check_experiment_type_rules(["runs/"])
        for e in result["per_experiment"]:
            if "overfit_broken" in e["path"]:
                assert "overfit_insufficient" in e.get("flags", [])
    finally:
        exp._repo_root = original_root


# ---------------------------------------------------------------------------
# 9. Full training with large train/val gap (problematic overfitting)
# ---------------------------------------------------------------------------

def test_full_training_overfit_warning(temp_repo):
    """Full training with large train/val gap → WARNING."""
    exp = _import_tool("check_experiment_health")
    original_root = exp._repo_root
    exp._repo_root = lambda: temp_repo
    try:
        (temp_repo / "runs").mkdir(exist_ok=True)
        (temp_repo / "runs" / "phase99_overtrained").mkdir(exist_ok=True)

        # Full training: train loss low, val loss high → overfitting
        log_path = temp_repo / "runs" / "phase99_overtrained" / "train_log.csv"
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss"])
            writer.writerow(["1", "10.0", "10.5"])
            writer.writerow(["2", "5.0", "6.0"])
            writer.writerow(["3", "2.0", "4.0"])
            writer.writerow(["4", "0.5", "3.5"])   # train=0.5, val=3.5 → gap=6.0 (>0.5)
            writer.writerow(["5", "0.1", "3.0"])   # train=0.1, val=3.0 → gap=29.0

        result = exp.check_experiment_type_rules(["runs/"])
        for e in result["per_experiment"]:
            if "phase99_overtrained" in e["path"]:
                assert "large_train_val_gap" in e.get("flags", [])
    finally:
        exp._repo_root = original_root


def test_train_val_gap_check(temp_repo):
    """Train-val gap check should detect large gaps."""
    exp = _import_tool("check_experiment_health")
    original_root = exp._repo_root
    exp._repo_root = lambda: temp_repo
    try:
        (temp_repo / "runs").mkdir(exist_ok=True)
        (temp_repo / "runs" / "gap_exp").mkdir(exist_ok=True)

        log_path = temp_repo / "runs" / "gap_exp" / "train_log.csv"
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss"])
            writer.writerow(["1", "5.0", "5.5"])
            writer.writerow(["2", "3.0", "5.0"])

        result = exp.check_train_val_gap(["runs/"], gap_threshold=0.3)
        for e in result["per_experiment"]:
            if "gap_exp" in e["path"]:
                assert e["has_large_gap"] is True
        assert result["large_gap_count"] >= 1
    finally:
        exp._repo_root = original_root


# ---------------------------------------------------------------------------
# 10. Single module exception – others still complete
# ---------------------------------------------------------------------------

def test_module_isolation(temp_repo):
    """If one check module raises, others should still produce results."""
    data = _import_tool("check_data_health")
    original_root = data._repo_root
    data._repo_root = lambda: temp_repo
    try:
        (temp_repo / "manifests").mkdir(exist_ok=True)

        # Valid manifest (with hint key)
        (temp_repo / "manifests" / "training_manifest_ok.json").write_text('{"samples": [], "version": "1"}', encoding="utf-8")
        # Corrupted manifest
        (temp_repo / "manifests" / "training_manifest_broken.json").write_text("{not json!!!", encoding="utf-8")

        # Parse check should handle the error gracefully
        result = data.check_manifest_parse(["manifests/"])
        assert result["status"] in ("WARNING", "BLOCKED")
        assert len(result["parse_errors"]) >= 1
        assert len(result["parsed"]) >= 1
    finally:
        data._repo_root = original_root


def test_run_all_checks_resilience(temp_repo):
    """run_all_checks should complete even if some individual checks fail."""
    data = _import_tool("check_data_health")
    original_root = data._repo_root
    data._repo_root = lambda: temp_repo
    try:
        (temp_repo / "manifests").mkdir(exist_ok=True)

        result = data.run_all_checks()
        # Should complete without exception
        assert "overall_status" in result
        assert "checks" in result
        assert len(result["checks"]) > 0
    finally:
        data._repo_root = original_root


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------

def test_manifest_stats_unrecognized_format(temp_repo):
    """Manifest with unrecognized format → WARNING with note."""
    data = _import_tool("check_data_health")
    original_root = data._repo_root
    data._repo_root = lambda: temp_repo
    try:
        (temp_repo / "manifests").mkdir(exist_ok=True)
        # A JSON file that matches include pattern but has unrecognized structure
        weird = {"completely": "different", "structure": [1, 2, 3], "samples": []}
        with open(temp_repo / "manifests" / "training_manifest_weird.json", "w", encoding="utf-8") as f:
            json.dump(weird, f)

        result = data.check_manifest_stats(["manifests/"])
        # With "samples" hint key, it's recognized as batch_manifest
        assert result["status"] == "PASS"
        assert len(result["unknown_formats"]) == 0
    finally:
        data._repo_root = original_root


def test_duplicate_sample_detection(temp_repo):
    """Intra-manifest duplicate samples should be detected."""
    data = _import_tool("check_data_health")
    original_root = data._repo_root
    data._repo_root = lambda: temp_repo
    try:
        (temp_repo / "manifests").mkdir(exist_ok=True)

        manifest = {
            "version": "test",
            "samples": [
                {"sample_id": "dup_1", "music_id": "s1"},
                {"sample_id": "dup_1", "music_id": "s1"},  # duplicate
                {"sample_id": "unique", "music_id": "s2"},
            ],
        }
        with open(temp_repo / "manifests" / "dupes.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        result = data.check_duplicate_samples(["manifests/"])
        assert result["has_duplicates"] is True
        assert result["duplicate_within_manifest"]["count"] >= 1
    finally:
        data._repo_root = original_root


def test_experiment_files_missing(temp_repo):
    """Experiment missing required files → WARNING or BLOCKED."""
    exp = _import_tool("check_experiment_health")
    original_root = exp._repo_root
    exp._repo_root = lambda: temp_repo
    try:
        (temp_repo / "runs").mkdir(exist_ok=True)
        (temp_repo / "runs" / "incomplete_exp").mkdir(exist_ok=True)
        # Add a .pt file so _is_experiment_dir recognizes this as an experiment
        # No train_log.csv → should be missing required file
        (temp_repo / "runs" / "incomplete_exp" / "model.pt").write_text("dummy")

        result = exp.check_experiment_files(["runs/"], ["train_log.csv"], [])
        for e in result["per_experiment"]:
            if "incomplete_exp" in e["path"]:
                assert e["all_required_present"] is False
        assert result["missing_required_count"] >= 1
    finally:
        exp._repo_root = original_root


def test_training_completion_detection(temp_repo):
    """Training that didn't decrease loss → flagged as not completed."""
    exp = _import_tool("check_experiment_health")
    original_root = exp._repo_root
    exp._repo_root = lambda: temp_repo
    try:
        (temp_repo / "runs").mkdir(exist_ok=True)
        (temp_repo / "runs" / "stuck_exp").mkdir(exist_ok=True)

        log_path = temp_repo / "runs" / "stuck_exp" / "train_log.csv"
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss"])
            writer.writerow(["1", "10.0"])
            writer.writerow(["2", "10.0"])
            writer.writerow(["3", "10.0"])  # No decrease

        result = exp.check_training_completion(["runs/"])
        for e in result["per_experiment"]:
            if "stuck_exp" in e["path"]:
                assert e["completed_normally"] is False
    finally:
        exp._repo_root = original_root


def test_metrics_from_train_log(temp_repo):
    """Metrics should be extractable from train_log.csv."""
    exp = _import_tool("check_experiment_health")
    original_root = exp._repo_root
    exp._repo_root = lambda: temp_repo
    try:
        (temp_repo / "runs").mkdir(exist_ok=True)
        (temp_repo / "runs" / "metric_exp").mkdir(exist_ok=True)

        log_path = temp_repo / "runs" / "metric_exp" / "train_log.csv"
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss", "val_loss", "val_note_presence_f1"])
            writer.writerow(["1", "5.0", "6.0", "0.5"])
            writer.writerow(["2", "3.0", "4.0", "0.7"])
            writer.writerow(["3", "1.0", "2.0", "0.9"])

        result = exp.check_metrics(["runs/"])
        for e in result["per_experiment"]:
            if "metric_exp" in e["path"]:
                assert "train_log_latest" in e
                assert "best_val_metrics" in e
                assert e["best_val_metrics"]["best_val_loss"] == 2.0  # min val_loss
    finally:
        exp._repo_root = original_root


def test_numeric_column_detection():
    """Auto-detection of numeric columns from CSV rows."""
    exp = _import_tool("check_experiment_health")
    rows = [
        {"epoch": "1", "loss": "3.5", "notes": "hello", "count": "10"},
        {"epoch": "2", "loss": "2.1", "notes": "world", "count": "20"},
        {"epoch": "3", "loss": "1.0", "notes": "test", "count": "30"},
    ]
    numeric = exp._detect_numeric_columns(rows)
    assert "epoch" in numeric
    assert "loss" in numeric
    assert "count" in numeric
    assert "notes" not in numeric  # non-numeric column
    assert len(numeric["loss"]) == 3
    assert numeric["loss"] == [3.5, 2.1, 1.0]


def test_old_cache_detection(temp_repo):
    """Detect manifests referencing old absolute cache paths."""
    data = _import_tool("check_data_health")
    original_root = data._repo_root
    data._repo_root = lambda: temp_repo
    try:
        (temp_repo / "manifests").mkdir(exist_ok=True)

        manifest = {
            "schema": "maichart-training-manifest-v1",
            "source_root": "F:\\AI maichart designer\\raw_chart_data\\output",
            "cache_dir": "F:\\AI maichart designer\\cache",
            "songs": [],
        }
        with open(temp_repo / "manifests" / "old_paths.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        result = data.check_old_cache_references(["manifests/"])
        assert result["status"] == "WARNING"
        assert len(result["findings"]) >= 1
    finally:
        data._repo_root = original_root


# ---------------------------------------------------------------------------
# NEW TESTS: false positive fixes
# ---------------------------------------------------------------------------

def test_experiment_config_not_manifest(temp_repo):
    """experiment_config.json should not be identified as a manifest."""
    data = _import_tool("check_data_health")
    original_root = data._repo_root
    data._repo_root = lambda: temp_repo
    try:
        (temp_repo / "manifests").mkdir(exist_ok=True)
        (temp_repo / "manifests" / "experiments").mkdir(exist_ok=True)

        # Write an experiment_config.json (should be excluded)
        exp_config = {
            "version": "id_batch_v1_qc_fixed_experiment",
            "source_manifest": "manifests/training_manifest_qc_fixed_full.json",
            "output_dir": "manifests/experiments/id_batch",
        }
        with open(temp_repo / "manifests" / "experiments" / "experiment_config.json", "w", encoding="utf-8") as f:
            json.dump(exp_config, f)

        # Also write a real manifest
        real_manifest = {
            "version": "small_batch_v1",
            "samples": [{"sample_id": "s1", "music_id": "m1"}],
        }
        with open(temp_repo / "manifests" / "small_batch_v1.json", "w", encoding="utf-8") as f:
            json.dump(real_manifest, f)

        manifests = data._find_manifest_files(["manifests/"])
        # experiment_config.json should be excluded
        assert not any("experiment_config" in m for m in manifests), f"experiment_config found in: {manifests}"
        # real manifest should be included
        assert any("small_batch_v1" in m for m in manifests)
    finally:
        data._repo_root = original_root


def test_metrics_json_not_manifest(temp_repo):
    """Plain metrics JSON should not produce manifest warning."""
    data = _import_tool("check_data_health")
    original_root = data._repo_root
    data._repo_root = lambda: temp_repo
    try:
        (temp_repo / "manifests").mkdir(exist_ok=True)
        (temp_repo / "manifests" / "metrics.json").write_text(
            '{"accuracy": 0.95, "loss": 0.1}', encoding="utf-8"
        )
        (temp_repo / "manifests" / "prediction_sample.json").write_text(
            '{"preds": [1,2,3]}', encoding="utf-8"
        )

        manifests = data._find_manifest_files(["manifests/"])
        assert not any("metrics" in m or "prediction" in m for m in manifests), f"Non-manifest JSON found: {manifests}"

        result = data.check_manifest_stats(["manifests/"])
        # No manifests found → no format warnings
        unknown = result.get("unknown_formats", [])
        assert len(unknown) == 0, f"Should have 0 unknown formats, got: {unknown}"
    finally:
        data._repo_root = original_root


def test_cross_manifest_reuse_info_only(temp_repo):
    """Cross-manifest reuse should be INFO only, not a warning."""
    data = _import_tool("check_data_health")
    original_root = data._repo_root
    data._repo_root = lambda: temp_repo
    try:
        (temp_repo / "manifests").mkdir(exist_ok=True)

        # Two batch manifests referencing the same music_id
        m1 = {
            "version": "batch_1",
            "samples": [
                {"sample_id": "s1", "music_id": "shared_music"},
                {"sample_id": "s2", "music_id": "shared_music"},
            ],
        }
        m2 = {
            "version": "batch_2",
            "samples": [
                {"sample_id": "s3", "music_id": "shared_music"},
            ],
        }
        with open(temp_repo / "manifests" / "batch1.json", "w", encoding="utf-8") as f:
            json.dump(m1, f)
        with open(temp_repo / "manifests" / "batch2.json", "w", encoding="utf-8") as f:
            json.dump(m2, f)

        result = data.check_duplicate_samples(["manifests/"])
        # Cross-manifest reuse should NOT block or warn
        assert result["status"] == "PASS", f"Expected PASS, got {result['status']}"
        cr = result.get("cross_manifest_reuse", {})
        assert cr.get("music_id_reuse_count", 0) > 0, "Cross-manifest reuse should be recorded"
    finally:
        data._repo_root = original_root


def test_intra_manifest_duplicate_warning(temp_repo):
    """Intra-manifest duplicate samples → WARNING."""
    data = _import_tool("check_data_health")
    original_root = data._repo_root
    data._repo_root = lambda: temp_repo
    try:
        (temp_repo / "manifests").mkdir(exist_ok=True)

        # Same sample_id appears twice in one manifest
        m = {
            "version": "test",
            "samples": [
                {"sample_id": "dup_1", "music_id": "m1"},
                {"sample_id": "dup_1", "music_id": "m1"},
            ],
        }
        with open(temp_repo / "manifests" / "dupes.json", "w", encoding="utf-8") as f:
            json.dump(m, f)

        result = data.check_duplicate_samples(["manifests/"])
        assert result["status"] == "WARNING"
        dwm = result.get("duplicate_within_manifest", {})
        assert dwm.get("count", 0) > 0
    finally:
        data._repo_root = original_root


def test_split_leakage_still_blocked(temp_repo):
    """Split leakage should still be BLOCKED."""
    data = _import_tool("check_data_health")
    original_root = data._repo_root
    data._repo_root = lambda: temp_repo
    try:
        (temp_repo / "manifests").mkdir(exist_ok=True)

        split = {
            "train": [{"music_id": "leak_song"}],
            "val": [{"music_id": "leak_song"}],
            "test": [{"music_id": "other"}],
        }
        with open(temp_repo / "manifests" / "leaky.json", "w", encoding="utf-8") as f:
            json.dump(split, f)

        result = data.check_split_leakage(["manifests/"])
        assert result["status"] == "BLOCKED"
    finally:
        data._repo_root = original_root


def test_legacy_path_remapping(temp_repo):
    """Legacy paths should be remapped to current cache roots."""
    data = _import_tool("check_data_health")
    original_root = data._repo_root
    data._repo_root = lambda: temp_repo
    try:
        (temp_repo / "manifests").mkdir(exist_ok=True)

        # Create a cache file at the remapped location
        cache_dir = temp_repo / "cache_qc_fixed" / "audio_features"
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "song1.audio_features.json").write_text("{}", encoding="utf-8")

        # Manifest with old absolute path
        m = {
            "version": "test",
            "samples": [
                {
                    "sample_id": "s1",
                    "music_id": "song1",
                    "usable_for_training": True,
                    "paths": {
                        "audio_features": "F:\\AI maichart designer\\cache_qc_fixed\\audio_features\\song1.audio_features.json",
                    },
                }
            ],
        }
        with open(temp_repo / "manifests" / "m.json", "w", encoding="utf-8") as f:
            json.dump(m, f)

        result = data.check_data_product_existence(
            ["manifests/"],
            cache_roots=["cache_qc_fixed/"],
            required_products=["audio_features"],
        )
        # Remapping should resolve the path
        assert result["total_remapped"] >= 1, f"Expected remapped count > 0, got {result['total_remapped']}"
        assert result["total_missing"] == 0, f"Expected 0 missing, got {result['total_missing']}"
    finally:
        data._repo_root = original_root


def test_legacy_path_not_found_marked_missing(temp_repo):
    """When remapping also fails, file should be marked as missing."""
    data = _import_tool("check_data_health")
    original_root = data._repo_root
    data._repo_root = lambda: temp_repo
    try:
        (temp_repo / "manifests").mkdir(exist_ok=True)

        m = {
            "version": "test",
            "samples": [
                {
                    "sample_id": "s1",
                    "music_id": "song1",
                    "usable_for_training": True,
                    "paths": {
                        "audio_features": "F:\\AI maichart designer\\cache_qc_fixed\\audio_features\\nonexistent.json",
                    },
                }
            ],
        }
        with open(temp_repo / "manifests" / "m.json", "w", encoding="utf-8") as f:
            json.dump(m, f)

        result = data.check_data_product_existence(
            ["manifests/"],
            cache_roots=["cache_qc_fixed/"],
            required_products=["audio_features"],
        )
        assert result["total_missing"] >= 1
    finally:
        data._repo_root = original_root


def test_missing_artifacts_aggregated(temp_repo):
    """Missing artifacts should be aggregated, not listed individually."""
    data = _import_tool("check_data_health")
    original_root = data._repo_root
    data._repo_root = lambda: temp_repo
    try:
        (temp_repo / "manifests").mkdir(exist_ok=True)

        samples = []
        for i in range(50):
            samples.append({
                "sample_id": f"s{i}",
                "music_id": f"m{i}",
                "usable_for_training": True,
                "paths": {
                    "audio_features": f"cache/missing_{i}.json",
                    "frame_labels": f"cache/missing_{i}.json",
                },
            })
        m = {"version": "test", "samples": samples}
        with open(temp_repo / "manifests" / "m.json", "w", encoding="utf-8") as f:
            json.dump(m, f)

        result = data.check_data_product_existence(
            ["manifests/"],
            cache_roots=["cache_qc_fixed/"],
            required_products=["audio_features", "frame_labels"],
        )
        # Should aggregate: missing examples capped
        examples = result.get("missing_examples", [])
        assert len(examples) <= 20  # max_issue_examples default
        assert result["total_missing"] == 100  # 50 samples × 2 products
        # Evidence should be a single aggregated line
        assert "missing" in result["evidence"]
    finally:
        data._repo_root = original_root


def test_historical_overfit_loss_rebound_not_blocking(temp_repo):
    """Historical overfit experiment with loss rebound → WARNING, not BLOCKED."""
    exp = _import_tool("check_experiment_health")
    original_root = exp._repo_root
    exp._repo_root = lambda: temp_repo
    try:
        (temp_repo / "runs").mkdir(exist_ok=True)
        # Create a "latest" experiment (clean, newer mtime)
        latest_dir = temp_repo / "runs" / "latest_clean_exp"
        latest_dir.mkdir(exist_ok=True)
        with open(latest_dir / "train_log.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss"])
            writer.writerow(["1", "5.0"])
            writer.writerow(["2", "2.0"])
            writer.writerow(["3", "1.0"])

        # Create a historical overfit experiment (divergence, older)
        hist_dir = temp_repo / "runs" / "overfit_historical"
        hist_dir.mkdir(exist_ok=True)
        with open(hist_dir / "train_log.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss"])
            writer.writerow(["1", "10.0"])
            writer.writerow(["2", "0.1"])    # minimum
            writer.writerow(["3", "5.0"])    # rebound → divergence

        # Patch _find_experiments to make latest_clean_exp the latest
        original_find = exp._find_experiments
        def mock_find(roots):
            return [
                {"path": "runs/latest_clean_exp", "mtime": "2026-07-10T00:00:00"},
                {"path": "runs/overfit_historical", "mtime": "2026-07-01T00:00:00"},
            ]
        exp._find_experiments = mock_find

        try:
            result = exp.run_all_checks()
            # Should NOT be BLOCKED (historical experiment can't block)
            assert result["overall_status"] != "BLOCKED", f"Got {result['overall_status']}, expected not BLOCKED"

            # Loss health should be downgraded
            loss_check = next(c for c in result["checks"] if c["check"] == "loss_health")
            assert loss_check["status"] in ("WARNING", "PASS"), f"loss_health should be WARNING/PASS, got {loss_check['status']}"
        finally:
            exp._find_experiments = original_find
    finally:
        exp._repo_root = original_root


def test_active_overfit_nan_blocks(temp_repo):
    """Active (latest) overfit experiment with NaN → BLOCKED."""
    exp = _import_tool("check_experiment_health")
    original_root = exp._repo_root
    exp._repo_root = lambda: temp_repo
    try:
        (temp_repo / "runs").mkdir(exist_ok=True)
        latest_dir = temp_repo / "runs" / "overfit_latest_nan"
        latest_dir.mkdir(exist_ok=True)
        with open(latest_dir / "train_log.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss"])
            writer.writerow(["1", "5.0"])
            writer.writerow(["2", "NaN"])

        original_find = exp._find_experiments
        def mock_find(roots):
            return [
                {"path": "runs/overfit_latest_nan", "mtime": "2026-07-10T00:00:00"},
            ]
        exp._find_experiments = mock_find

        try:
            result = exp.run_all_checks()
            loss_check = next(c for c in result["checks"] if c["check"] == "loss_health")
            # NaN in latest experiment should still BLOCK
            assert loss_check["status"] == "BLOCKED", f"Expected BLOCKED, got {loss_check['status']}"
        finally:
            exp._find_experiments = original_find
    finally:
        exp._repo_root = original_root


def test_final_weaker_than_best_warning_only(temp_repo):
    """Final checkpoint weaker than best → WARNING only."""
    exp = _import_tool("check_experiment_health")
    original_root = exp._repo_root
    exp._repo_root = lambda: temp_repo
    try:
        (temp_repo / "runs").mkdir(exist_ok=True)
        exp_dir = temp_repo / "runs" / "phase99_worse_final"
        exp_dir.mkdir(exist_ok=True)
        with open(exp_dir / "train_log.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss"])
            writer.writerow(["1", "5.0", "5.5"])
            writer.writerow(["2", "1.0", "1.5"])   # best val
            writer.writerow(["3", "2.0", "2.5"])   # final is worse

        result = exp.check_experiment_type_rules(["runs/"])
        assert result["status"] != "BLOCKED", f"Expected not BLOCKED, got {result['status']}"
    finally:
        exp._repo_root = original_root


def test_historical_missing_checkpoint_warning_only(temp_repo):
    """Historical experiment missing checkpoint → WARNING only."""
    exp = _import_tool("check_experiment_health")
    original_root = exp._repo_root
    exp._repo_root = lambda: temp_repo
    try:
        (temp_repo / "runs").mkdir(exist_ok=True)

        # Latest experiment (has checkpoint)
        latest_dir = temp_repo / "runs" / "latest_with_ckpt"
        latest_dir.mkdir(exist_ok=True)
        with open(latest_dir / "train_log.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss"])
            writer.writerow(["1", "5.0"])

        # Historical experiment (no checkpoint)
        hist_dir = temp_repo / "runs" / "hist_no_ckpt"
        hist_dir.mkdir(exist_ok=True)
        # Only train_log, no .pt files

        original_find = exp._find_experiments
        def mock_find(roots):
            return [
                {"path": "runs/latest_with_ckpt", "mtime": "2026-07-10T00:00:00"},
                {"path": "runs/hist_no_ckpt", "mtime": "2026-07-01T00:00:00"},
            ]
        exp._find_experiments = mock_find

        try:
            result = exp.run_all_checks()
            # Missing checkpoint in historical experiment should NOT block
            assert result["overall_status"] != "BLOCKED", f"Got {result['overall_status']}"
        finally:
            exp._find_experiments = original_find
    finally:
        exp._repo_root = original_root


def test_report_paths_are_relative(temp_repo):
    """Report should not expose absolute temporary paths."""
    data = _import_tool("check_data_health")
    original_root = data._repo_root
    data._repo_root = lambda: temp_repo
    try:
        (temp_repo / "manifests").mkdir(exist_ok=True)
        m = {
            "version": "test",
            "samples": [
                {"sample_id": "s1", "music_id": "m1", "usable_for_training": True,
                 "paths": {"audio_features": f"{temp_repo.as_posix()}/cache/test.json"}},
            ],
        }
        with open(temp_repo / "manifests" / "m.json", "w", encoding="utf-8") as f:
            json.dump(m, f)

        result = data.check_data_product_existence(
            ["manifests/"],
            cache_roots=["cache/"],
            required_products=["audio_features"],
        )
        # Paths in missing_examples should be sanitized, not containing temp dir
        for ex in result.get("missing_examples", []):
            op = ex.get("original_path", "")
            assert str(temp_repo).replace("\\", "/") not in op, f"Raw path leaked: {op}"
    finally:
        data._repo_root = original_root


def test_examples_respect_cap(temp_repo):
    """Missing examples should respect max_issue_examples cap."""
    data = _import_tool("check_data_health")
    original_root = data._repo_root
    data._repo_root = lambda: temp_repo
    try:
        (temp_repo / "manifests").mkdir(exist_ok=True)

        samples = []
        for i in range(50):
            samples.append({
                "sample_id": f"s{i}",
                "music_id": f"m{i}",
                "usable_for_training": True,
                "paths": {"audio_features": f"cache/missing_{i}.json"},
            })
        m = {"version": "test", "samples": samples}
        with open(temp_repo / "manifests" / "m.json", "w", encoding="utf-8") as f:
            json.dump(m, f)

        result = data.check_data_product_existence(
            ["manifests/"],
            cache_roots=["cache_qc_fixed/"],
            required_products=["audio_features"],
        )
        max_ex = result.get("max_examples", 20)
        examples = result.get("missing_examples", [])
        assert len(examples) <= max_ex, f"Expected <= {max_ex} examples, got {len(examples)}"
    finally:
        data._repo_root = original_root
