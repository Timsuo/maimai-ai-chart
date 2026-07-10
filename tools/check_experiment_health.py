"""Experiment and model health check module for repository health system.

Read-only: scans experiment directories, parses logs, metrics, and configs.
Never modifies experiment data.
"""
from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_policy(path: str) -> dict[str, Any]:
    policy_path = _repo_root() / ".ai" / path
    if not policy_path.exists():
        return {}
    with open(policy_path, encoding="utf-8") as fh:
        return json.load(fh)


STATUS_PASS = "PASS"
STATUS_WARNING = "WARNING"
STATUS_BLOCKED = "BLOCKED"
STATUS_UNKNOWN = "UNKNOWN"

def _worst(a: str, b: str) -> str:
    order = {STATUS_UNKNOWN: 0, STATUS_PASS: 1, STATUS_WARNING: 2, STATUS_BLOCKED: 3}
    return a if order.get(a, 0) >= order.get(b, 0) else b


# ---------------------------------------------------------------------------
# Experiment discovery
# ---------------------------------------------------------------------------

def _is_experiment_dir(path: Path) -> bool:
    """Heuristic: a directory is an experiment if it contains train_log.csv or checkpoints.

    Excludes directories named 'checkpoints' (which are checkpoint storage, not experiments)
    and 'eval' (which are evaluation result subdirectories).
    """
    if not path.is_dir():
        return False
    # Skip well-known subdirectories that are not experiments themselves
    if path.name in ("checkpoints", "eval", "__pycache__"):
        return False
    contents = os.listdir(path)
    has_train_log = "train_log.csv" in contents
    has_checkpoints = any(f.endswith(".pt") for f in contents)
    has_checkpoint_dir = os.path.isdir(path / "checkpoints")
    has_eval_dir = os.path.isdir(path / "eval")
    return has_train_log or has_checkpoints or has_checkpoint_dir or has_eval_dir


def _find_experiments(experiment_roots: list[str]) -> list[dict[str, Any]]:
    """Scan experiment root directories and discover experiments recursively."""
    root = _repo_root()
    experiments = []

    for exp_root in experiment_roots:
        exp_path = root / exp_root.rstrip("/")
        if not exp_path.exists():
            continue

        for entry in sorted(exp_path.rglob("*")):
            if not entry.is_dir():
                continue
            if _is_experiment_dir(entry):
                rel = str(entry.relative_to(root)).replace("\\", "/")
                try:
                    mtime = datetime.fromtimestamp(entry.stat().st_mtime).isoformat()
                except OSError:
                    mtime = None
                experiments.append({
                    "path": rel,
                    "mtime": mtime,
                })
            # Check eval subdirectory for metrics
            elif entry.name == "eval" and _is_experiment_dir(entry.parent):
                pass  # Already counted via parent

    # Sort by mtime descending
    experiments.sort(key=lambda x: x.get("mtime") or "", reverse=True)
    return experiments


def _read_csv_columns(csv_path: str) -> list[str] | None:
    """Read column headers from a CSV file."""
    try:
        with open(csv_path, encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            return next(reader, None)
    except Exception:
        return None


def _read_csv_dict(path: str, max_rows: int = 3000) -> list[dict[str, str]]:
    """Read CSV as list of dicts, limited to max_rows."""
    rows = []
    try:
        with open(path, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for i, row in enumerate(reader):
                if i >= max_rows:
                    break
                rows.append(row)
    except Exception:
        pass
    return rows


def _detect_numeric_columns(rows: list[dict[str, str]]) -> dict[str, list[float]]:
    """From a list of row dicts, detect which columns are numeric and return their values.

    Includes NaN and Inf in the returned lists so they can be detected by callers.
    A column is considered numeric if at least 50% of rows parse as float
    (including NaN/Inf) so that pathological columns like all-Inf are still caught.
    """
    if not rows:
        return {}
    numeric: dict[str, list[float]] = {}
    for col in rows[0].keys():
        values: list[float] = []
        parse_failures = 0
        for row in rows:
            try:
                v = float(row.get(col, ""))
                values.append(v)
            except (ValueError, TypeError):
                parse_failures += 1
        # Include column if at least 50% of rows parse (NaN/Inf count as parsed)
        if len(values) >= len(rows) * 0.5:
            numeric[col] = values
    return numeric


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------

def check_latest_experiment(experiment_roots: list[str]) -> dict[str, Any]:
    """Identify the latest experiment by file modification time."""
    experiments = _find_experiments(experiment_roots)
    if not experiments:
        return {
            "check": "latest_experiment",
            "status": STATUS_WARNING,
            "latest": None,
            "total_found": 0,
            "evidence": "No experiments found",
        }
    latest = experiments[0]
    return {
        "check": "latest_experiment",
        "status": STATUS_PASS,
        "latest": latest,
        "total_found": len(experiments),
        "all_experiments": experiments[:50],  # Limit to top 50
        "evidence": f"Latest: {latest['path']} (mtime: {latest['mtime']})",
    }


def check_experiment_files(
    experiment_roots: list[str],
    required_files: list[str] | None = None,
    desired_files: list[str] | None = None,
) -> dict[str, Any]:
    """Check which required and desired files exist in each experiment."""
    if required_files is None:
        required_files = ["train_log.csv"]
    if desired_files is None:
        desired_files = ["metrics.json", "eval_test.csv", "prediction_sample.json"]

    experiments = _find_experiments(experiment_roots)
    root = _repo_root()
    per_exp: list[dict[str, Any]] = []
    missing_required_count = 0

    for exp in experiments:
        exp_path = root / exp["path"]
        files_status: dict[str, bool] = {}
        for f in required_files:
            # Check in exp dir and eval subdir
            exists = (exp_path / f).exists() or (exp_path / "eval" / f).exists()
            files_status[f] = exists
        for f in desired_files:
            exists = (exp_path / f).exists() or (exp_path / "eval" / f).exists()
            files_status[f] = exists

        has_all_required = all(files_status.get(f, False) for f in required_files)
        if not has_all_required:
            missing_required_count += 1

        per_exp.append({
            "path": exp["path"],
            "files": files_status,
            "all_required_present": has_all_required,
        })

    status = STATUS_PASS
    if missing_required_count > 0:
        if missing_required_count == len(per_exp):
            status = STATUS_BLOCKED
        else:
            status = STATUS_WARNING

    return {
        "check": "experiment_files",
        "status": status,
        "required_files": list(required_files),
        "desired_files": list(desired_files),
        "per_experiment": per_exp,
        "missing_required_count": missing_required_count,
        "evidence": f"{missing_required_count}/{len(per_exp)} experiments missing required files",
    }


def check_checkpoints(experiment_roots: list[str]) -> dict[str, Any]:
    """Check for existence of model checkpoints in experiment directories."""
    experiments = _find_experiments(experiment_roots)
    root = _repo_root()
    per_exp: list[dict[str, Any]] = []
    no_checkpoint_count = 0

    for exp in experiments:
        exp_path = root / exp["path"]
        checkpoints = list(exp_path.glob("*.pt")) + list(exp_path.glob("*.pth"))
        checkpoint_dir = exp_path / "checkpoints"
        if checkpoint_dir.is_dir():
            checkpoints += list(checkpoint_dir.glob("*.pt")) + list(checkpoint_dir.glob("*.pth"))

        cp_names = [c.name for c in checkpoints]
        has_best = any(n.startswith("v25_best") or "best" in n.lower() for n in cp_names)
        has_last = any(n.startswith("v25_last") or "last" in n.lower() for n in cp_names)

        if not checkpoints:
            no_checkpoint_count += 1

        per_exp.append({
            "path": exp["path"],
            "checkpoint_count": len(checkpoints),
            "has_best": has_best,
            "has_last": has_last,
            "checkpoint_names": cp_names[:10],
        })

    status = STATUS_PASS
    if no_checkpoint_count > 0:
        if no_checkpoint_count == len(per_exp):
            status = STATUS_WARNING
        else:
            status = STATUS_WARNING

    return {
        "check": "checkpoints",
        "status": status,
        "per_experiment": per_exp,
        "no_checkpoint_count": no_checkpoint_count,
        "evidence": f"{no_checkpoint_count}/{len(per_exp)} experiments have no checkpoints",
    }


def check_training_completion(experiment_roots: list[str]) -> dict[str, Any]:
    """Check if training ran to completion based on train_log.csv."""
    experiments = _find_experiments(experiment_roots)
    root = _repo_root()
    per_exp: list[dict[str, Any]] = []

    for exp in experiments:
        exp_path = root / exp["path"]
        log_path = exp_path / "train_log.csv"
        result: dict[str, Any] = {
            "path": exp["path"],
            "has_log": False,
            "epochs_completed": 0,
            "completed_normally": None,
            "evidence": "",
        }

        if not log_path.exists():
            result["evidence"] = "No train_log.csv found"
            per_exp.append(result)
            continue

        result["has_log"] = True
        rows = _read_csv_dict(str(log_path), max_rows=10000)
        if not rows:
            result["evidence"] = "train_log.csv empty or unreadable"
            per_exp.append(result)
            continue

        result["epochs_completed"] = len(rows)

        # Check if the last row has decreasing loss (still training normally)
        # or if loss has plateaued/diverged
        last_row = rows[-1]
        first_row = rows[0]
        try:
            last_loss = float(last_row.get("loss", last_row.get("train_loss", "0")))
            first_loss = float(first_row.get("loss", first_row.get("train_loss", "9")))
        except (ValueError, TypeError):
            last_loss = 0
            first_loss = 9

        # Heuristic: training completed normally if loss decreased significantly
        # from start and the final rows are stable
        if last_loss < first_loss * 0.9:
            result["completed_normally"] = True
            result["evidence"] = f"Loss decreased from {first_loss:.2f} to {last_loss:.2f} over {len(rows)} epochs"
        else:
            result["completed_normally"] = False
            result["evidence"] = f"Loss did not decrease significantly: {first_loss:.2f} -> {last_loss:.2f}"

        per_exp.append(result)

    not_completed = [e for e in per_exp if e.get("completed_normally") is False]
    status = STATUS_WARNING if not_completed else STATUS_PASS

    return {
        "check": "training_completion",
        "status": status,
        "per_experiment": per_exp,
        "not_completed_count": len(not_completed),
        "evidence": f"{len(not_completed)} experiments may not have completed normally",
    }


def check_loss_health(
    experiment_roots: list[str],
    divergence_factor: float = 3.0,
) -> dict[str, Any]:
    """Check for NaN, Inf, or sudden divergence in loss values."""
    experiments = _find_experiments(experiment_roots)
    root = _repo_root()
    per_exp: list[dict[str, Any]] = []
    any_bad = False

    for exp in experiments:
        log_path = root / exp["path"] / "train_log.csv"
        result: dict[str, Any] = {
            "path": exp["path"],
            "has_nan": False,
            "has_inf": False,
            "has_divergence": False,
            "nan_columns": [],
            "inf_columns": [],
            "divergence_details": [],
            "numeric_columns_checked": [],
        }

        if not log_path.exists():
            per_exp.append(result)
            continue

        rows = _read_csv_dict(str(log_path), max_rows=10000)
        if not rows:
            per_exp.append(result)
            continue

        numeric = _detect_numeric_columns(rows)
        if not numeric:
            per_exp.append(result)
            continue

        result["numeric_columns_checked"] = list(numeric.keys())

        for col, values in numeric.items():
            # Skip epoch-like columns
            if col.lower() == "epoch":
                continue

            # Check NaN
            nan_indices = [i for i, v in enumerate(values) if math.isnan(v)]
            if nan_indices:
                result["has_nan"] = True
                result["nan_columns"].append(col)

            # Check Inf
            inf_indices = [i for i, v in enumerate(values) if math.isinf(v)]
            if inf_indices:
                result["has_inf"] = True
                result["inf_columns"].append(col)

            # Check divergence: loss suddenly jumps UP after decreasing.
            # Normal training starts high and decreases – that is NOT divergence.
            # Divergence = loss was low, then unexpectedly spikes.
            if len(values) >= 3 and col.lower().startswith(("loss", "train_loss", "val_loss")):
                # Use only finite values for min detection
                finite_vals = [v for v in values if math.isfinite(v)]
                if finite_vals:
                    min_val = min(finite_vals)
                    if min_val > 0:
                        # Find the first index where min was reached
                        try:
                            min_idx = values.index(min_val)
                        except ValueError:
                            min_idx = 0
                        # Only check epochs AFTER the minimum was first reached
                        for i in range(min_idx + 1, len(values)):
                            if math.isfinite(values[i]) and values[i] > divergence_factor * min_val:
                                result["has_divergence"] = True
                                factor = values[i] / min_val
                                result["divergence_details"].append({
                                    "column": col,
                                    "row_index": i,
                                    "value": values[i],
                                    "min_observed": min_val,
                                    "factor": factor,
                                })
                                break

        if result["has_nan"] or result["has_inf"] or result["has_divergence"]:
            any_bad = True

        per_exp.append(result)

    status = STATUS_BLOCKED if any_bad else STATUS_PASS
    bad_count = sum(1 for e in per_exp if e["has_nan"] or e["has_inf"] or e["has_divergence"])

    return {
        "check": "loss_health",
        "status": status,
        "per_experiment": per_exp,
        "affected_experiments": bad_count,
        "evidence": f"{bad_count} experiments have NaN/Inf/divergence issues",
    }


def check_metrics(experiment_roots: list[str]) -> dict[str, Any]:
    """Extract latest and best metrics from train_log.csv and metrics.json."""
    experiments = _find_experiments(experiment_roots)
    root = _repo_root()
    per_exp: list[dict[str, Any]] = []

    for exp in experiments:
        exp_path = root / exp["path"]
        result: dict[str, Any] = {
            "path": exp["path"],
            "latest_metrics": {},
            "best_metrics": {},
            "metrics_source": None,
        }

        # Try metrics.json first
        for loc in [exp_path / "eval" / "metrics.json", exp_path / "metrics.json"]:
            if loc.exists():
                try:
                    with open(loc, encoding="utf-8") as fh:
                        result["latest_metrics"] = json.load(fh)
                    result["metrics_source"] = str(loc.relative_to(root)).replace("\\", "/")
                except Exception:
                    pass
                break

        # Also extract from train_log.csv (last row)
        log_path = exp_path / "train_log.csv"
        if log_path.exists():
            rows = _read_csv_dict(str(log_path), max_rows=10000)
            if rows:
                # Last row = latest metrics
                try:
                    result["train_log_latest"] = {
                        k: float(v) for k, v in rows[-1].items()
                        if k != "epoch"
                    }
                except (ValueError, TypeError):
                    result["train_log_latest"] = rows[-1]

                # Find best val_loss row
                best_val_row = None
                best_val = float("inf")
                for row in rows:
                    for key in ["val_loss", "val_loss_total"]:
                        if key in row:
                            try:
                                v = float(row[key])
                                if v < best_val and math.isfinite(v):
                                    best_val = v
                                    best_val_row = row
                            except (ValueError, TypeError):
                                pass
                if best_val_row:
                    try:
                        result["best_val_metrics"] = {
                            "epoch": best_val_row.get("epoch"),
                            "best_val_loss": best_val,
                        }
                        # Add validation metric columns
                        for k, v in best_val_row.items():
                            if k.startswith("val_") and k not in ("val_loss", "val_loss_total"):
                                try:
                                    result["best_val_metrics"][k] = float(v)
                                except (ValueError, TypeError):
                                    pass
                    except Exception:
                        pass

        per_exp.append(result)

    return {
        "check": "metrics",
        "status": STATUS_PASS,
        "per_experiment": per_exp,
        "evidence": f"Metrics extracted for {len(per_exp)} experiments",
    }


def check_train_val_gap(experiment_roots: list[str], gap_threshold: float = 0.5) -> dict[str, Any]:
    """Check the gap between train and validation loss in the final epoch."""
    experiments = _find_experiments(experiment_roots)
    root = _repo_root()
    per_exp: list[dict[str, Any]] = []
    large_gap_count = 0

    for exp in experiments:
        log_path = root / exp["path"] / "train_log.csv"
        result: dict[str, Any] = {
            "path": exp["path"],
            "train_loss_final": None,
            "val_loss_final": None,
            "gap_ratio": None,
            "has_large_gap": False,
        }

        if not log_path.exists():
            per_exp.append(result)
            continue

        rows = _read_csv_dict(str(log_path), max_rows=10000)
        if not rows:
            per_exp.append(result)
            continue

        last = rows[-1]

        # Detect train loss column
        for col in ["train_loss", "train_loss_total", "loss"]:
            if col in last:
                try:
                    result["train_loss_final"] = float(last[col])
                    break
                except (ValueError, TypeError):
                    pass

        # Detect val loss column
        for col in ["val_loss", "val_loss_total"]:
            if col in last:
                try:
                    result["val_loss_final"] = float(last[col])
                    break
                except (ValueError, TypeError):
                    pass

        if result["train_loss_final"] is not None and result["val_loss_final"] is not None:
            train = result["train_loss_final"]
            val = result["val_loss_final"]
            if train > 0:
                gap = abs(val - train) / train
                result["gap_ratio"] = gap
                if gap > gap_threshold:
                    result["has_large_gap"] = True
                    large_gap_count += 1

        per_exp.append(result)

    status = STATUS_WARNING if large_gap_count > 0 else STATUS_PASS

    return {
        "check": "train_val_gap",
        "status": status,
        "gap_threshold": gap_threshold,
        "per_experiment": per_exp,
        "large_gap_count": large_gap_count,
        "evidence": f"{large_gap_count} experiments have large train/val gap (> {gap_threshold})",
    }


def check_experiment_type_rules(experiment_roots: list[str]) -> dict[str, Any]:
    """Apply experiment-type-specific rules based on declared experiment_type.

    Detects experiment_type from:
    1. experiment_config.json in experiment directory
    2. experiment_config.json in parent directory (for nested experiments)
    3. From split manifest names
    4. Name-based heuristics (as last resort, with caveat)

    Uses per-experiment-type severity rules from policy to determine whether
    each flag is BLOCKING, WARNING, or INFO. Historical (non-latest) experiments
    default to WARNING even for BLOCKED-severity flags, unless the policy
    explicitly allows historical experiments to block.
    """
    experiments = _find_experiments(experiment_roots)
    root = _repo_root()
    policy = _load_policy("model_policy.json")
    type_rules = policy.get("experiment_type_rules", {})
    severity_rules = policy.get("experiment_rule_severities", {})
    historical_can_block = policy.get("historical_experiments_can_block", False)
    per_exp: list[dict[str, Any]] = []

    # Determine which experiments are "latest" (first in mtime-sorted list)
    latest_paths: set[str] = set()
    if experiments:
        latest_paths.add(experiments[0]["path"])

    for exp in experiments:
        exp_path = root / exp["path"]
        result: dict[str, Any] = {
            "path": exp["path"],
            "experiment_type": None,
            "type_source": None,
            "analysis": "",
            "flags": [],
        }

        # Try to find experiment_type from configs
        experiment_type = None
        type_source = None

        # 1. Check for experiment_config.json in experiment dir
        for cfg_name in ["experiment_config.json", "config.json", "train_config.json"]:
            cfg_path = exp_path / cfg_name
            if cfg_path.exists():
                try:
                    with open(cfg_path, encoding="utf-8") as fh:
                        cfg = json.load(fh)
                    experiment_type = cfg.get("experiment_type") or cfg.get("type")
                    if experiment_type:
                        type_source = str(cfg_path.relative_to(root)).replace("\\", "/")
                        break
                except Exception:
                    pass

        # 2. Check parent directory (for nested experiments like phase26_quick/baseline_audio7)
        if experiment_type is None:
            parent = exp_path.parent
            for cfg_name in ["experiment_config.json", "config.json"]:
                cfg_path = parent / cfg_name
                if cfg_path.exists():
                    try:
                        with open(cfg_path, encoding="utf-8") as fh:
                            cfg = json.load(fh)
                        experiment_type = cfg.get("experiment_type") or cfg.get("type")
                        if experiment_type:
                            type_source = str(cfg_path.relative_to(root)).replace("\\", "/")
                            break
                    except Exception:
                        pass
                if experiment_type:
                    break

        # 3. Check grandparent (e.g., runs/phase26_quick/ -> manifests/experiments/...)
        if experiment_type is None:
            grandparent = exp_path.parent.parent if exp_path.parent != root else None
            if grandparent and grandparent != root:
                for cfg_name in ["experiment_config.json", "config.json"]:
                    cfg_path = grandparent / cfg_name
                    if cfg_path.exists():
                        try:
                            with open(cfg_path, encoding="utf-8") as fh:
                                cfg = json.load(fh)
                            experiment_type = cfg.get("experiment_type") or cfg.get("type")
                            if experiment_type:
                                type_source = str(cfg_path.relative_to(root)).replace("\\", "/")
                                break
                        except Exception:
                            pass
                    if experiment_type:
                        break

        # 4. Name-based heuristics (last resort)
        if experiment_type is None:
            name = exp_path.name.lower()
            if "overfit" in name:
                experiment_type = "overfit_smoke"
                type_source = f"name_heuristic: '{exp_path.name}' contains 'overfit'"
            elif "smoke" in name:
                experiment_type = "batch_smoke"
                type_source = f"name_heuristic: '{exp_path.name}' contains 'smoke'"
            elif "phase" in name or "full" in name:
                experiment_type = "full_training"
                type_source = f"name_heuristic: '{exp_path.name}' contains 'phase'/'full'"
            elif "ablation" in name or "aux" in name:
                experiment_type = "ablation"
                type_source = f"name_heuristic: '{exp_path.name}' contains 'ablation'/'aux'"
            elif "eval" in name and "train" not in name:
                experiment_type = "evaluation"
                type_source = f"name_heuristic: '{exp_path.name}' contains 'eval'"

        result["experiment_type"] = experiment_type
        result["type_source"] = type_source
        result["is_historical"] = exp["path"] not in latest_paths

        if experiment_type is None:
            result["analysis"] = "Experiment type not declared - cannot apply type-specific rules. Marking as UNKNOWN for type-specific checks."
            result["flags"].append("experiment_type_undeclared")
        elif experiment_type not in type_rules:
            result["analysis"] = f"Experiment type '{experiment_type}' has no defined rules. Treating with generic checks only."
            result["flags"].append(f"unknown_experiment_type: {experiment_type}")
        else:
            rules = type_rules[experiment_type]
            result["applied_rules"] = rules

            # Read train log for analysis
            log_path = exp_path / "train_log.csv"
            if log_path.exists():
                rows = _read_csv_dict(str(log_path), max_rows=10000)
                if rows:
                    first_row = rows[0]
                    last_row = rows[-1]

                    try:
                        first_loss = float(first_row.get("loss", first_row.get("train_loss", "9")))
                        last_loss = float(last_row.get("loss", last_row.get("train_loss", "9")))
                    except (ValueError, TypeError):
                        first_loss = 9
                        last_loss = 9

                    # Apply type-specific rules
                    if experiment_type == "overfit_smoke":
                        # Expect overfitting: loss should decrease dramatically
                        if last_loss < first_loss * 0.1:
                            result["analysis"] = "PASS: Overfit smoke test succeeded - loss decreased significantly (expected overfitting)."
                        elif last_loss < first_loss * 0.5:
                            result["analysis"] = "PASS: Overfit smoke test shows reasonable loss decrease."
                            result["flags"].append("moderate_overfit")
                        else:
                            result["analysis"] = "WARNING: Overfit smoke test - loss did not decrease as expected for overfitting."
                            result["flags"].append("overfit_insufficient")

                        # Check no NaN in overfit (critical)
                        numeric = _detect_numeric_columns(rows)
                        has_nan = any(
                            any(math.isnan(v) for v in vals)
                            for col, vals in numeric.items()
                            if col.lower() != "epoch"
                        )
                        if has_nan:
                            result["analysis"] += " BLOCKED: NaN detected in overfit training."
                            result["flags"].append("nan_in_overfit")

                    elif experiment_type == "full_training":
                        # Check for train/val gap (overfitting concern)
                        has_val = "val_loss" in rows[0] or "val_loss_total" in rows[0]
                        if has_val:
                            try:
                                val_last = float(last_row.get("val_loss", last_row.get("val_loss_total", "9")))
                                train_last = float(last_row.get("train_loss", last_row.get("train_loss_total", last_row.get("loss", "9"))))
                            except (ValueError, TypeError):
                                val_last = 9
                                train_last = 9

                            if train_last > 0:
                                gap = abs(val_last - train_last) / train_last
                                if gap > 0.5:
                                    result["analysis"] = "WARNING: Full training shows large train/val gap - possible overfitting."
                                    result["flags"].append("large_train_val_gap")
                                elif gap > 0.2:
                                    result["analysis"] = "PASS (with note): Full training shows moderate train/val gap."
                                    result["flags"].append("moderate_train_val_gap")
                                else:
                                    result["analysis"] = "PASS: Full training - train/val gap is reasonable."

                            if val_last > train_last * 1.1:
                                result["analysis"] += " Val loss higher than train loss (expected for generalization)."

                            # Check if val_loss is increasing (degradation)
                            val_values = []
                            for row in rows:
                                try:
                                    val_values.append(float(row.get("val_loss", row.get("val_loss_total", ""))))
                                except (ValueError, TypeError):
                                    pass
                            if len(val_values) > 5:
                                first_half = sum(val_values[:len(val_values)//2]) / (len(val_values)//2)
                                second_half = sum(val_values[len(val_values)//2:]) / (len(val_values) - len(val_values)//2)
                                if second_half > first_half * 1.1:
                                    result["analysis"] += " WARNING: Validation loss is trending upward (degradation)."
                                    result["flags"].append("val_loss_degradation")
                        else:
                            result["analysis"] = "WARNING: Full training has no validation metrics. Cannot assess generalization."
                            result["flags"].append("no_validation_metrics")

                    elif experiment_type == "batch_smoke":
                        result["analysis"] = "PASS: Batch smoke test completed" if len(rows) > 1 else "WARNING: Batch smoke test - minimal training data."
                        if len(rows) <= 1:
                            result["flags"].append("minimal_training")

                    elif experiment_type == "ablation":
                        # Check if baseline exists nearby
                        parent = exp_path.parent
                        siblings = [
                            d for d in (parent.iterdir() if parent != root else [])
                            if d.is_dir() and d.name != exp_path.name
                        ] if parent != root else []
                        has_baseline = any(
                            "baseline" in s.name.lower() or "base" in s.name.lower()
                            for s in siblings
                        )
                        if has_baseline:
                            result["analysis"] = "PASS: Ablation study has a baseline for comparison."
                        else:
                            result["analysis"] = "WARNING: Ablation study - no baseline experiment found in same directory."
                            result["flags"].append("no_baseline_found")

                    elif experiment_type == "evaluation":
                        # Check checkpoint and eval data exist
                        has_pt = any((exp_path / f).suffix == ".pt" for f in os.listdir(exp_path)) if exp_path.is_dir() else False
                        has_pt = has_pt or (exp_path / "checkpoints").is_dir()
                        has_eval = (exp_path / "eval_test.csv").exists() or (exp_path / "eval" / "eval_test.csv").exists()
                        if has_pt and has_eval:
                            result["analysis"] = "PASS: Evaluation - checkpoint and evaluation data present."
                        elif has_pt:
                            result["analysis"] = "WARNING: Evaluation - checkpoint exists but no eval_test.csv found."
                            result["flags"].append("missing_eval_data")
                        else:
                            result["analysis"] = "WARNING: Evaluation - no checkpoint found."
                            result["flags"].append("no_checkpoint")

        per_exp.append(result)

    undeclared = sum(1 for e in per_exp if e.get("experiment_type") is None)
    flagged = sum(1 for e in per_exp if e.get("flags"))

    # Determine overall status using severity rules
    # Map each flag to its configured severity, considering historical status
    overall_status = STATUS_PASS
    for e in per_exp:
        exp_type = e.get("experiment_type", "")
        is_historical = e.get("is_historical", False)
        for flag in e.get("flags", []):
            severity = _get_flag_severity(
                flag, exp_type, severity_rules, is_historical, historical_can_block
            )
            if severity == "BLOCKED":
                overall_status = _worst(overall_status, STATUS_BLOCKED)
            elif severity == "WARNING":
                overall_status = _worst(overall_status, STATUS_WARNING)

    return {
        "check": "experiment_type_rules",
        "status": overall_status,
        "per_experiment": per_exp,
        "undeclared_type_count": undeclared,
        "flagged_count": flagged,
        "latest_experiment_path": experiments[0]["path"] if experiments else None,
        "historical_experiments_can_block": historical_can_block,
        "evidence": (
            f"{len(per_exp)} experiments analyzed: {undeclared} with undeclared type, "
            f"{flagged} with type-specific flags"
        ),
    }


def _get_flag_severity(
    flag: str,
    experiment_type: str,
    severity_rules: dict[str, Any],
    is_historical: bool,
    historical_can_block: bool,
) -> str:
    """Determine the severity level for a flag based on experiment type rules.

    Mapping of common flag names to severity keys:
    - nan_in_overfit, nan_in_training → nan_or_inf
    - overfit_insufficient, insufficient_overfit → insufficient_overfit
    - large_train_val_gap → large_train_val_gap
    - val_loss_degradation → val_loss_degradation
    - no_validation_metrics → no_validation_metrics
    - missing_checkpoint, no_checkpoint → missing_checkpoint
    - missing_eval_data → missing_eval_data
    - no_baseline_found → missing_baseline
    - minimal_training, pipeline_incomplete → pipeline_incomplete
    - experiment_type_undeclared → always WARNING
    """
    # Map internal flag names to policy flag names
    flag_map = {
        "nan_in_overfit": "nan_or_inf",
        "nan_in_training": "nan_or_inf",
        "overfit_insufficient": "insufficient_overfit",
        "large_train_val_gap": "large_train_val_gap",
        "val_loss_degradation": "val_loss_degradation",
        "no_validation_metrics": "no_validation_metrics",
        "missing_checkpoint": "missing_checkpoint",
        "no_checkpoint": "missing_checkpoint",
        "missing_eval_data": "missing_eval_data",
        "no_baseline_found": "missing_baseline",
        "minimal_training": "pipeline_incomplete",
        "pipeline_incomplete": "pipeline_incomplete",
        "moderate_overfit": "late_loss_rebound",
        "moderate_train_val_gap": "large_train_val_gap",
        "late_loss_rebound": "late_loss_rebound",
        "final_weaker_than_best": "final_weaker_than_best",
    }
    policy_flag = flag_map.get(flag, flag)
    rules = severity_rules.get(experiment_type, {})

    # Get configured severity
    configured = rules.get(policy_flag, rules.get(flag, "WARNING"))

    # Historical experiments: downgrade BLOCKED to WARNING
    if configured == "BLOCKED" and is_historical and not historical_can_block:
        configured = "WARNING"

    return configured


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def run_all_checks() -> dict[str, Any]:
    """Run all experiment health checks and return structured result."""
    policy = _load_policy("model_policy.json")
    experiment_roots = policy.get("experiment_roots", ["runs/", "experiments/", "outputs/"])
    required_files = policy.get("required_experiment_files", ["train_log.csv"])
    desired_files = policy.get("desired_experiment_files", [
        "metrics.json", "eval_test.csv", "prediction_sample.json",
    ])
    divergence_factor = policy.get("loss_divergence_threshold_factor", 3.0)
    gap_threshold = policy.get("overfit_train_val_gap_threshold", 0.5)
    historical_can_block = policy.get("historical_experiments_can_block", False)

    checks = [
        check_latest_experiment(experiment_roots),
        check_experiment_files(experiment_roots, required_files, desired_files),
        check_checkpoints(experiment_roots),
        check_training_completion(experiment_roots),
        check_loss_health(experiment_roots, divergence_factor),
        check_metrics(experiment_roots),
        check_train_val_gap(experiment_roots, gap_threshold),
        check_experiment_type_rules(experiment_roots),
    ]

    # Post-process: if historical experiments can't block, downgrade
    # divergence-only BLOCKED from loss_health for non-latest experiments.
    # NaN/Inf always stays BLOCKED regardless.
    if not historical_can_block:
        loss_check = next((c for c in checks if c["check"] == "loss_health"), None)
        if loss_check and loss_check["status"] == STATUS_BLOCKED:
            experiments = _find_experiments(experiment_roots)
            latest_path = experiments[0]["path"] if experiments else None
            per_exp = loss_check.get("per_experiment", [])
            has_nan_inf = any(
                (e.get("has_nan") or e.get("has_inf"))
                for e in per_exp
            )
            has_div_only = any(
                e.get("has_divergence") and not e.get("has_nan") and not e.get("has_inf")
                for e in per_exp
            )
            # Check if NaN/Inf is in a non-latest experiment (downgrade if so)
            nan_inf_in_latest = any(
                (e.get("has_nan") or e.get("has_inf")) and e.get("path") == latest_path
                for e in per_exp
            )
            div_in_latest = any(
                e.get("has_divergence") and e.get("path") == latest_path
                for e in per_exp
            )

            # If the only issues are in non-latest experiments, downgrade
            if has_nan_inf and not nan_inf_in_latest:
                # NaN/Inf only in historical → downgrade to WARNING
                loss_check["status"] = STATUS_WARNING
                loss_check["_downgraded_from"] = "BLOCKED"
                loss_check["_downgraded_reason"] = "NaN/Inf in historical experiment only; does not block repository"
            elif not has_nan_inf and has_div_only and not div_in_latest:
                # Divergence only in historical → downgrade to WARNING
                loss_check["status"] = STATUS_WARNING
                loss_check["_downgraded_from"] = "BLOCKED"
                loss_check["_downgraded_reason"] = "Divergence in historical experiments only; does not block repository"

    # Recompute overall after any downgrades
    overall = STATUS_UNKNOWN
    for c in checks:
        overall = _worst(overall, c.get("status", STATUS_UNKNOWN))

    blocking_issues = []
    warnings = []
    for c in checks:
        status = c.get("status", STATUS_UNKNOWN)
        check_name = c.get("check", "unknown")
        entry = {
            "check": check_name,
            "criteria": f"Experiment check: {check_name}",
            "actual": c.get("evidence", ""),
            "impact": f"{status}: {check_name}",
            "suggested_action": "Review experiment health report for details",
            "evidence_source": "experiment directories and log files",
        }
        if status == STATUS_BLOCKED:
            blocking_issues.append(entry)
        elif status == STATUS_WARNING:
            warnings.append(entry)

    return {
        "overall_status": overall,
        "checks": checks,
        "blocking_issues": blocking_issues,
        "warnings": warnings,
        "historical_experiments_can_block": historical_can_block,
    }


if __name__ == "__main__":
    print(json.dumps(run_all_checks(), indent=2, ensure_ascii=False, default=str))
