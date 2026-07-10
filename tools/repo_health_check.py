#!/usr/bin/env python3
"""Repository Health Check – main entry point.

Run:  python tools/repo_health_check.py

Generates:
  reports/repository_health/latest.json   (structured, machine-readable)
  reports/repository_health/latest.md     (human-readable summary)

This tool is READ-ONLY. It never modifies repository state, deletes files,
commits, merges, rebases, resets, or pushes.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _ensure_report_dir() -> Path:
    report_dir = _repo_root() / "reports" / "repository_health"
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir


# ---------------------------------------------------------------------------
# Status ordering
# ---------------------------------------------------------------------------

STATUS_PASS = "PASS"
STATUS_WARNING = "WARNING"
STATUS_BLOCKED = "BLOCKED"
STATUS_UNKNOWN = "UNKNOWN"

def _worst(a: str, b: str) -> str:
    order = {STATUS_UNKNOWN: 0, STATUS_PASS: 1, STATUS_WARNING: 2, STATUS_BLOCKED: 3}
    return a if order.get(a, 0) >= order.get(b, 0) else b


# ---------------------------------------------------------------------------
# Import check modules
# ---------------------------------------------------------------------------

def _import_module(module_name: str):
    """Import a module from the tools/ directory."""
    import importlib.util
    tools_dir = _repo_root() / "tools"
    module_path = tools_dir / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _generate_markdown(report: dict[str, Any]) -> str:
    """Generate a human-readable Markdown report from the structured JSON report."""
    lines = []
    lines.append("# Repository Health Report")
    lines.append("")
    lines.append(f"**Generated:** {report.get('generated_at', 'unknown')}")
    lines.append(f"**Repository:** {report.get('repository_root', 'unknown')}")
    lines.append("")

    # Overall status
    overall = report.get("overall_status", STATUS_UNKNOWN)
    icon = {"PASS": "✅", "WARNING": "⚠️", "BLOCKED": "🚫", "UNKNOWN": "❓"}.get(overall, "❓")
    lines.append(f"## Overall Status: {icon} {overall}")
    lines.append("")

    # Module statuses
    lines.append("| Module | Status |")
    lines.append("|--------|--------|")
    for mod in ["git", "data", "model"]:
        mod_data = report.get(mod, {})
        mod_status = mod_data.get("overall_status", STATUS_UNKNOWN) if mod_data else "NOT RUN"
        mod_icon = {"PASS": "✅", "WARNING": "⚠️", "BLOCKED": "🚫", "UNKNOWN": "❓"}.get(mod_status, "❓")
        lines.append(f"| {mod.title()} | {mod_icon} {mod_status} |")
    lines.append("")

    # Blocking issues
    blocking = report.get("blocking_issues", [])
    lines.append(f"## 🚫 Blocking Issues ({len(blocking)})")
    lines.append("")
    if blocking:
        for i, issue in enumerate(blocking, 1):
            lines.append(f"### {i}. {issue.get('check', 'unknown')}")
            lines.append("")
            lines.append(f"- **Criteria:** {issue.get('criteria', 'N/A')}")
            lines.append(f"- **Actual Result:** {issue.get('actual', 'N/A')}")
            lines.append(f"- **Impact:** {issue.get('impact', 'N/A')}")
            lines.append(f"- **Suggested Action:** {issue.get('suggested_action', 'N/A')}")
            lines.append(f"- **Evidence Source:** {issue.get('evidence_source', 'N/A')}")
            lines.append("")
    else:
        lines.append("No blocking issues detected.")
        lines.append("")

    # Warnings
    warns = report.get("warnings", [])
    lines.append(f"## ⚠️ Warnings ({len(warns)})")
    lines.append("")
    if warns:
        for i, w in enumerate(warns, 1):
            lines.append(f"### {i}. {w.get('check', 'unknown')}")
            lines.append("")
            lines.append(f"- **Criteria:** {w.get('criteria', 'N/A')}")
            lines.append(f"- **Actual Result:** {w.get('actual', 'N/A')}")
            lines.append(f"- **Impact:** {w.get('impact', 'N/A')}")
            lines.append(f"- **Suggested Action:** {w.get('suggested_action', 'N/A')}")
            lines.append(f"- **Evidence Source:** {w.get('evidence_source', 'N/A')}")
            lines.append("")
    else:
        lines.append("No warnings.")
        lines.append("")

    # Recommended actions
    actions = report.get("recommended_actions", [])
    lines.append(f"## 📋 Recommended Actions ({len(actions)})")
    lines.append("")
    if actions:
        for i, a in enumerate(actions, 1):
            lines.append(f"{i}. {a}")
    else:
        lines.append("No specific actions recommended at this time.")
    lines.append("")

    # Evidence summary per module
    for mod_name in ["git", "data", "model"]:
        mod_data = report.get(mod_name, {})
        if not mod_data:
            continue
        lines.append(f"## {mod_name.title()} Health – Evidence")
        lines.append("")
        checks = mod_data.get("checks", [])
        for check in checks:
            check_name = check.get("check", "unknown")
            check_status = check.get("status", STATUS_UNKNOWN)
            evidence = check.get("evidence", "")
            icon = {"PASS": "✅", "WARNING": "⚠️", "BLOCKED": "🚫", "UNKNOWN": "❓"}.get(check_status, "❓")
            lines.append(f"- {icon} **{check_name}**: {evidence}")
        lines.append("")

    # Errors
    errors = report.get("errors", [])
    if errors:
        lines.append(f"## 🔧 Execution Errors ({len(errors)})")
        lines.append("")
        for err in errors:
            lines.append(f"- **{err.get('module', '?')}**: {err.get('error', '?')}")
        lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by `tools/repo_health_check.py` — read-only health check.*")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Run all health checks and generate reports."""
    generated_at = datetime.now(timezone.utc).isoformat()
    root = _repo_root()
    report_dir = _ensure_report_dir()

    # Overall report structure
    report: dict[str, Any] = {
        "generated_at": generated_at,
        "repository_root": str(root),
        "overall_status": STATUS_UNKNOWN,
        "git": {},
        "data": {},
        "model": {},
        "blocking_issues": [],
        "warnings": [],
        "recommended_actions": [],
        "errors": [],
    }

    # --- Git health ---
    try:
        git_module = _import_module("check_git_health")
        git_result = git_module.run_all_checks()
        report["git"] = git_result
    except Exception as exc:
        report["git"] = {"overall_status": STATUS_UNKNOWN, "error": str(exc)}
        report["errors"].append({"module": "git", "error": str(exc)})
        print(f"[ERROR] Git health check failed: {exc}", file=sys.stderr)

    # --- Data health ---
    try:
        data_module = _import_module("check_data_health")
        data_result = data_module.run_all_checks()
        report["data"] = data_result
    except Exception as exc:
        report["data"] = {"overall_status": STATUS_UNKNOWN, "error": str(exc)}
        report["errors"].append({"module": "data", "error": str(exc)})
        print(f"[ERROR] Data health check failed: {exc}", file=sys.stderr)

    # --- Experiment/model health ---
    try:
        exp_module = _import_module("check_experiment_health")
        exp_result = exp_module.run_all_checks()
        report["model"] = exp_result
    except Exception as exc:
        report["model"] = {"overall_status": STATUS_UNKNOWN, "error": str(exc)}
        report["errors"].append({"module": "model", "error": str(exc)})
        print(f"[ERROR] Experiment health check failed: {exc}", file=sys.stderr)

    # --- Compute overall status ---
    overall = STATUS_UNKNOWN
    for mod in ["git", "data", "model"]:
        mod_status = report.get(mod, {}).get("overall_status", STATUS_UNKNOWN)
        overall = _worst(overall, mod_status)
    report["overall_status"] = overall

    # --- Collect blocking issues and warnings from all modules ---
    for mod in ["git", "data", "model"]:
        mod_data = report.get(mod, {})
        report["blocking_issues"].extend(mod_data.get("blocking_issues", []))
        report["warnings"].extend(mod_data.get("warnings", []))

    # --- Generate recommended actions ---
    actions = []
    if report["blocking_issues"]:
        actions.append("🔴 Address blocking issues immediately – these may prevent reliable training or evaluation.")
    if report["warnings"]:
        actions.append("🟡 Review warnings and address those relevant to current work.")

    # Git-specific recommendations
    git_result = report.get("git", {})
    for check in git_result.get("checks", []):
        name = check.get("check", "")
        if name == "workspace_clean" and not check.get("is_clean", True):
            actions.append("💾 Working tree has uncommitted changes – consider committing or stashing before switching branches.")
        if name == "upstream_status" and (check.get("behind") or 0) > 0:
            actions.append("📥 Branch is behind upstream – consider pulling latest changes.")
        if name == "stale_branches" and check.get("stale"):
            actions.append("🧹 Stale branches detected – consider cleaning up merged or abandoned branches.")
        if name == "large_files" and check.get("large_files"):
            actions.append("📦 Large files detected in repository – consider using Git LFS or adding to .gitignore.")
        if name == "generated_directories":
            tracked = check.get("tracked_files", {})
            if tracked:
                actions.append("🗑️ Generated files tracked in Git – add generated directories to .gitignore and remove from tracking.")

    # Data-specific recommendations
    data_result = report.get("data", {})
    for check in data_result.get("checks", []):
        name = check.get("check", "")
        if name == "data_product_existence" and check.get("total_missing", 0) > 0:
            remapped = check.get("total_remapped", 0)
            msg = f"📊 {check.get('total_missing', 0)} missing data products detected"
            if remapped > 0:
                msg += f" ({remapped} resolved via legacy path remapping)"
            msg += " – re-run preprocessing for affected samples."
            actions.append(msg)
        if name == "split_leakage" and any(r.get("has_leakage") for r in check.get("results", [])):
            actions.append("🔀 Split leakage detected – review split logic to ensure train/val/test separation by music_id.")
        if name == "old_cache_references" and check.get("findings"):
            actions.append("📁 Old cache root references detected – update manifest paths to relative or current cache structure.")
        if name == "duplicate_samples":
            info = check.get("cross_manifest_reuse_info", "")
            if info and check.get("cross_manifest_reuse", {}).get("music_id_reuse_count", 0) > 0:
                # Cross-manifest reuse is normal for batch subsets → not a warning, just info
                pass

    # Model-specific recommendations
    model_result = report.get("model", {})
    for check in model_result.get("checks", []):
        name = check.get("check", "")
        if name == "loss_health" and check.get("affected_experiments", 0) > 0:
            actions.append("🔥 NaN/Inf/divergence in loss detected – investigate training instability in affected experiments.")
        if name == "checkpoints" and check.get("no_checkpoint_count", 0) > 0:
            actions.append("💿 Experiments without checkpoints – ensure checkpoints are saved for resumability.")
        if name == "experiment_type_rules":
            for e in check.get("per_experiment", []):
                if "experiment_type_undeclared" in e.get("flags", []):
                    actions.append(f"🏷️ Experiment '{e['path']}' has no declared experiment_type – add to config for accurate health assessment.")
                    break

    report["recommended_actions"] = actions

    # --- Write reports ---
    json_path = report_dir / "latest.json"
    md_path = report_dir / "latest.md"

    # Write JSON
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)

    # Write Markdown
    md_content = _generate_markdown(report)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(md_content)

    # --- Summary to stdout ---
    print(f"=== Repository Health Check ===")
    print(f"Generated: {generated_at}")
    print(f"Overall:   {report['overall_status']}")
    print(f"  Git:     {report.get('git', {}).get('overall_status', 'NOT RUN')}")
    print(f"  Data:    {report.get('data', {}).get('overall_status', 'NOT RUN')}")
    print(f"  Model:   {report.get('model', {}).get('overall_status', 'NOT RUN')}")
    print(f"Blocking:  {len(report['blocking_issues'])}")
    print(f"Warnings:  {len(report['warnings'])}")
    print(f"Actions:   {len(report['recommended_actions'])}")
    print(f"")
    print(f"Report written to:")
    print(f"  {json_path}")
    print(f"  {md_path}")

    return 0 if report["overall_status"] != STATUS_BLOCKED else 1


if __name__ == "__main__":
    sys.exit(main())
