"""Git health check module for repository health system.

Read-only: only executes safe Git queries. Never modifies the repository.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    """Return the repository root (two levels up from tools/)."""
    return Path(__file__).resolve().parent.parent


def _git(*args: str, cwd: str | None = None) -> subprocess.CompletedProcess:
    """Run a git command and return the CompletedProcess.

    All commands are read-only (log, status, branch, config, etc.).
    """
    cmd = ["git"] + list(args)
    return subprocess.run(
        cmd,
        capture_output=True, text=True,
        cwd=cwd or str(_repo_root()),
        encoding="utf-8", errors="replace",
    )


def _load_policy(path: str) -> dict[str, Any]:
    """Load a JSON-encoded policy file from .ai/."""
    policy_path = _repo_root() / ".ai" / path
    if not policy_path.exists():
        return {}
    with open(policy_path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

STATUS_PASS = "PASS"
STATUS_WARNING = "WARNING"
STATUS_BLOCKED = "BLOCKED"
STATUS_UNKNOWN = "UNKNOWN"

def _worst(a: str, b: str) -> str:
    order = {STATUS_UNKNOWN: 0, STATUS_PASS: 1, STATUS_WARNING: 2, STATUS_BLOCKED: 3}
    return a if order.get(a, 0) >= order.get(b, 0) else b


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------

def check_current_branch() -> dict[str, Any]:
    """Return current branch name."""
    result: dict[str, Any] = {
        "check": "current_branch",
        "status": STATUS_UNKNOWN,
        "branch": None,
        "evidence": "",
    }
    try:
        r = _git("rev-parse", "--abbrev-ref", "HEAD")
        if r.returncode == 0:
            branch = r.stdout.strip()
            result["branch"] = branch
            result["status"] = STATUS_PASS
            result["evidence"] = f"HEAD is at '{branch}'"
        else:
            result["status"] = STATUS_WARNING
            result["evidence"] = f"git rev-parse failed: {r.stderr.strip()}"
    except Exception as exc:
        result["status"] = STATUS_WARNING
        result["evidence"] = f"Exception: {exc}"
    return result


def check_workspace_clean() -> dict[str, Any]:
    """Check whether the working tree is clean."""
    result: dict[str, Any] = {
        "check": "workspace_clean",
        "status": STATUS_UNKNOWN,
        "is_clean": None,
        "modified_tracked": [],
        "staged": [],
        "untracked": [],
        "evidence": "",
    }
    try:
        r = _git("status", "--porcelain")
        if r.returncode != 0:
            result["status"] = STATUS_WARNING
            result["evidence"] = f"git status failed: {r.stderr.strip()}"
            return result

        lines = [l for l in r.stdout.splitlines() if l.strip()]
        if not lines:
            result["is_clean"] = True
            result["status"] = STATUS_PASS
            result["evidence"] = "Working tree clean"
            return result

        result["is_clean"] = False
        for line in lines:
            status_col = line[:2]
            filename = line[3:]
            if status_col[0] in "MRC" and status_col[1] in "MRC ":
                result["staged"].append(filename)
            elif status_col[0] in " MRC":
                result["modified_tracked"].append(filename)
            elif status_col == "??":
                result["untracked"].append(filename)
            else:
                result["modified_tracked"].append(filename)

        result["status"] = STATUS_WARNING
        result["evidence"] = (
            f"{len(result['modified_tracked'])} modified, "
            f"{len(result['staged'])} staged, "
            f"{len(result['untracked'])} untracked"
        )
    except Exception as exc:
        result["status"] = STATUS_WARNING
        result["evidence"] = f"Exception: {exc}"
    return result


def check_upstream_status() -> dict[str, Any]:
    """Check ahead/behind relative to upstream tracking branch."""
    result: dict[str, Any] = {
        "check": "upstream_status",
        "status": STATUS_UNKNOWN,
        "branch": None,
        "upstream": None,
        "ahead": None,
        "behind": None,
        "evidence": "",
    }
    try:
        r_branch = _git("rev-parse", "--abbrev-ref", "HEAD")
        if r_branch.returncode != 0:
            result["evidence"] = "Cannot determine current branch"
            return result
        branch = r_branch.stdout.strip()
        result["branch"] = branch

        r_upstream = _git("rev-parse", "--abbrev-ref", f"{branch}@{{upstream}}")
        if r_upstream.returncode != 0:
            result["status"] = STATUS_WARNING
            result["evidence"] = f"No upstream configured for '{branch}': {r_upstream.stderr.strip()}"
            return result

        upstream = r_upstream.stdout.strip()
        result["upstream"] = upstream

        r_cmp = _git("rev-list", "--left-right", "--count", f"{branch}...{upstream}")
        if r_cmp.returncode == 0:
            parts = r_cmp.stdout.strip().split()
            if len(parts) >= 2:
                result["ahead"] = int(parts[0])
                result["behind"] = int(parts[1])
                if result["ahead"] == 0 and result["behind"] == 0:
                    result["status"] = STATUS_PASS
                    result["evidence"] = f"Branch '{branch}' is in sync with '{upstream}'"
                elif result["behind"] > 0:
                    result["status"] = STATUS_WARNING
                    result["evidence"] = (
                        f"Branch '{branch}' is {result['behind']} behind '{upstream}'"
                    )
                else:
                    result["status"] = STATUS_PASS
                    result["evidence"] = (
                        f"Branch '{branch}' is {result['ahead']} ahead of '{upstream}'"
                    )
            else:
                result["status"] = STATUS_WARNING
                result["evidence"] = f"Unexpected rev-list output: {r_cmp.stdout.strip()}"
        else:
            result["status"] = STATUS_WARNING
            result["evidence"] = f"rev-list failed: {r_cmp.stderr.strip()}"
    except Exception as exc:
        result["status"] = STATUS_WARNING
        result["evidence"] = f"Exception: {exc}"
    return result


def check_local_branches() -> dict[str, Any]:
    """List local branches and their upstreams."""
    result: dict[str, Any] = {
        "check": "local_branches",
        "status": STATUS_PASS,
        "branches": [],
        "evidence": "",
    }
    try:
        r = _git("branch", "-vv")
        if r.returncode != 0:
            result["status"] = STATUS_WARNING
            result["evidence"] = f"git branch -vv failed: {r.stderr.strip()}"
            return result

        for line in r.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            # Format: "* branch_name  hash message" or "  branch_name  hash [upstream] message"
            current = line.startswith("*")
            name_part = line[2:].strip()  # Remove "* " or "  "
            name = name_part.split()[0] if name_part else ""

            upstream = None
            if "[" in line:
                bracket_start = line.index("[")
                bracket_end = line.index("]", bracket_start)
                upstream = line[bracket_start + 1:bracket_end]

            result["branches"].append({
                "name": name,
                "current": current,
                "upstream": upstream,
            })
        result["evidence"] = f"{len(result['branches'])} local branches"
    except Exception as exc:
        result["status"] = STATUS_WARNING
        result["evidence"] = f"Exception: {exc}"
    return result


def check_merged_branches(base_branch: str = "main") -> dict[str, Any]:
    """List branches merged into the base branch."""
    result: dict[str, Any] = {
        "check": "merged_branches",
        "status": STATUS_UNKNOWN,
        "base_branch": base_branch,
        "merged": [],
        "total_local": 0,
        "evidence": "",
    }
    try:
        # Get all local branches
        r_all = _git("branch", "--format=%(refname:short)")
        if r_all.returncode != 0:
            result["status"] = STATUS_WARNING
            result["evidence"] = f"Cannot list branches: {r_all.stderr.strip()}"
            return result
        all_branches = [b for b in r_all.stdout.splitlines() if b.strip() and b.strip() != base_branch]
        result["total_local"] = len(all_branches)

        r = _git("branch", "--merged", base_branch)
        if r.returncode != 0:
            result["status"] = STATUS_WARNING
            result["evidence"] = f"Cannot list merged branches: {r.stderr.strip()}"
            return result

        merged = []
        for line in r.stdout.splitlines():
            line = line.strip().lstrip("*").strip()
            if line and line != base_branch:
                merged.append(line)
        result["merged"] = merged
        result["status"] = STATUS_PASS
        result["evidence"] = f"{len(merged)} branches merged into '{base_branch}'"
    except Exception as exc:
        result["status"] = STATUS_WARNING
        result["evidence"] = f"Exception: {exc}"
    return result


def check_stale_branches(stale_days: int = 30) -> dict[str, Any]:
    """Find local branches with no commits in the last N days."""
    result: dict[str, Any] = {
        "check": "stale_branches",
        "status": STATUS_UNKNOWN,
        "stale_days": stale_days,
        "stale": [],
        "evidence": "",
    }
    try:
        r = _git("branch", "--format=%(refname:short)")
        if r.returncode != 0:
            result["status"] = STATUS_WARNING
            result["evidence"] = f"Cannot list branches: {r.stderr.strip()}"
            return result

        now = datetime.now(timezone.utc)
        stale = []
        for branch in r.stdout.splitlines():
            branch = branch.strip()
            if not branch:
                continue
            r_date = _git("log", "-1", "--format=%aI", branch)
            if r_date.returncode != 0:
                continue
            date_str = r_date.stdout.strip()
            if not date_str:
                continue
            try:
                last_commit = datetime.fromisoformat(date_str)
                age_days = (now - last_commit).days
                if age_days > stale_days:
                    stale.append({
                        "branch": branch,
                        "last_commit": date_str,
                        "age_days": age_days,
                    })
            except ValueError:
                pass

        result["stale"] = stale
        if stale:
            result["status"] = STATUS_WARNING
            result["evidence"] = f"{len(stale)} branches stale (> {stale_days} days)"
        else:
            result["status"] = STATUS_PASS
            result["evidence"] = "No stale branches found"
    except Exception as exc:
        result["status"] = STATUS_WARNING
        result["evidence"] = f"Exception: {exc}"
    return result


def check_generated_directories(generated_dirs: list[str] | None = None) -> dict[str, Any]:
    """Check for tracked or untracked generated files in specified directories.

    Uses a single git ls-files call for all tracked files, and capped file-
    system scans for untracked files to avoid O(N) git subprocess calls.
    """
    MAX_UNTACKED_SCAN_PER_DIR = 5000  # safety cap per directory

    if generated_dirs is None:
        generated_dirs = [
            "cache/", "cache_qc_fixed/", "cache_event_v2/",
            "downloads/", "_backup/", "archives/",
        ]

    result: dict[str, Any] = {
        "check": "generated_directories",
        "status": STATUS_UNKNOWN,
        "directories_checked": list(generated_dirs),
        "tracked_files": {},
        "untracked_files": {},
        "evidence": "",
    }
    root = _repo_root()
    try:
        # --- Collect ALL tracked files once ---
        r_all = _git("ls-files", "--")
        all_tracked: set[str] = set()
        if r_all.returncode == 0:
            all_tracked = {line.strip() for line in r_all.stdout.splitlines() if line.strip()}

        has_any = False
        for gen_dir in generated_dirs:
            gen_path = root / gen_dir.rstrip("/")
            if not gen_path.exists():
                continue
            gen_dir_norm = gen_dir.rstrip("/") + "/"

            # Tracked files: filter from the pre-collected set
            tracked = sorted(f for f in all_tracked if f.startswith(gen_dir_norm) or f == gen_dir.rstrip("/"))
            if tracked:
                result["tracked_files"][gen_dir] = tracked
                has_any = True

            # Untracked files: walk the filesystem with a cap
            untracked_in_dir: list[str] = []
            count = 0
            try:
                for file_path in gen_path.rglob("*"):
                    if not file_path.is_file():
                        continue
                    count += 1
                    if count > MAX_UNTACKED_SCAN_PER_DIR:
                        untracked_in_dir.append(f"... (capped at {MAX_UNTACKED_SCAN_PER_DIR} files)")
                        break
                    rel = str(file_path.relative_to(root)).replace("\\", "/")
                    if rel not in all_tracked:
                        untracked_in_dir.append(rel)
            except (OSError, PermissionError):
                pass

            if untracked_in_dir:
                # Cap output: store count + first few examples
                output_entries = untracked_in_dir[:50]  # max 50 examples
                if len(untracked_in_dir) > 50:
                    output_entries.append(f"... and {len(untracked_in_dir) - 50} more files")
                result["untracked_files"][gen_dir] = output_entries
                result["_untracked_total"] = result.get("_untracked_total", 0) + len(untracked_in_dir)
                has_any = True

        if has_any:
            total_tracked = sum(len(v) for v in result["tracked_files"].values())
            total_untracked = sum(
                len(v) for v in result["untracked_files"].values()
                if not (len(v) == 1 and v[0].startswith("..."))
            )
            result["status"] = STATUS_WARNING
            result["evidence"] = (
                f"Generated directories: {total_tracked} tracked + {total_untracked} untracked files"
            )
        else:
            result["status"] = STATUS_PASS
            result["evidence"] = "No generated files detected in generated directories"
    except Exception as exc:
        result["status"] = STATUS_WARNING
        result["evidence"] = f"Exception: {exc}"
    return result


def check_large_files(threshold_bytes: int = 10_485_760) -> dict[str, Any]:
    """Find tracked files exceeding the size threshold."""
    result: dict[str, Any] = {
        "check": "large_files",
        "status": STATUS_UNKNOWN,
        "threshold_bytes": threshold_bytes,
        "large_files": [],
        "evidence": "",
    }
    if threshold_bytes <= 0:
        result["status"] = STATUS_PASS
        result["evidence"] = "Large file check disabled (threshold <= 0)"
        return result

    root = _repo_root()
    try:
        r = _git("ls-files", "--")
        if r.returncode != 0:
            result["status"] = STATUS_WARNING
            result["evidence"] = f"git ls-files failed: {r.stderr.strip()}"
            return result

        large = []
        for line in r.stdout.splitlines():
            filepath = line.strip()
            if not filepath:
                continue
            full = root / filepath
            try:
                size = full.stat().st_size
                if size > threshold_bytes:
                    large.append({"path": filepath, "size_bytes": size})
            except FileNotFoundError:
                pass

        result["large_files"] = large
        if large:
            result["status"] = STATUS_WARNING
            result["evidence"] = f"{len(large)} files exceed {threshold_bytes:,} bytes"
        else:
            result["status"] = STATUS_PASS
            result["evidence"] = f"No files exceed {threshold_bytes:,} bytes"
    except Exception as exc:
        result["status"] = STATUS_WARNING
        result["evidence"] = f"Exception: {exc}"
    return result


# ---------------------------------------------------------------------------
# Main entry: run all checks and return structured result
# ---------------------------------------------------------------------------

def run_all_checks() -> dict[str, Any]:
    """Run all Git health checks and return a structured result."""
    policy = _load_policy("branch_policy.json")
    repo_policy = _load_policy("repository_policy.json")

    protected = policy.get("protected_branches", ["main"])
    stale_days = policy.get("stale_branch_days", 30)
    generated_dirs = repo_policy.get("generated_directories", [
        "cache/", "cache_qc_fixed/", "cache_event_v2/",
        "downloads/", "_backup/", "archives/",
    ])
    large_threshold = repo_policy.get("large_file_threshold_bytes", 10_485_760)

    checks = []

    # 1. Current branch
    checks.append(check_current_branch())

    # 2. Workspace clean
    checks.append(check_workspace_clean())

    # 3. Upstream status
    checks.append(check_upstream_status())

    # 4. Local branches
    checks.append(check_local_branches())

    # 5. Merged branches (for each protected branch)
    for base in protected:
        checks.append(check_merged_branches(base))

    # 6. Stale branches
    checks.append(check_stale_branches(stale_days))

    # 7. Generated directories
    checks.append(check_generated_directories(generated_dirs))

    # 8. Large files
    checks.append(check_large_files(large_threshold))

    # Compute overall status
    overall = STATUS_UNKNOWN
    for c in checks:
        overall = _worst(overall, c.get("status", STATUS_UNKNOWN))

    # Collect blocking issues and warnings
    blocking_issues = []
    warnings = []
    for c in checks:
        status = c.get("status", STATUS_UNKNOWN)
        check_name = c.get("check", "unknown")
        if status == STATUS_BLOCKED:
            blocking_issues.append({
                "check": check_name,
                "criteria": f"Git check: {check_name}",
                "actual": c.get("evidence", ""),
                "impact": f"BLOCKED: {check_name} check could not pass",
                "suggested_action": "Investigate and resolve the blocking condition",
                "evidence_source": "git CLI output",
            })
        elif status == STATUS_WARNING:
            warnings.append({
                "check": check_name,
                "criteria": f"Git check: {check_name}",
                "actual": c.get("evidence", ""),
                "impact": f"WARNING: {check_name} check flagged an issue",
                "suggested_action": "Review the warning and take appropriate action",
                "evidence_source": "git CLI output",
            })

    return {
        "overall_status": overall,
        "checks": checks,
        "blocking_issues": blocking_issues,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(json.dumps(run_all_checks(), indent=2, ensure_ascii=False, default=str))
