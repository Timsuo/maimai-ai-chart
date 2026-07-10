"""Data health check module for repository health system.

Read-only: scans manifest files, checks data product existence,
validates JSON, and detects split leakage.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
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
        # Support both .yaml (JSON content) and .json files
        return json.load(fh)


STATUS_PASS = "PASS"
STATUS_WARNING = "WARNING"
STATUS_BLOCKED = "BLOCKED"
STATUS_UNKNOWN = "UNKNOWN"
STATUS_INFO = "INFO"

def _worst(a: str, b: str) -> str:
    order = {STATUS_INFO: 0, STATUS_UNKNOWN: 1, STATUS_PASS: 2, STATUS_WARNING: 3, STATUS_BLOCKED: 4}
    return a if order.get(a, 0) >= order.get(b, 0) else b


# ---------------------------------------------------------------------------
# Manifest scanning
# ---------------------------------------------------------------------------

# Fields that hint a JSON file might be a manifest (even if filename doesn't match)
MANIFEST_HINT_KEYS = {
    "samples",
    "songs",
    "entries",
    "difficulties",
    "train",
    "validation",
    "test",
    "usable_difficulties",
    "source_manifest",
    "schema",
}


def _matches_pattern(name: str, patterns: list[str]) -> bool:
    """Simple glob-style pattern matching for filenames."""
    import fnmatch
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def _find_manifest_files(manifest_paths: list[str]) -> list[str]:
    """Find manifest JSON files using include/exclude patterns and structural hints.

    Resolution order:
    1. Exclude pattern match → skip
    2. Include pattern match → candidate
    3. No filename rule match but JSON has manifest hint keys → candidate
    4. Otherwise → skip silently
    """
    policy = _load_policy("data_policy.json")
    include_patterns = policy.get("manifest_include_patterns", ["*.json"])
    exclude_patterns = policy.get("manifest_exclude_patterns", ["*config*.json", "*metrics*.json", "*report*.json", "*prediction*.json", "*summary*.json"])

    found = []
    skipped_by_exclude = []
    root = _repo_root()
    for mp in manifest_paths:
        p = root / mp.rstrip("/")
        if not p.exists():
            continue
        for item in sorted(p.rglob("*.json")):
            if not item.is_file():
                continue
            rel = str(item.relative_to(root))
            fname = item.name

            # 1. Exclude pattern
            if _matches_pattern(fname, exclude_patterns):
                skipped_by_exclude.append(rel)
                continue

            # 2. Include pattern
            if _matches_pattern(fname, include_patterns):
                found.append(rel)
                continue

            # 3. Structural hint
            try:
                with open(item, encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict) and (set(data.keys()) & MANIFEST_HINT_KEYS):
                    found.append(rel)
            except Exception:
                # Unparseable files are handled by check_manifest_parse
                pass

    return found


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------

def check_manifest_list(manifest_paths: list[str]) -> dict[str, Any]:
    """List all manifest files found."""
    manifests = _find_manifest_files(manifest_paths)
    return {
        "check": "manifest_list",
        "status": STATUS_PASS if manifests else STATUS_WARNING,
        "manifests": manifests,
        "count": len(manifests),
        "evidence": f"Found {len(manifests)} manifest files" if manifests else "No manifest files found",
    }


def check_manifest_parse(manifest_paths: list[str]) -> dict[str, Any]:
    """Parse all manifest JSON files and report errors."""
    manifests = _find_manifest_files(manifest_paths)
    parsed_ok = []
    parse_errors = []
    root = _repo_root()

    for mf in manifests:
        fp = root / mf
        try:
            with open(fp, encoding="utf-8") as fh:
                data = json.load(fh)
            parsed_ok.append({
                "path": mf,
                "keys": list(data.keys()),
                "schema": data.get("schema"),
                "version": data.get("version"),
            })
        except json.JSONDecodeError as exc:
            parse_errors.append({"path": mf, "error": str(exc)})
        except UnicodeDecodeError as exc:
            parse_errors.append({"path": mf, "error": f"Encoding error: {exc}"})
        except Exception as exc:
            parse_errors.append({"path": mf, "error": f"Unexpected: {exc}"})

    status = STATUS_PASS
    if parse_errors:
        status = STATUS_BLOCKED if len(parse_errors) == len(manifests) else STATUS_WARNING

    evidence = f"{len(parsed_ok)} okay, {len(parse_errors)} failed to parse"
    if parse_errors:
        evidence += " | Errors: " + "; ".join(f"{e['path']}: {e['error'][:80]}" for e in parse_errors[:3])

    return {
        "check": "manifest_parse",
        "status": status,
        "parsed": parsed_ok,
        "parse_errors": parse_errors,
        "evidence": evidence,
    }


def check_manifest_stats(manifest_paths: list[str]) -> dict[str, Any]:
    """Compute per-manifest statistics: song count, difficulty count, usable/rejected."""
    manifests = _find_manifest_files(manifest_paths)
    stats = []
    root = _repo_root()

    for mf in manifests:
        fp = root / mf
        try:
            with open(fp, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue

        stat: dict[str, Any] = {"path": mf}

        # Try different manifest structures
        if "songs" in data:
            songs = data["songs"]
            stat["song_count"] = len(songs)
            total_diffs = sum(len(s.get("difficulties", [])) for s in songs)
            stat["difficulty_count"] = total_diffs
            usable = sum(
                1 for s in songs for d in s.get("difficulties", [])
                if d.get("usable_for_training", False)
            )
            stat["usable_count"] = usable
            stat["rejected_count"] = total_diffs - usable
            stat["format"] = "training_manifest_v1"
        elif "samples" in data:
            samples = data["samples"]
            stat["song_count"] = len(set(s.get("music_id", "") for s in samples))
            stat["difficulty_count"] = len(samples)
            usable = sum(1 for s in samples if s.get("usable_for_training", True))
            stat["usable_count"] = usable
            stat["rejected_count"] = len(samples) - usable
            stat["format"] = "batch_manifest"
        else:
            stat["song_count"] = 0
            stat["difficulty_count"] = 0
            stat["usable_count"] = 0
            stat["rejected_count"] = 0
            stat["format"] = "unknown"
            stat["warning"] = f"Unrecognized format, keys: {list(data.keys())}"

        stats.append(stat)

    status = STATUS_PASS
    unknown_formats = [s["path"] for s in stats if s.get("format") == "unknown"]
    if unknown_formats:
        status = STATUS_WARNING

    return {
        "check": "manifest_stats",
        "status": status,
        "stats": stats,
        "unknown_formats": unknown_formats,
        "evidence": f"{len(stats)} manifests analyzed, {len(unknown_formats)} with unrecognized format",
    }


def check_data_product_existence(
    manifest_paths: list[str],
    cache_roots: list[str] | None = None,
    required_products: list[str] | None = None,
) -> dict[str, Any]:
    """Check that required data products exist on disk for each usable difficulty.

    Resolution order:
    1. Manifest's original path (as-is).
    2. Legacy path mapping (from policy).
    3. Try each configured cache_root with stable relative path suffix.
    4. If still not found → record as missing (aggregated, not per-file).
    """
    policy = _load_policy("data_policy.json")
    if cache_roots is None:
        cache_roots = policy.get("cache_roots", ["cache_qc_fixed/", "cache_event_v2/"])
    if required_products is None:
        required_products = policy.get("required_data_products", [
            "audio_features", "chart_ir", "frame_labels", "alignment_report",
        ])
    legacy_mappings = policy.get("legacy_path_mappings", [])
    max_examples = policy.get("max_issue_examples", 20)

    manifests = _find_manifest_files(manifest_paths)
    root = _repo_root()

    # Aggregated tracking
    total_checked = 0
    total_missing = 0
    total_remapped = 0
    missing_by_product: dict[str, int] = defaultdict(int)
    missing_examples: list[dict[str, Any]] = []  # capped
    affected_manifests: set[str] = set()

    # Pre-build legacy mapping lookup: normalize paths for cross-OS comparison
    def _norm(p: str) -> str:
        return p.replace("\\", "/").rstrip("/")

    legacy_map: dict[str, str] = {}
    for lm in legacy_mappings:
        legacy_map[_norm(lm["from"])] = _norm(lm["to"])

    def _resolve_path(orig_path: str) -> tuple[str | None, bool]:
        """Resolve a path to a real file. Returns (resolved_rel_path, was_remapped)."""
        np = _norm(orig_path)

        # 1. Original path
        if (root / np).exists():
            return np, False

        # 2. Legacy mapping
        for legacy_from, legacy_to in legacy_map.items():
            if np.startswith(legacy_from):
                remapped = legacy_to + np[len(legacy_from):]
                if (root / remapped).exists():
                    return remapped, True

        # 3. Try each cache root with the stable suffix
        # Extract the stable relative part: e.g. "audio_features/song_id.audio_features.json"
        parts = np.split("/")
        # Find where a known product directory appears, then take everything from there
        product_dirs = {"audio_features", "chart_ir", "frame_labels", "alignment_reports"}
        for i, part in enumerate(parts):
            if part in product_dirs:
                suffix = "/".join(parts[i:])
                for cr in cache_roots:
                    candidate = _norm(cr) + "/" + suffix
                    if (root / candidate).exists():
                        return candidate, True
                break

        return None, False

    for mf in manifests:
        fp = root / mf
        try:
            with open(fp, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue

        mf_missing = 0
        mf_checked = 0
        mf_remapped = 0

        def check_one(orig_path: str, product_name: str, sample_label: str = "") -> None:
            nonlocal mf_missing, mf_checked, mf_remapped, total_missing, total_checked, total_remapped
            if not orig_path:
                mf_checked += 1
                total_checked += 1
                mf_missing += 1
                total_missing += 1
                missing_by_product[product_name] += 1
                return
            mf_checked += 1
            total_checked += 1
            resolved, was_remapped = _resolve_path(orig_path)
            if resolved:
                if was_remapped:
                    mf_remapped += 1
                    total_remapped += 1
            else:
                mf_missing += 1
                total_missing += 1
                missing_by_product[product_name] += 1
                if len(missing_examples) < max_examples:
                    missing_examples.append({
                        "manifest": mf,
                        "product": product_name,
                        "original_path": _sanitize_path(orig_path),
                        "sample": sample_label,
                    })

        # Handle batch manifests (samples with paths dict)
        if "samples" in data:
            for sample in data["samples"]:
                if sample.get("usable_for_training", True):
                    paths = sample.get("paths", {})
                    for product in required_products:
                        p = paths.get(product, "")
                        check_one(p, product, sample.get("sample_id", ""))

        # Handle training manifests (songs with difficulties)
        elif "songs" in data:
            for song in data["songs"]:
                song_id = song.get("song_id", "")
                for diff in song.get("difficulties", []):
                    if diff.get("usable_for_training", False):
                        # Audio features
                        audio_path = f"{data.get('cache_dir', 'cache')}/audio_features/{song_id}.audio_features.json"
                        check_one(audio_path, "audio_features", song_id)
                        check_one(diff.get("chart_ir_path", ""), "chart_ir", f"{song_id}_d{diff.get('difficulty_index', '?')}")
                        check_one(diff.get("frame_labels_path", ""), "frame_labels", f"{song_id}_d{diff.get('difficulty_index', '?')}")
                        check_one(diff.get("alignment_report_path", ""), "alignment_report", f"{song_id}_d{diff.get('difficulty_index', '?')}")

        if mf_missing > 0:
            affected_manifests.add(mf)

    status = STATUS_PASS
    if total_missing > 0:
        # Only BLOCKED if >50% of checked products are missing AND >100 total
        if total_missing > max(total_checked * 0.5, 100) and total_checked > 50:
            status = STATUS_BLOCKED
        else:
            status = STATUS_WARNING

    return {
        "check": "data_product_existence",
        "status": status,
        "total_checked": total_checked,
        "total_missing": total_missing,
        "total_remapped": total_remapped,
        "affected_manifest_count": len(affected_manifests),
        "missing_by_product": dict(missing_by_product),
        "missing_examples": missing_examples,
        "max_examples": max_examples,
        "evidence": (
            f"Checked {total_checked} data products: "
            f"{total_missing} missing, {total_remapped} remapped via legacy paths, "
            f"affecting {len(affected_manifests)} manifests"
        ),
    }


def _sanitize_path(path: str) -> str:
    """Replace known absolute roots with relative forms for report safety."""
    sanitized = path.replace("\\", "/")
    # Replace the real repo root
    try:
        real_root = str(_repo_root()).replace("\\", "/")
        sanitized = sanitized.replace(real_root + "/", "<REPO_ROOT>/")
    except Exception:
        pass
    # Replace known legacy roots
    for legacy in ["F:/AI maichart designer/", "F:\\AI maichart designer\\"]:
        legacy_norm = legacy.replace("\\", "/")
        if legacy_norm in sanitized:
            sanitized = sanitized.replace(legacy_norm, "<LEGACY_ROOT>/")
    return sanitized


def check_split_leakage(manifest_paths: list[str]) -> dict[str, Any]:
    """Check for music_id overlap between train/val/test splits."""
    manifests = _find_manifest_files(manifest_paths)
    root = _repo_root()
    policy = _load_policy("data_policy.json")
    split_keys = policy.get("split_keys_to_check", ["splits", "split_info"])

    results: list[dict[str, Any]] = []

    for mf in manifests:
        fp = root / mf
        try:
            with open(fp, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue

        # Try to find split definitions
        splits_found = {}
        for sk in split_keys:
            if isinstance(data.get(sk), dict):
                splits_found = data[sk]
                break

        # Also check if this is a split manifest (has split_name)
        if "split_name" in data:
            splits_found = {
                data.get("split_name", "unknown"): data,
            }
        if "split_strategy" in data:
            split_name = data.get("split_name", "unknown")
            # Extract music_ids from songs
            if "songs" in data:
                music_ids = [s.get("song_id") for s in data["songs"] if s.get("song_id")]
                results.append({
                    "manifest": mf,
                    "split_name": split_name,
                    "split_strategy": data.get("split_strategy"),
                    "music_id_count": len(set(music_ids)),
                    "note": "single split file - cross-reference with other splits needed",
                })

        # If we have train/val/test as lists of music_ids
        if isinstance(data, dict) and any(k in data for k in ["train", "val", "test"]):
            train_ids = set()
            val_ids = set()
            test_ids = set()

            if isinstance(data.get("train"), list):
                # Could be list of music_ids or list of samples
                for item in data["train"]:
                    if isinstance(item, dict):
                        mid = item.get("music_id") or item.get("song_id")
                        if mid: train_ids.add(mid)
                    elif isinstance(item, str):
                        train_ids.add(item)
            if isinstance(data.get("val"), list):
                for item in data["val"]:
                    if isinstance(item, dict):
                        mid = item.get("music_id") or item.get("song_id")
                        if mid: val_ids.add(mid)
                    elif isinstance(item, str):
                        val_ids.add(item)
            if isinstance(data.get("test"), list):
                for item in data["test"]:
                    if isinstance(item, dict):
                        mid = item.get("music_id") or item.get("song_id")
                        if mid: test_ids.add(mid)
                    elif isinstance(item, str):
                        test_ids.add(item)

            if train_ids or val_ids or test_ids:
                leakage = {
                    "train_val": len(train_ids & val_ids),
                    "train_test": len(train_ids & test_ids),
                    "val_test": len(val_ids & test_ids),
                }
                total_leakage = sum(leakage.values())
                results.append({
                    "manifest": mf,
                    "train_ids": len(train_ids),
                    "val_ids": len(val_ids),
                    "test_ids": len(test_ids),
                    "leakage": leakage,
                    "has_leakage": total_leakage > 0,
                })

    has_any_leakage = any(r.get("has_leakage", False) for r in results)
    status = STATUS_BLOCKED if has_any_leakage else STATUS_PASS
    if not results:
        status = STATUS_UNKNOWN

    return {
        "check": "split_leakage",
        "status": status,
        "results": results,
        "evidence": (
            f"Checked {len(results)} manifests with splits, {'LEAKAGE DETECTED' if has_any_leakage else 'no leakage found'}"
            if results else "No manifest splits found to check"
        ),
    }


def check_duplicate_samples(manifest_paths: list[str]) -> dict[str, Any]:
    """Check for sample duplication at four levels:

    1. within_manifest: same sample_id appears >1 time in one manifest → WARNING
    2. within_split: same music_id appears >1 time in one split partition → WARNING
    3. split_leakage: music_id appears across different split partitions → BLOCKED
    4. cross_manifest_reuse: same sample/music_id across different manifests → INFO
    """
    manifests = _find_manifest_files(manifest_paths)
    root = _repo_root()
    policy = _load_policy("data_policy.json")
    max_examples = policy.get("max_issue_examples", 20)

    # Trackers
    intra_dup_examples: list[dict[str, Any]] = []
    intra_dup_count = 0
    split_dup_examples: list[dict[str, Any]] = []
    split_dup_count = 0
    leakage_examples: list[dict[str, Any]] = []
    leakage_count = 0
    global_music_ids: dict[str, list[str]] = defaultdict(list)  # music_id → list of manifest paths
    global_sample_ids: dict[str, list[str]] = defaultdict(list)  # sample_id → list of manifest paths

    for mf in manifests:
        fp = root / mf
        try:
            with open(fp, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue

        local_sample_ids: dict[str, int] = defaultdict(int)
        local_music_ids: dict[str, int] = defaultdict(int)

        # Determine if this manifest has an explicit parent
        source_manifest = data.get("source_manifest", "")
        manifest_type = data.get("version", data.get("schema", ""))

        def record_sample(sid: str, mid: str):
            local_sample_ids[sid] += 1
            local_music_ids[mid] += 1
            global_sample_ids[sid].append(mf)
            global_music_ids[mid].append(mf)

        if "samples" in data:
            for s in data["samples"]:
                sid = s.get("sample_id", "")
                mid = s.get("music_id", "")
                record_sample(sid, mid)
        elif "songs" in data:
            for song in data["songs"]:
                song_id = song.get("song_id", "")
                for diff in song.get("difficulties", []):
                    sid = f"{song_id}_difficulty_{diff.get('difficulty_index', '?')}"
                    record_sample(sid, song_id)

        # --- 1. Within-manifest duplicates ---
        internal_dupes = {k: v for k, v in local_sample_ids.items() if v > 1}
        if internal_dupes:
            intra_dup_count += 1
            if len(intra_dup_examples) < max_examples:
                intra_dup_examples.append({
                    "manifest": mf,
                    "duplicate_sample_ids": list(internal_dupes.keys())[:5],
                    "count": sum(internal_dupes.values()) - len(internal_dupes),
                })

        # --- 2. Within-split duplicates ---
        # Check train/val/test partitions within this manifest
        for split_name in ["train", "val", "test"]:
            if isinstance(data.get(split_name), list):
                split_mids: dict[str, int] = defaultdict(int)
                for item in data[split_name]:
                    if isinstance(item, dict):
                        mid = item.get("music_id") or item.get("song_id", "")
                    else:
                        mid = str(item)
                    if mid:
                        split_mids[mid] += 1
                mid_dupes = {k: v for k, v in split_mids.items() if v > 1}
                if mid_dupes:
                    split_dup_count += 1
                    if len(split_dup_examples) < max_examples:
                        split_dup_examples.append({
                            "manifest": mf,
                            "split": split_name,
                            "duplicate_music_ids": list(mid_dupes.keys())[:5],
                        })

    # --- 3. Split leakage (already handled in check_split_leakage, but also check here) ---
    leakage_result = check_split_leakage(manifest_paths)
    leakage_count = sum(1 for r in leakage_result.get("results", []) if r.get("has_leakage"))
    leakage_examples = leakage_result.get("results", [])[:max_examples]

    # --- 4. Cross-manifest reuse (INFO only, not a warning) ---
    cross_reuse_music = {k: v for k, v in global_music_ids.items() if len(v) > 1}
    cross_reuse_sample = {k: v for k, v in global_sample_ids.items() if len(v) > 1}

    has_intra_dupes = intra_dup_count > 0
    has_split_dupes = split_dup_count > 0
    has_leakage = leakage_count > 0

    status = STATUS_PASS
    if has_leakage:
        status = STATUS_BLOCKED
    elif has_intra_dupes or has_split_dupes:
        status = STATUS_WARNING

    return {
        "check": "duplicate_samples",
        "status": status,
        "duplicate_within_manifest": {
            "count": intra_dup_count,
            "examples": intra_dup_examples,
        },
        "duplicate_within_split": {
            "count": split_dup_count,
            "examples": split_dup_examples,
        },
        "split_leakage": {
            "count": leakage_count,
            "examples": leakage_examples,
        },
        "cross_manifest_reuse": {
            "music_id_reuse_count": len(cross_reuse_music),
            "sample_id_reuse_count": len(cross_reuse_sample),
        },
        "has_duplicates": has_intra_dupes or has_split_dupes or has_leakage,
        "evidence": (
            f"Intra-manifest dupes: {intra_dup_count}, "
            f"within-split dupes: {split_dup_count}, "
            f"split leakage: {leakage_count}, "
            f"cross-manifest reused music_ids: {len(cross_reuse_music)}"
        ),
        "cross_manifest_reuse_info": (
            f"{len(cross_reuse_music)} music_ids and {len(cross_reuse_sample)} sample_ids "
            f"appear across multiple manifests (normal for batch subsets)"
        ),
    }


def check_old_cache_references(manifest_paths: list[str]) -> dict[str, Any]:
    """Detect manifests that reference old/absolute cache root directories."""
    manifests = _find_manifest_files(manifest_paths)
    root = _repo_root()
    root_str = str(root).replace("\\", "/")
    suspicious_patterns = [
        "F:\\AI maichart designer\\",
        "F:/AI maichart designer/",
    ]
    findings = []

    for mf in manifests:
        fp = root / mf
        try:
            with open(fp, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue

        issues = []

        # Check source_root
        source_root = data.get("source_root", "")
        if source_root:
            for pat in suspicious_patterns:
                if pat in str(source_root).replace("\\", "/"):
                    issues.append(f"source_root contains old path: {source_root}")
                    break

        # Check cache_dir
        cache_dir = data.get("cache_dir", "")
        if cache_dir:
            for pat in suspicious_patterns:
                if pat in str(cache_dir).replace("\\", "/"):
                    issues.append(f"cache_dir contains old path: {cache_dir}")
                    break

        # Check paths in samples
        if "samples" in data:
            for sample in data["samples"]:
                paths = sample.get("paths", {})
                for k, v in paths.items():
                    if isinstance(v, str):
                        for pat in suspicious_patterns:
                            if pat in v.replace("\\", "/"):
                                issues.append(f"sample {sample.get('sample_id', '?')} path {k}: {v}")
                                break
                        if len(issues) >= 5:
                            break
                if len(issues) >= 5:
                    break

        if issues:
            findings.append({"manifest": mf, "issues": issues[:10]})

    status = STATUS_WARNING if findings else STATUS_PASS
    return {
        "check": "old_cache_references",
        "status": status,
        "findings": findings,
        "evidence": (
            f"{len(findings)} manifests reference old cache paths"
            if findings else "No old cache references detected"
        ),
    }


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def run_all_checks() -> dict[str, Any]:
    """Run all data health checks and return structured result."""
    policy = _load_policy("data_policy.json")
    manifest_paths = policy.get("manifest_paths", ["manifests/"])
    cache_roots = policy.get("cache_roots", ["cache/", "cache_qc_fixed/", "cache_event_v2/"])
    required_products = policy.get("required_data_products", [
        "audio_features", "chart_ir", "frame_labels", "alignment_report",
    ])

    checks = [
        check_manifest_list(manifest_paths),
        check_manifest_parse(manifest_paths),
        check_manifest_stats(manifest_paths),
        check_data_product_existence(manifest_paths, cache_roots, required_products),
        check_split_leakage(manifest_paths),
        check_duplicate_samples(manifest_paths),
        check_old_cache_references(manifest_paths),
    ]

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
            "criteria": f"Data check: {check_name}",
            "actual": c.get("evidence", ""),
            "impact": f"{status}: {check_name}",
            "suggested_action": "Review data health report for details",
            "evidence_source": "manifest files and file system",
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
    }


if __name__ == "__main__":
    print(json.dumps(run_all_checks(), indent=2, ensure_ascii=False, default=str))
