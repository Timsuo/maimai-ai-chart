# Cleanup Stage 1 Archive

Date: 2026-07-10

This stage performs a conservative archive of old experiment artifacts only.
No original files or directories were deleted, moved, or renamed.

## Archived Scope

- `runs/overfit_v25_*`
- `runs/v25_phase2_overfit_limit2/`
- `reports/event_label_compare_limit2/`
- `reports/event_label_compare_limit2_validation_smoke/`

## Archive Outputs

- `archives/v25_overfit_runs_20260710.7z`
- `archives/event_ir_smoke_reports_20260710.zip`

## Explicitly Untouched

- `cache/`
- `cache_qc_fixed/`
- `raw_chart_data/`
- `manifests/training_manifest_qc_fixed_full.json`
- `manifests/experiments/id_batch_v1_qc_fixed/`
- `runs/v25_phase2_split_list_none/`
- `runs/v25_phase2_split_list_sqrt/`
- `runs/v25_phase25_quick_*/`
- `runs/id_batch_v1_qc_fixed/`
- `reports/event_label_compare_quick/`
- `reports/event_label_compare_large_sample/`
- `reports/pattern_label_stats/`
- `.git/`
- `.venv/`
- `tools/python312-embed/`
- `downloads/`

## Important Source And Tool Files

These are current important source/tool files and are not part of this archive:

- `src/maichart/event_ir.py`
- `src/maichart/event_labels.py`
- `src/maichart/event_ir_converter.py`
- `docs/event_ir.md`
- `tests/test_event_ir.py`
- `tests/test_event_labels.py`
- `tools/analyze_pattern_labels.py`
- `tools/compare_event_labels.py`

## Restore Notes

To restore archived files, extract the relevant archive from the repository root
so the stored relative paths are recreated under their original locations.
Because original files were not removed during this stage, restore should only
be needed after a later cleanup step removes or relocates the source directories.
