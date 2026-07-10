# Model Phase 2.6 Quick Split Report

This report summarizes the Phase 2.6 quick split comparison using:

- `runs/phase26_quick/manifests/train.json`
- `runs/phase26_quick/manifests/val.json`
- `runs/phase26_quick/manifests/test.json`

The local manifest copies only remap cache paths to this workspace. Training artifacts,
checkpoints, generated manifests, and evaluation CSVs are intentionally kept under
`runs/` and `reports/` and are not tracked.

## Feature Notes

`audio7` preserves the original seven audio inputs.

`audio7_plus_grid` appends:

- `beat_phase_sin`
- `beat_phase_cos`
- `bar_phase_sin`
- `bar_phase_cos`
- `song_progress`
- `local_bpm_norm`
- `difficulty_norm`
- `level_norm`

Bar phase is a 4/4 fallback because the current frame labels do not carry a meter map.
`local_bpm_norm` uses cached audio tempo or manifest BPM metadata as the default BPM
fallback; there is no local BPM map in this phase. `level_norm` uses manifest
`level / 15.0` when available and falls back to `difficulty_norm` when level is missing.

## Results

| experiment | feature_set | note_start_pos_weight | val_note_start_f1@0.5 | val_note_start_best_f1 | val_note_start_best_threshold | val_note_start_pred_positive_rate@0.5 | val_note_start_target_positive_rate | val_pattern_start_macro_f1 | val_chord_size_macro_f1 | val_button_best_f1 | test_button_best_f1 | test_note_type_accuracy | test_density_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_audio7 | audio7 | 1.000000 | 0.493092 | 0.552658 | 0.162500 | 0.722369 | 0.393867 | 0.091685 | 0.386556 | 0.115066 | 0.115639 | 0.454142 | 0.610168 |
| aux_audio7 | audio7 | 1.000000 | 0.127529 | 0.560094 | 0.197500 | 0.092909 | 0.393867 | 0.114109 | 0.407614 | 0.114642 | 0.116006 | 0.451917 | 0.612729 |
| aux_audio7_posauto | audio7 | 1.803509 | 0.462404 | 0.560324 | 0.357500 | 0.692186 | 0.393867 | 0.113288 | 0.403556 | 0.113986 | 0.116235 | 0.431725 | 0.599952 |
| aux_grid_posauto | audio7_plus_grid | 1.803509 | 0.733522 | 0.750887 | 0.643750 | 0.510536 | 0.393867 | 0.126438 | 0.450164 | 0.201584 | 0.217778 | 0.552449 | 0.396456 |
| baseline_grid | audio7_plus_grid | 1.000000 | 0.376535 | 0.687440 | 0.352500 | 0.133177 | 0.393867 | 0.082419 | 0.339372 | 0.200166 | 0.212579 | 0.553066 | 0.402225 |

## Takeaways

- `note_start_pos_weight=auto` improves the auxiliary audio7 run at threshold 0.5
  relative to fixed `1.0` and moves the best threshold upward from about `0.20` to
  `0.36`.
- `audio7_plus_grid` is the clear winner on this quick split for button, note_start,
  note type, and density metrics.
- Grid features help the baseline strongly for button, note type, and density metrics,
  though baseline note_start at threshold 0.5 remains conservative without the
  auxiliary positive weighting.
- The best quick split combination is `aux_grid_posauto`.
- The result is strong enough to justify a larger split run before considering any
  Phase 3 decode work.
