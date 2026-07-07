# V2.3 Alignment Report

The chart-audio alignment report compares ChartIR frame labels with audio
features. It is an analysis artifact for understanding how mature charts place
notes against onsets, beats, energy, and local density.

This stage does not train models, generate charts, repair charts, run neural
networks, classify drums, separate vocals, use Demucs, or provide a GUI.

## Inputs

Single-report mode needs:

```text
chart_ir.json
frame_labels.json
audio_features.json
```

Dataset batch mode expects each manifest song/difficulty to contain:

```json
{
  "audio_features_path": "v2_audio_features/513.audio_features.json",
  "difficulties": [
    {
      "chart_ir_path": "v2_chart_ir_cache/513/difficulty_5.chart_ir.json",
      "frame_labels_path": "v2_frame_labels/513/difficulty_5.frame_labels.json"
    }
  ]
}
```

## CLI

Build one report:

```bash
maichart align chart_ir.json frame_labels.json audio_features.json -o alignment_report.json --onset-tolerance-ms 50
```

Build reports for a dataset manifest:

```bash
maichart dataset align sample_runs/v2_dataset_manifest.json --out-dir sample_runs/v2_alignment_reports --onset-tolerance-ms 50
```

Batch mode skips difficulties missing `frame_labels_path`, skips songs missing
`audio_features_path`, records individual failures in manifest `errors`, and
adds `alignment_report_path` to successfully processed difficulties. Use
`--force` to rebuild reports that already exist.

## Schema

Top-level payload:

```json
{
  "schema": "maichart-alignment-report-v1",
  "song_id": "513",
  "difficulty": 5,
  "onset_tolerance_ms": 50.0,
  "summary": {},
  "by_note_type": {},
  "density": {},
  "frames": []
}
```

Summary fields:

```json
{
  "note_count": 469,
  "frames_with_notes": 312,
  "nearest_onset_mean_delta_ms": 12.4,
  "nearest_onset_median_delta_ms": 8.1,
  "onset_hit_rate_25ms": 0.62,
  "onset_hit_rate_50ms": 0.78,
  "onset_hit_rate_100ms": 0.89
}
```

`onset_hit_rate_50ms` is the fraction of note-bearing chart frames whose
nearest audio onset is within 50 milliseconds. The 25 ms and 100 ms rates give
a stricter and looser view of the same alignment behavior.

By-note-type fields are reported for `tap`, `break`, `hold_start`,
`slide_start`, and `touch`:

```json
{
  "count": 0,
  "onset_hit_rate_50ms": 0.0,
  "mean_delta_ms": null,
  "median_delta_ms": null,
  "mean_onset_strength": null
}
```

Each frame records the chart frame, nearest onset for note frames, and nearest
audio feature sample:

```json
{
  "frame_index": 128,
  "time_sec": 32.0,
  "beat": "64",
  "has_note": true,
  "note_types": ["tap"],
  "nearest_onset": {
    "time_sec": 32.01,
    "delta_ms": 10.0,
    "strength": 0.82
  },
  "audio": {
    "onset_strength": 0.73,
    "rms": 0.2,
    "percussive_rms": 0.14,
    "harmonic_rms": 0.08
  }
}
```

## Density Curves

`density` contains:

- `notes_per_second`: note count in each one-second window.
- `notes_per_4beat_window`: note count in each four-beat window.
- `notes_per_16beat_window`: note count in each sixteen-beat phrase-like window.

These curves are intended for inspecting high-density sections and comparing
difficulty layers against the same audio.

## Next Step

V2.4 uses alignment reports to design a rule baseline rhythm generator. V2.5
can then use the rule baseline as a comparison point for supervised rhythm
modeling.
