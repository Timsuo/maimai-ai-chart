# V2 Dataset Manifest

The V2 dataset builder prepares local Maidata samples for later audio feature
extraction and training-sample generation. It scans a sample directory, parses
each `maidata.txt` with the V1 parser, writes one ChartIR JSON cache file per
successfully parsed difficulty, and emits a dataset manifest that indexes the
results.

This stage does not perform audio analysis, beat detection, onset detection,
model training, model inference, tensor generation, or chart generation.

## CLI

```bash
maichart dataset build sample_runs -o sample_runs/v2_dataset_manifest.json --cache-dir sample_runs/v2_chart_ir_cache
```

Use `--force` to rebuild ChartIR cache files even when they already exist:

```bash
maichart dataset build sample_runs -o sample_runs/v2_dataset_manifest.json --cache-dir sample_runs/v2_chart_ir_cache --force
```

Use `--encoding` to force a text encoding. If omitted, the builder reuses the
existing Maidata reader's encoding detection.

## Input Layout

The scanner recursively finds files named `maidata.txt`. Typical layouts:

```text
sample_runs/
  maichart_417_20260705_233958/
    maidata.txt
    track.mp3
    bg.png
  maichart_513/
    maidata.txt
    track.mp3
    bg.png
```

```text
data/raw/
  417/
    maidata.txt
    track.mp3
  513/
    maidata.txt
    track.mp3
```

The song id is inferred from the parent directory name. If `track.mp3` or
`track.wav` is present next to `maidata.txt`, the manifest records it as audio.
If no audio file is present, the build still succeeds and records
`has_audio=false`. If `bg.png` is present, the manifest records it as the
background path.

## Manifest Shape

Top-level fields:

```json
{
  "schema": "maichart-dataset-manifest-v1",
  "source_root": "sample_runs",
  "cache_dir": "sample_runs/v2_chart_ir_cache",
  "song_count": 0,
  "difficulty_count": 0,
  "songs": [],
  "errors": []
}
```

Each song entry:

```json
{
  "song_id": "513",
  "title": "Example Title",
  "artist": "Example Artist",
  "maidata_path": "sample_runs/maichart_513/maidata.txt",
  "audio_path": "sample_runs/maichart_513/track.mp3",
  "background_path": "sample_runs/maichart_513/bg.png",
  "has_audio": true,
  "difficulties": []
}
```

Each difficulty entry:

```json
{
  "index": 5,
  "level": "12.0",
  "designer": "Chart Designer",
  "chart_ir_path": "sample_runs/v2_chart_ir_cache/513/difficulty_5.chart_ir.json",
  "timing_points": 387,
  "note_count": 473,
  "type_counts": {
    "tap": 310,
    "hold": 85,
    "slide": 74
  },
  "parse_coverage": 1.0,
  "unknown_token_count": 0,
  "validate_errors": 0,
  "validate_warnings": 0,
  "duration_kind_counts": {},
  "slide_pattern_counts": {}
}
```

Each error entry is non-fatal and includes the failed stage, song id, optional
difficulty index, source path, exception type, and message. A bad `maidata.txt`
does not stop the whole dataset build.

## Paths

When built from the CLI, manifest paths are written relative to the output
manifest's parent directory when possible. Paths outside that base are written
as absolute paths. The ChartIR cache files themselves are written under
`--cache-dir` with stable names:

```text
<cache-dir>/<song_id>/difficulty_<index>.chart_ir.json
```

Existing cache files are reused by default. Pass `--force` to rebuild them.

## Frame Labels

V2.1 can add frame labels for each cached ChartIR:

```bash
maichart dataset labels sample_runs/v2_dataset_manifest.json --out-dir sample_runs/v2_frame_labels --division 16
```

The command preserves existing manifest fields and adds `frame_labels_path` to
each successfully processed difficulty:

```json
{
  "chart_ir_path": "v2_chart_ir_cache/513/difficulty_5.chart_ir.json",
  "frame_labels_path": "v2_frame_labels/513/difficulty_5.frame_labels.json"
}
```

## Next Step

V2.2 can add `audio_features_path` to each song. V2.3 should consume
`audio_path`, `chart_ir_path`, `frame_labels_path`, and `audio_features_path`
to build chart-audio alignment reports while staying separate from parser,
exporter, validator, label builder, audio feature extraction, and generator
code.
