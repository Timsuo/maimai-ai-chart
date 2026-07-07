# V2.2 Audio Features

The audio feature extractor prepares basic song-level audio features for later
chart-audio alignment and ML dataset preparation. It reads an audio file,
extracts timing and spectral summaries, and writes `audio_features.json`.

This stage does not train models, generate charts, classify drums, perform
vocal separation, run Demucs, create alignment reports, or modify ChartIR.

## Install Optional Dependencies

Audio analysis dependencies are optional so V1 parser, stats, validate, export,
dataset manifest, and frame-label workflows still work without them.

```bash
pip install -e ".[audio]"
```

For development with tests:

```bash
pip install -e ".[dev,audio]"
```

If the audio extra is missing, audio commands report a clear install message.

## CLI

Analyze one audio file:

```bash
maichart audio analyze path/to/track.mp3 -o path/to/audio_features.json --sample-rate 22050
```

Analyze every song with audio in a dataset manifest:

```bash
maichart dataset audio-features sample_runs/v2_dataset_manifest.json --out-dir sample_runs/v2_audio_features --sample-rate 22050 --division 16
```

Use `--force` to rebuild existing feature files. Songs without audio are
skipped and marked in the manifest. Audio read failures are recorded in
manifest `errors` without stopping the batch.

## Schema

Top-level payload:

```json
{
  "schema": "maichart-audio-features-v1",
  "audio_path": "track.mp3",
  "sample_rate": 22050,
  "duration_sec": 135.2,
  "tempo_bpm": 120.0,
  "beats": [],
  "onsets": [],
  "feature_frames": [],
  "hop_length": 512,
  "division": 16
}
```

Beat entries:

```json
{
  "index": 0,
  "time_sec": 0.53
}
```

Onset entries:

```json
{
  "index": 0,
  "time_sec": 1.02,
  "strength": 0.83
}
```

Feature frames:

```json
{
  "frame_index": 0,
  "time_sec": 0.0,
  "onset_strength": 0.0,
  "rms": 0.0,
  "percussive_rms": 0.0,
  "harmonic_rms": 0.0,
  "spectral_centroid": 0.0,
  "spectral_bandwidth": 0.0,
  "zero_crossing_rate": 0.0
}
```

The frame grid is a fixed audio hop grid, not the final chart-frame alignment.
V2.3 will align audio feature frames with chart frame labels.

## Extracted Features

- audio duration
- estimated tempo
- beat times
- onset times and onset strengths
- RMS energy
- harmonic/percussive separation via HPSS
- harmonic RMS
- percussive RMS
- spectral centroid
- spectral bandwidth
- zero crossing rate

All NumPy values are converted to normal Python `float`/`int` values before
JSON serialization. The extractor limits saved feature frames for very long
audio rather than writing raw arrays.

## Manifest Integration

Batch mode adds song-level fields:

```json
{
  "audio_features_path": "v2_audio_features/513.audio_features.json",
  "audio_features_status": "processed"
}
```

Skipped songs include `audio_features_reason`, such as `"missing audio"` or
`"audio features already exist"`. The top-level `audio_feature_summary` records
`processed`, `skipped`, and `failed` counts.

## Next Step

V2.3 should build a chart-audio alignment report that reads `frame_labels_path`
and `audio_features_path` from the manifest and compares chart frames with
audio feature frames.
