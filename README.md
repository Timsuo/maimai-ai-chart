# maimai-ai-chart

`maimai-ai-chart` is a Python foundation for parsing Maidata/Simai-like maimai
fan charts into a raw-preserving `ChartIR`, validating that IR, computing stats,
and exporting original-like Maidata when raw source text is available.

V1 is deliberately scoped to parser, IR, stats, validation, CLI, tests, docs,
and a raw-preserving exporter. V2 starts with a dataset manifest, ChartIR cache
builder, frame-label builder, basic audio feature extractor, and chart-audio
alignment report. V2.4 adds a rule-based rhythm skeleton baseline. V2.5 data
preprocessing adds dataset QC, reusable caches, a training manifest, and
leakage-safe train/val/test splits. This project does not yet implement full
chart generation, neural network training, inference, tensor dataset
generation, or a GUI.

## V1 Supports

- Maidata metadata parsing for `&title`, `&artist`, `&wholebpm`, `&first`,
  `&lv_N`, `&des_N`, and raw multi-line `&inote_N` blocks.
- Timing tokenization for comma-delimited timing points, `{N}` grids, `(BPM)`
  directives, and `E`.
- Button taps, break taps, holds, touch taps, touch holds, slides, break slide
  heads, and chained slides.
- Slide semantics as predefined trajectory references, not arbitrary geometry.
- Unified hold/slide duration parsing for `[8:1]`, `[4:3]`, `[352:49]`,
  `[#0.8057]`, `[0.6383##2.0455]`, and `[1:0]`.
- Unknown token preservation with timing index, beat, tick, seconds, raw token,
  raw timing point, and reason.
- ChartIR JSON round-trip for supported V1 fields.
- Stats for note counts, parser coverage, duration kinds, and slide patterns.
- Validation warnings by default and strict validation errors for unknown tokens.
- A raw-preserving exporter for `parse original maidata -> ChartIR -> export
  original-like maidata`.

## V1 Does Not Support

- Advanced audio analysis, drum transcription, vocal separation, or chart/audio
  alignment reports.
- AI generation, neural network training, model inference, or batch dataset
  tensor processing.
- GUI editing.
- Full normalized export from arbitrary generated ChartIR.
- Free-form slide geometry generation.
- Complete Maidata/Simai syntax coverage beyond the V1 parser surface.

## Install

```bash
pip install -e ".[dev]"
```

Audio feature extraction is optional:

```bash
pip install -e ".[audio]"
```

Run tests:

```bash
python -m pytest -q
```

## CLI

Parse raw metadata:

```bash
maichart parse path/to/maidata.txt -o raw-metadata.json
```

Inspect one difficulty:

```bash
maichart parse path/to/maidata.txt --difficulty 5 --timing -o timing.json
maichart parse path/to/maidata.txt --difficulty 5 --notes -o notes.json
maichart parse path/to/maidata.txt --difficulty 5 --ir -o chart-ir.json
```

Stats and validation:

```bash
maichart stats path/to/maidata.txt --difficulty 5
maichart validate path/to/maidata.txt --difficulty 5
maichart validate path/to/maidata.txt --difficulty 5 --strict
```

Export ChartIR JSON back to Maidata-like text:

```bash
maichart export path/to/chart-ir.json -o maidata.txt
```

Build a V2 dataset manifest and ChartIR cache:

```bash
maichart dataset build sample_runs -o sample_runs/v2_dataset_manifest.json --cache-dir sample_runs/v2_chart_ir_cache
maichart dataset build sample_runs -o sample_runs/v2_dataset_manifest.json --cache-dir sample_runs/v2_chart_ir_cache --force
```

Build V2.1 frame labels from ChartIR:

```bash
maichart labels build sample_runs/v2_chart_ir_cache/maichart_513/difficulty_5.chart_ir.json -o sample_runs/v2_frame_labels/maichart_513/difficulty_5.frame_labels.json --division 16
maichart dataset labels sample_runs/v2_dataset_manifest.json --out-dir sample_runs/v2_frame_labels --division 16
```

Build V2.2 audio features:

```bash
maichart audio analyze sample_runs/maichart_513/track.mp3 -o sample_runs/maichart_513/audio_features.json --sample-rate 22050
maichart dataset audio-features sample_runs/v2_dataset_manifest.json --out-dir sample_runs/v2_audio_features --sample-rate 22050 --division 16
```

Build V2.3 chart-audio alignment reports:

```bash
maichart align sample_runs/v2_chart_ir_cache/maichart_513/difficulty_5.chart_ir.json sample_runs/v2_frame_labels/maichart_513/difficulty_5.frame_labels.json sample_runs/v2_audio_features/maichart_513.audio_features.json -o sample_runs/v2_alignment_reports/maichart_513/difficulty_5.alignment_report.json
maichart dataset align sample_runs/v2_dataset_manifest.json --out-dir sample_runs/v2_alignment_reports --onset-tolerance-ms 50
```

Build V2.4 rhythm profiles and skeletons:

```bash
maichart rhythm profile sample_runs/v2_dataset_manifest.json -o sample_runs/v2_rhythm_profile.json
maichart rhythm generate sample_runs/v2_audio_features/maichart_513.audio_features.json --level 12.0 --profile sample_runs/v2_rhythm_profile.json -o sample_runs/v2_rhythm_skeleton_513.json
maichart rhythm evaluate sample_runs/v2_rhythm_skeleton_513.json sample_runs/v2_frame_labels/maichart_513/difficulty_5.frame_labels.json -o sample_runs/v2_rhythm_eval_513.json
maichart dataset rhythm sample_runs/v2_dataset_manifest.json --out-dir sample_runs/v2_rhythm_skeletons --profile sample_runs/v2_rhythm_profile.json --level-source difficulty
```

Build V2.5 preprocessing manifests and splits:

```bash
maichart preprocess scan raw_chart_data/output -o manifests/raw_manifest.json
maichart preprocess build raw_chart_data/output --cache-dir cache -o manifests/training_manifest.json --division 16 --limit 100
maichart preprocess split manifests/training_manifest.json --out-dir manifests/splits --seed 42 --split-by-song
```

`parse`, `stats`, `validate`, and `dataset build` accept `--encoding`. If
omitted, the reader tries common encodings: `utf-8-sig`, `utf-8`, `shift_jis`,
`cp932`, and `gbk`.

## V2 Dataset Manifest

The dataset builder recursively scans for `maidata.txt`, infers `song_id` from
the parent directory, records sibling `track.mp3`, `track.wav`, and `bg.png`
files when present, and keeps going if audio is missing or one chart fails to
parse. Successfully parsed difficulties are cached as ChartIR JSON under:

```text
<cache-dir>/<song_id>/difficulty_<index>.chart_ir.json
```

The manifest schema is `maichart-dataset-manifest-v1`. It includes
`source_root`, `cache_dir`, `song_count`, `difficulty_count`, `songs`, and
non-fatal `errors`. Each difficulty records stats, parser coverage, unknown
token count, validation issue counts, duration kind counts, and slide pattern
counts. CLI-generated paths are relative to the manifest output directory when
possible.

See [docs/dataset_manifest.md](docs/dataset_manifest.md) for the full field
shape and path conventions. This V2 step only prepares ChartIR caches and a
manifest; audio feature extraction is the next separate phase.

## V2.1 Frame Labels

The frame-label builder converts one ChartIR difficulty into fixed beat-grid
labels for later ML data preparation. The default `division=16` means 16 frames
per 4/4 measure. With `TICKS_PER_BEAT = 1920`, each frame is `480` ticks, or
one quarter beat.

The frame-label schema is `maichart-frame-labels-v1`. Each frame stores
`frame_index`, `beat`, `tick`, estimated `time_sec`, and counts/lists for taps,
breaks, hold starts, hold activity, slide starts, slide activity, touches,
touch holds, note types, positions, slide patterns, and duration kinds. Dataset
batch mode adds `frame_labels_path` to each manifest difficulty without
removing existing fields.

See [docs/frame_labels.md](docs/frame_labels.md) for the full field shape and
grid rules. Frame labels do not contain audio features; V2.2 will add audio
feature extraction as a separate module.

## V2.2 Audio Features

The audio feature extractor is optional and requires:

```bash
pip install -e ".[audio]"
```

It writes `maichart-audio-features-v1` JSON containing audio metadata, duration,
estimated tempo, beat times, onset times, onset strength, RMS, harmonic RMS,
percussive RMS, spectral centroid, spectral bandwidth, zero crossing rate, and
fixed-hop `feature_frames`. Dataset batch mode adds `audio_features_path`,
`audio_features_status`, and an `audio_feature_summary` to the manifest.

See [docs/audio_features.md](docs/audio_features.md) for the schema and batch
behavior. This stage only extracts basic features; V2.3 will build the
chart-audio alignment report.

## V2.3 Alignment Report

The alignment report compares frame labels with audio features. It measures
nearest-onset deltas, onset hit rates at 25/50/100 ms, by-note-type alignment
for taps, breaks, hold starts, slide starts, and touches, plus density curves
per second, per 4 beats, and per 16 beats. Dataset batch mode adds
`alignment_report_path`, `alignment_status`, and `alignment_summary` to the
manifest.

See [docs/alignment_report.md](docs/alignment_report.md) for the full schema.
This stage is analysis only; V2.4 can use these reports to design a rule
baseline rhythm generator.

## V2.4 Rhythm Skeleton

The rule-based rhythm skeleton generator selects candidate note times from
audio features and assigns broad event types: `tap`, `break`, `hold_start`,
`slide_start`, and `touch`. It does not generate positions, slide patterns,
hold durations, full ChartIR, or Maidata.

`maichart rhythm profile` derives level-band density and event-type targets
from frame labels and alignment reports. `maichart rhythm generate` scores
audio frames with onset strength, percussive RMS, RMS, and beat proximity, then
selects a density-limited set of frames. `maichart rhythm evaluate` compares the
baseline with human frame labels using precision, recall, F1, note-count error,
density error, and event distribution difference.

See [docs/rhythm_skeleton.md](docs/rhythm_skeleton.md) for schemas and
limitations. V2.5 can use these outputs as a baseline for a supervised rhythm
model.

## V2.5 Data Preprocessing

The preprocessing pipeline is the model-data gate for collected raw samples:

```text
raw samples -> QC -> ChartIR cache -> frame labels -> audio features -> alignment reports -> training_manifest.json -> train/val/test split
```

Raw samples usually contain `convert_report.json`, `maidata.txt`, and
`track.mp3`/`track.wav`/`track.ogg`. `convert_report.status` is not treated as
the source of truth for dataset usability: a conversion can report `failed`
because files already existed, while the sample is still usable if `maidata`
and audio are present and readable. The manifests keep `conversion_status`,
`dataset_usable`, and per-difficulty `usable_for_training` separate.

`maichart preprocess build` writes:

```text
cache/chart_ir/<song_id>/difficulty_<index>.chart_ir.json
cache/frame_labels/<song_id>/difficulty_<index>.frame_labels.json
cache/audio_features/<song_id>.audio_features.json
cache/alignment_reports/<song_id>/difficulty_<index>.alignment_report.json
```

Training filters reject missing charts, `level=?`, too-low note count,
validation errors, parser coverage below 1.0, unknown tokens, unreadable audio,
and failed frame/audio cache generation. Easy/basic/advanced charts are kept
when valid but receive lower `training_weight`. Splits default to
`--split-by-song` so difficulties from the same song cannot leak across train,
validation, and test.

See [docs/data_preprocessing.md](docs/data_preprocessing.md) for schemas,
filter rules, cache behavior, and CLI examples.

## ChartIR

`ChartIR` is the source of truth after parsing. It stores metadata, difficulty
metadata, timing events, notes, unknown tokens, duration expressions, slide
segments, and raw source text. See [docs/ir_schema.md](docs/ir_schema.md).

Unknown tokens are intentional V1 data, not parser garbage. The rule is:
parsing can fail for a token, but raw content must not be silently dropped.

## Slide Model

Slides are represented as references to known maimai trajectory patterns:

```text
start position + predefined trajectory pattern + path arguments + duration
```

They are not arbitrary free curves. Future generators must select from a slide
trajectory catalog. If visual path sampling is needed, samples should come from
a predefined trajectory table, not model-generated geometry.

The V1 catalog includes `-`, `<`, `>`, `p`, `q`, `pp`, `qq`, `s`, `z`, `v`,
`V`, and `w`. See [docs/slide_model.md](docs/slide_model.md).

## Duration Model

Holds and slides use the same duration parser. Raw duration text is preserved
alongside parsed kind, beats, seconds, ticks, and pair values when available.
Non-integer tick durations are preserved with `ticks = null`. See
[docs/duration_model.md](docs/duration_model.md).

## Raw-Preserving Exporter

The V1 exporter is a raw-preserving exporter. It is intended for:

```text
parse original maidata -> preserve raw -> export original-like maidata
```

When `ChartIR.raw` contains the original `&inote_N=` block, the exporter emits
that raw block. If raw text is missing, the fallback renderer can emit simple
tap, hold, and slide tokens for inspection, but it is not a complete normalized
Maidata generator and should not be treated as one.

## Regression Samples

Local sample runs live under `sample_runs/` when available. The V1 regression
check covers the core known sample IDs:

```text
417 513 786 799 803 812 820 825 833 834 432 471 496 773 777 835
```

New-note regression samples, including charts with touch and touch-hold note
elements, are:

```text
11821 11898 11956 11051 11106 11152 11184 11206 11222 11228 11448 11451 11818 11820
```

Missing samples should be skipped, not treated as failures. Regression reports
belong under `sample_runs/` or `reports/` and should include pipeline failures,
validation errors, unknown token counts, parser coverage, duration kind
distribution, and slide pattern distribution.

Latest local V1 regression results:

- Core samples: `16` samples, `64` difficulties, `0` pipeline failures,
  `0` validation errors, `0` unknown tokens, minimum parse coverage `1.0`.
- New-note samples: `14` samples, `56` difficulties, `0` pipeline failures,
  `0` validation errors, `0` unknown tokens, minimum parse coverage `1.0`.
  Parsed note distribution was `tap=23573`, `hold=2417`, `slide=2377`,
  `touch=2749`, and `touch_hold=91`.
- Combined `all` sample set: `30` samples, `120` difficulties, `0` pipeline
  failures, `0` validation errors, `0` unknown tokens, minimum parse coverage
  `1.0`.

Run the checks:

```bash
python scripts/v1_regression.py --sample-set core -o sample_runs/v1_regression_report.json
python scripts/v1_regression.py --sample-set new-notes -o sample_runs/v1_new_note_regression_report.json
python scripts/v1_regression.py --sample-set all -o sample_runs/v1_all_regression_report.json
```

See [docs/regression_samples.md](docs/regression_samples.md) for the report
fields and current sample-set definitions.

## Roadmap

V1 freeze target:

- Keep parser, ChartIR, stats, validation, CLI, tests, docs, and raw-preserving
  exporter stable.
- Preserve unknown syntax and raw source text.
- Avoid expanding into V2 features during V1 stabilization.

V2/V3 candidates:

- Broader Maidata/Simai syntax coverage.
- A normalized exporter with explicit limits and tests.
- A complete slide trajectory catalog with visual sampling tables.
- Supervised rhythm modeling based on V2 audio features, labels, alignment
  reports, and rule skeletons.
- AI generation only after parser, IR, cache, and feature foundations are
  stable.
