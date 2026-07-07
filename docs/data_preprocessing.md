# V2 Data Preprocessing

This stage turns raw converted maimai chart samples into a QC'd training
manifest for later V2.5 supervised learning. It does not train models, run
inference, generate charts, or export normalized Maidata.

## Raw Dataset Layout

Each song should normally live in its own directory:

```text
raw_dataset/
  A000_music000101/
    convert_report.json
    maidata.txt
    track.mp3
```

`track.wav` and `track.ogg` are also accepted. `bg.png`, `bg.jpg`, and other
conversion artifacts may be present and are recorded when useful.

## Conversion Status Is Not Dataset Usability

`convert_report.json` describes the converter run, not the current sample
state. A converter may report `status="failed"` because `maidata.txt` or
`track.mp3` already existed. If the files are present and readable by the
parser/audio analyzer, the sample can still be `dataset_usable=true`.

The preprocessing manifests therefore keep three concepts separate:

- `conversion_status`: converter status from `convert_report.json`.
- `dataset_usable`: raw files needed by the dataset exist.
- `usable_for_training`: one difficulty passed parser, cache, audio, labels,
  validation, and QC filters.

Only difficulties with `usable_for_training=true` are model-ready.

## Commands

Scan raw samples without heavy analysis:

```bash
maichart preprocess scan raw_dataset -o manifests/raw_manifest.json
```

Build caches and the training manifest:

```bash
maichart preprocess build raw_dataset \
  --cache-dir cache \
  -o manifests/training_manifest.json \
  --division 16
```

Useful debug options:

```bash
maichart preprocess build raw_dataset --cache-dir cache -o manifests/training_manifest.json --limit 100
maichart preprocess build raw_dataset --cache-dir cache -o manifests/training_manifest.json --force
maichart preprocess build raw_dataset --cache-dir cache -o manifests/training_manifest.json --skip-audio --skip-alignment
```

Build leakage-safe splits:

```bash
maichart preprocess split manifests/training_manifest.json \
  --out-dir manifests/splits \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 42 \
  --split-by-song
```

## Cache Layout

`preprocess build` writes stable cache paths:

```text
cache/
  chart_ir/<song_id>/difficulty_<index>.chart_ir.json
  frame_labels/<song_id>/difficulty_<index>.frame_labels.json
  audio_features/<song_id>.audio_features.json
  alignment_reports/<song_id>/difficulty_<index>.alignment_report.json
```

Readable caches are reused unless `--force` is passed. Corrupt cache files are
rebuilt when possible. One failed song or difficulty is recorded in `errors`
and does not stop the whole batch.

## Raw Manifest

The raw scan schema is `maichart-raw-sample-manifest-v1`. Each sample records
paths, file presence, `conversion_status`, `dataset_usable`,
`dataset_usable_reasons`, warnings, and errors. Duplicate `song_id` values get
stable suffixes and warnings instead of silently overwriting entries.

## Training Manifest

The training schema is `maichart-training-manifest-v1`. Song entries contain
metadata, source paths, conversion/dataset usability, and audio QC:

- `readable`
- `duration_sec`
- `estimated_bpm`
- `metadata_bpm`
- `bpm_delta`
- `audio_features_path`

Difficulty entries contain:

- `difficulty_index`, `difficulty_name`, `level_raw`, `level`, `designer`
- `chart_ir_path`, `frame_labels_path`, `alignment_report_path`
- note counts, type counts, parser coverage, unknown token count
- validation error/warning counts
- duration and slide-pattern counts
- alignment summary
- `usable_for_training`, `training_weight`, `filter_reasons`, `warning_codes`

The top-level `summary` includes raw counts, usable counts, filter/warning
histograms, distributions, failure counts, and lists such as failed conversion
but usable samples, `level=?` samples, missing audio, and parse failures.

## Filters

A difficulty is filtered out when any required condition fails:

- missing or empty chart
- non-numeric level such as `?`
- note count below 20
- validation errors
- parse coverage below 1.0
- unknown tokens
- unreadable/missing audio
- frame-label generation failure
- audio feature generation failure

`level=?` is kept in the manifest as `level_raw="?"`, but `level=null` and
`usable_for_training=false`.

## Training Weights

Default weights by difficulty name:

```text
easy:      0.25
basic:    0.35
advanced: 0.6
expert:   1.0
master:   1.1
remaster: 1.1
```

Difficulty indexes map to names as `1 easy`, `2 basic`, `3 advanced`,
`4 expert`, and `5 master`. Validation warnings and very low onset hit rate
can reduce the final weight.

## Splits

Splits default to song-level grouping. All usable difficulties for one
`song_id` stay in the same split, preventing train/test leakage across
difficulties of the same song. Split files only contain
`usable_for_training=true` difficulties.
