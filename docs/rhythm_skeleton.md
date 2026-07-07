# V2.4 Rhythm Skeleton

The rhythm skeleton generator is a rule-based baseline. It answers two early
generation questions:

```text
Which times in the song are good note candidates?
What broad event type should each candidate use?
```

It does not generate button positions, touch regions, slide patterns, hold
durations, full ChartIR, Maidata text, neural-network predictions, or a GUI.

## Rhythm Profile

A rhythm profile is built from an existing dataset manifest with frame labels
and alignment reports:

```bash
maichart rhythm profile sample_runs/v2_dataset_manifest.json -o sample_runs/v2_rhythm_profile.json
```

The profile groups difficulties into level bands and stores:

- average note density per second
- selected frame ratio
- event type distribution for `tap`, `break`, `hold_start`, `slide_start`, and `touch`
- mean 50 ms onset hit rate when alignment reports are available
- sample count per level band

Schema:

```json
{
  "schema": "maichart-rhythm-profile-v1",
  "level_bands": [
    {
      "min_level": 12.0,
      "max_level": 14.0,
      "target_note_density_per_sec": 4.0,
      "target_selected_frame_ratio": 0.35,
      "event_type_distribution": {
        "tap": 0.75,
        "break": 0.04,
        "hold_start": 0.08,
        "slide_start": 0.11,
        "touch": 0.02
      },
      "onset_hit_rate_50ms": 0.6,
      "sample_count": 12
    }
  ],
  "global_stats": {}
}
```

## Generate Skeleton

Generate from audio features:

```bash
maichart rhythm generate sample_runs/v2_audio_features/maichart_513.audio_features.json --level 12.0 --profile sample_runs/v2_rhythm_profile.json -o sample_runs/v2_rhythm_skeleton_513.json
```

Each audio feature frame receives an explainable score:

```text
score =
  0.45 * normalized_onset_strength
+ 0.25 * normalized_percussive_rms
+ 0.15 * normalized_rms
+ 0.15 * beat_proximity_bonus
```

A small onset-proximity bonus is added when the frame is close to a detected
onset. The target note count comes from the rhythm profile for the requested
level, or from an internal fallback density table. A minimum time interval is
applied so the baseline does not select overly dense clusters.

Event types are broad labels only:

- `tap`
- `break`
- `hold_start`
- `slide_start`
- `touch`

They are assigned by simple suitability rules and the target distribution. The
output is not a full playable chart.

## Skeleton Schema

```json
{
  "schema": "maichart-rhythm-skeleton-v1",
  "song_id": "maichart_513",
  "target_level": 12.5,
  "division": 16,
  "duration_sec": 135.2,
  "summary": {
    "selected_frame_count": 480,
    "estimated_note_density_per_sec": 3.55,
    "event_type_counts": {
      "tap": 420,
      "break": 20,
      "hold_start": 20,
      "slide_start": 20,
      "touch": 0
    }
  },
  "frames": [
    {
      "frame_index": 128,
      "time_sec": 32.0,
      "beat": "64",
      "tick": null,
      "selected": true,
      "event_type": "tap",
      "score": 0.87,
      "reasons": ["near_onset", "strong_percussive_rms", "beat_aligned"]
    }
  ]
}
```

## Evaluate Skeleton

Evaluate against reference frame labels:

```bash
maichart rhythm evaluate sample_runs/v2_rhythm_skeleton_513.json sample_runs/v2_frame_labels/maichart_513/difficulty_5.frame_labels.json -o sample_runs/v2_rhythm_eval_513.json
```

Evaluation fields:

- `predicted_selected_frames`: selected skeleton frames
- `reference_note_frames`: human-chart frames with notes
- `precision`: selected frames matching a reference note frame
- `recall`: reference note frames matched by the skeleton
- `f1`: harmonic mean of precision and recall
- `note_count_error`: predicted minus reference frame count
- `density_error`: predicted density minus reference density
- `event_type_distribution_difference`: predicted distribution minus reference distribution
- `onset_supported_rate`: selected frames justified by onset-related reasons

## Dataset Batch

Generate skeletons for a manifest:

```bash
maichart dataset rhythm sample_runs/v2_dataset_manifest.json --out-dir sample_runs/v2_rhythm_skeletons --profile sample_runs/v2_rhythm_profile.json --level-source difficulty
```

Batch mode reads each song's `audio_features_path` and each difficulty's
`level`. It writes `rhythm_skeleton_path` to each difficulty. Missing audio
features are skipped; individual failures are recorded without stopping the
batch.

## Limitations

This is intentionally conservative. It does not learn style, infer hand
patterns, assign positions, generate holds or slide geometry, or know maimai
playability. It is a transparent baseline and an evaluation target for future
models.

## Next Step

V2.5 can build a supervised rhythm model that predicts skeleton frames from
audio features and uses this rule baseline as a comparison point.
