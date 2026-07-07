# V2.1 Frame Labels

Frame labels convert one ChartIR difficulty into a fixed beat-grid sequence for
later machine-learning data preparation. The output is still chart-only data:
it does not include audio features, beat detection, onset detection, model
training, inference, chart generation, or normalized export.

## CLI

Build labels for one ChartIR JSON file:

```bash
maichart labels build path/to/difficulty_5.chart_ir.json -o path/to/difficulty_5.frame_labels.json --division 16
```

Build labels for every difficulty listed in a dataset manifest:

```bash
maichart dataset labels sample_runs/v2_dataset_manifest.json --out-dir sample_runs/v2_frame_labels --division 16
```

The manifest command adds `frame_labels_path` to each successfully processed
difficulty. By default it updates the input manifest in place. Pass `-o` to
write the updated manifest to another file.

## Grid

The default `division` is `16`, meaning 16 frames per 4/4 measure. Since V1 uses
`TICKS_PER_BEAT = 1920`, the frame size is:

```text
ticks_per_frame = TICKS_PER_BEAT * 4 / division
```

For `division=16`:

```text
ticks_per_frame = 1920 * 4 / 16 = 480
```

Tick is the primary alignment unit. Frame index is stable and reproducible:

```text
frame_index = floor(note_tick / ticks_per_frame)
```

Beat is stored as a string fraction such as `"0"`, `"1/4"`, or `"129/4"`.
`time_sec` is estimated from ChartIR BPM events when available.

## Schema

Top-level payload:

```json
{
  "schema": "maichart-frame-labels-v1",
  "song_id": "513",
  "difficulty": 5,
  "grid": {
    "division": 16,
    "ticks_per_beat": 1920,
    "ticks_per_frame": 480
  },
  "frames": []
}
```

Each frame:

```json
{
  "frame_index": 0,
  "beat": "0",
  "tick": 0,
  "time_sec": 0.0,
  "labels": {
    "has_note": false,
    "note_count": 0,
    "tap_count": 0,
    "break_count": 0,
    "hold_start_count": 0,
    "hold_active_count": 0,
    "slide_start_count": 0,
    "slide_active_count": 0,
    "touch_count": 0,
    "touch_hold_start_count": 0,
    "note_types": [],
    "positions": [],
    "slide_patterns": [],
    "duration_kinds": [],
    "has_validation_warning": false,
    "warning_codes": []
  }
}
```

## Label Fields

- `has_note`: true when the frame has a note start or an active hold/slide.
- `note_count`: number of note starts in this frame.
- `tap_count`: tap starts, including break taps.
- `break_count`: note starts carrying a break modifier.
- `hold_start_count`: hold starts.
- `hold_active_count`: hold or touch-hold duration coverage across frames.
- `slide_start_count`: slide starts.
- `slide_active_count`: slide duration coverage across frames.
- `touch_count`: touch tap starts.
- `touch_hold_start_count`: touch hold starts.
- `note_types`: unique note types starting in this frame.
- `positions`: unique start positions in this frame.
- `slide_patterns`: unique slide segment patterns starting in this frame.
- `duration_kinds`: unique duration kinds from the note or slide segments.
- `has_validation_warning`: reserved for future validation-aware labels.
- `warning_codes`: reserved warning code list.

Tap, break, hold, slide, chained slide, touch, touch hold, seconds duration,
timing-pair duration, zero-duration hold, and non-integer tick duration inputs
are all accepted without crashing. Active duration labels use integer ticks when
available, otherwise they conservatively estimate from beats or seconds.

## Next Step

V2.2 adds audio feature extraction beside these labels. V2.3 should read
`frame_labels_path` and `audio_features_path` from the dataset manifest and
produce a chart-audio alignment report.
