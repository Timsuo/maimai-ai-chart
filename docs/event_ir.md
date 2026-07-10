# EventIR

EventIR is an additive event-level representation for future chart generation,
validation, export, and visualization work. It does not replace the V1 ChartIR
parser output, frame-label builder, exporter, dataset pipeline, or training
code.

## Relationship To ChartIR

ChartIR remains the source of truth for current V1 parsing and existing
training flows. EventIR is derived from ChartIR in Phase A2 with
`chart_ir_to_event_ir`.

The first EventIR implementation preserves the existing metadata, difficulty
metadata, explicit BPM events, parsed note events, unknown tokens, and raw
source text where available. It does not add inferred information that ChartIR
does not already know.

## Event Types

EventIR stores note-like objects in `ChartEventIR.events`:

- `TapEvent`
- `HoldEvent`
- `SlideEvent`
- `TouchEvent`
- `TouchHoldEvent`

Timing changes are stored separately in `ChartEventIR.timing_events`.
`ChartEventIR.meter_events` is present for future meter support, but the current
converter leaves it empty because ChartIR does not model true meter events.

## SlideEvent

`SlideEvent` separates the slide star head from its motion payload without
duplicating the head as a separate `TapEvent`.

- `head_tick`: tick where the star head appears and is tapped.
- `start_position`: star head button position.
- `launch_offset_ticks`: delay from head tick to motion start. `None` means
  unknown, not one beat and not zero.
- `travel_duration_ticks`: motion duration when ChartIR has a reliable duration.
- `segments`: ordered `SlideSegment` list.
- `end_position`: final segment endpoint. For chained slides, this is the last
  segment endpoint, not the first segment endpoint.
- `raw_notation`: original slide token when available.
- `duration_raw`, `duration_kind`, and `timing_pair_values`: raw-preserving
  duration metadata from ChartIR.

`SlideSegment` keeps the segment start position, path type, path arguments,
endpoint, segment duration, raw notation, and compound `path_parts` when
available.

## Chained Slides

A chained slide such as:

```text
1-4[8:1]*-6[8:1]
```

is represented as one `SlideEvent` with two `SlideSegment` entries. The first
segment ends at `4`, the second starts at `4` and ends at `6`, so the event
`end_position` is `6`.

## Current Limits

Phase A1/A2 intentionally does not:

- infer slide launch offsets;
- synthesize a default `wholebpm` timing event when ChartIR has no explicit BPM;
- synthesize a default 4/4 `MeterEvent`;
- simulate maimai touch-area judgment paths or sample visual slide geometry;
- generate frame labels from EventIR;
- export EventIR back to Maidata;
- decode model outputs to EventIR.

Those steps are reserved for later phases after EventIR has stable tests and
validation rules.

## EventIR To frame_labels_v2

Phase A3-prep adds a side-path builder:

```python
build_frame_labels_from_event_ir(chart, division=16, slide_launch_policy="legacy")
```

The output schema is:

```text
maichart-frame-labels-v2
```

This builder keeps the V1-compatible label fields so old ChartIR labels and
EventIR labels can be compared frame by frame:

- `has_note`
- `note_count`
- `tap_count`
- `break_count`
- `hold_start_count`
- `hold_active_count`
- `slide_start_count`
- `slide_active_count`
- `touch_count`
- `touch_hold_start_count`
- `note_types`
- `positions`
- `slide_patterns`
- `duration_kinds`
- `has_validation_warning`
- `warning_codes`

It does not replace `build_frame_labels_from_chart_ir`, manifest cache building,
or training data loading.

## Slide Lifecycle Labels

frame_labels_v2 adds lifecycle fields for slide-specific analysis:

- `note_start_count`: all note-like event starts in the frame.
- `slide_head_count`: slide star heads in the frame.
- `slide_motion_start_count`: slide motion starts in the frame.
- `slide_motion_active_count`: frames covered by slide motion.
- `slide_motion_end_count`: slide motion ends in the frame.
- `slide_launch_offset_kinds`: launch offset source labels.
- `slide_travel_duration_kinds`: travel duration source labels.
- `slide_unknown_launch_offset_count`: slide heads whose launch offset is not
  known from EventIR.

These labels describe event lifecycle timing only. They do not simulate maimai
touch-area judgment paths or sample visual trajectory geometry.

## slide_launch_policy

`slide_launch_policy` controls how `SlideEvent.launch_offset_ticks=None` is
handled:

- `legacy`: treat unknown launch offset as zero ticks. This is the default and
  is intended to align with legacy ChartIR frame labels as closely as possible.
- `unknown`: do not generate slide motion coverage for unknown launch offsets.
  The slide head is still counted, and `slide_unknown_launch_offset_count`
  records the missing lifecycle data.
- `default_one_beat`: treat unknown launch offset as `TICKS_PER_BEAT`. This is
  an experiment mode, not the default training strategy.

If a slide has an explicit `launch_offset_ticks`, all policies use that explicit
value and label the source as `explicit`.

## Why A3-prep Stays Side-path

EventIR labels are currently for comparison and validation. They should not be
fed into the existing training pipeline yet because most slides derived from
legacy ChartIR have `launch_offset_ticks=None`. The `legacy` policy can match
old labels, but it preserves old assumptions; the `unknown` policy exposes the
missing lifecycle information; and `default_one_beat` is only a controlled
experiment.

Before replacing caches or training data, run comparison reports over real
manifests and decide how launch offset should be obtained or inferred.

## Phase A3-validation results

Phase A3-validation compared legacy ChartIR frame labels with side-path EventIR
frame_labels_v2. The comparison tool writes per-run reports and aggregate CSV
tables under `reports/event_label_compare_quick` and
`reports/event_label_compare_large_sample`.

The compare tool supports:

```text
--slide-launch-policy legacy|unknown|default_one_beat
--limit N
```

It also writes `diff_details.csv` with `song_id`, difficulty, frame index,
field name, V1/V2 values, and nearby raw note/event text when any compared field
differs.

### Quick split

The quick split used:

- `manifests/experiments/id_batch_v1_qc_fixed/train.json`
- `manifests/experiments/id_batch_v1_qc_fixed/val.json`
- `manifests/experiments/id_batch_v1_qc_fixed/test.json`

Results:

| Split | Policy | Samples | Frames | Diff frames | Slide heads | Motion active | Unknown launch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | legacy | 35 | 44587 | 0 | 2567 | 8433 | 0 |
| train | unknown | 35 | 44587 | 7343 slide-active | 2567 | 0 | 2567 |
| train | default_one_beat | 35 | 44627 | 7741 slide-active | 2567 | 8433 | 0 |
| val | legacy | 8 | 11804 | 0 | 353 | 1766 | 0 |
| val | unknown | 8 | 11804 | 1548 slide-active | 353 | 0 | 353 |
| val | default_one_beat | 8 | 11824 | 1156 slide-active | 353 | 1766 | 0 |
| test | legacy | 7 | 9946 | 0 | 466 | 1687 | 0 |
| test | unknown | 7 | 9946 | 1517 slide-active | 466 | 0 | 466 |
| test | default_one_beat | 7 | 9954 | 1439 slide-active | 466 | 1687 | 0 |

In the quick split, `legacy` reproduces the compared V1-compatible fields with
zero diffs. `unknown` shows that all slide heads derived from current ChartIR
lack explicit launch offset data. `default_one_beat` keeps the same amount of
motion-active coverage as legacy, but shifts it later, which changes the
frame-by-frame `slide_active_count` distribution.

### Larger split sample

The larger sample used `--limit 100` for:

- `train_exclude_id_batch.json`
- `val_exclude_id_batch.json`
- `test_id_batch.json`

Results:

| Split | Policy | Samples | Frames | Diff frames | Slide heads | Motion active | Unknown launch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train_exclude_id_batch | legacy | 100 | 93377 | 0 | 2040 | 8121 | 0 |
| train_exclude_id_batch | unknown | 100 | 93377 | 7362 slide-active | 2040 | 0 | 2040 |
| train_exclude_id_batch | default_one_beat | 100 | 93484 | 9202 slide-active | 2040 | 8121 | 0 |
| val_exclude_id_batch | legacy | 100 | 109889 | 0 | 2818 | 10805 | 0 |
| val_exclude_id_batch | unknown | 100 | 109889 | 9708 slide-active | 2818 | 0 | 2818 |
| val_exclude_id_batch | default_one_beat | 100 | 110019 | 11608 slide-active | 2818 | 10804 | 0 |
| test_id_batch | legacy | 50 | 66337 | 0 | 3386 | 11886 | 0 |
| test_id_batch | unknown | 50 | 66337 | 10408 slide-active | 3386 | 0 | 3386 |
| test_id_batch | default_one_beat | 50 | 66405 | 10336 slide-active | 3386 | 11886 | 0 |

The larger sample confirms the same pattern: `legacy` can reproduce the
compared V1-compatible labels, while `unknown` reports a 1.0 unknown launch
offset ratio because current EventIR is derived from ChartIR and ChartIR does
not carry explicit launch offsets. `default_one_beat` changes the temporal
placement of slide motion coverage and sometimes extends the frame sequence.

### Recommendation

It is reasonable to proceed to the real A3 cache-generation design as a
side-path only. Do not connect frame_labels_v2 to training yet. The next step
should generate optional EventIR/frame_labels_v2 caches and keep validating
launch-offset policy choices before any dataset or model consumes them.
