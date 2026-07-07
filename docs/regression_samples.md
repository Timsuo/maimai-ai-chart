# Regression Samples

The V1 regression script is `scripts/v1_regression.py`. It runs the same
pipeline for each available sample and difficulty:

1. metadata parse
2. timing tokenize
3. notes parse
4. ChartIR build
5. ChartIR JSON serialization
6. stats
7. validate

Missing samples are skipped. A sample that exists but fails any pipeline stage
is reported in `pipeline_failure_details`.

## Sample Sets

Core V1 samples:

```text
417 513 786 799 803 812 820 825 833 834 432 471 496 773 777 835
```

New-note samples:

```text
11821 11898 11956 11051 11106 11152 11184 11206 11222 11228 11448 11451 11818 11820
```

These new-note samples include touch and touch-hold syntax. They are useful
for checking that the parser does not regress on post-classic note elements
while still preserving the V1 boundary.

## Latest Local Results

Core sample report: `sample_runs/v1_regression_report.json`

- sample_count: 16
- difficulty_count: 64
- pipeline_failures: 0
- validate_errors: 0
- unknown_token_count: 0
- min_parse_coverage: 1.0

New-note sample report: `sample_runs/v1_new_note_regression_report.json`

- sample_count: 14
- difficulty_count: 56
- pipeline_failures: 0
- validate_errors: 0
- unknown_token_count: 0
- min_parse_coverage: 1.0
- touch_note_count: 2840
- note_type_distribution: `tap=23573`, `hold=2417`, `slide=2377`,
  `touch=2749`, `touch_hold=91`

Combined report: `sample_runs/v1_all_regression_report.json`

- sample_count: 30
- difficulty_count: 120
- pipeline_failures: 0
- validate_errors: 0
- unknown_token_count: 0
- min_parse_coverage: 1.0
- touch_note_count: 2840

Validation warnings are allowed for V1 when they represent preserved but
non-fatal semantics such as zero durations or non-integer tick durations.
Unknown tokens are still expected to be zero for these curated local samples.
