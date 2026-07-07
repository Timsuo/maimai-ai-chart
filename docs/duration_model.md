# Duration Model

V1 uses one raw-preserving duration parser for hold and slide durations.

## Supported Forms

| Example | Kind | Notes |
| --- | --- | --- |
| `[8:1]` | `grid_fraction` | `4 * 1 / 8` beats |
| `[4:3]` | `grid_fraction` | `3` beats |
| `[352:49]` | `grid_fraction` | exact `49/88` beats; may not map to integer ticks |
| `[#0.8057]` | `seconds` | explicit seconds; beats derived when BPM is known |
| `[0.6383##2.0455]` | `timing_pair` | both values preserved; V1 uses the second as conservative seconds duration |
| `[1:0]` | `grid_fraction` | zero beats; valid to parse, validation warning |

## Stored Fields

`DurationExpr` keeps:

- `raw`: original duration block.
- `kind`: parser classification.
- `beats`: exact beat fraction when known.
- `seconds`: seconds when known or derivable from BPM.
- `ticks`: integer ticks when the beat duration maps exactly.
- `values`: raw numeric pair values for `timing_pair`.

Non-integer tick durations are preserved with `ticks = null`; they should not
crash parsing or validation.

## Validation

Validation reports:

- errors for negative durations or malformed duration blocks in note-like
  tokens.
- warnings for zero-duration notes.
- warnings for grid or compound durations that cannot map to integer ticks.
- warnings for unknown tokens by default, or errors for unknown tokens in
  strict mode.

Old ChartIR JSON without a `duration` object remains readable because the core
note duration fields are optional.
