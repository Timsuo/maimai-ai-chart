# ChartIR Schema

`ChartIR` is the source of truth after parsing a Maidata/Simai-like chart. The
schema is intentionally small for V1 and keeps raw source text available so the
parser and exporter can preserve details that are not normalized yet.

## Top-Level Object

`ChartIR`

- `schema_version`: version of the IR schema, currently `1`.
- `metadata`: chart-level metadata.
- `difficulty`: one chart difficulty.
- `timing`: BPM, grid/division, and hspeed events.
- `notes`: ordered note objects.
- `unknown_tokens`: raw note tokens that V1 preserved because they were not
  understood.
- `raw`: optional raw source text for the whole chart or relevant source block.

## Metadata

`ChartMetadata`

- `title`: chart title.
- `artist`: song artist.
- `audio_file`: source audio filename or path from the chart metadata.
- `background_file`: background image or movie filename/path.
- `offset`: chart offset in seconds.
- `raw`: optional raw metadata text.

## Difficulty

`DifficultyMetadata`

- `index`: numeric difficulty index when available.
- `name`: difficulty name, such as `Basic`, `Advanced`, `Expert`, `Master`, or a
  fan-chart-specific label.
- `level`: displayed level string.
- `designer`: chart designer.
- `raw`: optional raw difficulty metadata text.

## Timing Events

Timing events all carry the shared time fields:

- `time_sec`: seconds from chart start when known.
- `beat`: beat position when known.
- `tick`: integer tick position for ordering and equality checks.
- `raw`: optional source token or line.

`TimingData`

- `bpms`: `BpmEvent` entries with `bpm`.
- `grids`: `GridEvent` entries with `division`.
- `hspeeds`: `HSpeedEvent` entries with `speed`.

## Notes

`Note` supports these `note_type` values:

- `tap`
- `hold`
- `slide`
- `touch`
- `touch_hold`

Each note has the shared time fields plus:

- `position`: start position, button, or touch area.
- `end_position`: end position for slides or other moving notes.
- `duration_sec`: note duration in seconds when known.
- `duration_beats`: note duration in beats when known.
- `duration_ticks`: note duration in integer ticks when known.
- `duration`: optional raw-preserving duration expression object with `raw`,
  `kind`, `beats`, `seconds`, `ticks`, and `values`.
- `path`: optional slide path or touch movement tokens.
- `segments`: structured slide segments. For slide notes, each segment contains
  a `trajectory` reference with `trajectory_id`, `pattern`, `start_position`,
  `end_position`, `path_args`, and `raw`.
- `modifiers`: optional structured details that should not become core fields
  yet.
- `raw`: optional raw source token.

For slide notes, `segments` is the preferred V1 shape. Legacy compatibility
fields such as `path`, `modifiers.slide_pattern`, and
`modifiers.slide_segments` are retained for existing callers. A slide segment
references a predefined maimai trajectory pattern; it does not describe an
arbitrary generated curve. See `docs/slide_model.md`.

For touch notes, `position` stores raw touch lanes such as `A1`, `B5`, `E4`, or
`C`. Touch firework markers are preserved in `modifiers.firework`.

## Unknown Tokens

`UnknownChartToken` stores:

- `raw_token`: original token text.
- `timing_index`: timing point index.
- `beat`, `tick`, and `time_sec`: timing context when known.
- `raw_timing_point`: original timing point text.
- `reason`: parser reason.

V1 allows unknown tokens because preserving raw syntax is safer than silently
dropping unsupported chart content.

The schema should not depend on MajdataView or MajdataPlay internals. Those
tools can be used later for compatibility validation.
