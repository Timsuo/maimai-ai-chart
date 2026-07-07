# Slide Model

V1 treats a maimai slide as a chart object that references predefined maimai
trajectory patterns. A slide is not an arbitrary generated path or free-form
geometry.

## Definition

A slide token is modeled as:

```text
start position + predefined trajectory pattern + path arguments + duration
```

Examples:

- `1-4[8:1]`: start at `1`, use pattern `-`, path arg `4`, duration `[8:1]`.
- `2V84[8:1]`: start at `2`, use pattern `V`, path args `8` and `4`,
  duration `[8:1]`.
- `3b-1[8:1]*-7[8:1]`: one break slide note with two chained segments.

## Token Parts

- `start_position`: the button where the slide starts.
- `pattern`: a known maimai slide pattern such as `-`, `<`, `>`, `p`, `q`,
  `pp`, `qq`, `s`, `z`, `v`, `V`, or `w`.
- `path_args`: button numbers consumed by the pattern. Most V1 patterns use
  one path arg. `V` uses two.
- `duration`: a raw-preserving duration expression parsed by the unified
  duration parser.
- `segments`: chained slide pieces separated by top-level `*`.
- `raw`: original source text.

## ChartIR Shape

V1 stores slide details in `Note.segments`. Each segment has a `trajectory`
object:

```json
{
  "trajectory_id": "1-4",
  "pattern": "-",
  "start_position": 1,
  "end_position": 4,
  "path_args": [4],
  "raw": "1-4"
}
```

For `2V84[8:1]`, the trajectory is:

```json
{
  "trajectory_id": "2V84",
  "pattern": "V",
  "start_position": 2,
  "end_position": 4,
  "path_args": [8, 4],
  "raw": "2V84"
}
```

The final endpoint is retained for ordering and chain continuation, but the
semantics are the selected `V` trajectory plus both path args, not a free line
from `2` to `4`.

## Basic And Chained Slides

A basic slide has one segment. A chained slide is still one slide note, with
multiple ordered segments. Later segments infer their start position from the
previous segment endpoint.

Festival-era and other complex slide syntax should be treated as combinations
or chains of known patterns when V1 can identify every piece. V1 preserves the
raw text and `path_parts` for these compound segments; it does not invent new
geometry.

## V1 Supported Patterns

| Pattern | Category | Path args | Supported | Meaning |
| --- | --- | ---: | --- | --- |
| `-` | line | 1 | yes | Straight slide to one button |
| `<` | arc | 1 | yes | Counter-clockwise arc-like slide |
| `>` | arc | 1 | yes | Clockwise arc-like slide |
| `p` | curve | 1 | yes | Known p-shaped maimai slide |
| `q` | curve | 1 | yes | Known q-shaped maimai slide |
| `pp` | curve | 1 | yes | Known pp-shaped maimai slide |
| `qq` | curve | 1 | yes | Known qq-shaped maimai slide |
| `s` | zigzag | 1 | yes | Known s-shaped maimai slide |
| `z` | zigzag | 1 | yes | Known z-shaped maimai slide |
| `v` | composite | 1 | yes | Known lowercase-v slide |
| `V` | composite | 2 | yes | V-shaped slide with two path args |
| `w` | composite | 1 | yes | Known w-shaped maimai slide |

The source of truth for this table is `src/maichart/slides.py`.

## Not Supported In V1

V1 does not compute real screen coordinates, visual path samples, collision
geometry, slide animation timing, or normalized generated slide syntax.

Future generators must choose from a slide trajectory catalog. If visual path
sampling is needed, samples should come from a predefined trajectory table, not
from a model freely drawing arbitrary curves.
