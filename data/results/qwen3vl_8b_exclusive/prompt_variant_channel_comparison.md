# Prompt-robustness check: CHANNEL_WITHHELD articulator-list wording

Main dataset tag: `qwen3vl_8b` (the reported prompt). Variant tag: `qwen3vl_8b_exclusive`.
Scope: `clip_kind=combined`, the two CHANNEL_WITHHELD modes only (the variant is byte-identical to the main prompt under CHANNEL_GIVEN).

## `ablate_channel` (n=55)

| | main prompt | variant prompt |
|---|---|---|
| `channel_judged` matches manifest | 33/55 (60%) | 33/55 (60%) |
| selected `non_dominant_hand` | 0 | 0 |
| schema-valid | 55/55 | 54/55 |
| distribution | {'manual': 52, 'face': 2, 'head_torso': 1} | {'manual': 47, 'face': 7, None: 1} |

`channel_judged` changed on **6/55** items (the manipulation's power -- it did move judgments):

- C21/event2: `manual` -> `None` (manifest target: `manual`)
- C22/event1: `head_torso` -> `face` (manifest target: `head_torso`)
- C27/event1: `manual` -> `face` (manifest target: `face`)
- C31/event1: `manual` -> `face` (manifest target: `face`)
- C35/event2: `manual` -> `face` (manifest target: `manual`)
- C39/event1: `manual` -> `face` (manifest target: `face`)

The 7 `non_dominant_hand`-target events:

- C01/event1: main `manual`, variant `manual`
- C06/event1: main `manual`, variant `manual`
- C08/event1: main `manual`, variant `manual`
- C12/event1: main `manual`, variant `manual`
- C13/event1: main `manual`, variant `manual`
- C36/event1: main `manual`, variant `manual`
- C38/event1: main `manual`, variant `manual`

Sanity check only, **not an attribution result** (this set is not perception-gated or human-rated): D1 mean 1.87 -> 1.91 (identical on 48/55); D2 mean 2.11 -> 2.02 (identical on 42/55).

## `ablate_both` (n=55)

| | main prompt | variant prompt |
|---|---|---|
| `channel_judged` matches manifest | 32/55 (58%) | 33/55 (60%) |
| selected `non_dominant_hand` | 0 | 0 |
| schema-valid | 55/55 | 55/55 |
| distribution | {'manual': 53, 'face': 1, 'head_torso': 1} | {'manual': 49, 'face': 6} |

`channel_judged` changed on **5/55** items (the manipulation's power -- it did move judgments):

- C22/event1: `head_torso` -> `face` (manifest target: `head_torso`)
- C26/event1: `manual` -> `face` (manifest target: `face`)
- C27/event1: `manual` -> `face` (manifest target: `face`)
- C31/event1: `manual` -> `face` (manifest target: `face`)
- C35/event2: `manual` -> `face` (manifest target: `manual`)

The 7 `non_dominant_hand`-target events:

- C01/event1: main `manual`, variant `manual`
- C06/event1: main `manual`, variant `manual`
- C08/event1: main `manual`, variant `manual`
- C12/event1: main `manual`, variant `manual`
- C13/event1: main `manual`, variant `manual`
- C36/event1: main `manual`, variant `manual`
- C38/event1: main `manual`, variant `manual`

Sanity check only, **not an attribution result** (this set is not perception-gated or human-rated): D1 mean 1.84 -> 1.73 (identical on 47/55); D2 mean 2.02 -> 1.93 (identical on 42/55).
