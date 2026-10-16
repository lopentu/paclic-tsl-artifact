# Prompt-robustness check: CHANNEL_WITHHELD articulator-list wording

Main dataset tag: `qwen3vl_8b` (the reported prompt). Variant tag: `qwen3vl_8b_altlist`.
Scope: `clip_kind=combined`, the two CHANNEL_WITHHELD modes only (the variant is byte-identical to the main prompt under CHANNEL_GIVEN).

## `ablate_channel` (n=55)

| | main prompt | variant prompt |
|---|---|---|
| `channel_judged` matches manifest | 33/55 (60%) | 31/55 (56%) |
| selected `non_dominant_hand` | 0 | 0 |
| schema-valid | 55/55 | 55/55 |
| distribution | {'manual': 52, 'face': 2, 'head_torso': 1} | {'head_torso': 8, 'manual': 47} |

`channel_judged` changed on **7/55** items (the manipulation's power -- it did move judgments):

- C01/event1: `manual` -> `head_torso` (manifest target: `non_dominant_hand`)
- C02/event2: `manual` -> `head_torso` (manifest target: `manual`)
- C21/event1: `face` -> `head_torso` (manifest target: `manual`)
- C26/event1: `face` -> `head_torso` (manifest target: `face`)
- C27/event1: `manual` -> `head_torso` (manifest target: `face`)
- C31/event1: `manual` -> `head_torso` (manifest target: `face`)
- C39/event1: `manual` -> `head_torso` (manifest target: `face`)

The 7 `non_dominant_hand`-target events:

- C01/event1: main `manual`, variant `head_torso`
- C06/event1: main `manual`, variant `manual`
- C08/event1: main `manual`, variant `manual`
- C12/event1: main `manual`, variant `manual`
- C13/event1: main `manual`, variant `manual`
- C36/event1: main `manual`, variant `manual`
- C38/event1: main `manual`, variant `manual`

Sanity check only, **not an attribution result** (this set is not perception-gated or human-rated): D1 mean 1.87 -> 1.85 (identical on 50/55); D2 mean 2.11 -> 2.05 (identical on 43/55).

## `ablate_both` (n=55)

| | main prompt | variant prompt |
|---|---|---|
| `channel_judged` matches manifest | 32/55 (58%) | 30/55 (55%) |
| selected `non_dominant_hand` | 0 | 0 |
| schema-valid | 55/55 | 54/55 |
| distribution | {'manual': 53, 'face': 1, 'head_torso': 1} | {'manual': 47, 'head_torso': 7, None: 1} |

`channel_judged` changed on **7/55** items (the manipulation's power -- it did move judgments):

- C02/event2: `manual` -> `head_torso` (manifest target: `manual`)
- C21/event1: `face` -> `head_torso` (manifest target: `manual`)
- C21/event2: `manual` -> `None` (manifest target: `manual`)
- C26/event1: `manual` -> `head_torso` (manifest target: `face`)
- C27/event1: `manual` -> `head_torso` (manifest target: `face`)
- C31/event1: `manual` -> `head_torso` (manifest target: `face`)
- C39/event1: `manual` -> `head_torso` (manifest target: `face`)

The 7 `non_dominant_hand`-target events:

- C01/event1: main `manual`, variant `manual`
- C06/event1: main `manual`, variant `manual`
- C08/event1: main `manual`, variant `manual`
- C12/event1: main `manual`, variant `manual`
- C13/event1: main `manual`, variant `manual`
- C36/event1: main `manual`, variant `manual`
- C38/event1: main `manual`, variant `manual`

Sanity check only, **not an attribution result** (this set is not perception-gated or human-rated): D1 mean 1.84 -> 1.67 (identical on 45/55); D2 mean 2.02 -> 1.96 (identical on 46/55).
