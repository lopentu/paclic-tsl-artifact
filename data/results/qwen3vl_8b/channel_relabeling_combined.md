# Does the channel cue redirect the model, or only relabel what it already saw?

Clip kind: `combined`. Model outputs: `qwen3vl_8b`. No human rating is used, so this is independent of Tier 2/3 and of the perception gate.

## 1. Similarity of the described ACTION with vs without the channel cue

Token-Jaccard and character-level ratio over `handshape`+`movement`+`location` (the fields that identify which action was seen). The words `hand`/`hands`/`signer` are excluded from the token measure so that a pure change of LABEL cannot register as a change in what was described.

### `grounded` vs `ablate_channel` (translation given in both)

| target_channel | n | mean token-Jaccard | mean char ratio | identical `movement` string |
|---|---|---|---|---|
| `manual` | 32 | 0.648 | 0.640 | 7/32 |
| `face` | 15 | 0.098 | 0.177 | 1/15 |
| `non_dominant_hand` | 7 | 0.501 | 0.326 | 2/7 |
| `head_torso` | 1 | 0.227 | 0.225 | 0/1 |

### `ablate_translation` vs `ablate_both` (translation withheld in both)

| target_channel | n | mean token-Jaccard | mean char ratio | identical `movement` string |
|---|---|---|---|---|
| `manual` | 32 | 0.641 | 0.650 | 10/32 |
| `face` | 15 | 0.032 | 0.101 | 0/15 |
| `non_dominant_hand` | 7 | 0.341 | 0.212 | 0/7 |
| `head_torso` | 1 | 0.238 | 0.329 | 0/1 |

## 2. Item-level detail, `non_dominant_hand` targets

### `grounded` vs `ablate_channel`

| item | Jaccard | identical `movement` | `hands` (grounded) | `hands` (ablate_channel) | `channel_judged` given/withheld |
|---|---|---|---|---|---|
| C01/event1 | 0.82 | yes | non-dominant hand | dominant hand | non_dominant_hand / manual |
| C06/event1 | 0.36 | no | non-dominant hand | Both hands | non_dominant_hand / manual |
| C08/event1 | 0.68 | yes | non-dominant hand | dominant hand | non_dominant_hand / manual |
| C12/event1 | 0.31 | no | non-dominant hand | Both hands | non_dominant_hand / manual |
| C13/event1 | 0.41 | no | non-dominant hand | dominant hand and non-dominant hand | non_dominant_hand / manual |
| C36/event1 | 0.61 | no | non-dominant hand | Both hands | non_dominant_hand / manual |
| C38/event1 | 0.31 | no | non-dominant hand | dominant hand (right) and non-dominant hand (left) | non_dominant_hand / manual |

### `ablate_translation` vs `ablate_both`

| item | Jaccard | identical `movement` | `hands` (ablate_translation) | `hands` (ablate_both) | `channel_judged` given/withheld |
|---|---|---|---|---|---|
| C01/event1 | 0.38 | no | non-dominant hand | dominant hand | non_dominant_hand / manual |
| C06/event1 | 0.29 | no | non-dominant hand | Both hands | non_dominant_hand / manual |
| C08/event1 | 0.41 | no | non-dominant hand | dominant hand | non_dominant_hand / manual |
| C12/event1 | 0.20 | no | non-dominant hand | Both hands | non_dominant_hand / manual |
| C13/event1 | 0.29 | no | non-dominant hand | dominant hand and non-dominant hand | non_dominant_hand / manual |
| C36/event1 | 0.52 | no | non-dominant hand | Both hands | non_dominant_hand / manual |
| C38/event1 | 0.30 | no | non-dominant hand | dominant hand (right) and non-dominant hand (left) | non_dominant_hand / manual |

## 2b. Decisive test: does the cued description match the withheld run's self-selected percept, or its own non-dominant-hand report?

For each event: **X** = the channel-given run's target-action text (`handshape`+`movement`+`location`); **A** = the channel-withheld run's target-action text (the articulator it self-selected); **B** = the same withheld run's own `non_dominant_hand` field. `J(X,A) > J(X,B)` means the cued description tracks the self-selected percept rather than the model's own report of the non-dominant hand. A and B come from the same run, which removes the "withheld run described both hands" confound in section 1.

### `grounded` vs `ablate_channel` (translation given in both)

| target_channel | n | mean J(X,A) self-selected percept | mean J(X,B) own NDH report | J(X,A) > J(X,B) |
|---|---|---|---|---|
| `manual` | 32 | 0.648 | 0.074 | 32/32 |
| `face` | 15 | 0.098 | 0.029 | 10/15 |
| `non_dominant_hand` | 7 | 0.501 | 0.111 | 7/7 |
| `head_torso` | 1 | 0.227 | 0.067 | 1/1 |

Per item, `non_dominant_hand` targets:

| item | J(X,A) | J(X,B) |
|---|---|---|
| C01/event1 | 0.82 | 0.00 |
| C06/event1 | 0.36 | 0.21 |
| C08/event1 | 0.68 | 0.06 |
| C12/event1 | 0.31 | 0.10 |
| C13/event1 | 0.41 | 0.05 |
| C36/event1 | 0.61 | 0.25 |
| C38/event1 | 0.31 | 0.12 |

### `ablate_translation` vs `ablate_both` (translation withheld in both)

| target_channel | n | mean J(X,A) self-selected percept | mean J(X,B) own NDH report | J(X,A) > J(X,B) |
|---|---|---|---|---|
| `manual` | 32 | 0.641 | 0.065 | 32/32 |
| `face` | 15 | 0.032 | 0.044 | 7/15 |
| `non_dominant_hand` | 7 | 0.341 | 0.069 | 7/7 |
| `head_torso` | 1 | 0.238 | 0.062 | 1/1 |

Per item, `non_dominant_hand` targets:

| item | J(X,A) | J(X,B) |
|---|---|---|
| C01/event1 | 0.38 | 0.00 |
| C06/event1 | 0.29 | 0.00 |
| C08/event1 | 0.41 | 0.15 |
| C12/event1 | 0.20 | 0.12 |
| C13/event1 | 0.29 | 0.00 |
| C36/event1 | 0.52 | 0.08 |
| C38/event1 | 0.30 | 0.14 |

## 3. The `non_dominant_hand` field: independent report, or restatement of the target action?

Token-Jaccard between the run's own `non_dominant_hand` field and its `handshape`+`movement` (high = the field restates the target action rather than reporting a second articulator), and how often that field uses resting/supporting vocabulary.

### `grounded`

| target_channel | n | NDH-field vs target-action Jaccard | NDH field says resting/supporting |
|---|---|---|---|
| `manual` | 32 | 0.082 | 24/32 |
| `face` | 15 | 0.152 | 15/15 |
| `non_dominant_hand` | 7 | 0.579 | 3/7 |
| `head_torso` | 1 | 0.083 | 1/1 |

### `ablate_translation`

| target_channel | n | NDH-field vs target-action Jaccard | NDH field says resting/supporting |
|---|---|---|---|
| `manual` | 32 | 0.090 | 21/32 |
| `face` | 15 | 0.193 | 15/15 |
| `non_dominant_hand` | 7 | 0.274 | 1/7 |
| `head_torso` | 1 | 0.083 | 1/1 |

### `ablate_channel`

| target_channel | n | NDH-field vs target-action Jaccard | NDH field says resting/supporting |
|---|---|---|---|
| `manual` | 32 | 0.103 | 23/32 |
| `face` | 15 | 0.106 | 10/15 |
| `non_dominant_hand` | 7 | 0.193 | 5/7 |
| `head_torso` | 1 | 0.167 | 1/1 |

### `ablate_both`

| target_channel | n | NDH-field vs target-action Jaccard | NDH field says resting/supporting |
|---|---|---|---|
| `manual` | 32 | 0.101 | 23/32 |
| `face` | 15 | 0.123 | 10/15 |
| `non_dominant_hand` | 7 | 0.165 | 5/7 |
| `head_torso` | 1 | 0.182 | 1/1 |

## 4. `hands` field (what the model says is performing the action)

- `grounded`: **manual** -> both hands 16, dominant hand 16; **face** -> not visible / N/A 15; **non_dominant_hand** -> non-dominant hand 7; **head_torso** -> not visible / N/A 1

- `ablate_translation`: **manual** -> both hands 16, dominant hand 16; **face** -> not visible / N/A 15; **non_dominant_hand** -> non-dominant hand 7; **head_torso** -> not visible / N/A 1

- `ablate_channel`: **manual** -> both hands 17, dominant hand 14, not visible / N/A 1; **face** -> both hands 10, dominant hand 4, not visible / N/A 1; **non_dominant_hand** -> both hands 3, dominant hand 2, non-dominant hand 2; **head_torso** -> hands clasped together in fr 1

- `ablate_both`: **manual** -> both hands 17, dominant hand 14, not visible / N/A 1; **face** -> both hands 10, dominant hand 5; **non_dominant_hand** -> both hands 3, dominant hand 2, non-dominant hand 2; **head_torso** -> hands clasped or held togeth 1
