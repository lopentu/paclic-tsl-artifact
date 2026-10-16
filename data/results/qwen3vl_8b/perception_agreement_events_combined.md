# Tier-2 perception scores: inter-rater agreement and score analysis

Clip kind: `combined`. Threshold for "adequate": mean rating >= 2 (`attribution_scoring.perception_adequate_min_score`).

## 1. Inter-rater agreement on perception_score (0-2)

Per cue mode first: 55 items is the real unit of analysis, and each mode is a separate set of model descriptions being judged.

### `grounded` (translation + channel) (items=55, raters=2)

- Krippendorff's alpha (ordinal): 0.329
- Weighted kappa A01 vs A02 (n=55): 0.279
- Raw agreement A01 vs A02 (n=55): exact 0.600, within 1 0.927
- A01 marginal: mean 1.62, distribution {0: 1, 1: 19, 2: 35}
- A02 marginal: mean 1.36, distribution {0: 8, 1: 19, 2: 28}

### `ablate_translation` (channel only) (items=55, raters=2)

- Krippendorff's alpha (ordinal): 0.241
- Weighted kappa A01 vs A02 (n=55): 0.208
- Raw agreement A01 vs A02 (n=55): exact 0.545, within 1 0.927
- A01 marginal: mean 1.60, distribution {1: 22, 2: 33}
- A02 marginal: mean 1.33, distribution {0: 9, 1: 19, 2: 27}

### `ablate_channel` (translation only) (items=55, raters=2)

- Krippendorff's alpha (ordinal): 0.607
- Weighted kappa A01 vs A02 (n=55): 0.493
- Raw agreement A01 vs A02 (n=55): exact 0.582, within 1 0.982
- A01 marginal: mean 1.02, distribution {0: 15, 1: 24, 2: 16}
- A02 marginal: mean 0.84, distribution {0: 23, 1: 18, 2: 14}

### `ablate_both` (neither (span only)) (items=55, raters=2)

- Krippendorff's alpha (ordinal): 0.618
- Weighted kappa A01 vs A02 (n=55): 0.525
- Raw agreement A01 vs A02 (n=55): exact 0.618, within 1 0.982
- A01 marginal: mean 1.05, distribution {0: 12, 1: 28, 2: 15}
- A02 marginal: mean 0.84, distribution {0: 23, 1: 18, 2: 14}

### Pooled over all four cue modes

_Read second, not first: these rows are four ratings of the same 55 items (one per cue mode), so pooling raises n without adding items. Reported because it is the agreement figure over the whole rated set, not because it is a stronger reliability estimate than the per-mode numbers._

### all modes (items=220, raters=2)

- Krippendorff's alpha (ordinal): 0.537
- Weighted kappa A01 vs A02 (n=220): 0.446
- Raw agreement A01 vs A02 (n=220): exact 0.586, within 1 0.955
- A01 marginal: mean 1.32, distribution {0: 28, 1: 93, 2: 99}
- A02 marginal: mean 1.09, distribution {0: 63, 1: 74, 2: 83}


## 2. What the ratings say about the model's form descriptions

### Adequacy by cue mode

| mode | cues given | mean rating | per-rating adequate | per-item gated (mean >= 2) |
|---|---|---|---|---|
| `grounded` | translation + channel | 1.49 | 63/110 (57.3%) | 23/55 (41.8%) |
| `ablate_translation` | channel only | 1.46 | 60/110 (54.5%) | 20/55 (36.4%) |
| `ablate_channel` | translation only | 0.93 | 30/110 (27.3%) | 11/55 (20.0%) |
| `ablate_both` | neither (span only) | 0.95 | 29/110 (26.4%) | 11/55 (20.0%) |

### Mean rating by target channel (rows = 2 raters x 55 items per mode)

| target_channel | n items | `grounded` | `ablate_translation` | `ablate_channel` | `ablate_both` |
|---|---|---|---|---|---|
| `manual` | 32 | 1.48 | 1.41 | 1.22 | 1.28 |
| `face` | 15 | 1.67 | 1.70 | 0.40 | 0.27 |
| `non_dominant_hand` | 7 | 1.07 | 1.14 | 0.57 | 0.71 |
| `head_torso` | 1 | 2.00 | 2.00 | 2.00 | 2.00 |

### Mean rating by whether the model's `channel_judged` matched `target_channel`

| mode | matched: n / mean | mismatched: n / mean |
|---|---|---|
| `grounded` | 110 / 1.49 | 0 / -- |
| `ablate_translation` | 110 / 1.46 | 0 / -- |
| `ablate_channel` | 66 / 1.30 | 44 / 0.36 |
| `ablate_both` | 64 / 1.33 | 46 / 0.41 |
