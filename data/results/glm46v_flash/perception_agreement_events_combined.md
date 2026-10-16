# Tier-2 perception scores: inter-rater agreement and score analysis

Clip kind: `combined`. Threshold for "adequate": mean rating >= 2 (`attribution_scoring.perception_adequate_min_score`).

**Missing sheets** for: ablate_translation, ablate_both -- run `09b_score_perception_adequacy_events.py aggregate --mode <mode> --clip-kind combined` first.

## 1. Inter-rater agreement on perception_score (0-2)

Per cue mode first: 55 items is the real unit of analysis, and each mode is a separate set of model descriptions being judged.

### `grounded` (translation + channel) (items=55, raters=2)

- Krippendorff's alpha (ordinal): 0.579
- Weighted kappa A01 vs A02 (n=55): 0.503
- Raw agreement A01 vs A02 (n=55): exact 0.673, within 1 0.982
- A01 marginal: mean 1.51, distribution {0: 3, 1: 21, 2: 31}
- A02 marginal: mean 1.38, distribution {0: 9, 1: 16, 2: 30}

### `ablate_channel` (translation only) (items=55, raters=2)

- Krippendorff's alpha (ordinal): 0.717
- Weighted kappa A01 vs A02 (n=55): 0.570
- Raw agreement A01 vs A02 (n=55): exact 0.618, within 1 1.000
- A01 marginal: mean 0.98, distribution {0: 18, 1: 20, 2: 17}
- A02 marginal: mean 0.96, distribution {0: 20, 1: 17, 2: 18}

### Pooled over all four cue modes

_Read second, not first: these rows are four ratings of the same 55 items (one per cue mode), so pooling raises n without adding items. Reported because it is the agreement figure over the whole rated set, not because it is a stronger reliability estimate than the per-mode numbers._

### all modes (items=110, raters=2)

- Krippendorff's alpha (ordinal): 0.674
- Weighted kappa A01 vs A02 (n=110): 0.570
- Raw agreement A01 vs A02 (n=110): exact 0.645, within 1 0.991
- A01 marginal: mean 1.25, distribution {0: 21, 1: 41, 2: 48}
- A02 marginal: mean 1.17, distribution {0: 29, 1: 33, 2: 48}


## 2. What the ratings say about the model's form descriptions

### Adequacy by cue mode

| mode | cues given | mean rating | per-rating adequate | per-item gated (mean >= 2) |
|---|---|---|---|---|
| `grounded` | translation + channel | 1.45 | 61/110 (55.5%) | 24/55 (43.6%) |
| `ablate_channel` | translation only | 0.97 | 35/110 (31.8%) | 11/55 (20.0%) |

### Mean rating by target channel (rows = 2 raters x 55 items per mode)

| target_channel | n items | `grounded` | `ablate_channel` |
|---|---|---|---|
| `manual` | 32 | 1.55 | 1.50 |
| `face` | 15 | 1.40 | 0.17 |
| `non_dominant_hand` | 7 | 1.00 | 0.43 |
| `head_torso` | 1 | 2.00 | 0.00 |

### Mean rating by whether the model's `channel_judged` matched `target_channel`

| mode | matched: n / mean | mismatched: n / mean |
|---|---|---|
| `grounded` | 98 / 1.56 | 12 / 0.50 |
| `ablate_channel` | 64 / 1.50 | 46 / 0.24 |
