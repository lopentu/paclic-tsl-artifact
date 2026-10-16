# Uncapped Tier-2 re-scoring of the articulator-mismatched rows

Appendix A Step 1 caps a Tier-2 row at <=1 when `channel_judged` != `target_channel`. In the two articulator-withheld cells the model chose the articulator itself, so the cap partly pre-determines the mediation claim. These are the same rows re-scored with Step 1 suspended: how good the form description is about the articulator it actually describes.

**These scores were collected unblinded** -- the raters re-scored inside the merged sheet, with `target_channel` and their own capped score visible on the row. The anchoring diagnostic below is what bounds that exposure.

## Adequacy per cell, capped vs un-capped (paper's gate: mean of both raters $\ge$ 2)

| Cell | events | adequate capped | adequate un-capped | mismatched events clearing the gate |
|---|---|---|---|---|
| `ablate_channel` | 55 | 11/55 = 0.200 | 13/55 = 0.236 | 2/22 (C12, C13) |
| `ablate_both` | 55 | 11/55 = 0.200 | 13/55 = 0.236 | 2/23 (C13, C39) |

Compare `grounded` at 23/55 = 0.418. Un-capping barely moves the withheld cells, so the Step-1 cap was not producing the cue effect.

## Anchoring diagnostic (the unblinding exposure)

| rater | rows | uncapped == capped | uncapped > capped | uncapped < capped |
|---|---|---|---|---|
| A01 | 45 | 26 (57.8%) | 13 | 6 |
| A02 | 45 | 25 (55.6%) | 20 | 0 |

A row scored *down* is the clearest sign the rater was judging description quality rather than re-deriving the cap: the cap only ever pushes a score toward <=1, so it cannot explain a decrease.

## Inter-rater agreement on the uncapped score

- Exact agreement: 31/45 = 0.689
- Within 1 point: 45/45 = 1.000

## Uncapped adequacy by the withheld `target_channel`

Both raters pooled. The model answered MANUAL on nearly all these rows, so this asks: when it defaulted to MANUAL instead of the specified articulator, how good was the manual description it produced?

| withheld target | rows | adequate uncapped |
|---|---|---|
| `face` | 58 | 8/58 = 0.138 |
| `non_dominant_hand` | 28 | 8/28 = 0.286 |
| `manual` | 4 | 0/4 = 0.000 |
