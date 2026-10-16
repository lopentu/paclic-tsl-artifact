# Tier-3 attribution: gated results, gate composition, and baselines

Clip kind: `combined`. All rows are gated on Tier 1 (span matched) and Tier 2 (mean perception_score >= 2).

## 1. Gate composition: which items reach Tier 3

The gated n differs per mode, so the cross-mode numbers in section 2 are partly a change of sample. This table is what makes that visible.

| target_channel | items rated | `grounded` | `ablate_translation` | `ablate_channel` | `ablate_both` |
|---|---|---|---|---|---|
| `manual` | 32 | 13 | 9 | 9 | 10 |
| `face` | 15 | 9 | 9 | 1 | 0 |
| `non_dominant_hand` | 7 | 0 | 1 | 0 | 0 |
| `head_torso` | 1 | 1 | 1 | 1 | 1 |
| **total** | **55** | **23** | **20** | **11** | **11** |

## 2. Per-mode gated results

`best const.` is the MAE of always answering the reference set's own mean or median (whichever is better) -- an MAE at or above this is no better than a constant guess. `distinct` is how many different values the model actually used.

| mode | cues | n | D1 MAE | D1 best const. | D1 bias | D1 rho (p) | D1 within .5 | D1 distinct | D2 MAE | D2 best const. | D2 bias | D2 rho (p) | D2 within .5 | D2 distinct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `grounded` | translation + channel | 23 | 0.98 | 0.76 | -0.67 | 0.81 (0.000) | 3/23 (13%) | 2 | 1.52 | 0.78 | +1.26 | -0.10 (0.651) | 4/23 (17%) | 3 |
| `ablate_translation` | channel only | 20 | 1.25 | 0.95 | -0.75 | 0.57 (0.009) | 2/20 (10%) | 3 | 1.40 | 0.95 | +0.95 | 0.03 (0.894) | 4/20 (20%) | 2 |
| `ablate_channel` | translation only | 11 | 1.05 | 0.23 | -0.95 | 0.67 (0.024) | 1/11 (9%) | 2 | 1.86 | 0.23 | +1.86 | -0.38 (0.249) | 2/11 (18%) | 3 |
| `ablate_both` | neither (span only) | 11 | 1.05 | 0.23 | -0.95 | 0.67 (0.024) | 1/11 (9%) | 2 | 1.68 | 0.23 | +1.68 | -0.34 (0.313) | 2/11 (18%) | 3 |

### Directional error rates and semiotic modes

| mode | n | D1 under-attrib. | D2 over-attrib. | description MAE / within .5 | indication MAE / within .5 | depiction MAE / within .5 |
|---|---|---|---|---|---|---|
| `grounded` | 23 | 17.4% | 65.2% | 0.91 / 7/23 | 0.78 / 13/23 | 0.33 / 18/23 |
| `ablate_translation` | 20 | 35.0% | 60.0% | 1.00 / 5/20 | 0.70 / 13/20 | 0.23 / 18/20 |
| `ablate_channel` | 11 | 9.1% | 72.7% | 1.18 / 1/11 | 1.27 / 4/11 | 0.18 / 9/11 |
| `ablate_both` | 11 | 9.1% | 72.7% | 0.86 / 2/11 | 1.27 / 4/11 | 0.14 / 10/11 |

## 3. Common subset: items passing the gate in all 4 modes (n=7)

Channel composition: `head_torso` 1, `manual` 6.

The only cue comparison here that holds the item set fixed. Small n; read as a consistency check on section 2, not as an independent result.

| mode | D1 MAE | D1 rho | D2 MAE | D2 rho | description MAE | indication MAE | depiction MAE |
|---|---|---|---|---|---|---|---|
| `grounded` | 0.93 | 1.00 | 1.50 | 0.25 | 1.00 | 1.43 | 0.29 |
| `ablate_translation` | 1.21 | 0.47 | 1.50 | 0.25 | 1.00 | 1.57 | 0.14 |
| `ablate_channel` | 1.07 | 0.65 | 1.50 | -0.27 | 1.29 | 1.43 | 0.14 |
| `ablate_both` | 1.07 | 0.65 | 1.36 | -0.20 | 1.00 | 1.43 | 0.14 |

## 4. Score distributions (model vs reference), gated rows

- `grounded` (n=23): D1 model mean 1.57 {1: 10, 2: 13}, reference mean 2.24; D2 model mean 2.04 {1: 1, 2: 20, 3: 2}, reference mean 0.78
- `ablate_translation` (n=20): D1 model mean 1.30 {0: 1, 1: 12, 2: 7}, reference mean 2.05; D2 model mean 1.90 {1: 2, 2: 18}, reference mean 0.95
- `ablate_channel` (n=11): D1 model mean 1.82 {1: 2, 2: 9}, reference mean 2.77; D2 model mean 2.09 {1: 2, 2: 6, 3: 3}, reference mean 0.23
- `ablate_both` (n=11): D1 model mean 1.82 {1: 2, 2: 9}, reference mean 2.77; D2 model mean 1.91 {1: 2, 2: 8, 3: 1}, reference mean 0.23

## 5. Semiotic-mode distributions (gated rows)

Read the MAEs in section 2 against these: when reference and model both sit at the same end of the scale, a low MAE reflects a shared floor rather than recovered signal.

| mode | semiotic mode | reference distribution | model distribution | rho (p) |
|---|---|---|---|---|
| `grounded` | description | {0: 3, 1: 2, 1.5: 2, 2: 2, 2.5: 2, 3: 12} | {1: 10, 2: 12, 3: 1} | 0.58 (0.004) |
| `grounded` | indication | {0: 16, 2: 1, 3: 6} | {0: 15, 1: 8} | 0.55 (0.007) |
| `grounded` | depiction | {0: 21, 0.5: 1, 3: 1} | {0: 19, 1: 4} | -0.14 (0.520) |
| `ablate_translation` | description | {0: 3, 1: 3, 1.5: 1, 2: 1, 2.5: 1, 3: 11} | {0: 1, 1: 12, 2: 5, 3: 2} | 0.61 (0.005) |
| `ablate_translation` | indication | {0: 15, 3: 5} | {0: 15, 1: 5} | 0.47 (0.038) |
| `ablate_translation` | depiction | {0: 18, 0.5: 1, 3: 1} | {0: 19, 1: 1} | -0.08 (0.749) |
| `ablate_channel` | description | {0: 3, 2: 1, 3: 7} | {1: 2, 2: 9} | 0.17 (0.607) |
| `ablate_channel` | indication | {0: 5, 2: 1, 3: 5} | {0: 6, 1: 5} | 0.57 (0.065) |
| `ablate_channel` | depiction | {0: 11} | {0: 9, 1: 2} | undefined (no variance) |
| `ablate_both` | description | {0: 2, 2: 1, 2.5: 1, 3: 7} | {1: 2, 2: 9} | 0.78 (0.005) |
| `ablate_both` | indication | {0: 6, 2: 1, 3: 4} | {0: 7, 1: 2, 2: 2} | 0.13 (0.702) |
| `ablate_both` | depiction | {0: 10, 0.5: 1} | {0: 10, 1: 1} | -0.10 (0.770) |

## 6. Channel-only baseline: does the model add anything to the channel label?

`channel-determined` = fraction of rows whose value equals the modal value for their target channel; for the model this is computed with no reference involved. `LOO channel-mean` predicts each item from the reference mean of the *other* gated items sharing its target channel, scored on the same rows as the model -- so it measures what is achievable from the channel label alone.

| mode | dim | model MAE | LOO channel-mean MAE | model rho | LOO channel-mean rho | model channel-determined | reference channel-determined |
|---|---|---|---|---|---|---|---|
| `grounded` | D1 | 0.98 | 0.54 | 0.81 | 0.61 | 100.0% | 73.9% |
| `grounded` | D2 | 1.52 | 0.53 | -0.10 | 0.51 | 87.0% | 69.6% |
| `ablate_translation` | D1 | 1.25 | 0.73 | 0.57 | 0.52 | 85.0% | 75.0% |
| `ablate_translation` | D2 | 1.40 | 0.70 | 0.03 | 0.42 | 90.0% | 65.0% |
| `ablate_channel` | D1 | 1.05 | 0.25 | 0.67 | -0.10 | 90.9% | 100.0% |
| `ablate_channel` | D2 | 1.86 | 0.29 | -0.38 | -0.84 | 63.6% | 90.9% |
| `ablate_both` | D1 | 1.05 | 0.23 | 0.67 | -- | 90.9% | 100.0% |
| `ablate_both` | D2 | 1.68 | 0.27 | -0.34 | -0.98 | 72.7% | 90.9% |

### Model score by target channel, all 55 rated items per mode (ungated)

Ungated on purpose: this is a description of what the model emitted, not a model-vs-reference accuracy claim, so the perception gate does not apply. The same 55 videos and spans are used in every mode -- only the cue text differs.

Two provenance facts, because this is the one section reading a file that `--suffix` does not select. (i) The `reference D1` column comes from whatever reference 10b was last run against for the `_ungated` files, not necessarily the one behind the gated sections above; both must be the adjudicated `events_checked.csv`, and if the manual/face/non-dominant-hand reference D1 below is not 3.00 / 1.13 / 2.14 they are stale and 10b needs re-running with `--events-csv .../events_checked.csv`. (ii) `_ungated` lifts the perception gate only -- 10b always applies the localization gate -- so a cell can hold fewer than 55 rows where the model's self-reported span drifted off target. Figures quoted over all 55 events regardless of span match come from `18_analyze_robustness_checks.py` section 8, not from here.

| target channel | items | reference D1 | model D1 `grounded` | model D1 `ablate_translation` | model D1 `ablate_channel` | model D1 `ablate_both` |
|---|---|---|---|---|---|---|
| `manual` | 32 | 3.00 | {1: 3, 2: 29} | {1: 12, 2: 20} | {1: 5, 2: 27} | {1: 7, 2: 24} (n=31) |
| `face` | 15 | 1.13 | {1: 15} | {0: 2, 1: 13} | {2: 15} | {1: 1, 2: 14} |
| `non_dominant_hand` | 7 | 2.14 | {1: 6, 2: 1} | {1: 6, 2: 1} | {1: 1, 2: 6} | {2: 6} (n=6) |
| `head_torso` | 1 | 0.50 | {1: 1} | {1: 1} | {1: 1} | {1: 1} |

## 7. Model score within $\pm$0.5 of the expert mean

With two experts scoring integers 0--3, the reference mean is an integer when they agreed and a .5 when they differed by one, so this counts the model landing inside the interval the two experts bracketed: an exact match where they agreed, either of their values where they did not.

| mode | n | D1 | D2 | description | indication | depiction |
|---|---|---|---|---|---|---|
| `grounded` | 23 | 3/23 (13%) | 4/23 (17%) | 7/23 (30%) | 13/23 (57%) | 18/23 (78%) |
| `ablate_translation` | 20 | 2/20 (10%) | 4/20 (20%) | 5/20 (25%) | 13/20 (65%) | 18/20 (90%) |
| `ablate_channel` | 11 | 1/11 (9%) | 2/11 (18%) | 1/11 (9%) | 4/11 (36%) | 9/11 (82%) |
| `ablate_both` | 11 | 1/11 (9%) | 2/11 (18%) | 2/11 (18%) | 4/11 (36%) | 10/11 (91%) |
