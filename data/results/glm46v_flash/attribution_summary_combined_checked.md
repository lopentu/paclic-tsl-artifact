# Tier-3 attribution: gated results, gate composition, and baselines

Clip kind: `combined`. All rows are gated on Tier 1 (span matched) and Tier 2 (mean perception_score >= 2).

## 1. Gate composition: which items reach Tier 3

The gated n differs per mode, so the cross-mode numbers in section 2 are partly a change of sample. This table is what makes that visible.

| target_channel | items rated | `grounded` | `ablate_channel` |
|---|---|---|---|
| `manual` | 32 | 13 | 11 |
| `face` | 15 | 9 | 0 |
| `non_dominant_hand` | 7 | 1 | 0 |
| `head_torso` | 1 | 1 | 0 |
| **total** | **55** | **24** | **11** |

## 2. Per-mode gated results

`best const.` is the MAE of always answering the reference set's own mean or median (whichever is better) -- an MAE at or above this is no better than a constant guess. `distinct` is how many different values the model actually used.

| mode | cues | n | D1 MAE | D1 best const. | D1 bias | D1 rho (p) | D1 within .5 | D1 distinct | D2 MAE | D2 best const. | D2 bias | D2 rho (p) | D2 within .5 | D2 distinct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `grounded` | translation + channel | 24 | 1.25 | 1.00 | -0.67 | 0.30 (0.153) | 3/24 (12%) | 2 | 1.29 | 0.92 | +1.04 | 0.60 (0.002) | 3/24 (12%) | 3 |
| `ablate_channel` | translation only | 11 | 1.45 | 0.00 | -1.45 | nan (nan) | 0/11 (0%) | 2 | 1.27 | 0.00 | +1.27 | nan (nan) | 0/11 (0%) | 2 |

### Directional error rates and semiotic modes

| mode | n | D1 under-attrib. | D2 over-attrib. | description MAE / within .5 | indication MAE / within .5 | depiction MAE / within .5 |
|---|---|---|---|---|---|---|
| `grounded` | 24 | 37.5% | 45.8% | 1.40 / 7/24 | 0.75 / 17/24 | 0.79 / 14/24 |
| `ablate_channel` | 11 | 45.5% | 27.3% | 1.27 / 2/11 | 1.18 / 6/11 | 0.73 / 8/11 |

## 3. Common subset: items passing the gate in all 2 modes (n=9)

Channel composition: `manual` 9.

The only cue comparison here that holds the item set fixed. Small n; read as a consistency check on section 2, not as an independent result.

| mode | D1 MAE | D1 rho | D2 MAE | D2 rho | description MAE | indication MAE | depiction MAE |
|---|---|---|---|---|---|---|---|
| `grounded` | 1.56 | nan | 1.56 | nan | 1.83 | 1.44 | 1.11 |
| `ablate_channel` | 1.44 | nan | 1.33 | nan | 1.39 | 1.44 | 0.89 |

## 4. Score distributions (model vs reference), gated rows

- `grounded` (n=24): D1 model mean 1.33 {1: 16, 2: 8}, reference mean 2.00; D2 model mean 1.96 {1: 5, 2: 15, 3: 4}, reference mean 0.92
- `ablate_channel` (n=11): D1 model mean 1.55 {1: 5, 2: 6}, reference mean 3.00; D2 model mean 1.27 {1: 8, 2: 3}, reference mean 0.00

## 5. Semiotic-mode distributions (gated rows)

Read the MAEs in section 2 against these: when reference and model both sit at the same end of the scale, a low MAE reflects a shared floor rather than recovered signal.

| mode | semiotic mode | reference distribution | model distribution | rho (p) |
|---|---|---|---|---|
| `grounded` | description | {0: 6, 1: 2, 1.5: 3, 2: 2, 2.5: 2, 3: 9} | {0: 17, 2: 7} | 0.25 (0.231) |
| `grounded` | indication | {0: 17, 2: 1, 3: 6} | {0: 22, 1: 2} | 0.49 (0.014) |
| `grounded` | depiction | {0: 19, 0.5: 2, 3: 3} | {0: 16, 1: 4, 2: 4} | 0.05 (0.801) |
| `ablate_channel` | description | {1.5: 1, 2: 1, 2.5: 1, 3: 8} | {0: 3, 2: 8} | 0.16 (0.629) |
| `ablate_channel` | indication | {0: 6, 2: 1, 3: 4} | {0: 10, 1: 1} | 0.39 (0.233) |
| `ablate_channel` | depiction | {0: 7, 0.5: 2, 3: 2} | {0: 9, 1: 2} | 0.04 (0.899) |

## 6. Channel-only baseline: does the model add anything to the channel label?

`channel-determined` = fraction of rows whose value equals the modal value for their target channel; for the model this is computed with no reference involved. `LOO channel-mean` predicts each item from the reference mean of the *other* gated items sharing its target channel, scored on the same rows as the model -- so it measures what is achievable from the channel label alone.

| mode | dim | model MAE | LOO channel-mean MAE | model rho | LOO channel-mean rho | model channel-determined | reference channel-determined |
|---|---|---|---|---|---|---|---|
| `grounded` | D1 | 1.25 | 0.46 | 0.30 | 0.70 | 66.7% | 75.0% |
| `grounded` | D2 | 1.29 | 0.45 | 0.60 | 0.73 | 62.5% | 83.3% |
| `ablate_channel` | D1 | 1.45 | 0.00 | -- | -- | 54.5% | 100.0% |
| `ablate_channel` | D2 | 1.27 | 0.00 | -- | -- | 72.7% | 100.0% |

### Model score by target channel, all 55 rated items per mode (ungated)

Ungated on purpose: this is a description of what the model emitted, not a model-vs-reference accuracy claim, so the perception gate does not apply. The same 55 videos and spans are used in every mode -- only the cue text differs.

Two provenance facts, because this is the one section reading a file that `--suffix` does not select. (i) The `reference D1` column comes from whatever reference 10b was last run against for the `_ungated` files, not necessarily the one behind the gated sections above; both must be the adjudicated `events_checked.csv`, and if the manual/face/non-dominant-hand reference D1 below is not 3.00 / 1.13 / 2.14 they are stale and 10b needs re-running with `--events-csv .../events_checked.csv`. (ii) `_ungated` lifts the perception gate only -- 10b always applies the localization gate -- so a cell can hold fewer than 55 rows where the model's self-reported span drifted off target. Figures quoted over all 55 events regardless of span match come from `18_analyze_robustness_checks.py` section 8, not from here.

| target channel | items | reference D1 | model D1 `grounded` | model D1 `ablate_channel` |
|---|---|---|---|---|
| `manual` | 32 | 3.00 | {1: 12, 2: 18} (n=30) | {1: 12, 2: 19} (n=31) |
| `face` | 15 | 1.13 | {0: 2, 1: 10, 2: 3} | {1: 5, 2: 9} (n=14) |
| `non_dominant_hand` | 7 | 2.00 | {1: 3} (n=3) | {1: 3, 2: 2} (n=5) |
| `head_torso` | 1 | 0.50 | {1: 1} | {1: 1} |

## 7. Model score within $\pm$0.5 of the expert mean

With two experts scoring integers 0--3, the reference mean is an integer when they agreed and a .5 when they differed by one, so this counts the model landing inside the interval the two experts bracketed: an exact match where they agreed, either of their values where they did not.

| mode | n | D1 | D2 | description | indication | depiction |
|---|---|---|---|---|---|---|
| `grounded` | 24 | 3/24 (12%) | 3/24 (12%) | 7/24 (29%) | 17/24 (71%) | 14/24 (58%) |
| `ablate_channel` | 11 | 0/11 (0%) | 0/11 (0%) | 2/11 (18%) | 6/11 (55%) | 8/11 (73%) |
