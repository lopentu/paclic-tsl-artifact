# Output stability under sampling (T = 1.0, k = 5)

55 events x 2 cells x 5 draws = 550 sampled runs, against the greedy (T = 0) runs the paper reports.

Every item supplies 5 draws in every cell: no gaps in coverage.

**Schema validity under sampling: 548/550 (99.6%).** The greedy runs are 220/220. 2 sampled run(s) emitted unparseable JSON (C21__event2__combined__sample1__ablate_channel.json, C33__event1__combined__sample4__ablate_channel.json) and are dropped from every readout below; the schema is enforced identically in both settings, so this is a property of sampling, not of the parser.

## 1. Is D1 articulator-determined in the distribution?

`modal(greedy)` is the target articulator's modal D1 among the greedy runs -- the value the paper's 'articulator-determined' counts are computed against. `draws at modal` pools every draw from every item of that articulator.


**grounded**

| target articulator | n | modal(greedy) | draws at modal | mean per-item modal share | mean entropy (bits) | values seen |
|---|---|---|---|---|---|---|
| manual | 32 | 2 | 138/160 = 86.2% | 0.91 | 0.25 | 1/2 |
| face | 15 | 1 | 57/75 = 76.0% | 0.76 | 0.65 | 0/1/2 |
| non_dominant_hand | 7 | 1 | 24/35 = 68.6% | 0.83 | 0.48 | 1/2 |
| head_torso | 1 | 1 | 3/5 = 60.0% | 0.60 | 1.37 | 0/1/2 |

**ablate_channel**

| target articulator | n | modal(greedy) | draws at modal | mean per-item modal share | mean entropy (bits) | values seen |
|---|---|---|---|---|---|---|
| manual | 32 | 2 | 139/158 = 88.0% | 0.92 | 0.25 | 1/2/3 |
| face | 15 | 2 | 66/75 = 88.0% | 0.88 | 0.37 | 1/2/3 |
| non_dominant_hand | 7 | 2 | 31/35 = 88.6% | 0.89 | 0.28 | 1/2 |
| head_torso | 1 | 1 | 4/5 = 80.0% | 0.80 | 0.72 | 1/2 |

## 2. Is the greedy output the sampled mode?

If it is, the paper's T=0 runs are representative draws rather than an artifact of the decoding rule.

| cell | score | greedy == sampled mode | mean per-item modal share | mean entropy |
|---|---|---|---|---|
| grounded | D1 | 51/55 | 0.85 | 0.41 |
| grounded | D2 | 47/55 | 0.80 | 0.57 |
| grounded | description | 49/55 | 0.79 | 0.59 |
| grounded | indication | 45/55 | 0.83 | 0.49 |
| grounded | depiction | 49/55 | 0.86 | 0.43 |
| ablate_channel | D1 | 51/55 | 0.90 | 0.30 |
| ablate_channel | D2 | 44/55 | 0.85 | 0.49 |
| ablate_channel | description | 48/55 | 0.84 | 0.46 |
| ablate_channel | indication | 42/55 | 0.78 | 0.65 |
| ablate_channel | depiction | 44/55 | 0.80 | 0.59 |

## 3. The face flip under sampling

Greedy: the 15 face targets score D1=1 on 15/15 when the face is named and D1=2 on 15/15 when the articulator is withheld. Pooling draws turns that into a difference of distributions.

| target articulator | cell | mean D1 | D1 distribution over draws | channel_judged = cued/self-selected |
|---|---|---|---|---|
| manual | grounded | 1.86 | 1:22, 2:138 | 160/160 at cued |
| manual | ablate_channel | 1.89 | 1:18, 2:139, 3:1 | 153/158 at manual |
| face | grounded | 1.13 | 0:4, 1:57, 2:14 | 75/75 at cued |
| face | ablate_channel | 1.91 | 1:8, 2:66, 3:1 | 70/75 at manual |
| non_dominant_hand | grounded | 1.31 | 1:24, 2:11 | 35/35 at cued |
| non_dominant_hand | ablate_channel | 1.89 | 1:4, 2:31 | 35/35 at manual |
| head_torso | grounded | 1.00 | 0:1, 1:3, 2:1 | 5/5 at cued |
| head_torso | ablate_channel | 1.20 | 1:4, 2:1 | 5/5 at head_torso |

## 4. Does sampling reach the endpoints the greedy runs never use?

The paper reports that the model never emits D1=0 or D1=3, and flags that a compressed mapping is confounded with a failure to anchor an unfamiliar scale. If the endpoints are absent from 440 undamped draws too, the compression is a property of the distribution.

| cell | score | draws | 0 | 1 | 2 | 3 | endpoint share |
|---|---|---|---|---|---|---|---|
| grounded | D1 | 275 | 5 | 106 | 164 | 0 | 1.8% |
| grounded | D2 | 275 | 3 | 20 | 199 | 53 | 20.4% |
| grounded | description | 275 | 5 | 93 | 160 | 17 | 8.0% |
| grounded | indication | 275 | 193 | 53 | 18 | 11 | 74.2% |
| grounded | depiction | 275 | 232 | 37 | 6 | 0 | 84.4% |
| ablate_channel | D1 | 273 | 0 | 34 | 237 | 2 | 0.7% |
| ablate_channel | D2 | 273 | 0 | 21 | 194 | 58 | 21.2% |
| ablate_channel | description | 273 | 0 | 29 | 224 | 20 | 7.3% |
| ablate_channel | indication | 273 | 146 | 88 | 27 | 12 | 57.9% |
| ablate_channel | depiction | 273 | 189 | 65 | 17 | 2 | 70.0% |

## 5. Items whose D1 is not unanimous across draws


**grounded**: 25/55 items split across draws:

| item | articulator | draws |
|---|---|---|
| C02 event1 | manual | 2, 2, 2, 1, 1 |
| C04 event1 | face | 1, 1, 0, 0, 1 |
| C05 event1 | face | 1, 1, 1, 0, 1 |
| C07 event1 | manual | 2, 2, 1, 2, 2 |
| C12 event1 | non_dominant_hand | 2, 2, 2, 1, 2 |
| C13 event1 | non_dominant_hand | 1, 1, 2, 1, 1 |
| C14 event1 | face | 1, 1, 0, 1, 1 |
| C16 event1 | manual | 1, 2, 2, 2, 1 |
| C16 event2 | manual | 1, 1, 2, 1, 2 |
| C19 event2 | manual | 2, 1, 2, 1, 2 |
| C20 event2 | manual | 2, 2, 2, 1, 2 |
| C21 event1 | manual | 1, 2, 1, 1, 2 |
| C22 event1 | head_torso | 1, 1, 2, 1, 0 |
| C25 event1 | face | 1, 2, 1, 1, 1 |
| C26 event1 | face | 2, 1, 1, 2, 1 |
| C27 event1 | face | 2, 1, 1, 2, 1 |
| C28 event1 | manual | 2, 2, 1, 2, 2 |
| C31 event1 | face | 1, 2, 1, 2, 1 |
| C34 event1 | face | 1, 1, 2, 1, 1 |
| C35 event2 | manual | 2, 1, 2, 2, 1 |
| C36 event1 | non_dominant_hand | 1, 2, 2, 1, 2 |
| C37 event1 | face | 1, 1, 1, 2, 2 |
| C37 event2 | face | 1, 1, 1, 2, 2 |
| C38 event1 | non_dominant_hand | 2, 1, 1, 2, 2 |
| C39 event2 | face | 1, 1, 2, 1, 2 |

**ablate_channel**: 20/55 items split across draws:

| item | articulator | draws |
|---|---|---|
| C01 event1 | non_dominant_hand | 1, 2, 2, 2, 1 |
| C03 event1 | face | 2, 2, 2, 1, 2 |
| C04 event1 | face | 2, 1, 2, 1, 2 |
| C08 event1 | non_dominant_hand | 2, 2, 2, 1, 1 |
| C10 event1 | manual | 1, 2, 2, 2, 2 |
| C11 event1 | manual | 1, 1, 1, 1, 2 |
| C15 event1 | face | 2, 1, 2, 2, 2 |
| C16 event2 | manual | 1, 1, 2, 1, 1 |
| C18 event2 | manual | 2, 2, 1, 2, 2 |
| C19 event2 | manual | 1, 2, 2, 1, 2 |
| C20 event2 | manual | 2, 1, 2, 2, 2 |
| C21 event1 | manual | 1, 2, 2, 1, 2 |
| C21 event2 | manual | 2, 3, 2, 2 |
| C22 event1 | head_torso | 1, 2, 1, 1, 1 |
| C24 event2 | manual | 2, 1, 2, 1, 2 |
| C25 event1 | face | 2, 2, 2, 2, 1 |
| C26 event1 | face | 2, 2, 2, 2, 1 |
| C28 event1 | manual | 2, 1, 2, 2, 2 |
| C31 event1 | face | 1, 2, 2, 1, 2 |
| C37 event1 | face | 2, 2, 3, 2, 2 |

## 6. Stated confidence

| cell | mean stated confidence over draws |
|---|---|
| grounded | 0.93 |
| ablate_channel | 0.94 |
