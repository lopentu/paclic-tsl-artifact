# Inter-annotator agreement

# Part 1 -- reference scores (annotators' own judgments)

Items with >=2 annotators: 55

## D1 (semantic/syntactic load), 0-3
- Krippendorff's alpha (ordinal, 2 annotators): 0.953
- Weighted kappa A01 vs A02 (n=55): 0.876
- Raw agreement A01 vs A02 (n=55): exact 0.909, within 1 0.964

## D2 (pragmatic/depictive/affective enrichment), 0-3
- Krippendorff's alpha (ordinal, 2 annotators): 0.932
- Weighted kappa A01 vs A02 (n=55): 0.864
- Raw agreement A01 vs A02 (n=55): exact 0.891, within 1 0.982

## Semiotic mode: description, 0-3
- Krippendorff's alpha (ordinal, 2 annotators): 0.855
- Weighted kappa A01 vs A02 (n=55): 0.794
- Raw agreement A01 vs A02 (n=55): exact 0.800, within 1 0.964

## Semiotic mode: indication, 0-3
- Krippendorff's alpha (ordinal, 2 annotators): 0.972
- Weighted kappa A01 vs A02 (n=55): 0.983
- Raw agreement A01 vs A02 (n=55): exact 0.982, within 1 1.000

## Semiotic mode: depiction, 0-3
- Krippendorff's alpha (ordinal, 2 annotators): 0.742
- Weighted kappa A01 vs A02 (n=55): 0.827
- Raw agreement A01 vs A02 (n=55): exact 0.945, within 1 1.000

## Confidence (optional, 0.0-1.0)
- Krippendorff's alpha (interval, 2 annotators): 0.095
- Raw agreement A01 vs A02 (n=55): exact 0.236, within 0.2 0.509

## Semantic dependence (optional, 0.0-1.0)
- Krippendorff's alpha (interval, 2 annotators): -0.031
- Raw agreement A01 vs A02 (n=55): exact 0.109, within 0.2 0.364


# Part 2 -- ratings of the MODEL's form descriptions (mode=grounded)

_Agreement on the 0-2 perception_score each rater gave the model's form description. Since 2026-07-27 this is where description-side agreement comes from: the annotators no longer write competing free-text descriptions, they rate one shared text, so the number is interpretable._

- Not yet available: `perception_scores_grounded.csv` has no item rated by >=2 raters with a `rater_id`. Build the sheet with `09_score_perception_adequacy.py build --mode grounded`, have each rater fill their own copy (setting `rater_id`), then `aggregate` them.
