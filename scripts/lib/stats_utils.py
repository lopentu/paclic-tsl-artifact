"""Agreement and error metrics used across the scoring scripts (the project design log ).

Ordinal agreement (weighted kappa, Krippendorff's alpha) rather than raw accuracy, per the
doc's explicit instruction not to treat the two-dimensional scores as nominal categories.
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer


def weighted_cohen_kappa(ratings_a, ratings_b, weights="linear"):
    """Weighted Cohen's kappa for two annotators' ordinal ratings on the same items."""
    return cohen_kappa_score(ratings_a, ratings_b, weights=weights)


def krippendorff_alpha_ordinal(value_matrix, level_of_measurement="ordinal"):
    """Krippendorff's alpha, ordinal by default.

    value_matrix: list of rows, one per annotator, one column per item, np.nan for missing.
    `level_of_measurement="interval"` is the right choice for the continuous 0.0-1.0 fields
    (confidence, semantic_dependence), where the 0-3 ordinal treatment would discard the fact
    that 0.9 vs 0.8 is a smaller disagreement than 0.9 vs 0.1.

    Requires the `krippendorff` package (pip install krippendorff). Raises a clear error
    if it is not installed rather than silently falling back to a different statistic.
    """
    try:
        import krippendorff
    except ImportError as exc:
        raise ImportError(
            "krippendorff_alpha_ordinal requires `pip install krippendorff`"
        ) from exc
    data = np.array(value_matrix, dtype=float)
    return krippendorff.alpha(reliability_data=data, level_of_measurement=level_of_measurement)


def jaccard_index(set_a, set_b):
    set_a, set_b = set(set_a), set(set_b)
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def multilabel_micro_f1(labels_a, labels_b, all_labels):
    """Micro-F1 between two annotators' multi-label form-tag assignments (list of label-lists)."""
    mlb = MultiLabelBinarizer(classes=sorted(all_labels))
    y_a = mlb.fit_transform(labels_a)
    y_b = mlb.transform(labels_b)
    return f1_score(y_a, y_b, average="micro", zero_division=0)


def mean_absolute_error(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def spearman_correlation(y_true, y_pred):
    if len(set(y_true)) < 2 or len(set(y_pred)) < 2:
        return float("nan"), float("nan")
    rho, p = spearmanr(y_true, y_pred)
    return float(rho), float(p)


def categorical_agreement(labels_a, labels_b):
    """Exact-match accuracy and (nominal) Cohen's kappa between two aligned label sequences.

    General-purpose helper for any nominal label agreement (unweighted kappa, unlike the ordinal
    D1/D2/semiotic-mode kappa). Returns (accuracy, kappa); kappa is nan when fewer than two
    distinct labels are present (undefined) or sklearn declines.
    """
    n = len(labels_a)
    if n == 0:
        return float("nan"), float("nan")
    acc = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    if len(set(labels_a) | set(labels_b)) < 2:
        return acc, float("nan")
    try:
        kappa = float(cohen_kappa_score(labels_a, labels_b))
    except Exception:
        kappa = float("nan")
    return acc, kappa


def interval_overlap_ratio(a_start, a_end, b_start, b_end):
    """Intersection-over-union of two [start, end) time intervals."""
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0
