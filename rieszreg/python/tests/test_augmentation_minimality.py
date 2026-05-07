"""Pin the per-subject augmented-row counts for built-in estimands.

`Estimand.augment` should emit exactly the rows the loss needs and no more.
For ATT and LocalShift, subjects whose multiplicative factor (`a` or
`1{a < threshold}`) is zero contribute only one augmented row (the original
observation), not phantom counterfactual rows with zero coefficients.
"""

from __future__ import annotations

import collections

import numpy as np

from rieszreg import ATE, ATT, LocalShift


def _make_features(a: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.column_stack([a.astype(float), x.astype(float)])


def test_ate_two_rows_per_subject():
    """ATE always touches both treatment levels → 2 augmented rows per subject."""
    rng = np.random.default_rng(0)
    n = 50
    a = rng.binomial(1, 0.5, n).astype(float)
    x = rng.uniform(0, 1, n)
    aug = ATE().augment(_make_features(a, x))

    assert aug.features.shape[0] == 2 * n
    counts = collections.Counter(aug.origin_index.tolist())
    assert set(counts.values()) == {2}, counts


def test_att_treated_two_rows_untreated_one_row():
    """ATT (partial) skips counterfactuals for untreated subjects.

    Untreated rows have `a_i = 0`, so `m(α)(z) = a · (α(1, x) − α(0, x))`
    contributes only the original row. Treated rows contribute the original
    plus the (0, x) counterfactual.

    Total expected augmented rows = n + n_treated.
    """
    rng = np.random.default_rng(0)
    n = 200
    a = rng.binomial(1, 0.5, n).astype(float)
    x = rng.uniform(0, 1, n)
    n_treated = int(a.sum())

    aug = ATT().augment(_make_features(a, x))
    assert aug.features.shape[0] == n + n_treated, (
        f"expected {n + n_treated} augmented rows (1 per untreated, 2 per "
        f"treated); got {aug.features.shape[0]}"
    )

    counts = collections.Counter(aug.origin_index.tolist())
    treated_counts = {counts[i] for i in range(n) if a[i] == 1}
    untreated_counts = {counts[i] for i in range(n) if a[i] == 0}
    assert treated_counts == {2}, f"treated row counts: {treated_counts}"
    assert untreated_counts == {1}, f"untreated row counts: {untreated_counts}"

    for i in range(n):
        if a[i] == 0:
            mask = aug.origin_index == i
            j = int(np.where(mask)[0][0])
            assert aug.is_original[j] == 1.0
            assert aug.potential_deriv_coef[j] == 0.0


def test_local_shift_skips_above_threshold_subjects():
    """LocalShift multiplies by 1{a < threshold}, so subjects above the
    threshold contribute only their original row."""
    rng = np.random.default_rng(0)
    n = 200
    a = rng.uniform(-1, 1, n)
    x = rng.uniform(0, 1, n)
    threshold = 0.0
    n_below = int((a < threshold).sum())

    aug = LocalShift(delta=0.1, threshold=threshold).augment(_make_features(a, x))
    expected = n + n_below
    assert aug.features.shape[0] == expected, (
        f"expected {expected} augmented rows; got {aug.features.shape[0]}"
    )

    counts = collections.Counter(aug.origin_index.tolist())
    above = {counts[i] for i in range(n) if a[i] >= threshold}
    below = {counts[i] for i in range(n) if a[i] < threshold}
    assert above == {1}, above
    assert below == {2}, below
