"""Tests for the OutcomeRegNormSq estimand.

`OutcomeRegNormSq` has m(α)(z, y) = α(x) · y, Riesz representer
μ_0(x) = E[Y | X=x], and estimand value E[μ_0(X)²]. Under the squared
Bregman-Riesz loss the augmented dataset reduces to the original X with
`is_original = 1`, `potential_deriv_coef = -y` — i.e. the empirical loss
equals ∑(α(x_i) − y_i)². This file pins the augmentation contract and the
empirical-`m_bar` fallback; per-backend parity vs. stock MSE regressors
lives in each implementation package's tests.
"""
from __future__ import annotations

import numpy as np
import pytest

from rieszreg import FiniteEvalEstimand, OutcomeRegNormSq, trace


def _make_xy(n=50, p=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, p))
    y = rng.normal(0.0, 1.0, size=n)
    return X, y


def _signature(aug):
    out = []
    for i in range(aug.features.shape[0]):
        row = tuple(round(x, 9) for x in aug.features[i])
        out.append((
            int(aug.origin_index[i]),
            row,
            round(float(aug.is_original[i]), 9),
            round(float(aug.potential_deriv_coef[i]), 9),
        ))
    return sorted(out)


def test_augment_shape_and_values():
    e = OutcomeRegNormSq(covariates=("x0", "x1"))
    X, y = _make_xy(n=20, p=2)
    aug = e.augment(X, ys=y)
    assert aug.n_rows == 20
    assert aug.features.shape == (20, 2)
    np.testing.assert_array_equal(aug.features, X)
    np.testing.assert_array_equal(aug.is_original, np.ones(20))
    np.testing.assert_allclose(aug.potential_deriv_coef, -y)
    np.testing.assert_array_equal(aug.origin_index, np.arange(20))


def test_augment_requires_ys():
    e = OutcomeRegNormSq(covariates=("x",))
    X, _ = _make_xy(n=10, p=1)
    with pytest.raises(ValueError, match="needs the outcome"):
        e.augment(X)


def test_vectorised_augment_matches_base_default():
    e = OutcomeRegNormSq(covariates=("x0", "x1"))
    X, y = _make_xy(n=30, p=2, seed=1)
    fast = e.augment(X, ys=y)
    slow = FiniteEvalEstimand.augment(e, X, ys=y)  # bypass override → tracer path
    assert _signature(slow) == _signature(fast)


def test_m_bar_from_augmentation_equals_mean_y():
    """The orchestrator's default init reads m̄ = −ΣC / n off the augmented
    data; for OutcomeRegNormSq that is E[Y]."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 1))
    y = rng.normal(loc=2.5, scale=1.5, size=200)
    aug = OutcomeRegNormSq(covariates=("x",)).augment(X, ys=y)
    assert abs(-aug.potential_deriv_coef.sum() / aug.n_rows - y.mean()) < 1e-12


def test_trace_yields_single_point_per_row():
    """m(α)(z, y) = α(x) · y traces to one (y, {x}) pair."""
    e = OutcomeRegNormSq(covariates=("x",))
    pairs = trace(e, {"x": 1.5}, y=2.0)
    assert len(pairs) == 1
    coef, point = pairs[0]
    assert coef == pytest.approx(2.0)
    assert point == {"x": 1.5}
