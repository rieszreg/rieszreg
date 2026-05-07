"""Parity tests for built-in subclass `augment()` overrides.

For each built-in estimand, the vectorised override must produce an
:class:`AugmentedDataset` equivalent (modulo row order) to the inherited
Tracer-based default implementation.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from rieszreg import (
    ATE,
    ATT,
    AdditiveShift,
    FiniteEvalEstimand,
    LocalShift,
    TSM,
)


def _make_features(n=100, p=2, seed=0, treatment_levels=(0.0, 1.0)):
    rng = np.random.default_rng(seed)
    a = rng.choice(treatment_levels, size=n).astype(float)
    x = rng.normal(0.0, 1.0, size=(n, p - 1))
    return np.column_stack([a, x])


def _signature(aug):
    """Stable representation of an AugmentedDataset modulo row order."""
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


def _assert_equivalent(slow, fast):
    assert slow.n_rows == fast.n_rows
    assert slow.features.shape == fast.features.shape
    assert _signature(slow) == _signature(fast)


@pytest.mark.parametrize(
    "estimand_factory, treatment_levels",
    [
        (lambda: ATE(treatment="a", covariates=("x0", "x1")), (0.0, 1.0)),
        (lambda: ATT(treatment="a", covariates=("x0", "x1")), (0.0, 1.0)),
        (lambda: TSM(treatment="a", covariates=("x0", "x1"), level=1.0), (0.0, 1.0)),
        (lambda: TSM(treatment="a", covariates=("x0", "x1"), level=0.0), (0.0, 1.0)),
        (
            lambda: AdditiveShift(delta=0.5, treatment="a", covariates=("x0", "x1")),
            (0.0, 1.0, 2.0),
        ),
        (
            lambda: LocalShift(
                delta=0.3, threshold=1.5, treatment="a", covariates=("x0", "x1")
            ),
            (0.0, 1.0, 2.0),
        ),
    ],
)
def test_subclass_augment_matches_base_default(estimand_factory, treatment_levels):
    """For every built-in subclass, the override produces the same augmented
    dataset (modulo row order) as the inherited base-class default."""
    estimand = estimand_factory()
    features = _make_features(n=50, p=3, treatment_levels=treatment_levels)
    fast = estimand.augment(features)
    slow = FiniteEvalEstimand.augment(estimand, features)  # bypass MRO → base default
    _assert_equivalent(slow, fast)


def test_custom_estimand_uses_inherited_default():
    """Custom `FiniteEvalEstimand(...)` instances inherit the base `augment`,
    which traces `m` row-by-row and emits the correct augmented dataset."""

    def m(alpha):
        def inner(z, y=None):
            return alpha(a=z["a"], x=z["x"])  # identity at the original (a, x)
        return inner

    custom = FiniteEvalEstimand(feature_keys=("a", "x"), m=m, name="identity-mean")
    features = _make_features(n=20, p=2)
    aug = custom.augment(features)
    # Each row's m gives one (coef=+1, point=(a, x)) which merges with the
    # original at the same point → 1 augmented row per input.
    assert aug.features.shape[0] == 20
    assert (aug.is_original == 1.0).all()
    assert np.allclose(aug.potential_deriv_coef, -1.0)


def test_y_does_not_change_built_in_augmentation():
    """Built-in `m`s ignore y; passing `ys` must produce the same output as
    omitting it."""
    features = _make_features(n=30, p=2)
    ys = np.linspace(-1.0, 1.0, 30)
    estimand = ATE()
    a = estimand.augment(features)
    b = estimand.augment(features, ys=ys)
    _assert_equivalent(a, b)


def test_subclass_augment_is_meaningfully_faster_than_base():
    """The vectorised override should be ≥ 20× faster than the Tracer-based
    base default on a moderate dataset. Headline ratio is much higher (~100×);
    20× is a generous floor that survives noise across machines."""
    estimand = ATE(treatment="a", covariates=tuple(f"x{j}" for j in range(10)))
    features = _make_features(n=2000, p=11)

    n_iters = 5
    t0 = time.perf_counter()
    for _ in range(n_iters):
        estimand.augment(features)
    fast_s = (time.perf_counter() - t0) / n_iters

    t0 = time.perf_counter()
    for _ in range(n_iters):
        FiniteEvalEstimand.augment(estimand, features)
    slow_s = (time.perf_counter() - t0) / n_iters

    assert fast_s * 20 < slow_s, (
        f"subclass augment should be ≥ 20× faster; "
        f"got fast={fast_s*1e3:.2f} ms vs slow={slow_s*1e3:.2f} ms"
    )


@pytest.mark.parametrize(
    "estimand, expected",
    [
        (ATE(), 0.0),
        (ATT(), 0.0),
        (TSM(level=1.0), 1.0),
        (TSM(level=0.0), 1.0),
        (AdditiveShift(delta=0.5), 0.0),
        (LocalShift(delta=0.3, threshold=1.5), 0.0),
    ],
)
def test_m_bar_analytic_matches_empirical(estimand, expected):
    """The class-level closed-form `m_bar` matches the empirical mean of
    `sum(coef for coef, _ in trace(estimand, z))` for built-ins."""
    from rieszreg import trace

    rng = np.random.default_rng(0)
    n = 200
    a = rng.choice([0.0, 1.0, 2.0], size=n).astype(float)
    x = rng.normal(size=n)
    rows = [{"a": float(a[i]), "x": float(x[i])} for i in range(n)]
    empirical = float(np.mean([sum(c for c, _ in trace(estimand, z)) for z in rows]))
    assert estimand.m_bar == expected
    np.testing.assert_allclose(empirical, expected)


def test_additive_shift_zero_delta_raises():
    with pytest.raises(ValueError, match="delta != 0"):
        AdditiveShift(delta=0.0)


def test_local_shift_zero_delta_raises():
    with pytest.raises(ValueError, match="delta != 0"):
        LocalShift(delta=0.0, threshold=1.0)
