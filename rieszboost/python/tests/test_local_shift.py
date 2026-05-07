"""LocalShift partial-parameter estimand."""

import numpy as np
import pandas as pd

import rieszboost
from rieszboost import RieszBooster
from rieszboost.tracer import trace


def test_local_shift_traces_below_threshold():
    m = rieszboost.LocalShift(delta=1.0, threshold=0.0)
    pairs = trace(m, {"a": -0.5, "x": 0.3})
    coefs = sorted(c for c, _ in pairs)
    assert coefs == [-1.0, 1.0]
    sample_a = sorted(p["a"] for _, p in pairs)
    assert sample_a == [-0.5, 0.5]


def test_local_shift_above_threshold_contributes_nothing():
    m = rieszboost.LocalShift(delta=1.0, threshold=0.0)
    assert trace(m, {"a": 0.5, "x": 0.3}) == []


def test_local_shift_at_threshold_excluded():
    m = rieszboost.LocalShift(delta=1.0, threshold=0.0)
    assert trace(m, {"a": 0.0, "x": 0.0}) == []


def test_local_shift_recovers_truth_on_continuous_dgp():
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.uniform(0, 2, n)
    a = rng.normal(x**2 - 1.0, np.sqrt(2.0))
    df = pd.DataFrame({"a": a, "x": x})

    delta, t = 1.0, 0.0
    booster = RieszBooster(
        estimand=rieszboost.LocalShift(delta=delta, threshold=t),
        n_estimators=2000,
        early_stopping_rounds=20,
        validation_fraction=0.2,
        learning_rate=0.05,
        max_depth=3,
        reg_lambda=10.0,
    ).fit(df)
    alpha_hat = booster.predict(df)

    density_ratio = np.exp((2 * (a - x**2) + 1) / 4)
    alpha_true = (a < t + delta).astype(float) * density_ratio - (a < t).astype(float)
    rmse = float(np.sqrt(np.mean((alpha_hat - alpha_true) ** 2)))
    assert rmse < 0.6


def test_local_shift_augmentation_skips_above_threshold():
    features = np.array([[-0.5, 0.0], [0.5, 0.0]])
    aug = rieszboost.LocalShift(delta=1.0, threshold=0.0).augment(features)
    assert aug.features.shape == (3, 2)
    above_idx = np.where(aug.origin_index == 1)[0]
    assert aug.is_original[above_idx].sum() == 1.0
    assert aug.potential_deriv_coef[above_idx].sum() == 0.0
