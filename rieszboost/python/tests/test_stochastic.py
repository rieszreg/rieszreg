"""StochasticIntervention: Monte Carlo over an intervention density."""

import numpy as np
import pandas as pd
import pytest

import rieszboost
from rieszboost import RieszBooster
from rieszboost.augmentation import build_augmented
from rieszboost.tracer import trace


def test_stochastic_with_one_deterministic_sample():
    delta = 1.0
    z = {"a": 0.5, "x": 0.3, "samples": [0.5 + delta]}
    pairs = trace(rieszboost.StochasticIntervention(samples_key="samples"), z)
    assert len(pairs) == 1
    coef, point = pairs[0]
    assert coef == pytest.approx(1.0)
    assert point["a"] == pytest.approx(0.5 + delta)


def test_stochastic_with_multiple_samples_averages():
    z = {"a": 0.0, "x": 0.5, "samples": [1.0, 1.5, 2.0, 2.5]}
    pairs = trace(rieszboost.StochasticIntervention(samples_key="samples"), z)
    assert len(pairs) == 4
    assert all(c == pytest.approx(0.25) for c, _ in pairs)


def test_stochastic_empty_samples_returns_zero():
    z = {"a": 0.0, "x": 0.5, "samples": []}
    pairs = trace(rieszboost.StochasticIntervention(samples_key="samples"), z)
    assert pairs == []


def test_stochastic_works_end_to_end_via_dataframe():
    rng = np.random.default_rng(0)
    n = 1000
    x = rng.uniform(0, 2, n)
    a = rng.normal(x**2 - 1.0, np.sqrt(2.0))
    df = pd.DataFrame({"a": a, "x": x})
    df["shift_samples"] = [rng.normal(a[i] + 1.0, 0.5, size=20).tolist() for i in range(n)]

    booster = RieszBooster(
        estimand=rieszboost.StochasticIntervention(),
        n_estimators=50, learning_rate=0.05, max_depth=3,
    ).fit(df)
    alpha_hat = booster.predict(df)
    assert alpha_hat.shape == (n,)
    assert np.all(np.isfinite(alpha_hat))


def test_stochastic_augmentation_size():
    n_mc = 5
    rng = np.random.default_rng(0)
    rows = [
        {
            "a": float(rng.normal(0, 1)),
            "x": float(rng.uniform()),
            "shift_samples": rng.normal(1.0, 0.5, size=n_mc).tolist(),
        }
        for _ in range(50)
    ]
    aug = build_augmented(rows, rieszboost.StochasticIntervention())
    assert aug.features.shape == (50 * (1 + n_mc), 2)
    for i in range(50):
        idx = np.where(aug.origin_index == i)[0]
        assert pytest.approx(aug.b[idx].sum()) == -2.0
        assert pytest.approx(aug.a[idx].sum()) == 1.0
