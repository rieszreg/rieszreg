"""StochasticIntervention factory: Monte Carlo over an intervention density."""

import numpy as np
import pytest

import rieszboost
from rieszboost.engine import build_augmented, fit
from rieszboost.estimands import AdditiveShift, StochasticIntervention
from rieszboost.tracer import trace


def test_stochastic_with_one_deterministic_sample_matches_additive_shift():
    """K=1 sample at exactly a+delta should produce the same augmented dataset
    as AdditiveShift(delta), modulo the missing -alpha(a, x) term that
    AdditiveShift includes (StochasticIntervention only has the +alpha(a+delta, x) part)."""
    delta = 1.0
    z = {"a": 0.5, "x": 0.3, "samples": [0.5 + delta]}
    pairs = trace(StochasticIntervention(samples_key="samples"), z)
    # K=1, coefficient 1/K = 1.0
    assert len(pairs) == 1
    coef, point = pairs[0]
    assert coef == pytest.approx(1.0)
    assert point["a"] == pytest.approx(0.5 + delta)
    assert point["x"] == pytest.approx(0.3)


def test_stochastic_with_multiple_samples_averages():
    z = {"a": 0.0, "x": 0.5, "samples": [1.0, 1.5, 2.0, 2.5]}
    pairs = trace(StochasticIntervention(samples_key="samples"), z)
    assert len(pairs) == 4
    coefs = sorted(c for c, _ in pairs)
    assert all(c == pytest.approx(0.25) for c in coefs)
    sample_a = sorted(p["a"] for _, p in pairs)
    assert sample_a == pytest.approx([1.0, 1.5, 2.0, 2.5])


def test_stochastic_empty_samples_returns_zero():
    z = {"a": 0.0, "x": 0.5, "samples": []}
    pairs = trace(StochasticIntervention(samples_key="samples"), z)
    assert pairs == []


def test_stochastic_works_end_to_end_on_continuous_dgp():
    """Smoke test: fit a stochastic shift Riesz representer with a simple
    Gaussian intervention. Just check it runs and predictions are finite."""
    rng = np.random.default_rng(0)
    n = 1000
    x = rng.uniform(0, 2, n)
    a = rng.normal(x**2 - 1.0, np.sqrt(2.0))

    # Stochastic shift: A' = A + 1 + N(0, 0.5²)
    n_mc = 20
    rows = []
    for i in range(n):
        rows.append({
            "a": float(a[i]),
            "x": float(x[i]),
            "shift_samples": rng.normal(a[i] + 1.0, 0.5, size=n_mc).tolist(),
        })

    booster = fit(
        rows,
        StochasticIntervention(),
        feature_keys=("a", "x"),
        num_boost_round=50,
        learning_rate=0.05,
        max_depth=3,
        seed=0,
    )
    alpha_hat = booster.predict(rows)
    assert alpha_hat.shape == (n,)
    assert np.all(np.isfinite(alpha_hat))
    # Boosting from init=0: predictions shouldn't all be exactly zero.
    assert np.std(alpha_hat) > 0


def test_stochastic_augmentation_size():
    """K samples per row → K augmented counterfactual rows per original row,
    plus the original itself. So 1 + K rows per subject (some may merge)."""
    n_mc = 5
    rng = np.random.default_rng(0)
    rows = []
    for i in range(50):
        rows.append({
            "a": float(rng.normal(0, 1)),
            "x": float(rng.uniform()),
            "shift_samples": rng.normal(1.0, 0.5, size=n_mc).tolist(),
        })
    aug = build_augmented(rows, StochasticIntervention(), feature_keys=("a", "x"))
    # Each row contributes 1 original + n_mc counterfactuals (continuous A
    # makes coincidences vanishingly rare, so no merges in practice).
    assert aug.features.shape == (50 * (1 + n_mc), 2)
    # Sum of b over each row should be -2 * sum(coefficients) = -2 * 1.0 = -2.
    for i in range(50):
        idx = np.where(aug.origin_index == i)[0]
        assert pytest.approx(aug.b[idx].sum()) == -2.0
        assert pytest.approx(aug.a[idx].sum()) == 1.0


def test_public_api_export():
    assert hasattr(rieszboost, "StochasticIntervention")
