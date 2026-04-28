"""ATT estimand: m(z, alpha) = (a/p_treated) * (alpha(1,x) - alpha(0,x))."""

import numpy as np

from rieszboost.engine import build_augmented, fit
from rieszboost.estimands import ATT
from rieszboost.tracer import trace


def test_att_traces_to_zero_for_controls():
    p = 0.4
    m = ATT(p_treated=p)
    # treated row: full ATE-style contrast
    treated_pairs = trace(m, {"a": 1, "x": 0.5})
    coefs = sorted(c for c, _ in treated_pairs)
    assert coefs == [-1.0 / p, 1.0 / p]
    # control row: weight is 0, no terms
    assert trace(m, {"a": 0, "x": 0.5}) == []


def test_att_augmentation_skips_controls():
    rows = [{"a": 1, "x": 0.5}, {"a": 0, "x": 0.7}]
    aug = build_augmented(rows, ATT(p_treated=0.5), feature_keys=("a", "x"))
    # Treated row: 2 unique points (the (1,x) one merges with the original).
    # Control row: just the original alpha^2 row at (0, 0.7).
    assert aug.features.shape == (3, 2)
    # Control row's contributions are pure quadratic (a=1, b=0).
    ctrl_idx = np.where(aug.origin_index == 1)[0]
    assert aug.a[ctrl_idx].sum() == 1.0
    assert aug.b[ctrl_idx].sum() == 0.0


def _logit(z):
    return 1.0 / (1.0 + np.exp(-z))


def test_att_recovers_truth_on_lee_schuler_dgp():
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.uniform(0, 1, n)
    pi = _logit(-0.02 * x - x**2 + 4 * np.log(x + 0.3) + 1.5)
    a = rng.binomial(1, pi)
    rows = [{"a": int(ai), "x": float(xi)} for ai, xi in zip(a, x)]
    p_treated = float(a.mean())

    n_tr = int(0.8 * n)
    booster = fit(
        rows[:n_tr],
        ATT(p_treated=p_treated),
        feature_keys=("a", "x"),
        valid_rows=rows[n_tr:],
        num_boost_round=2000,
        early_stopping_rounds=20,
        learning_rate=0.05,
        max_depth=3,
        reg_lambda=10.0,
        seed=0,
    )
    alpha_hat = booster.predict(rows)
    # True ATT representer: A/P(A=1) - (1-A) * pi(X) / ((1-pi(X)) * P(A=1))
    alpha_true = a / p_treated - (1 - a) * pi / ((1 - pi) * p_treated)
    rmse = float(np.sqrt(np.mean((alpha_hat - alpha_true) ** 2)))
    # Lee-Schuler Table 1 reports RieszBoost RMSE on ATT at n=500: 0.435.
    assert rmse < 0.5, f"RMSE {rmse:.3f} too high"
