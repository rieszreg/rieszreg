"""ATT partial-parameter estimand: m(α)(z) = a * (α(1,x) - α(0,x))."""

import numpy as np
import pandas as pd

import rieszboost
from rieszboost import RieszBooster
from rieszboost.augmentation import build_augmented
from rieszboost.tracer import trace


def test_att_traces_to_zero_for_controls():
    m = rieszboost.ATT()
    coefs = sorted(c for c, _ in trace(m, {"a": 1, "x": 0.5}))
    assert coefs == [-1.0, 1.0]
    assert trace(m, {"a": 0, "x": 0.5}) == []


def test_att_augmentation_skips_controls():
    rows = [{"a": 1, "x": 0.5}, {"a": 0, "x": 0.7}]
    aug = build_augmented(rows, rieszboost.ATT())
    assert aug.features.shape == (3, 2)
    ctrl_idx = np.where(aug.origin_index == 1)[0]
    assert aug.is_original[ctrl_idx].sum() == 1.0
    assert aug.potential_deriv_coef[ctrl_idx].sum() == 0.0


def _logit(z):
    return 1.0 / (1.0 + np.exp(-z))


def test_att_partial_recovers_truth_on_lee_schuler_dgp():
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.uniform(0, 1, n)
    pi = _logit(-0.02 * x - x**2 + 4 * np.log(x + 0.3) + 1.5)
    a = rng.binomial(1, pi)
    df = pd.DataFrame({"a": a.astype(float), "x": x.astype(float)})

    booster = RieszBooster(
        estimand=rieszboost.ATT(),
        n_estimators=2000,
        early_stopping_rounds=20,
        validation_fraction=0.2,
        learning_rate=0.05,
        max_depth=3,
        reg_lambda=10.0,
    ).fit(df)

    alpha_hat = booster.predict(df)
    alpha_true = a - (1 - a) * pi / (1 - pi)
    rmse = float(np.sqrt(np.mean((alpha_hat - alpha_true) ** 2)))
    assert rmse < 0.5
