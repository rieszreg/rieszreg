"""Bregman-Riesz losses: SquaredLoss equivalence + KLLoss density-ratio support."""

import numpy as np
import pandas as pd
import pytest

import rieszboost
from rieszboost import RieszBooster
from rieszboost.losses import KLLoss, SquaredLoss


def _logit(z):
    return 1.0 / (1.0 + np.exp(-z))


def _simulate_df(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    pi = _logit(-0.02 * x - x**2 + 4 * np.log(x + 0.3) + 1.5)
    a = rng.binomial(1, pi)
    return pd.DataFrame({"a": a.astype(float), "x": x.astype(float)}), pi


def test_squared_loss_explicit_matches_default():
    df, _ = _simulate_df(500, seed=0)
    common = dict(estimand=rieszboost.ATE(), n_estimators=20,
                  learning_rate=0.1, max_depth=3, random_state=0)
    b1 = RieszBooster(**common).fit(df)
    b2 = RieszBooster(loss=SquaredLoss(), **common).fit(df)
    np.testing.assert_array_equal(b1.predict(df), b2.predict(df))


def test_kl_loss_rejects_signed_coefficients():
    df, _ = _simulate_df(200, seed=1)
    booster = RieszBooster(estimand=rieszboost.ATE(), loss=KLLoss(),
                           n_estimators=5, learning_rate=0.1, max_depth=3)
    with pytest.raises(ValueError, match="non-negative"):
        booster.fit(df)


def test_kl_loss_predicts_positive_alpha():
    df, _ = _simulate_df(1000, seed=0)
    booster = RieszBooster(
        estimand=rieszboost.TSM(level=1),
        loss=KLLoss(),
        n_estimators=50, learning_rate=0.05, max_depth=3,
    ).fit(df)
    alpha_hat = booster.predict(df)
    assert alpha_hat.min() > 0
    assert alpha_hat.max() > 0.5


def test_kl_correlates_with_truth_on_treated():
    df, pi = _simulate_df(4000, seed=0)
    a = df["a"].values
    booster = RieszBooster(
        estimand=rieszboost.TSM(level=1),
        loss=KLLoss(),
        n_estimators=2000,
        early_stopping_rounds=20,
        validation_fraction=0.2,
        learning_rate=0.05,
        max_depth=3,
        reg_lambda=10.0,
    ).fit(df)
    alpha_hat = booster.predict(df)
    treated = a == 1
    corr = float(np.corrcoef(alpha_hat[treated], 1.0 / pi[treated])[0, 1])
    assert corr > 0.3


def test_kl_riesz_loss_finite():
    df, _ = _simulate_df(200, seed=0)
    booster = RieszBooster(
        estimand=rieszboost.TSM(level=1),
        loss=KLLoss(),
        n_estimators=20, learning_rate=0.1, max_depth=3,
    ).fit(df)
    assert np.isfinite(booster.riesz_loss(df))
