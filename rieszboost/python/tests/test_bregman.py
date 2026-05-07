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


def test_bernoulli_predicts_in_zero_one():
    from rieszboost.losses import BernoulliLoss
    df, _ = _simulate_df(500, seed=0)
    # For TSM, m̄ = 1, which under Bernoulli sits on the upper boundary; the
    # loss-minimizing constant init clips to (1-ε) and saturates the sigmoid.
    # Pass `init=0.5` explicitly so the iterative fit has room to descend.
    booster = RieszBooster(
        estimand=rieszboost.TSM(level=1),
        loss=BernoulliLoss(),
        init=0.5,
        n_estimators=30, learning_rate=0.1, max_depth=3,
    ).fit(df)
    alpha_hat = booster.predict(df)
    assert alpha_hat.min() > 0.0
    assert alpha_hat.max() < 1.0


def test_bounded_squared_predicts_in_range():
    from rieszboost.losses import BoundedSquaredLoss
    df, _ = _simulate_df(800, seed=2)
    lo, hi = -8.0, 8.0
    booster = RieszBooster(
        estimand=rieszboost.ATE(),
        loss=BoundedSquaredLoss(lo=lo, hi=hi),
        n_estimators=200, learning_rate=0.05, max_depth=3,
        early_stopping_rounds=10, validation_fraction=0.2,
    ).fit(df)
    alpha_hat = booster.predict(df)
    assert alpha_hat.min() > lo
    assert alpha_hat.max() < hi


def test_bounded_squared_correlates_with_truth():
    """BoundedSquaredLoss with reasonably tight bounds should still track α₀.

    Note: very generous bounds make the sigmoid link saturate over most of η,
    which slows the boosting dynamics — pick bounds that closely fit α₀.
    """
    from rieszboost.losses import BoundedSquaredLoss
    df, pi = _simulate_df(2000, seed=3)
    a = df["a"].values
    alpha_true = a / pi - (1 - a) / (1 - pi)
    booster = RieszBooster(
        estimand=rieszboost.ATE(),
        loss=BoundedSquaredLoss(lo=alpha_true.min() - 1, hi=alpha_true.max() + 1),
        n_estimators=300, learning_rate=0.05, max_depth=3,
        early_stopping_rounds=20, validation_fraction=0.2, reg_lambda=10.0,
    ).fit(df)
    corr = float(np.corrcoef(booster.predict(df), alpha_true)[0, 1])
    assert corr > 0.7


def test_bounded_squared_init_validation():
    from rieszboost.losses import BoundedSquaredLoss
    with pytest.raises(ValueError, match="lo"):
        BoundedSquaredLoss(lo=2.0, hi=1.0)


def test_bernoulli_serialization_round_trip(tmp_path):
    from rieszboost.losses import BernoulliLoss
    df, _ = _simulate_df(300, seed=4)
    b = RieszBooster(
        estimand=rieszboost.TSM(level=1),
        loss=BernoulliLoss(max_abs_eta=20.0),
        n_estimators=20, learning_rate=0.1, max_depth=3,
    ).fit(df)
    pre = b.predict(df)
    b.save(tmp_path / "bern")
    loaded = RieszBooster.load(tmp_path / "bern")
    assert loaded.loss_.max_abs_eta == 20.0
    np.testing.assert_array_equal(pre, loaded.predict(df))


def test_bounded_squared_serialization_round_trip(tmp_path):
    from rieszboost.losses import BoundedSquaredLoss
    df, _ = _simulate_df(300, seed=5)
    b = RieszBooster(
        estimand=rieszboost.ATE(),
        loss=BoundedSquaredLoss(lo=-5.0, hi=5.0),
        n_estimators=20, learning_rate=0.1, max_depth=3,
    ).fit(df)
    pre = b.predict(df)
    b.save(tmp_path / "bounded")
    loaded = RieszBooster.load(tmp_path / "bounded")
    assert loaded.loss_.lo == -5.0
    assert loaded.loss_.hi == 5.0
    np.testing.assert_array_equal(pre, loaded.predict(df))
