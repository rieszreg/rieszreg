"""End-to-end tests for `RieszBooster` on the binary-treatment ATE DGP."""

import numpy as np
import pandas as pd
import pytest

import rieszboost
from rieszboost import RieszBooster
from rieszboost.augmentation import build_augmented


def _logit(z):
    return 1.0 / (1.0 + np.exp(-z))


def _simulate(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    pi = _logit(-0.02 * x - x**2 + 4.0 * np.log(x + 0.3) + 1.5)
    a = rng.binomial(1, pi)
    return x, a, pi


def _df(x, a):
    return pd.DataFrame({"a": a.astype(float), "x": x.astype(float)})


def test_augmentation_shape_for_ate():
    x, a, _ = _simulate(50, seed=1)
    rows = [{"a": float(ai), "x": float(xi)} for ai, xi in zip(a, x)]
    aug = build_augmented(rows, rieszboost.ATE())
    assert aug.features.shape == (2 * len(rows), 2)
    for i in range(len(rows)):
        idx = np.where(aug.origin_index == i)[0]
        assert pytest.approx(aug.potential_deriv_coef[idx].sum()) == 0.0
        assert pytest.approx(aug.is_original[idx].sum()) == 1.0


def test_ate_recovers_inverse_propensity_dataframe():
    n = 4000
    x, a, pi = _simulate(n, seed=42)
    df = _df(x, a)

    booster = RieszBooster(
        estimand=rieszboost.ATE(),
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=0,
    ).fit(df)

    alpha_hat = booster.predict(df)
    alpha_true = a / pi - (1 - a) / (1 - pi)
    rmse = float(np.sqrt(np.mean((alpha_hat - alpha_true) ** 2)))
    assert rmse < 1.0


def test_ate_recovers_inverse_propensity_ndarray():
    n = 4000
    x, a, pi = _simulate(n, seed=42)
    X = np.column_stack([a, x]).astype(float)

    booster = RieszBooster(
        estimand=rieszboost.ATE(),
        n_estimators=300, learning_rate=0.05, max_depth=4, random_state=0,
    ).fit(X)

    alpha_hat = booster.predict(X)
    alpha_true = a / pi - (1 - a) / (1 - pi)
    rmse = float(np.sqrt(np.mean((alpha_hat - alpha_true) ** 2)))
    assert rmse < 1.0


def test_default_init_for_ate_gives_zero_baseline():
    """For ATE, m(z, 1) = 1 + (-1) = 0 per row, so m̄ = 0 and the
    loss-minimizing constant init is 0 in α-space (identity link → η)."""
    x, a, _ = _simulate(100, seed=3)
    df = _df(x, a)
    booster = RieszBooster(estimand=rieszboost.ATE(), n_estimators=1).fit(df)
    assert booster.base_score_ == 0.0


def test_early_stopping_with_validation_fraction():
    n = 1000
    x, a, _ = _simulate(n, seed=10)
    df = _df(x, a)

    booster = RieszBooster(
        estimand=rieszboost.ATE(),
        n_estimators=2000,
        early_stopping_rounds=20,
        validation_fraction=0.2,
        learning_rate=0.05,
        max_depth=4,
        random_state=0,
    ).fit(df)
    assert booster.best_iteration_ is not None
    assert booster.best_iteration_ < 1500
    assert booster.best_score_ is not None


def test_early_stopping_with_explicit_eval_set():
    x, a, _ = _simulate(800, seed=21)
    df_train = _df(x[:600], a[:600])
    df_valid = _df(x[600:], a[600:])

    booster = RieszBooster(
        estimand=rieszboost.ATE(),
        n_estimators=200,
        early_stopping_rounds=20,
        learning_rate=0.05,
        max_depth=4,
    ).fit(df_train, eval_set=df_valid)
    assert booster.best_iteration_ is not None


def test_score_matches_negative_riesz_loss():
    x, a, _ = _simulate(500, seed=2)
    df = _df(x, a)
    booster = RieszBooster(estimand=rieszboost.ATE(), n_estimators=20).fit(df)
    score = booster.score(df)
    riesz = booster.riesz_loss(df)
    assert score == pytest.approx(-riesz, rel=1e-9)


def test_unfitted_booster_raises():
    booster = RieszBooster(estimand=rieszboost.ATE())
    with pytest.raises(RuntimeError):
        booster.predict(np.array([[1.0, 0.5]]))


def test_public_api_exports():
    assert hasattr(rieszboost, "RieszBooster")
    assert hasattr(rieszboost, "ATE")
    assert hasattr(rieszboost, "ATT")
    assert hasattr(rieszboost, "TSM")
    assert hasattr(rieszboost, "AdditiveShift")
    assert hasattr(rieszboost, "LocalShift")
    assert hasattr(rieszboost, "Estimand")
    assert hasattr(rieszboost, "XGBoostBackend")
    assert hasattr(rieszboost, "SklearnBackend")
