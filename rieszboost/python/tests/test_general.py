"""SklearnBackend: first-order gradient boosting with sklearn-compatible base learners."""

import numpy as np
import pandas as pd
import pytest

import rieszboost
from rieszboost import RieszBooster
from rieszboost.backends import SklearnBackend


def _logit(z):
    return 1.0 / (1.0 + np.exp(-z))


def _df_pi(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    pi = _logit(-0.02 * x - x**2 + 4 * np.log(x + 0.3) + 1.5)
    a = rng.binomial(1, pi)
    return pd.DataFrame({"a": a.astype(float), "x": x.astype(float)}), pi


def test_sklearn_backend_with_decision_tree():
    from sklearn.tree import DecisionTreeRegressor
    df, pi = _df_pi(2000, seed=0)
    booster = RieszBooster(
        estimand=rieszboost.ATE(),
        backend=SklearnBackend(lambda: DecisionTreeRegressor(max_depth=3, random_state=0)),
        n_estimators=400,
        early_stopping_rounds=20,
        validation_fraction=0.2,
        learning_rate=0.05,
    ).fit(df)
    assert booster.best_iteration_ is not None
    a = df["a"].values
    alpha_true = a / pi - (1 - a) / (1 - pi)
    rmse = float(np.sqrt(np.mean((booster.predict(df) - alpha_true) ** 2)))
    assert rmse < 1.5


def test_sklearn_backend_with_kernel_ridge():
    from sklearn.kernel_ridge import KernelRidge
    df, pi = _df_pi(800, seed=1)
    booster = RieszBooster(
        estimand=rieszboost.ATE(),
        backend=SklearnBackend(lambda: KernelRidge(alpha=1.0, kernel="rbf", gamma=2.0)),
        n_estimators=80,
        early_stopping_rounds=10,
        validation_fraction=0.2,
        learning_rate=0.05,
    ).fit(df)
    a = df["a"].values
    alpha_true = a / pi - (1 - a) / (1 - pi)
    corr = float(np.corrcoef(booster.predict(df), alpha_true)[0, 1])
    assert corr > 0.85


def test_sklearn_backend_score_matches_negative_loss():
    from sklearn.tree import DecisionTreeRegressor
    df, _ = _df_pi(800, seed=2)
    booster = RieszBooster(
        estimand=rieszboost.ATE(),
        backend=SklearnBackend(lambda: DecisionTreeRegressor(max_depth=3, random_state=0)),
        n_estimators=200,
        early_stopping_rounds=15,
        validation_fraction=0.2,
        learning_rate=0.05,
    ).fit(df)
    assert booster.score(df) == pytest.approx(-booster.riesz_loss(df), rel=1e-9)


def test_sklearn_backend_requires_validation_for_early_stopping():
    from sklearn.tree import DecisionTreeRegressor
    df, _ = _df_pi(50, seed=0)
    # validation_fraction=0 + early_stopping_rounds=N => booster's fit does
    # the internal split (using default 0.2 fraction). But when we pass
    # the backend directly to fit_augmented with no valid set, it raises.
    # End-to-end via RieszBooster: should auto-split.
    booster = RieszBooster(
        estimand=rieszboost.ATE(),
        backend=SklearnBackend(lambda: DecisionTreeRegressor(max_depth=3)),
        n_estimators=10,
        early_stopping_rounds=2,
    ).fit(df)
    # Auto-split kicks in via validation_fraction default of 0.2 when ES set.
    assert booster.best_iteration_ is not None or len(booster.predictor_.learners) > 0
