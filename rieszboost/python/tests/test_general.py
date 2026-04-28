"""Slow general path with sklearn base learners on the Lee-Schuler ATE DGP."""

import numpy as np
import pytest

from rieszboost.estimands import ATE
from rieszboost.general import GeneralRieszBooster, general_fit


def _logit(z):
    return 1.0 / (1.0 + np.exp(-z))


def _simulate(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    pi = _logit(-0.02 * x - x**2 + 4 * np.log(x + 0.3) + 1.5)
    a = rng.binomial(1, pi)
    return x, a, pi


def test_general_fit_with_decision_tree():
    from sklearn.tree import DecisionTreeRegressor

    x, a, pi = _simulate(2000, seed=0)
    rows = [{"a": int(ai), "x": float(xi)} for ai, xi in zip(a, x)]

    booster = general_fit(
        rows[:1600],
        ATE(),
        feature_keys=("a", "x"),
        base_learner=lambda: DecisionTreeRegressor(max_depth=3, random_state=0),
        valid_rows=rows[1600:],
        num_boost_round=400,
        early_stopping_rounds=20,
        learning_rate=0.05,
    )
    assert isinstance(booster, GeneralRieszBooster)
    assert booster.best_iteration is not None
    alpha_hat = booster.predict(rows)
    alpha_true = a / pi - (1 - a) / (1 - pi)
    rmse = float(np.sqrt(np.mean((alpha_hat - alpha_true) ** 2)))
    assert rmse < 1.5, f"DecisionTree slow path RMSE {rmse:.3f} too high"


def test_general_fit_with_kernel_ridge():
    """Kernel ridge regression as a smooth base learner — would not be available
    via xgboost custom obj. Demonstrates the slow path's whole reason for being."""
    from sklearn.kernel_ridge import KernelRidge

    x, a, pi = _simulate(800, seed=1)
    rows = [{"a": int(ai), "x": float(xi)} for ai, xi in zip(a, x)]

    booster = general_fit(
        rows[:640],
        ATE(),
        feature_keys=("a", "x"),
        base_learner=lambda: KernelRidge(alpha=1.0, kernel="rbf", gamma=2.0),
        valid_rows=rows[640:],
        num_boost_round=80,
        early_stopping_rounds=10,
        learning_rate=0.05,
    )
    alpha_hat = booster.predict(rows)
    alpha_true = a / pi - (1 - a) / (1 - pi)
    corr = float(np.corrcoef(alpha_hat, alpha_true)[0, 1])
    assert corr > 0.85, f"KernelRidge slow path correlation only {corr:.3f}"


def test_general_riesz_loss_matches_history_at_best_iteration():
    from sklearn.tree import DecisionTreeRegressor

    x, a, _ = _simulate(800, seed=2)
    rows = [{"a": int(ai), "x": float(xi)} for ai, xi in zip(a, x)]
    n_tr = 640

    booster = general_fit(
        rows[:n_tr],
        ATE(),
        feature_keys=("a", "x"),
        base_learner=lambda: DecisionTreeRegressor(max_depth=3, random_state=0),
        valid_rows=rows[n_tr:],
        num_boost_round=200,
        early_stopping_rounds=15,
        learning_rate=0.05,
    )
    held_out = booster.riesz_loss(rows[n_tr:], ATE())
    assert pytest.approx(held_out, rel=1e-6) == booster.best_score


def test_general_early_stopping_requires_valid_rows():
    from sklearn.tree import DecisionTreeRegressor

    rows = [{"a": 1, "x": 0.5}]
    with pytest.raises(ValueError):
        general_fit(
            rows,
            ATE(),
            feature_keys=("a", "x"),
            base_learner=lambda: DecisionTreeRegressor(),
            num_boost_round=5,
            early_stopping_rounds=2,
        )
