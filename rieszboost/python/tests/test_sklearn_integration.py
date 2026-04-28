"""sklearn integration acceptance tests: clone, GridSearchCV, cross_val_predict."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict

import rieszboost
from rieszboost import RieszBooster


def _logit(z):
    return 1.0 / (1.0 + np.exp(-z))


def _simulate_df(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    pi = _logit(-0.02 * x - x**2 + 4 * np.log(x + 0.3) + 1.5)
    a = rng.binomial(1, pi)
    return pd.DataFrame({"a": a.astype(float), "x": x.astype(float)}), pi


def test_clone_produces_unfitted_copy():
    df, _ = _simulate_df(200, seed=0)
    booster = RieszBooster(
        estimand=rieszboost.ATE(),
        n_estimators=20,
        learning_rate=0.1,
        max_depth=3,
    ).fit(df)
    assert hasattr(booster, "predictor_")
    cloned = clone(booster)
    assert not hasattr(cloned, "predictor_")
    # Hyperparameters preserved
    assert cloned.n_estimators == 20
    assert cloned.estimand.name == "ATE"


def test_gridsearchcv_runs_end_to_end():
    df, _ = _simulate_df(800, seed=1)
    grid = GridSearchCV(
        RieszBooster(estimand=rieszboost.ATE(), n_estimators=30),
        param_grid={
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 4],
        },
        cv=3,
        n_jobs=1,
    )
    grid.fit(df)
    assert grid.best_params_ is not None
    assert grid.best_score_ > 0  # negative-loss is positive at optimum


def test_cross_val_predict_returns_oof():
    df, pi = _simulate_df(800, seed=2)
    booster = RieszBooster(
        estimand=rieszboost.ATE(),
        n_estimators=50,
        learning_rate=0.05,
        max_depth=3,
    )
    oof = cross_val_predict(booster, df, cv=KFold(n_splits=5, shuffle=True, random_state=0))
    assert oof.shape == (800,)
    assert np.all(np.isfinite(oof))
    # OOF predictions should correlate with truth
    a = df["a"].values
    alpha_true = a / pi - (1 - a) / (1 - pi)
    corr = float(np.corrcoef(oof, alpha_true)[0, 1])
    assert corr > 0.5


def test_get_set_params_round_trip():
    booster = RieszBooster(
        estimand=rieszboost.ATE(),
        n_estimators=50,
        learning_rate=0.05,
    )
    params = booster.get_params(deep=False)
    assert params["n_estimators"] == 50
    booster.set_params(n_estimators=200, learning_rate=0.1)
    assert booster.n_estimators == 200
    assert booster.learning_rate == 0.1
