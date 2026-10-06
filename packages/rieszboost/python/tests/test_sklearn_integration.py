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


def test_n_jobs_sets_threads_without_changing_the_fit():
    import json

    df, _ = _simulate_df(500, seed=3)
    one = RieszBooster(estimand=rieszboost.ATE(), n_estimators=30, n_jobs=1).fit(df)
    every = RieszBooster(estimand=rieszboost.ATE(), n_estimators=30).fit(df)
    config = json.loads(one.predictor_.booster.save_config())
    assert config["learner"]["generic_param"]["nthread"] == "1"
    np.testing.assert_array_equal(one.predict(df), every.predict(df))


def test_booster_settings_match_explicit_backend():
    """RieszBooster forwards every tree setting to the XGBoostBackend it
    builds: the same settings set on the backend give the same fit."""
    import dataclasses

    from rieszboost import XGBoostBackend

    settings = dict(
        n_estimators=40, learning_rate=0.1, max_depth=2, reg_lambda=0.0,
        subsample=0.7, early_stopping_rounds=5, validation_fraction=0.2, n_jobs=1,
    )
    # Every setting RieszBooster forwards is covered here and is a backend field.
    assert set(settings) == set(RieszBooster._BACKEND_PARAMS)
    assert set(settings) <= {f.name for f in dataclasses.fields(XGBoostBackend)}

    df, _ = _simulate_df(500, seed=4)
    via_booster = RieszBooster(estimand=rieszboost.ATE(), **settings).fit(df)
    via_backend = RieszBooster(
        estimand=rieszboost.ATE(), backend=XGBoostBackend(**settings)
    ).fit(df)
    assert via_booster.best_iteration_ == via_backend.best_iteration_
    np.testing.assert_array_equal(via_booster.predict(df), via_backend.predict(df))


def test_gridsearchcv_tunes_xgboost_backend_fields():
    """backend__<field> reaches XGBoostBackend settings RieszBooster doesn't
    expose, such as hessian_floor. The backend passed in is left unchanged."""
    from rieszboost import XGBoostBackend

    df, _ = _simulate_df(600, seed=5)
    backend = XGBoostBackend(n_estimators=20)
    grid = GridSearchCV(
        RieszBooster(estimand=rieszboost.ATE(), backend=backend),
        param_grid={"backend__hessian_floor": ["auto", 2.0], "backend__max_depth": [2, 3]},
        cv=3,
    ).fit(df)
    best = grid.best_estimator_.backend
    assert best.hessian_floor == grid.best_params_["backend__hessian_floor"]
    assert best.max_depth == grid.best_params_["backend__max_depth"]
    assert backend == XGBoostBackend(n_estimators=20)
    assert grid.best_estimator_.best_iteration_ is None  # fitted, all 20 trees kept


def test_gridsearchcv_tunes_sklearn_backend_fields():
    from sklearn.tree import DecisionTreeRegressor

    from rieszboost import SklearnBackend

    df, _ = _simulate_df(400, seed=6)
    grid = GridSearchCV(
        RieszBooster(
            estimand=rieszboost.ATE(),
            backend=SklearnBackend(
                lambda: DecisionTreeRegressor(max_depth=2, random_state=0), n_estimators=15
            ),
        ),
        param_grid={"backend__learning_rate": [0.05, 0.2]},
        cv=3,
    ).fit(df)
    assert grid.best_estimator_.backend.learning_rate == grid.best_params_["backend__learning_rate"]
    assert len(grid.best_estimator_.predictor_.learners) == 15


@pytest.mark.parametrize("backend_name", ["xgboost", "sklearn"])
def test_backend_holds_out_rows_only_under_early_stopping(backend_name):
    """Both backends default to validation_fraction=0.1 and hold rows out only
    when early_stopping_rounds is set, so a fit without early stopping trains
    on every row."""
    from sklearn.tree import DecisionTreeRegressor

    from rieszboost import SklearnBackend, XGBoostBackend

    def make(**kw):
        if backend_name == "xgboost":
            return XGBoostBackend(n_estimators=200, **kw)
        return SklearnBackend(
            lambda: DecisionTreeRegressor(max_depth=3, random_state=0), n_estimators=200, **kw
        )

    df, _ = _simulate_df(500, seed=7)
    assert make().validation_fraction == 0.1
    assert make().holdout_fraction() == 0.0
    assert make(early_stopping_rounds=10).holdout_fraction() == 0.1

    no_es = RieszBooster(estimand=rieszboost.ATE(), backend=make()).fit(df)
    all_rows = RieszBooster(estimand=rieszboost.ATE(), backend=make(validation_fraction=0.0)).fit(df)
    np.testing.assert_array_equal(no_es.predict(df), all_rows.predict(df))

    # Early stopping works with the backend alone: without held-out rows the
    # fit would raise, and best_score_ is set only when early stopping ran.
    es = RieszBooster(estimand=rieszboost.ATE(), backend=make(early_stopping_rounds=10)).fit(df)
    assert es.best_iteration_ is not None and es.best_score_ is not None
