"""sklearn-conformance: clone, GridSearchCV, cross_val_predict."""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict

from rieszreg.testing.conformance import (
    assert_clone_roundtrip,
    assert_get_params_round_trip,
)

from riesznet import ATE, RieszNet, TSM


def test_clone_roundtrip():
    assert_clone_roundtrip(
        lambda: RieszNet(
            estimand=ATE(),
            hidden_sizes=(16,),
            epochs=10,
            learning_rate=2e-3,
            random_state=42,
        )
    )


def test_get_params_round_trip():
    assert_get_params_round_trip(
        lambda: RieszNet(
            estimand=ATE(),
            hidden_sizes=(16,),
            epochs=10,
            random_state=42,
        )
    )


def test_get_params_includes_simple_mlp_knobs():
    est = RieszNet(
        estimand=ATE(),
        hidden_sizes=(16,),
        activation="tanh",
        dropout=0.1,
        learning_rate=2e-3,
        weight_decay=1e-4,
        epochs=10,
        random_state=42,
    )
    p = est.get_params(deep=False)
    for k in (
        "estimand", "hidden_sizes", "activation", "dropout", "learning_rate",
        "weight_decay", "epochs", "device", "dtype", "grad_clip_norm",
        "loss", "init", "validation_fraction", "early_stopping_rounds",
        "random_state",
    ):
        assert k in p, f"missing param {k!r}"


def test_set_params_round_trip():
    est = RieszNet(estimand=ATE(), hidden_sizes=(8,), epochs=5)
    est.set_params(epochs=42, dropout=0.3, learning_rate=5e-4)
    assert est.epochs == 42
    assert est.dropout == 0.3
    assert est.learning_rate == 5e-4


def test_clone_preserves_constructor_args():
    est = RieszNet(
        estimand=ATE(),
        hidden_sizes=(8, 8),
        activation="tanh",
        dropout=0.1,
        learning_rate=2e-3,
        weight_decay=1e-4,
        epochs=15,
        random_state=123,
    )
    twin = clone(est)
    assert twin.hidden_sizes == est.hidden_sizes
    assert twin.activation == est.activation
    assert twin.dropout == est.dropout
    assert twin.learning_rate == est.learning_rate
    assert twin.weight_decay == est.weight_decay
    assert twin.epochs == est.epochs
    assert twin.random_state == est.random_state
    assert not hasattr(twin, "predictor_")


def test_grid_search_runs(logistic_tsm_df):
    est = RieszNet(
        estimand=TSM(level=1),
        hidden_sizes=(8,),
        epochs=10,
        random_state=0,
    )
    grid = {"learning_rate": [1e-3, 5e-3], "dropout": [0.0, 0.2]}
    gs = GridSearchCV(est, grid, cv=2, n_jobs=1, refit=True)
    gs.fit(logistic_tsm_df)
    assert hasattr(gs, "best_params_")
    pred = gs.predict(logistic_tsm_df)
    assert pred.shape == (len(logistic_tsm_df),)


def test_cross_val_predict_runs(logistic_tsm_df):
    est = RieszNet(
        estimand=TSM(level=1),
        hidden_sizes=(8,),
        epochs=10,
        random_state=0,
    )
    cv = KFold(n_splits=3, shuffle=True, random_state=0)
    pred = cross_val_predict(est, logistic_tsm_df, cv=cv, n_jobs=1)
    assert pred.shape == (len(logistic_tsm_df),)
    assert np.all(np.isfinite(pred))
