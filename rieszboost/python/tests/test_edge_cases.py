"""Edge cases — boundary inputs and degenerate configurations.

Each one is a one-liner that would have a non-obvious failure mode and
benefits from being pinned in the test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rieszboost import ATE, LocalShift, RieszBooster


def _logit(z):
    return 1.0 / (1.0 + np.exp(-z))


def _df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    pi = _logit(-0.02 * x - x**2 + 4 * np.log(x + 0.3) + 1.5)
    a = rng.binomial(1, pi)
    return pd.DataFrame({"a": a.astype(float), "x": x})


def test_extra_columns_in_dataframe_are_ignored():
    """Columns not in estimand.feature_keys should pass through silently —
    not raise, not affect the fit."""
    df = _df(seed=1)
    df["irrelevant_col"] = np.arange(len(df))
    df["another"] = "string_data"  # would crash if naively passed to xgboost
    booster = RieszBooster(
        estimand=ATE(), n_estimators=20, learning_rate=0.1, max_depth=3,
    ).fit(df)
    assert booster.predict(df).shape == (len(df),)


def test_single_row_dataframe():
    """Fitting on a single row should not crash; predictions are degenerate."""
    df = pd.DataFrame({"a": [1.0], "x": [0.5]})
    booster = RieszBooster(
        estimand=ATE(), n_estimators=5, learning_rate=0.1, max_depth=3,
    ).fit(df)
    pred = booster.predict(df)
    assert pred.shape == (1,) and np.isfinite(pred).all()


def test_all_treated_input():
    """ATE on data with no controls. The augmentation still emits both
    counterfactual rows; the booster fits but α̂ at A=0 is extrapolation."""
    df = pd.DataFrame({"a": np.ones(200), "x": np.linspace(0, 1, 200)})
    booster = RieszBooster(
        estimand=ATE(), n_estimators=20, learning_rate=0.1, max_depth=3,
    ).fit(df)
    pred = booster.predict(df)
    assert pred.shape == (200,) and np.isfinite(pred).all()


def test_all_control_input():
    df = pd.DataFrame({"a": np.zeros(200), "x": np.linspace(0, 1, 200)})
    booster = RieszBooster(
        estimand=ATE(), n_estimators=20, learning_rate=0.1, max_depth=3,
    ).fit(df)
    pred = booster.predict(df)
    assert pred.shape == (200,) and np.isfinite(pred).all()


def test_local_shift_all_above_threshold_returns_no_counterfactuals():
    """LocalShift only adds counterfactual rows for subjects with a < threshold;
    if every subject is above, augmentation is just the original points (no
    linear pull). Booster trains but should produce near-zero predictions."""
    df = pd.DataFrame({"a": np.full(200, 5.0), "x": np.linspace(0, 1, 200)})
    booster = RieszBooster(
        estimand=LocalShift(delta=1.0, threshold=0.0),
        n_estimators=20, learning_rate=0.1, max_depth=3,
    ).fit(df)
    pred = booster.predict(df)
    assert pred.shape == (200,) and np.isfinite(pred).all()
    # Without any linear pull, predictions should stay near init=0.
    assert float(np.max(np.abs(pred))) < 0.5


def test_early_stopping_without_validation_fraction_raises():
    """early_stopping_rounds without validation_fraction>0 (and no eval_set)
    raises a clear error: the orchestrator does not auto-split."""
    df = _df(n=400, seed=2)
    booster = RieszBooster(
        estimand=ATE(),
        n_estimators=200,
        early_stopping_rounds=10,
        validation_fraction=0.0,  # explicit
        learning_rate=0.1, max_depth=3,
    )
    with pytest.raises(ValueError, match="validation"):
        booster.fit(df)


def test_eval_set_overrides_internal_split():
    df = _df(n=400, seed=3)
    df_valid = _df(n=100, seed=4)
    booster = RieszBooster(
        estimand=ATE(),
        n_estimators=100, early_stopping_rounds=10,
        validation_fraction=0.0,  # ignored when eval_set provided
        learning_rate=0.1, max_depth=3,
    ).fit(df, eval_set=df_valid)
    # eval_set was used → best_iteration_ should be set (or model finished).
    assert booster.predict(df).shape == (len(df),)


def test_predict_on_unseen_extreme_x():
    """The booster should produce finite (if extrapolated) predictions on
    inputs outside the training range."""
    df = _df(n=300, seed=5)
    booster = RieszBooster(
        estimand=ATE(), n_estimators=30, learning_rate=0.1, max_depth=3,
    ).fit(df)
    extreme = pd.DataFrame({"a": [0.0, 1.0], "x": [-2.0, 3.0]})
    pred = booster.predict(extreme)
    assert pred.shape == (2,) and np.isfinite(pred).all()


def test_ndarray_input_with_wrong_n_features_errors():
    df = _df(n=100, seed=6)
    booster = RieszBooster(estimand=ATE(), n_estimators=5).fit(df)
    bad_Z = np.random.default_rng(0).uniform(size=(10, 5))  # 5 cols, expected 2
    with pytest.raises(ValueError, match="feature columns"):
        booster.predict(bad_Z)
