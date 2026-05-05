"""Save/load round-trip tests for RieszBooster."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import rieszboost
from rieszboost import RieszBooster
from rieszboost.backends import SklearnBackend
from rieszboost.estimand import FiniteEvalEstimand
from rieszboost.losses import KLLoss


def _logit(z):
    return 1.0 / (1.0 + np.exp(-z))


def _df(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    pi = _logit(-0.02 * x - x**2 + 4 * np.log(x + 0.3) + 1.5)
    a = rng.binomial(1, pi)
    return pd.DataFrame({"a": a.astype(float), "x": x})


def test_xgboost_squared_round_trip(tmp_path):
    df = _df(seed=0)
    booster = RieszBooster(
        estimand=rieszboost.ATE(),
        n_estimators=30, learning_rate=0.1, max_depth=3,
    ).fit(df)
    pre = booster.predict(df)
    booster.save(tmp_path / "m")
    loaded = RieszBooster.load(tmp_path / "m")
    np.testing.assert_array_equal(pre, loaded.predict(df))
    assert loaded.best_iteration_ == booster.best_iteration_
    assert loaded.score(df) == pytest.approx(booster.score(df), rel=1e-9)


def test_round_trip_preserves_best_iteration_with_early_stopping(tmp_path):
    df = _df(n=800, seed=1)
    booster = RieszBooster(
        estimand=rieszboost.ATE(),
        n_estimators=500,
        early_stopping_rounds=10,
        validation_fraction=0.2,
        learning_rate=0.05, max_depth=3,
    ).fit(df)
    assert booster.best_iteration_ is not None
    pre = booster.predict(df)
    booster.save(tmp_path / "m")
    loaded = RieszBooster.load(tmp_path / "m")
    assert loaded.best_iteration_ == booster.best_iteration_
    np.testing.assert_array_equal(pre, loaded.predict(df))


def test_round_trip_per_built_in_estimand(tmp_path):
    df = _df(seed=2)
    cases = [
        ("ATE", rieszboost.ATE()),
        ("ATT", rieszboost.ATT()),
        ("TSM", rieszboost.TSM(level=1)),
        ("AdditiveShift", rieszboost.AdditiveShift(delta=0.1)),
        ("LocalShift", rieszboost.LocalShift(delta=0.1, threshold=0.5)),
    ]
    for name, est in cases:
        b = RieszBooster(estimand=est, n_estimators=15, learning_rate=0.1).fit(df)
        b.save(tmp_path / name)
        loaded = RieszBooster.load(tmp_path / name)
        assert loaded.estimand.name == est.name, name
        np.testing.assert_array_equal(b.predict(df), loaded.predict(df))


def test_round_trip_with_kl_loss(tmp_path):
    df = _df(seed=3)
    b = RieszBooster(
        estimand=rieszboost.TSM(level=1),
        loss=KLLoss(max_eta=40.0),
        n_estimators=20, learning_rate=0.05, max_depth=3,
    ).fit(df)
    pre = b.predict(df)
    b.save(tmp_path / "kl")
    loaded = RieszBooster.load(tmp_path / "kl")
    assert loaded.loss_.max_eta == 40.0
    np.testing.assert_array_equal(pre, loaded.predict(df))


def test_round_trip_with_sklearn_backend(tmp_path):
    from sklearn.tree import DecisionTreeRegressor
    df = _df(seed=4)
    b = RieszBooster(
        estimand=rieszboost.ATE(),
        backend=SklearnBackend(lambda: DecisionTreeRegressor(max_depth=3, random_state=0)),
        n_estimators=20, learning_rate=0.05,
    ).fit(df)
    pre = b.predict(df)
    b.save(tmp_path / "sk")
    loaded = RieszBooster.load(tmp_path / "sk")
    np.testing.assert_array_equal(pre, loaded.predict(df))


def test_custom_estimand_requires_explicit_estimand_on_load(tmp_path):
    df = _df(seed=5)
    def m_custom(alpha):
        def inner(z, y=None):
            return alpha(a=1, x=z["x"]) - alpha(a=0, x=z["x"])
        return inner
    custom = FiniteEvalEstimand(feature_keys=("a", "x"), m=m_custom, name="my_custom")
    b = RieszBooster(estimand=custom, n_estimators=10).fit(df)
    b.save(tmp_path / "custom")

    # No estimand passed → raise
    with pytest.raises(ValueError, match="custom"):
        RieszBooster.load(tmp_path / "custom")

    # With estimand passed → succeeds
    loaded = RieszBooster.load(tmp_path / "custom", estimand=custom)
    np.testing.assert_array_equal(b.predict(df), loaded.predict(df))


def test_save_unfitted_raises(tmp_path):
    b = RieszBooster(estimand=rieszboost.ATE())
    with pytest.raises(RuntimeError, match="unfitted"):
        b.save(tmp_path / "bad")


# ---- joblib (sklearn idiom) ----

def test_joblib_round_trip_default(tmp_path):
    """`joblib.dump(booster, path)` / `joblib.load(path)` works on a fitted
    booster with the default backend. This is the sklearn-native way to
    persist any BaseEstimator."""
    import joblib
    df = _df(seed=10)
    b = RieszBooster(
        estimand=rieszboost.ATE(),
        n_estimators=30, learning_rate=0.1, max_depth=3,
    ).fit(df)
    pre = b.predict(df)
    p = tmp_path / "booster.pkl"
    joblib.dump(b, p)
    loaded = joblib.load(p)
    np.testing.assert_array_equal(pre, loaded.predict(df))
    assert loaded.best_iteration_ == b.best_iteration_


@pytest.mark.parametrize("name,estimand", [
    ("ATE", rieszboost.ATE()),
    ("ATT", rieszboost.ATT()),
    ("TSM", rieszboost.TSM(level=1)),
    ("AdditiveShift", rieszboost.AdditiveShift(delta=0.5)),
    ("LocalShift", rieszboost.LocalShift(delta=0.5, threshold=0.7)),
])
def test_joblib_round_trip_per_estimand(name, estimand, tmp_path):
    import joblib
    df = _df(seed=11)
    b = RieszBooster(
        estimand=estimand,
        n_estimators=15, learning_rate=0.1, max_depth=3,
    ).fit(df)
    pre = b.predict(df)
    p = tmp_path / f"{name}.pkl"
    joblib.dump(b, p)
    loaded = joblib.load(p)
    np.testing.assert_array_equal(pre, loaded.predict(df))


def test_joblib_round_trip_with_kl_loss(tmp_path):
    import joblib
    df = _df(seed=12)
    b = RieszBooster(
        estimand=rieszboost.TSM(level=1),
        loss=KLLoss(max_eta=40.0),
        n_estimators=20, learning_rate=0.1, max_depth=3,
    ).fit(df)
    pre = b.predict(df)
    p = tmp_path / "kl.pkl"
    joblib.dump(b, p)
    loaded = joblib.load(p)
    assert loaded.loss_.max_eta == 40.0
    np.testing.assert_array_equal(pre, loaded.predict(df))


def test_joblib_supports_clone_after_load(tmp_path):
    """Loaded booster should still compose with sklearn machinery."""
    import joblib
    from sklearn.base import clone
    df = _df(seed=13)
    b = RieszBooster(
        estimand=rieszboost.ATE(),
        n_estimators=10, learning_rate=0.1, max_depth=3,
    ).fit(df)
    p = tmp_path / "b.pkl"
    joblib.dump(b, p)
    loaded = joblib.load(p)
    cloned = clone(loaded)
    assert cloned.estimand.name == "ATE"
    assert not hasattr(cloned, "predictor_")
