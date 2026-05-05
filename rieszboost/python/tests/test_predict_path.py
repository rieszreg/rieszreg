"""Tests for `RieszBooster.predict_path` — α̂ at every tree count in a grid
extracted from a single boosting fit via xgboost's `iteration_range`.

The optimization's correctness claim is bit-equality with single-point refits:
fitting at `n_estimators=k` and predicting must give identical numbers to
`fit(n_estimators=K).predict_path(Z, [k])` for any k ≤ K. Same training data,
same seed.
"""

import numpy as np
import pandas as pd
import pytest

import rieszboost
from rieszboost import RieszBooster


def _logit(z):
    return 1.0 / (1.0 + np.exp(-z))


def _simulate(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    pi = _logit(-0.02 * x - x**2 + 4.0 * np.log(x + 0.3) + 1.5)
    a = rng.binomial(1, pi)
    return pd.DataFrame({"a": a.astype(float), "x": x.astype(float)})


def _booster(n_estimators: int) -> RieszBooster:
    return RieszBooster(
        estimand=rieszboost.ATE(),
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=4,
        random_state=0,
    )


def test_predict_path_matches_single_fit_bit_equal():
    df = _simulate(400, seed=1)
    big = _booster(50).fit(df)
    grid = [5, 17, 50]
    path = big.predict_path(df, grid)
    assert path.shape == (len(df), len(grid))
    for j, k in enumerate(grid):
        small = _booster(k).fit(df)
        np.testing.assert_array_equal(path[:, j], small.predict(df))


def test_predict_path_full_column_matches_predict():
    df = _simulate(300, seed=2)
    booster = _booster(40).fit(df)
    path = booster.predict_path(df, [40])
    np.testing.assert_array_equal(path[:, 0], booster.predict(df))


def test_predict_path_validates_grid_range():
    df = _simulate(200, seed=3)
    booster = _booster(20).fit(df)
    with pytest.raises(ValueError, match="must satisfy"):
        booster.predict_path(df, [0])
    with pytest.raises(ValueError, match="must satisfy"):
        booster.predict_path(df, [21])
    with pytest.raises(ValueError, match="non-empty"):
        booster.predict_path(df, [])


def test_predict_path_unfitted_raises():
    df = _simulate(100, seed=4)
    booster = _booster(10)
    with pytest.raises(RuntimeError, match="not fitted"):
        booster.predict_path(df, [5])


def test_predict_path_round_trip(tmp_path):
    df = _simulate(300, seed=5)
    booster = _booster(40).fit(df)
    grid = [10, 25, 40]
    expected = booster.predict_path(df, grid)
    booster.save(tmp_path / "m")
    loaded = RieszBooster.load(tmp_path / "m")
    actual = loaded.predict_path(df, grid)
    np.testing.assert_array_equal(actual, expected)
