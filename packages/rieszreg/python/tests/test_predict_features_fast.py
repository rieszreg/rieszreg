"""``_features_from_Z``: DataFrame / ndarray → ``feature_keys``-ordered floats."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rieszreg.estimands import ATE
from rieszreg.estimator import _features_from_Z


def _make_df(n=400, p=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, p))
    a = (rng.uniform(0, 1, size=n) > 0.5).astype(float)
    cols = {f"x{j}": X[:, j] for j in range(p)}
    cols["a"] = a
    return pd.DataFrame(cols)


def _ate(p):
    return ATE(treatment="a", covariates=tuple(f"x{j}" for j in range(p)))


def test_features_from_Z_orders_dataframe_columns_by_feature_keys():
    df = _make_df(n=200, p=5)
    estimand = _ate(5)
    expected = df[list(estimand.feature_keys)].to_numpy(dtype=float)
    np.testing.assert_array_equal(_features_from_Z(df, estimand), expected)
    np.testing.assert_array_equal(_features_from_Z(expected, estimand), expected)


def test_features_from_Z_matches_integer_column_labels():
    """``bind`` names columns by ``str(label)``; integer labels must resolve."""
    arr = _make_df(n=20, p=2)[["a", "x0", "x1"]].to_numpy()
    df = pd.DataFrame(arr)  # columns 0, 1, 2
    estimand = ATE(treatment="0").bind(list(df.columns))
    np.testing.assert_array_equal(_features_from_Z(df, estimand), arr)
    # The labels themselves work too, with or without explicit covariates.
    for est in (ATE(treatment=0, covariates=[1, 2]), ATE(treatment=0)):
        np.testing.assert_array_equal(_features_from_Z(df, est.bind(list(df.columns))), arr)


def test_features_from_Z_rejects_missing_columns():
    df = _make_df(n=50, p=3)
    estimand = ATE(treatment="a", covariates=("x0", "x1", "missing_col"))
    with pytest.raises(ValueError, match="missing"):
        _features_from_Z(df, estimand)


def test_features_from_Z_rejects_wrong_shape():
    df = _make_df(n=50, p=3)
    estimand = _ate(3)
    arr = df[list(estimand.feature_keys)].to_numpy()
    bad = arr[:, :-1]  # drop a column
    with pytest.raises(ValueError, match="expects"):
        _features_from_Z(bad, estimand)
