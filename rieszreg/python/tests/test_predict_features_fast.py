"""Predict-path vectorisation: ``_features_from_Z`` byte-equivalent to the
old ``_rows_from_Z + _features_from_rows`` round-trip, but cheaper.

These tests pin the new fast path so a future refactor can't silently
diverge from the legacy row-dict pivot. Speed is asserted as a *ratio*
to keep the test stable across machines.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from rieszreg.estimands import ATE
from rieszreg.estimator import (
    _features_from_rows,
    _features_from_Z,
    _rows_from_Z,
)


def _make_df(n=400, p=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, p))
    a = (rng.uniform(0, 1, size=n) > 0.5).astype(float)
    cols = {f"x{j}": X[:, j] for j in range(p)}
    cols["a"] = a
    return pd.DataFrame(cols)


def _ate(p):
    return ATE(treatment="a", covariates=tuple(f"x{j}" for j in range(p)))


def test_features_from_Z_dataframe_matches_legacy_pivot():
    df = _make_df(n=200, p=5)
    estimand = _ate(5)
    fast = _features_from_Z(df, estimand)
    legacy = _features_from_rows(_rows_from_Z(df, estimand), estimand)
    np.testing.assert_array_equal(fast, legacy)


def test_features_from_Z_ndarray_matches_legacy_pivot():
    df = _make_df(n=200, p=5)
    estimand = _ate(5)
    arr = df[list(estimand.feature_keys)].to_numpy()
    fast = _features_from_Z(arr, estimand)
    legacy = _features_from_rows(_rows_from_Z(arr, estimand), estimand)
    np.testing.assert_array_equal(fast, legacy)


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


def test_features_from_Z_is_meaningfully_faster_than_legacy():
    """The fast path should be at least 5× faster than the row-dict pivot
    on a moderately wide DataFrame. The headline ratio is much higher
    (~100×); 5× is a generous floor that survives noise across machines."""
    df = _make_df(n=2000, p=20)
    estimand = _ate(20)

    n_iters = 5
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _features_from_Z(df, estimand)
    fast_s = (time.perf_counter() - t0) / n_iters

    t0 = time.perf_counter()
    for _ in range(n_iters):
        _features_from_rows(_rows_from_Z(df, estimand), estimand)
    legacy_s = (time.perf_counter() - t0) / n_iters

    assert fast_s * 5 < legacy_s, (
        f"_features_from_Z should be ≥ 5× faster than the row-dict pivot; "
        f"got fast={fast_s*1e3:.2f} ms vs legacy={legacy_s*1e3:.2f} ms"
    )
