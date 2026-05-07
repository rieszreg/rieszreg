"""Tests for `KernelRieszRegressor.predict_path` — α̂ at every λ in the
sweep, returned in one call by reusing the test-side kernel slab.

The optimization's correctness claim is mathematical equivalence: column ``j``
of `predict_path(X)` equals a fresh fit at `lambda_grid=[lambdas[j]]` with the
same training data, since the per-λ dual γ comes from the same
eigendecomposition (direct solver) or the same Nyström landmarks (nystrom_cg).
"""

from __future__ import annotations

import numpy as np
import pytest

from krrr import ATE, Gaussian, KernelRieszRegressor


def _krr(lambda_grid, **kwargs) -> KernelRieszRegressor:
    return KernelRieszRegressor(
        estimand=ATE("a", ("x",)),
        kernel=Gaussian(length_scale=1.0),
        lambda_grid=lambda_grid,
        solver="direct",
        validation_fraction=0.25,
        random_state=0,
        **kwargs,
    )


def test_predict_path_shape_and_columns(binary_ate_data):
    df, _, _ = binary_ate_data
    grid = [1e-3, 1e-2, 1e-1, 1.0]
    krr = _krr(grid).fit(df)
    path = krr.predict_path(df)
    assert path.shape == (len(df), len(grid))
    assert np.all(np.isfinite(path))


def test_predict_path_columns_match_single_lambda_fits(binary_ate_data):
    df, _, _ = binary_ate_data
    grid = [1e-3, 1e-2, 1e-1, 1.0]
    full = _krr(grid).fit(df)
    path = full.predict_path(df)
    for j, lam in enumerate(grid):
        single = _krr([lam]).fit(df)
        np.testing.assert_allclose(path[:, j], single.predict(df), rtol=0, atol=1e-10)


def test_predict_path_best_column_matches_predict(binary_ate_data):
    df, _, _ = binary_ate_data
    grid = [1e-3, 1e-2, 1e-1, 1.0]
    krr = _krr(grid).fit(df)
    path = krr.predict_path(df)
    best_idx = int(krr.best_iteration_)
    np.testing.assert_array_equal(path[:, best_idx], krr.predict(df))


def test_predict_path_subset_lambdas(binary_ate_data):
    df, _, _ = binary_ate_data
    grid = [1e-3, 1e-2, 1e-1, 1.0]
    krr = _krr(grid).fit(df)
    full = krr.predict_path(df)
    subset = krr.predict_path(df, lambdas=[1e-3, 1e-1])
    assert subset.shape == (len(df), 2)
    np.testing.assert_array_equal(subset[:, 0], full[:, 0])
    np.testing.assert_array_equal(subset[:, 1], full[:, 2])


def test_predict_path_unknown_lambda_raises(binary_ate_data):
    df, _, _ = binary_ate_data
    krr = _krr([1e-3, 1e-2, 1e-1]).fit(df)
    with pytest.raises(ValueError, match="not in stored"):
        krr.predict_path(df, lambdas=[42.0])


def test_predict_path_keep_path_false_raises(binary_ate_data):
    df, _, _ = binary_ate_data
    krr = _krr([1e-3, 1e-2], keep_path=False).fit(df)
    # predict() still works
    krr.predict(df)
    with pytest.raises(RuntimeError, match="keep_path=True"):
        krr.predict_path(df)


def test_predict_unchanged_when_keep_path_default(binary_ate_data):
    df, _, _ = binary_ate_data
    grid = [1e-3, 1e-2, 1e-1, 1.0]
    a = _krr(grid, keep_path=True).fit(df).predict(df)
    b = _krr(grid, keep_path=False).fit(df).predict(df)
    np.testing.assert_array_equal(a, b)


def test_predict_path_round_trip(tmp_path, binary_ate_data):
    df, _, _ = binary_ate_data
    grid = [1e-3, 1e-2, 1e-1, 1.0]
    krr = _krr(grid).fit(df)
    expected = krr.predict_path(df)
    krr.save(tmp_path / "krr")
    loaded = KernelRieszRegressor.load(tmp_path / "krr")
    actual = loaded.predict_path(df)
    np.testing.assert_array_equal(actual, expected)


def test_rff_solver_predict_path_matches_single_fits(binary_ate_data):
    df, _, _ = binary_ate_data
    grid = [1e-3, 1e-2, 1e-1, 1.0]
    full = KernelRieszRegressor(
        estimand=ATE("a", ("x",)),
        kernel=Gaussian(length_scale=1.0),
        lambda_grid=grid,
        solver="rff",
        n_features=128,
        validation_fraction=0.25,
        random_state=0,
    ).fit(df)
    path = full.predict_path(df)
    for j, lam in enumerate(grid):
        single = KernelRieszRegressor(
            estimand=ATE("a", ("x",)),
            kernel=Gaussian(length_scale=1.0),
            lambda_grid=[lam],
            solver="rff",
            n_features=128,
            validation_fraction=0.25,
            random_state=0,
        ).fit(df)
        np.testing.assert_allclose(path[:, j], single.predict(df), rtol=0, atol=1e-10)
