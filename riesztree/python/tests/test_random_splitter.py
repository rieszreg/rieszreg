"""Phase 7: ``splitter='random'`` (sklearn ExtraTrees-style).

Random splitter draws a single uniform threshold per feature per leaf
and evaluates the gain there — no per-feature sweep. Useful for
ExtraTrees-style forests where extra randomization improves
decorrelation, and for "completely random" baselines.

Trees fit with ``splitter='random'`` are necessarily different from
``splitter='exact'`` trees (different threshold positions). The tests
below verify:
  - The fit completes without errors on all four built-in losses.
  - ``random_state`` makes the result reproducible.
  - Different seeds produce different trees.
  - Behaviour matches sklearn's convention: at least one split is
    found whenever the columns are non-constant.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riesztree import (
    ATE,
    BernoulliLoss,
    BoundedSquaredLoss,
    KLLoss,
    RieszTreeRegressor,
    SquaredLoss,
    TSM,
)
from riesztree.tree import n_leaves


def _make_df(n=600, p=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, p))
    logit = 0.6 * X[:, 0] + 0.4 * X[:, 1]
    pi = 1.0 / (1.0 + np.exp(-logit))
    a = (rng.uniform(0, 1, size=n) < pi).astype(float)
    cols = {f"x{j}": X[:, j] for j in range(p)}
    cols["a"] = a
    return pd.DataFrame(cols)


def _ate(p):
    return ATE(treatment="a", covariates=tuple(f"x{j}" for j in range(p)))


# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "loss_factory, estimand_factory",
    [
        (lambda: SquaredLoss(), lambda p: _ate(p)),
        (
            lambda: KLLoss(),
            lambda p: TSM(
                treatment="a", covariates=tuple(f"x{j}" for j in range(p)), level=1.0
            ),
        ),
        (lambda: BernoulliLoss(), lambda p: _ate(p)),
        (lambda: BoundedSquaredLoss(lo=-3.0, hi=3.0), lambda p: _ate(p)),
    ],
)
def test_random_splitter_fits_all_builtin_losses(loss_factory, estimand_factory):
    df = _make_df(n=800, p=5)
    estimand = estimand_factory(5)
    est = RieszTreeRegressor(
        estimand=estimand, loss=loss_factory(),
        max_depth=4, splitter="random", random_state=0,
    ).fit(df)
    a = est.predict(df)
    assert np.isfinite(a).all()


def test_random_splitter_grows_a_tree_on_signaled_dgp():
    df = _make_df(n=800, p=5)
    est = RieszTreeRegressor(
        estimand=_ate(5), max_depth=5, splitter="random", random_state=0,
    ).fit(df)
    # Random thresholds are sometimes unproductive; over depth=5 with 5
    # informative-ish features we should get at least 2 leaves with high
    # probability.
    assert n_leaves(est.predictor_.tree) >= 2


def test_random_splitter_random_state_reproducibility():
    df = _make_df(n=600, p=4)
    a = RieszTreeRegressor(
        estimand=_ate(4), max_depth=4, splitter="random", random_state=42
    ).fit(df).predict(df)
    b = RieszTreeRegressor(
        estimand=_ate(4), max_depth=4, splitter="random", random_state=42
    ).fit(df).predict(df)
    np.testing.assert_array_equal(a, b)


def test_random_splitter_different_seeds_produce_different_trees():
    df = _make_df(n=800, p=5)
    a_est = RieszTreeRegressor(
        estimand=_ate(5), max_depth=5, splitter="random", random_state=0
    ).fit(df)
    b_est = RieszTreeRegressor(
        estimand=_ate(5), max_depth=5, splitter="random", random_state=7
    ).fit(df)
    a_pred = a_est.predict(df)
    b_pred = b_est.predict(df)
    # Sanity: both fits grew non-trivially.
    assert n_leaves(a_est.predictor_.tree) > 1
    assert n_leaves(b_est.predictor_.tree) > 1
    # Different seeds → different threshold draws → different predictions.
    assert not np.array_equal(a_pred, b_pred)


def test_random_splitter_round_trips_through_save_load(tmp_path):
    df = _make_df(n=300, p=3)
    est = RieszTreeRegressor(
        estimand=_ate(3), max_depth=4, splitter="random"
    ).fit(df)
    path = tmp_path / "tree"
    est.save(str(path))
    loaded = RieszTreeRegressor.load(str(path))
    assert loaded.splitter == "random"


def test_random_splitter_handles_constant_column_gracefully():
    """A column with no variation should be skipped by the random splitter
    (the threshold range collapses); other columns can still contribute."""
    rng = np.random.default_rng(0)
    n = 400
    a = (rng.uniform(0, 1, size=n) > 0.5).astype(float)
    cols = {
        "a": a,
        "x0": rng.normal(0.0, 1.0, size=n),
        "x1": np.zeros(n),       # constant — should be skipped
        "x2": rng.normal(0.0, 1.0, size=n),
    }
    df = pd.DataFrame(cols)
    est = RieszTreeRegressor(
        estimand=ATE(treatment="a", covariates=("x0", "x1", "x2")),
        max_depth=4, splitter="random", random_state=0,
    ).fit(df)
    assert n_leaves(est.predictor_.tree) >= 2  # other features still split
    assert np.isfinite(est.predict(df)).all()
