"""Parity: Cython best-split sweep ≡ pure-Python reference sweep.

Locks the contract that the Cython kernel behind ``splitter='exact'``
produces the *same partition* as :func:`riesztree.splitter.best_split_continuous`
on the four built-in Bregman-Riesz losses. Any drift means the
optimisation also changed the algorithm.
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
)
from riesztree.fast._splitter import best_split_continuous_fast, loss_kind_for
from riesztree.splitter import best_split_continuous, make_leaf_solvers


def _make_DC(n=400, seed=0):
    """Synthetic (D, C, x) tuple with the augmented-data invariants:
    D >= 0; C unbounded; D > 0 ⇒ original row."""
    rng = np.random.default_rng(seed)
    D = (rng.uniform(0, 1, size=n) > 0.5).astype(float)  # bernoulli, ~50% original
    # C is anti-symmetric per pair on average. Use a feature-dependent C.
    x = rng.normal(0.0, 1.0, size=n)
    C = (rng.normal(0.0, 1.0, size=n) - 0.7 * x) * D + (rng.normal(0.0, 1.0, size=n) + 0.7 * x) * (1 - D)
    idx = np.arange(n, dtype=np.int64)
    return x.astype(np.float64), D, C, idx


@pytest.mark.parametrize(
    "loss",
    [SquaredLoss(), KLLoss(), BernoulliLoss(), BoundedSquaredLoss(lo=-3.0, hi=3.0)],
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cython_continuous_split_matches_python(loss, seed):
    x, D, C, idx = _make_DC(n=400, seed=seed)
    leaf_loss, _alpha = make_leaf_solvers(loss)
    py_split = best_split_continuous(x, D, C, idx, leaf_loss, min_orig_leaf=10)

    kind, lo, hi, addr = loss_kind_for(loss)
    cy_split = best_split_continuous_fast(
        x, D, C, idx,
        loss_kind=kind, bounded_lo=lo, bounded_hi=hi,
        min_orig_leaf=10, user_cfunc_addr=addr,
    )

    if py_split is None and cy_split is None:
        return
    assert py_split is not None and cy_split is not None
    py_gain, py_thr, py_l, py_r = py_split
    cy_gain, cy_thr, cy_l, cy_r = cy_split
    assert py_gain == pytest.approx(cy_gain, abs=1e-12)
    assert py_thr == pytest.approx(cy_thr, abs=1e-12)
    np.testing.assert_array_equal(py_l, cy_l)
    np.testing.assert_array_equal(py_r, cy_r)


def test_splitter_round_trips_through_save_load(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(300, 3)), columns=["x0", "x1", "x2"])
    df["a"] = (rng.uniform(size=300) < 1 / (1 + np.exp(-df["x0"]))).astype(float)
    est = RieszTreeRegressor(
        estimand=ATE(treatment="a", covariates=("x0", "x1", "x2")),
        max_depth=4, splitter="hist",
    ).fit(df)
    est.save(str(tmp_path / "tree"))
    loaded = RieszTreeRegressor.load(str(tmp_path / "tree"))
    assert loaded.splitter == "hist"
    np.testing.assert_array_equal(loaded.predict(df), est.predict(df))
