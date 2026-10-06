"""Parity tests for the Cython whole-tree histogram driver.

With ``splitter='hist'``, no categorical features, no per-split feature
subsampling and no early stopping, the tree is grown by the Cython
worklist driver (which uses parent-minus-sibling histogram subtraction).
Other configurations use the Python grower, calling the Cython histogram
kernel per node. Both must grow the same tree.
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


@pytest.mark.parametrize(
    "loss_factory, estimand_factory",
    [
        (lambda: SquaredLoss(), lambda p: _ate(p)),
        (
            lambda: KLLoss(),
            lambda p: TSM(treatment="a", covariates=tuple(f"x{j}" for j in range(p)), level=1.0),
        ),
        (lambda: BernoulliLoss(), lambda p: _ate(p)),
        (lambda: BoundedSquaredLoss(lo=-3.0, hi=3.0), lambda p: _ate(p)),
    ],
)
def test_cython_hist_driver_matches_python_grower(loss_factory, estimand_factory, monkeypatch):
    """The Cython driver and the Python grower pick identical splits."""
    import riesztree.grow as grow

    df = _make_df(n=800, p=5)
    estimand = estimand_factory(5)
    kwargs = dict(max_depth=4, splitter="hist", max_bins=255, random_state=0)
    driver = RieszTreeRegressor(estimand=estimand, loss=loss_factory(), **kwargs).fit(df)
    monkeypatch.setattr(grow._Grower, "compiled_eligible", property(lambda self: False))
    python = RieszTreeRegressor(estimand=estimand, loss=loss_factory(), **kwargs).fit(df)
    assert n_leaves(driver.predictor_.tree) == n_leaves(python.predictor_.tree)
    np.testing.assert_allclose(driver.predict(df), python.predict(df), rtol=1e-12, atol=1e-12)
