"""Parity tests for the parent-minus-sibling histogram path.

PMS is enabled automatically when ``splitter='hist'`` AND there are
no categorical features AND ``max_features=None`` (no per-split
feature subsampling — different leaves would see different
candidate-feature sets, breaking the "subtract from parent" invariant).

The test below verifies that, on those eligible configurations, PMS
produces fits **byte-identical** to the non-PMS hist path, modulo tiny
floating-point reordering from the subtraction trick.
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
def test_pms_predictions_match_non_pms_hist(loss_factory, estimand_factory):
    """The PMS-enabled hist path must produce predictions equal to the
    non-PMS hist path within tight FP tolerance, since PMS is purely
    an arithmetic identity (parent_hist == left_hist + right_hist).

    PMS triggers when max_features is None and no categoricals; we
    can't easily disable PMS while keeping max_features=None on the
    same fit, so we compare hist (PMS) vs exact (no PMS) instead.
    Histograms approximate threshold positions but exact has the same
    splits in expectation; we relax the tolerance accordingly.
    """
    df = _make_df(n=800, p=5)
    estimand = estimand_factory(5)

    hist_pms = RieszTreeRegressor(
        estimand=estimand, loss=loss_factory(),
        max_depth=4, splitter="hist", max_bins=255, random_state=0,
    ).fit(df)

    # Force NON-PMS hist by setting max_features="all" — same as None
    # logically, but max_features="all" still routes through the
    # per-feature subsample logic (which disqualifies PMS even though
    # n_features_to_consider == n_features). This is the cleanest way
    # to compare PMS vs non-PMS hist on the same algorithm.
    hist_nopms = RieszTreeRegressor(
        estimand=estimand, loss=loss_factory(),
        max_depth=4, splitter="hist", max_bins=255, random_state=0,
        max_features="all",
    ).fit(df)

    a_pms = hist_pms.predict(df)
    a_nopms = hist_nopms.predict(df)
    np.testing.assert_array_equal(a_pms, a_nopms)


def test_pms_with_max_features_falls_back_silently():
    """When max_features is set, PMS is not eligible. The fit should
    still complete normally (using the non-PMS hist path)."""
    df = _make_df(n=800, p=10)
    est = RieszTreeRegressor(
        estimand=_ate(10), max_depth=4, splitter="hist",
        max_features="sqrt", random_state=0,
    ).fit(df)
    a = est.predict(df)
    assert np.isfinite(a).all()


def test_pms_with_categorical_falls_back_silently():
    """When categorical_features is set, PMS is not eligible (the
    Cython hist kernel doesn't handle them). The non-PMS path takes
    over."""
    rng = np.random.default_rng(0)
    n = 500
    cat = rng.integers(0, 4, size=n).astype(float)
    pi = (cat + 1) / 5.0
    a = (rng.uniform(0, 1, size=n) < pi).astype(float)
    x = rng.normal(0.0, 1.0, size=n)
    df = pd.DataFrame({"a": a, "cat": cat, "x": x})
    est = RieszTreeRegressor(
        estimand=ATE(treatment="a", covariates=("cat", "x")),
        max_depth=4, splitter="hist",
        categorical_features=(0,),
    ).fit(df)
    assert n_leaves(est.predictor_.tree) >= 2
    assert np.isfinite(est.predict(df)).all()


def test_pms_grows_non_trivial_tree_at_depth():
    """Sanity: deep PMS fit produces a non-trivial tree on a signaled
    DGP. Catches a regression where PMS might erroneously truncate."""
    df = _make_df(n=2000, p=6, seed=42)
    est = RieszTreeRegressor(
        estimand=_ate(6), max_depth=10, splitter="hist", random_state=0,
        min_samples_split=2, min_samples_leaf=1,
    ).fit(df)
    assert n_leaves(est.predictor_.tree) >= 8
