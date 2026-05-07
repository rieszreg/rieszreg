"""Phase 6: histogram splitter (`splitter='hist'`).

Histograms are *approximate* — the optimal split is forced onto a bin
boundary rather than a distinct value. With ``max_bins=255`` on
quantile-sampled data the gap to the exact split is typically tiny.
The tests below pin two contracts:

1. The leaf α* values are exact (binning doesn't move sums of D and
   C within a leaf), so per-config RMSE drift from the exact splitter
   should be small.
2. The hist splitter always finds a non-trivial split when the exact
   splitter does — it shouldn't degenerate to a stump on
   non-degenerate data.
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
from riesztree.fast._binner import fit_bin_mapper, transform
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
# Binner correctness

def test_bin_mapper_low_cardinality_is_exact():
    """A feature with ≤ max_bins distinct values uses one bin per value."""
    X = np.array([[1.0], [2.0], [3.0], [2.0], [1.0]])
    mapper = fit_bin_mapper(X, max_bins=10)
    assert mapper.n_bins[0] == 3
    binned = transform(X, mapper)
    # bin 0 = value 1.0, bin 1 = 2.0, bin 2 = 3.0
    np.testing.assert_array_equal(binned.ravel(), [0, 1, 2, 1, 0])


def test_bin_mapper_high_cardinality_uses_quantiles():
    rng = np.random.default_rng(0)
    X = rng.normal(0.0, 1.0, size=(2000, 1))
    mapper = fit_bin_mapper(X, max_bins=10)
    assert 1 < mapper.n_bins[0] <= 10
    binned = transform(X, mapper)
    # Each bin should contain roughly equal counts (quantile-balanced).
    counts = np.bincount(binned.ravel(), minlength=mapper.n_bins[0])
    assert counts.min() > 0
    # Quantile-balanced: max count is at most ~3× the min.
    assert counts.max() <= 3 * counts.min()


def test_bin_mapper_max_bins_cap():
    rng = np.random.default_rng(0)
    X = rng.normal(0.0, 1.0, size=(2000, 3))
    mapper = fit_bin_mapper(X, max_bins=8)
    assert (mapper.n_bins <= 8).all()


# ---------------------------------------------------------------------------
# End-to-end fit: hist vs exact on the four built-in losses

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
def test_hist_fit_close_to_exact(loss_factory, estimand_factory):
    """Histogram and exact splitters should agree closely on predicted
    α (RMSE drift small) on a non-trivial DGP. Exact byte-equality is
    not expected because bins discretize thresholds."""
    df = _make_df(n=1500, p=5)
    estimand = estimand_factory(5)
    exact = RieszTreeRegressor(
        estimand=estimand, loss=loss_factory(),
        max_depth=4, splitter="exact", random_state=0,
    ).fit(df)
    hist = RieszTreeRegressor(
        estimand=estimand, loss=loss_factory(),
        max_depth=4, splitter="hist", max_bins=255, random_state=0,
    ).fit(df)
    a_exact = exact.predict(df)
    a_hist = hist.predict(df)
    rmse = float(np.sqrt(np.mean((a_hist - a_exact) ** 2)))
    range_ = float(a_exact.max() - a_exact.min())
    if range_ == 0.0:
        # Exact path didn't split (e.g. Bernoulli with all augmented
        # rows infeasible on this seed). Hist should agree exactly.
        assert rmse == 0.0
        return
    # Allow drift up to 20% of the prediction range — generous floor;
    # in practice the drift is much smaller.
    assert rmse < 0.20 * range_, (
        f"hist vs exact RMSE drift {rmse:.4f} exceeds 20% of range {range_:.4f}"
    )


def test_hist_grows_non_trivial_tree():
    """On a clearly-signaled DGP, the hist splitter must find at least
    one positive-gain split."""
    df = _make_df(n=800, p=3)
    est = RieszTreeRegressor(estimand=_ate(3), max_depth=4, splitter="hist").fit(df)
    assert n_leaves(est.predictor_.tree) >= 2


def test_hist_smaller_max_bins_still_works():
    """``max_bins=8`` is aggressively coarse but should still produce a
    fitted tree without errors."""
    df = _make_df(n=800, p=3)
    est = RieszTreeRegressor(
        estimand=_ate(3), max_depth=5, splitter="hist", max_bins=8
    ).fit(df)
    a = est.predict(df)
    assert np.isfinite(a).all()


# ---------------------------------------------------------------------------
# Hyperparameter handling

def test_max_bins_round_trips_through_save_load(tmp_path):
    df = _make_df(n=300, p=3)
    est = RieszTreeRegressor(
        estimand=_ate(3), max_depth=4, splitter="hist", max_bins=64
    ).fit(df)
    path = tmp_path / "tree"
    est.save(str(path))
    loaded = RieszTreeRegressor.load(str(path))
    assert loaded.max_bins == 64
    assert loaded.splitter == "hist"


def test_invalid_splitter_value_raises():
    df = _make_df(n=200, p=3)
    with pytest.raises(ValueError, match="splitter"):
        RieszTreeRegressor(estimand=_ate(3), splitter="approx").fit(df)


def test_max_bins_too_large_raises():
    df = _make_df(n=200, p=3)
    with pytest.raises(ValueError, match="max_bins"):
        RieszTreeRegressor(
            estimand=_ate(3), splitter="hist", max_bins=300
        ).fit(df)


# ---------------------------------------------------------------------------
# Mixed continuous + categorical: hist on continuous, Python on categorical

def test_hist_handles_mixed_continuous_categorical():
    rng = np.random.default_rng(0)
    n = 600
    cat = rng.integers(0, 4, size=n).astype(float)
    pi_per = rng.uniform(0.2, 0.8, size=4)
    pi = pi_per[cat.astype(int)]
    a = (rng.uniform(0, 1, size=n) < pi).astype(float)
    x = rng.normal(0.0, 1.0, size=n)
    df = pd.DataFrame({"a": a, "cat": cat, "x": x})
    est = RieszTreeRegressor(
        estimand=ATE(treatment="a", covariates=("cat", "x")),
        max_depth=4, splitter="hist",
        categorical_features=(0,),
    ).fit(df)
    a_hat = est.predict(df)
    assert n_leaves(est.predictor_.tree) >= 2
    assert np.isfinite(a_hat).all()
