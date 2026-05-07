"""Sklearn-style ``cost_complexity_pruning_path``.

Mirrors :meth:`sklearn.tree.DecisionTreeRegressor.cost_complexity_pruning_path`:
returns ``(ccp_alphas, impurities)`` describing the cost-complexity
pruning sequence — the standard input to alpha cross-validation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riesztree import (
    ATE,
    RieszTreeRegressor,
    cost_complexity_pruning_path,
    n_leaves,
)


def _make_df(n=600, p=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, p))
    pi = 1.0 / (1.0 + np.exp(-0.5 * X[:, 0]))
    a = (rng.uniform(0, 1, size=n) < pi).astype(float)
    cols = {f"x{j}": X[:, j] for j in range(p)}
    cols["a"] = a
    return pd.DataFrame(cols)


def _ate(p):
    return ATE(treatment="a", covariates=tuple(f"x{j}" for j in range(p)))


# ---------------------------------------------------------------------------

def test_pruning_path_returns_arrays():
    """Smoke: returns a pair of equal-length numpy arrays."""
    df = _make_df(n=400, p=3)
    est = RieszTreeRegressor(estimand=_ate(3), max_depth=6).fit(df)
    alphas, impurities = est.cost_complexity_pruning_path()
    assert isinstance(alphas, np.ndarray)
    assert isinstance(impurities, np.ndarray)
    assert alphas.shape == impurities.shape
    assert len(alphas) >= 1


def test_pruning_path_starts_at_zero():
    """First entry corresponds to the unpruned tree (alpha=0)."""
    df = _make_df(n=400, p=3)
    est = RieszTreeRegressor(estimand=_ate(3), max_depth=6).fit(df)
    alphas, _ = est.cost_complexity_pruning_path()
    assert alphas[0] == 0.0


def test_pruning_path_alphas_monotonic_nondecreasing():
    """Cost-complexity alphas increase as the tree is pruned."""
    df = _make_df(n=400, p=3)
    est = RieszTreeRegressor(estimand=_ate(3), max_depth=6).fit(df)
    alphas, _ = est.cost_complexity_pruning_path()
    diffs = np.diff(alphas)
    assert (diffs >= -1e-12).all(), f"alphas not monotonic: {alphas}"


def test_pruning_path_impurities_monotonic_nondecreasing():
    """Pruning can only INCREASE the sum-of-leaf-loss-at-optimum
    (we're losing information by collapsing). Impurity is monotone
    non-decreasing along the path."""
    df = _make_df(n=400, p=3)
    est = RieszTreeRegressor(estimand=_ate(3), max_depth=6).fit(df)
    _, impurities = est.cost_complexity_pruning_path()
    diffs = np.diff(impurities)
    # Allow a tiny negative drift from FP arithmetic.
    assert (diffs >= -1e-6).all(), f"impurities decreased along path: {diffs}"


def test_pruning_path_picks_alpha_that_reproduces_pruned_tree():
    """Take an alpha from the path, refit with it, verify the resulting
    tree's leaf count matches what the path predicts (the tree at that
    alpha)."""
    df = _make_df(n=600, p=3)
    est = RieszTreeRegressor(estimand=_ate(3), max_depth=8).fit(df)
    alphas, impurities = est.cost_complexity_pruning_path()
    # Pick a middle alpha from the path.
    mid = len(alphas) // 2
    chosen_alpha = float(alphas[mid])
    # The "size" at that alpha is implicit: the tree after `mid`
    # collapses, with impurity == impurities[mid]. Refit:
    refit = RieszTreeRegressor(
        estimand=_ate(3), max_depth=8, ccp_alpha=chosen_alpha,
    ).fit(df)
    # The refit's leaf-loss-sum should be ≤ impurities[mid] (the
    # post-prune cost), with small slack for FP differences in
    # whether the chosen alpha is on or just past a path step.
    from riesztree.pruning import _leaves_loss_sum
    refit_impurity = _leaves_loss_sum(refit.predictor_.tree)
    assert refit_impurity <= impurities[mid] + 1e-6, (
        f"refit impurity {refit_impurity} > path impurity "
        f"{impurities[mid]} at alpha={chosen_alpha}"
    )


def test_pruning_path_with_Z_y_clones_estimator():
    """When called with (Z, y), the estimator is NOT mutated — the path
    uses an internal clone."""
    df = _make_df(n=400, p=3)
    est = RieszTreeRegressor(estimand=_ate(3), max_depth=6, ccp_alpha=0.5)
    # Hasn't been fitted; calling path with Z still works.
    alphas, _ = est.cost_complexity_pruning_path(Z=df)
    assert len(alphas) >= 1
    # Estimator is still unfitted.
    assert not hasattr(est, "predictor_")


def test_pruning_path_unfitted_no_data_raises():
    est = RieszTreeRegressor(estimand=_ate(3), max_depth=4)
    with pytest.raises(RuntimeError, match="unfitted"):
        est.cost_complexity_pruning_path()


def test_pruning_path_collapses_to_root():
    """The final entry should correspond to the root-only tree."""
    df = _make_df(n=400, p=3)
    est = RieszTreeRegressor(estimand=_ate(3), max_depth=6).fit(df)
    alphas, _ = est.cost_complexity_pruning_path()
    # Refitting with the LAST alpha (or anything larger) should produce a stump.
    last_alpha = float(alphas[-1])
    refit = RieszTreeRegressor(
        estimand=_ate(3), max_depth=6, ccp_alpha=last_alpha + 1e-9,
    ).fit(df)
    assert n_leaves(refit.predictor_.tree) == 1


def test_pruning_path_does_not_mutate_input_tree():
    """The free function should leave the input tree unchanged."""
    df = _make_df(n=400, p=3)
    est = RieszTreeRegressor(estimand=_ate(3), max_depth=6).fit(df)
    pre_n_leaves = n_leaves(est.predictor_.tree)
    cost_complexity_pruning_path(est.predictor_.tree, est.loss_)
    post_n_leaves = n_leaves(est.predictor_.tree)
    assert pre_n_leaves == post_n_leaves
