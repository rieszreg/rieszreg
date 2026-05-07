"""sklearn-parity hyperparameters: max_features, min_impurity_decrease,
min_weight_fraction_leaf, random_state, plus the deprecation aliases for
``pruning_alpha`` (→ ``ccp_alpha``) and ``max_leaves`` (→ ``max_leaf_nodes``).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from riesztree import ATE, RieszTreeRegressor, n_leaves
from riesztree.grow import _resolve_max_features


# ---------------------------------------------------------------------------
# max_features

@pytest.mark.parametrize(
    "spec, n_features, expected",
    [
        (None, 10, 10),
        ("all", 10, 10),
        ("sqrt", 16, 4),
        ("sqrt", 10, 3),
        ("log2", 8, 3),
        ("log2", 10, 3),
        (5, 10, 5),
        (100, 10, 10),
        (0.5, 10, 5),
        (0.3, 10, 3),
        (1.0, 10, 10),
    ],
)
def test_resolve_max_features_matches_sklearn_convention(spec, n_features, expected):
    assert _resolve_max_features(spec, n_features) == expected


@pytest.mark.parametrize("spec", ["unknown", -1, 1.5, -0.1, 0.0])
def test_resolve_max_features_rejects_bad_inputs(spec):
    with pytest.raises((ValueError, TypeError)):
        _resolve_max_features(spec, 10)


def _make_multi_x_df(n=600, p=8, seed=0):
    """DGP with multiple informative features so `max_features='sqrt'`
    subsampling reliably finds positive-gain splits."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, p))
    # Spread signal across the first three covariates so a 'sqrt'
    # feature-subsample of 3 out of 8 has near-certain probability of
    # including at least one informative column.
    logit = 0.6 * X[:, 0] + 0.5 * X[:, 1] - 0.4 * X[:, 2]
    pi = 1.0 / (1.0 + np.exp(-logit))
    a = (rng.uniform(0, 1, size=n) < pi).astype(float)
    cols = {f"x{j}": X[:, j] for j in range(p)}
    cols["a"] = a
    return pd.DataFrame(cols)


def test_max_features_full_matches_default():
    """max_features=None and max_features='all' should produce the same tree
    as the default (no subsampling)."""
    df = _make_multi_x_df()
    estimand = ATE(treatment="a", covariates=tuple(f"x{j}" for j in range(8)))
    a_default = RieszTreeRegressor(estimand=estimand, max_depth=4, random_state=0).fit(df).predict(df)
    a_none = RieszTreeRegressor(estimand=estimand, max_depth=4, random_state=0, max_features=None).fit(df).predict(df)
    a_all = RieszTreeRegressor(estimand=estimand, max_depth=4, random_state=0, max_features="all").fit(df).predict(df)
    assert np.allclose(a_default, a_none)
    assert np.allclose(a_default, a_all)


def test_max_features_sqrt_changes_tree():
    """Subsampling features should produce a different tree than considering
    all features (with overwhelming probability on a non-trivial DGP)."""
    df = _make_multi_x_df(n=800, p=10)
    estimand = ATE(treatment="a", covariates=tuple(f"x{j}" for j in range(10)))
    full = RieszTreeRegressor(estimand=estimand, max_depth=5, random_state=0).fit(df)
    sub = RieszTreeRegressor(estimand=estimand, max_depth=5, random_state=0, max_features="sqrt").fit(df)
    assert not np.allclose(full.predict(df), sub.predict(df))


def test_max_features_random_state_reproducibility():
    """Same random_state ⇒ identical subsample ⇒ identical tree."""
    df = _make_multi_x_df(n=600, p=10)
    estimand = ATE(treatment="a", covariates=tuple(f"x{j}" for j in range(10)))
    a = RieszTreeRegressor(estimand=estimand, max_depth=4, max_features=5, random_state=42).fit(df).predict(df)
    b = RieszTreeRegressor(estimand=estimand, max_depth=4, max_features=5, random_state=42).fit(df).predict(df)
    assert np.allclose(a, b)


def test_max_features_different_seeds_differ():
    """With moderate subsampling (max_features=5 of 10) and three informative
    features, two different seeds produce two different fitted trees."""
    df = _make_multi_x_df(n=800, p=10)
    estimand = ATE(treatment="a", covariates=tuple(f"x{j}" for j in range(10)))
    a_est = RieszTreeRegressor(estimand=estimand, max_depth=5, max_features=5, random_state=3).fit(df)
    b_est = RieszTreeRegressor(estimand=estimand, max_depth=5, max_features=5, random_state=11).fit(df)
    # Sanity: both fits must actually grow, otherwise the test isn't measuring
    # what it claims.
    assert n_leaves(a_est.predictor_.tree) > 1
    assert n_leaves(b_est.predictor_.tree) > 1
    assert not np.allclose(a_est.predict(df), b_est.predict(df))


# ---------------------------------------------------------------------------
# min_impurity_decrease

def test_min_impurity_decrease_blocks_low_gain_splits():
    """A very large min_impurity_decrease should leave the tree at the root."""
    df = _make_multi_x_df(n=400, p=5)
    estimand = ATE(treatment="a", covariates=tuple(f"x{j}" for j in range(5)))
    huge = RieszTreeRegressor(
        estimand=estimand, max_depth=8, min_impurity_decrease=1e9
    ).fit(df)
    assert n_leaves(huge.predictor_.tree) == 1


def test_min_impurity_decrease_zero_default_grows_tree():
    df = _make_multi_x_df(n=600, p=5)
    estimand = ATE(treatment="a", covariates=tuple(f"x{j}" for j in range(5)))
    grown = RieszTreeRegressor(
        estimand=estimand, max_depth=4, min_impurity_decrease=0.0
    ).fit(df)
    assert n_leaves(grown.predictor_.tree) > 1


# ---------------------------------------------------------------------------
# min_weight_fraction_leaf

def test_min_weight_fraction_leaf_caps_tree_size():
    """Setting min_weight_fraction_leaf=0.4 forces leaves to hold ≥ 40% of
    original rows, so at most ~2 leaves are admissible."""
    df = _make_multi_x_df(n=400, p=4)
    estimand = ATE(treatment="a", covariates=tuple(f"x{j}" for j in range(4)))
    constrained = RieszTreeRegressor(
        estimand=estimand,
        max_depth=8,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.4,
    ).fit(df)
    assert n_leaves(constrained.predictor_.tree) <= 2


def test_min_weight_fraction_leaf_zero_is_identity():
    """min_weight_fraction_leaf=0.0 must behave exactly like the default."""
    df = _make_multi_x_df()
    estimand = ATE(treatment="a", covariates=tuple(f"x{j}" for j in range(8)))
    a = RieszTreeRegressor(estimand=estimand, max_depth=4).fit(df).predict(df)
    b = RieszTreeRegressor(estimand=estimand, max_depth=4, min_weight_fraction_leaf=0.0).fit(df).predict(df)
    assert np.allclose(a, b)


# ---------------------------------------------------------------------------
# Deprecation aliases: pruning_alpha → ccp_alpha, max_leaves → max_leaf_nodes

def test_pruning_alpha_alias_emits_future_warning():
    df = _make_multi_x_df(n=300, p=4)
    estimand = ATE(treatment="a", covariates=tuple(f"x{j}" for j in range(4)))
    with pytest.warns(FutureWarning, match="pruning_alpha.*ccp_alpha"):
        RieszTreeRegressor(estimand=estimand, max_depth=4, pruning_alpha=0.1).fit(df)


def test_pruning_alpha_alias_matches_ccp_alpha_behavior():
    df = _make_multi_x_df(n=400, p=4)
    estimand = ATE(treatment="a", covariates=tuple(f"x{j}" for j in range(4)))
    canonical = RieszTreeRegressor(estimand=estimand, max_depth=5, ccp_alpha=0.5).fit(df).predict(df)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        legacy = RieszTreeRegressor(estimand=estimand, max_depth=5, pruning_alpha=0.5).fit(df).predict(df)
    assert np.allclose(canonical, legacy)


def test_max_leaves_alias_emits_future_warning():
    df = _make_multi_x_df(n=300, p=4)
    estimand = ATE(treatment="a", covariates=tuple(f"x{j}" for j in range(4)))
    with pytest.warns(FutureWarning, match="max_leaves.*max_leaf_nodes"):
        RieszTreeRegressor(
            estimand=estimand,
            max_depth=4,
            growth_policy="leafwise",
            max_leaves=8,
        ).fit(df)


def test_max_leaves_alias_matches_max_leaf_nodes_behavior():
    df = _make_multi_x_df(n=400, p=4)
    estimand = ATE(treatment="a", covariates=tuple(f"x{j}" for j in range(4)))
    canonical = RieszTreeRegressor(
        estimand=estimand,
        max_depth=10,
        growth_policy="leafwise",
        max_leaf_nodes=8,
    ).fit(df).predict(df)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        legacy = RieszTreeRegressor(
            estimand=estimand,
            max_depth=10,
            growth_policy="leafwise",
            max_leaves=8,
        ).fit(df).predict(df)
    assert np.allclose(canonical, legacy)


def test_no_alias_no_warning():
    """Default construction with canonical names emits no FutureWarning."""
    df = _make_multi_x_df(n=300, p=4)
    estimand = ATE(treatment="a", covariates=tuple(f"x{j}" for j in range(4)))
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        RieszTreeRegressor(
            estimand=estimand,
            max_depth=4,
            ccp_alpha=0.1,
            max_leaf_nodes=8,
            growth_policy="leafwise",
        ).fit(df)


# ---------------------------------------------------------------------------
# save/load round-trips the new hyperparameters

def test_save_load_roundtrips_new_hyperparams(tmp_path):
    df = _make_multi_x_df(n=400, p=6)
    estimand = ATE(treatment="a", covariates=tuple(f"x{j}" for j in range(6)))
    est = RieszTreeRegressor(
        estimand=estimand,
        max_depth=4,
        max_features="sqrt",
        min_impurity_decrease=1e-3,
        min_weight_fraction_leaf=0.05,
        ccp_alpha=0.01,
        max_leaf_nodes=20,
        random_state=7,
    ).fit(df)
    path = tmp_path / "tree"
    est.save(str(path))
    loaded = RieszTreeRegressor.load(str(path))
    assert loaded.max_features == "sqrt"
    assert loaded.min_impurity_decrease == 1e-3
    assert loaded.min_weight_fraction_leaf == 0.05
    assert loaded.ccp_alpha == 0.01
    assert loaded.max_leaf_nodes == 20
    assert loaded.random_state == 7
    assert np.allclose(est.predict(df), loaded.predict(df))
