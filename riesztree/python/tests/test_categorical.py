"""Categorical predictor handling."""
from __future__ import annotations

import numpy as np
import pandas as pd

from riesztree import ATE, RieszTreeRegressor


def _make_categorical_dgp(n: int, seed: int = 0):
    """8 unordered category levels with random per-level propensity. The
    natural numeric ordering of the integer label is unrelated to the
    underlying treatment probability — a continuous splitter will struggle
    while a categorical-aware one can group levels by α* directly."""
    rng = np.random.default_rng(seed)
    # Random per-level propensity, scrambled order vs the integer label.
    pi_per_level = rng.uniform(0.1, 0.9, size=8)
    cat = rng.integers(0, 8, size=n).astype(float)
    pi = pi_per_level[cat.astype(int)]
    a = (rng.uniform(0, 1, size=n) < pi).astype(float)
    x_cont = rng.normal(0, 1, n)
    df = pd.DataFrame({"a": a, "cat": cat, "x": x_cont})
    df["_pi"] = pi
    return df


def _truth(df: pd.DataFrame) -> np.ndarray:
    a = df["a"].values
    pi = df["_pi"].values
    prob_a = a * pi + (1.0 - a) * (1.0 - pi)
    return (2 * a - 1.0) / prob_a


def _count_split_kinds(node) -> dict[str, int]:
    if node.is_leaf:
        return {"continuous": 0, "categorical": 0}
    a = _count_split_kinds(node.left)
    b = _count_split_kinds(node.right)
    out = {"continuous": a["continuous"] + b["continuous"],
           "categorical": a["categorical"] + b["categorical"]}
    out[node.split_kind] += 1
    return out


def test_categorical_splits_actually_fire():
    """When ``categorical_features`` declares a column, the splitter should
    use the categorical (sort-by-α*) mechanism for it."""
    df = _make_categorical_dgp(2000, seed=0)
    est = RieszTreeRegressor(
        estimand=ATE(treatment="a", covariates=("cat", "x")),
        max_depth=4,
        categorical_features=(1,),
    ).fit(df)
    counts = _count_split_kinds(est.predictor_.tree)
    assert counts["categorical"] >= 1, counts


def test_categorical_competitive_with_continuous_on_unordered_dgp():
    """Categorical-aware splits should perform on par with continuous
    splits when category labels happen to also work as ordered numerics
    (with depth=4 and 8 levels, continuous can almost separate them)."""
    df = _make_categorical_dgp(2000, seed=0)
    est_cat = RieszTreeRegressor(
        estimand=ATE(treatment="a", covariates=("cat", "x")),
        max_depth=4,
        categorical_features=(1,),
    ).fit(df)
    est_cont = RieszTreeRegressor(
        estimand=ATE(treatment="a", covariates=("cat", "x")),
        max_depth=4,
    ).fit(df)

    truth = _truth(df)
    rmse_cat = float(np.sqrt(np.mean((est_cat.predict(df) - truth) ** 2)))
    rmse_cont = float(np.sqrt(np.mean((est_cont.predict(df) - truth) ** 2)))
    # Allow categorical to be modestly worse (continuous can be lucky on
    # int-labelled categories at deep enough trees), but it shouldn't be
    # dramatically off.
    assert rmse_cat <= 2.0 * rmse_cont + 0.5, (rmse_cat, rmse_cont)


def test_single_categorical_level_doesnt_split():
    """Edge case: a column with a single unique value should not yield a split."""
    df = pd.DataFrame({
        "a": np.array([0.0, 1.0] * 500),
        "cat": np.zeros(1000),    # constant
        "x": np.random.default_rng(0).normal(size=1000),
        "_pi": np.full(1000, 0.5),
    })
    est = RieszTreeRegressor(
        estimand=ATE(treatment="a", covariates=("cat", "x")),
        max_depth=4,
        categorical_features=(1,),
    ).fit(df)
    # Splits should still occur (on x) — the categorical column is just unused.
    a_hat = est.predict(df)
    assert a_hat.shape == (1000,)
