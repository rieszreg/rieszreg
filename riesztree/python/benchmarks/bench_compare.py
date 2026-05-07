"""Speed comparison: riesztree vs state-of-the-art tree libraries.

Phase 6 closes the headline fit-time gap to state-of-the-art tree
libraries on equivalent (n_aug, p, max_depth) workloads. The
augmented-Riesz problem doesn't have an exact equivalent in standard
tree libraries (their objective is `(y - α)²`, not `D·α² + 2C·α`),
so this bench compares **infrastructure speed** rather than
end-to-end task speed:

  - riesztree: fit a tree on the augmented (n_aug × p) data, with
    SquaredLoss.
  - sklearn DecisionTreeRegressor: fit a regression tree on a
    fresh random regression with the same n_aug × p shape.
  - HistGradientBoostingRegressor (max_iter=1): single-tree
    histogram path; sklearn's analog of LightGBM.
  - LightGBM (n_estimators=1, num_leaves large): the reference
    histogram tree.
  - XGBoost (n_estimators=1, tree_method='hist'): same family.

Goal: the riesztree fit time should be in the same order of
magnitude as the comparator at the same (n_aug, p, max_depth). Any
gap larger than ~3× has a concrete attribution.

Run from anywhere outside the rieszreg / worktree directory::

    cd /tmp
    /Users/aschuler/Desktop/RieszReg/.venv/bin/python \\
        /Users/aschuler/Desktop/RieszReg/riesztree/python/benchmarks/bench_compare.py
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from rieszreg import ATE, SquaredLoss
from riesztree import RieszTreeRegressor


@dataclass
class Result:
    library: str
    n_aug: int
    p: int
    max_depth: int
    fit_seconds: float
    extra: str = ""


def _fit_riesztree(n: int, p: int, depth: int, splitter: str, seed: int = 0) -> Result:
    """Fit a riesztree on the augmented dataset of n original rows × p
    features. ``n_aug`` ≈ 2n for ATE."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, p))
    pi = 1.0 / (1.0 + np.exp(-0.5 * X[:, 0]))
    a = (rng.uniform(0, 1, size=n) < pi).astype(float)
    df = pd.DataFrame({**{f"x{j}": X[:, j] for j in range(p)}, "a": a})

    estimand = ATE(treatment="a", covariates=tuple(f"x{j}" for j in range(p)))
    est = RieszTreeRegressor(
        estimand=estimand, loss=SquaredLoss(),
        max_depth=depth, splitter=splitter, random_state=seed,
        # Effectively no leaf-size constraint so we get a fully-grown
        # tree to ``max_depth``.
        min_samples_split=2, min_samples_leaf=1,
        max_leaf_nodes=2 ** 30,
    )
    gc.collect()
    t0 = time.perf_counter()
    est.fit(df)
    fit_s = time.perf_counter() - t0
    # n_aug ≈ 2n for ATE; pull the actual count off the predictor's tree.
    n_aug = int(getattr(est.predictor_.tree, "n_aug", 2 * n))
    return Result(f"riesztree-{splitter}", n_aug, p, depth, fit_s)


def _fit_sklearn(n_aug: int, p: int, depth: int, seed: int = 0) -> Result:
    """Fit sklearn DecisionTreeRegressor on a random regression of
    matching ``(n_aug, p)`` shape."""
    from sklearn.tree import DecisionTreeRegressor
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n_aug, p))
    y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(0.0, 0.5, size=n_aug)
    est = DecisionTreeRegressor(
        max_depth=depth, min_samples_leaf=1, min_samples_split=2,
        random_state=seed,
    )
    gc.collect()
    t0 = time.perf_counter()
    est.fit(X, y)
    fit_s = time.perf_counter() - t0
    return Result("sklearn-DTR", n_aug, p, depth, fit_s)


def _fit_sklearn_hgb(n_aug: int, p: int, depth: int, seed: int = 0) -> Result:
    """Single-tree HistGradientBoostingRegressor."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n_aug, p))
    y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(0.0, 0.5, size=n_aug)
    est = HistGradientBoostingRegressor(
        max_iter=1, max_depth=depth, learning_rate=1.0,
        max_leaf_nodes=None, random_state=seed,
    )
    gc.collect()
    t0 = time.perf_counter()
    est.fit(X, y)
    fit_s = time.perf_counter() - t0
    return Result("sklearn-HGB(max_iter=1)", n_aug, p, depth, fit_s)


def _fit_lightgbm(n_aug: int, p: int, depth: int, seed: int = 0) -> Result | None:
    try:
        import lightgbm as lgb
    except ImportError:
        return None
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n_aug, p))
    y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(0.0, 0.5, size=n_aug)
    est = lgb.LGBMRegressor(
        n_estimators=1, max_depth=depth, num_leaves=2 ** depth,
        min_child_samples=1, random_state=seed, verbose=-1,
    )
    gc.collect()
    t0 = time.perf_counter()
    est.fit(X, y)
    fit_s = time.perf_counter() - t0
    return Result("lightgbm(n_estimators=1)", n_aug, p, depth, fit_s)


def _fit_xgboost(n_aug: int, p: int, depth: int, seed: int = 0) -> Result | None:
    try:
        import xgboost as xgb
    except ImportError:
        return None
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n_aug, p))
    y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(0.0, 0.5, size=n_aug)
    est = xgb.XGBRegressor(
        n_estimators=1, max_depth=depth, tree_method="hist",
        random_state=seed, verbosity=0,
    )
    gc.collect()
    t0 = time.perf_counter()
    est.fit(X, y)
    fit_s = time.perf_counter() - t0
    return Result("xgboost(n_estimators=1)", n_aug, p, depth, fit_s)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ns", type=int, nargs="+", default=[10_000, 50_000])
    parser.add_argument("--ps", type=int, nargs="+", default=[5, 20])
    parser.add_argument("--depths", type=int, nargs="+", default=[8, 16])
    args = parser.parse_args()

    print(f"# python={sys.version.split()[0]}")
    print(f"# {'library':<28} {'n_orig':>6} {'n_aug':>6} {'p':>4} {'depth':>5} {'fit (s)':>10}")
    print("# " + "-" * 70)

    results: list[Result] = []
    for n in args.ns:
        for p in args.ps:
            for depth in args.depths:
                # riesztree exact + hist
                rt_exact = _fit_riesztree(n, p, depth, "exact")
                rt_hist = _fit_riesztree(n, p, depth, "hist")
                n_aug = rt_exact.n_aug
                # Comparators on the same n_aug shape
                results_at_cell = [
                    rt_exact, rt_hist,
                    _fit_sklearn(n_aug, p, depth),
                    _fit_sklearn_hgb(n_aug, p, depth),
                    _fit_lightgbm(n_aug, p, depth),
                    _fit_xgboost(n_aug, p, depth),
                ]
                for r in results_at_cell:
                    if r is None:
                        continue
                    print(
                        f"  {r.library:<28} {n:>6} {r.n_aug:>6} {r.p:>4} "
                        f"{r.max_depth:>5} {r.fit_seconds:>10.3f}"
                    )
                    results.append(r)
                print("#")

    return 0


if __name__ == "__main__":
    sys.exit(main())
