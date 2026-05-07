"""Bench harness for ``riesztree`` ``fit`` and ``predict``.

Times ``RieszTreeRegressor.fit`` and ``RieszTreeRegressor.predict`` over a
grid of (loss, n, p, max_depth, growth_policy). Writes a tidy CSV and a
short summary to stdout. Used to lock a baseline before any optimisation
work, and re-run after each optimisation phase to report deltas.

Run from anywhere outside the rieszreg / worktree directory (the cwd must
not contain a ``rieszreg`` subdirectory; Python's namespace-package
machinery would shadow the editable install)::

    cd /tmp
    /Users/aschuler/Desktop/RieszReg/.venv/bin/python \\
        /Users/aschuler/Desktop/RieszReg/riesztree/python/benchmarks/bench_fit.py \\
        --out /Users/aschuler/Desktop/RieszReg/riesztree/python/benchmarks/results/baseline.csv

Default grid is the *small* grid: completes in a few minutes on the current
pure-Python path. ``--grid full`` enables the headline target grid; expect
runtime in the hours on the pure-Python path. ``--grid quick`` is a
~30-second smoke test for CI / iteration.
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd

# Imports that require the editable installs of rieszreg + riesztree on path.
from rieszreg import ATE, KLLoss, SquaredLoss, TSM
from riesztree import RieszTreeRegressor
from riesztree.tree import max_depth as _tree_max_depth
from riesztree.tree import n_leaves as _tree_n_leaves


# ---------------------------------------------------------------------------
# DGPs (multivariate-X versions of the canonical rieszreg.testing DGPs)
# ---------------------------------------------------------------------------

def linear_gaussian_ate(n: int, p: int, rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """Multivariate-X linear-Gaussian ATE DGP.

    A ~ Bernoulli(π(X)) with logit π(X) = β · X[0]; X ~ N(0, I_p). Closed-form
    Riesz representer: α₀(a, x) = (2a − 1) / [a · π(x) + (1−a)·(1−π(x))].
    """
    X = rng.normal(0.0, 1.0, size=(n, p))
    logit = 0.5 * X[:, 0]
    pi = 1.0 / (1.0 + np.exp(-logit))
    a = (rng.uniform(0, 1, size=n) < pi).astype(float)
    df = pd.DataFrame(X, columns=[f"x{j}" for j in range(p)])
    df.insert(0, "a", a)
    df["y"] = a + 0.5 * X[:, 0] + rng.normal(0.0, 1.0, size=n)
    prob_a = a * pi + (1.0 - a) * (1.0 - pi)
    sign = 2.0 * a - 1.0
    true_alpha = sign / prob_a
    return df, true_alpha


def logistic_tsm(n: int, p: int, rng: np.random.Generator, level: float = 1.0) -> tuple[pd.DataFrame, np.ndarray]:
    """Multivariate-X TSM DGP. α₀(a, x) = 1[a=level] / π(x|level)."""
    X = rng.normal(0.0, 1.0, size=(n, p))
    logit = 0.5 * X[:, 0]
    pi = 1.0 / (1.0 + np.exp(-logit))
    a = (rng.uniform(0, 1, size=n) < pi).astype(float)
    df = pd.DataFrame(X, columns=[f"x{j}" for j in range(p)])
    df.insert(0, "a", a)
    df["y"] = (a == level).astype(float) + 0.5 * X[:, 0] + rng.normal(0.0, 1.0, size=n)
    prob_a = a * pi + (1.0 - a) * (1.0 - pi)
    true_alpha = (a == level).astype(float) / prob_a
    return df, true_alpha


# ---------------------------------------------------------------------------
# Bench config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    loss: str               # "squared" | "kl"
    n: int
    p: int
    max_depth: int | None
    growth_policy: str      # "depthwise" | "leafwise"
    seed: int = 0
    n_predict: int = 10_000
    timeout_s: float = 600.0


@dataclass
class Result:
    loss: str
    n: int
    p: int
    max_depth: int | None
    growth_policy: str
    fit_seconds: float
    predict_seconds: float
    n_leaves: int
    max_depth_actual: int
    rmse: float
    n_aug_train: int
    timed_out: bool = False
    error: str = ""


def _make_estimator(cfg: Config) -> tuple[RieszTreeRegressor, "Estimand", object]:  # type: ignore[name-defined]
    if cfg.loss == "squared":
        estimand = ATE(treatment="a", covariates=tuple(f"x{j}" for j in range(cfg.p)))
        loss = SquaredLoss()
    elif cfg.loss == "kl":
        estimand = TSM(treatment="a", covariates=tuple(f"x{j}" for j in range(cfg.p)), level=1.0)
        loss = KLLoss()
    else:
        raise ValueError(f"unknown loss: {cfg.loss!r}")

    est = RieszTreeRegressor(
        estimand=estimand,
        loss=loss,
        max_depth=cfg.max_depth if cfg.max_depth is not None else 10**9,
        min_samples_split=20,
        min_samples_leaf=10,
        max_leaf_nodes=2**30,  # effectively unbounded for depthwise
        growth_policy=cfg.growth_policy,
        random_state=cfg.seed,
    )
    return est, estimand, loss


def run_one(cfg: Config) -> Result:
    rng = np.random.default_rng(cfg.seed)
    if cfg.loss == "squared":
        train_df, _ = linear_gaussian_ate(cfg.n, cfg.p, rng)
        test_df, true_alpha_test = linear_gaussian_ate(cfg.n_predict, cfg.p, rng)
    else:
        train_df, _ = logistic_tsm(cfg.n, cfg.p, rng)
        test_df, true_alpha_test = logistic_tsm(cfg.n_predict, cfg.p, rng)

    est, _, _ = _make_estimator(cfg)

    def _failed(e: Exception) -> Result:
        return Result(
            loss=cfg.loss, n=cfg.n, p=cfg.p, max_depth=cfg.max_depth,
            growth_policy=cfg.growth_policy, fit_seconds=float("nan"),
            predict_seconds=float("nan"), n_leaves=-1, max_depth_actual=-1,
            rmse=float("nan"), n_aug_train=-1, error=f"{type(e).__name__}: {e}",
        )

    gc.collect()
    t0 = time.perf_counter()
    try:
        est.fit(train_df)
    except Exception as e:  # noqa: BLE001 — bench records failures, doesn't crash
        return _failed(e)
    fit_s = time.perf_counter() - t0

    try:
        t0 = time.perf_counter()
        alpha_hat = est.predict(test_df)
        predict_s = time.perf_counter() - t0
        rmse = float(np.sqrt(np.mean((np.asarray(alpha_hat) - true_alpha_test) ** 2)))
        # Tree shape metrics are derivable from the predictor's tree directly,
        # without going through diagnose_tree (which calls riesz_loss and trips
        # on KLLoss leaves with α=0).
        root = getattr(est.predictor_, "tree", None)
        if root is not None:
            n_leaves = _tree_n_leaves(root)
            md_actual = _tree_max_depth(root)
            n_aug = int(getattr(root, "n_orig", -1))
        else:
            n_leaves = md_actual = n_aug = -1
    except Exception as e:  # noqa: BLE001
        return _failed(e)

    return Result(
        loss=cfg.loss, n=cfg.n, p=cfg.p, max_depth=cfg.max_depth,
        growth_policy=cfg.growth_policy,
        fit_seconds=fit_s, predict_seconds=predict_s,
        n_leaves=int(n_leaves),
        max_depth_actual=int(md_actual),
        rmse=rmse, n_aug_train=n_aug,
    )


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------

def grid_quick() -> list[Config]:
    cfgs = []
    for loss in ("squared",):
        for n in (1_000,):
            for p in (5,):
                for d in (4, 8):
                    for gp in ("depthwise",):
                        cfgs.append(Config(loss=loss, n=n, p=p, max_depth=d, growth_policy=gp))
    return cfgs


def grid_small() -> list[Config]:
    cfgs = []
    for loss in ("squared", "kl"):
        for n in (1_000, 10_000):
            for p in (5, 20):
                for d in (8, 16):
                    for gp in ("depthwise", "leafwise"):
                        cfgs.append(Config(loss=loss, n=n, p=p, max_depth=d, growth_policy=gp))
    return cfgs


def grid_full() -> list[Config]:
    """Headline target grid (per the plan).

    Pure-Python path will not finish many of these in reasonable time —
    exactly the point of the grid. Phases that introduce the fast paths
    will be re-evaluated against this grid.
    """
    cfgs = []
    for loss in ("squared", "kl"):
        for n in (10_000, 100_000, 1_000_000):
            for p in (10, 50):
                for d in (8, 16, None):
                    for gp in ("depthwise", "leafwise"):
                        cfgs.append(Config(loss=loss, n=n, p=p, max_depth=d, growth_policy=gp))
    return cfgs


GRIDS = {"quick": grid_quick, "small": grid_small, "full": grid_full}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _result_row(r: Result) -> dict:
    d = asdict(r)
    return d


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="riesztree fit/predict bench")
    parser.add_argument("--grid", choices=list(GRIDS), default="small")
    parser.add_argument("--out", type=str, default=None,
                        help="CSV path to write results to (created if missing)")
    parser.add_argument("--label", type=str, default="baseline",
                        help="Tag stored in every row to identify the run")
    parser.add_argument("--filter-loss", type=str, default=None,
                        choices=["squared", "kl"])
    parser.add_argument("--max-fit-seconds", type=float, default=300.0,
                        help="Skip remaining configs once any fit exceeds this wall time")
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfgs = GRIDS[args.grid]()
    if args.filter_loss is not None:
        cfgs = [c for c in cfgs if c.loss == args.filter_loss]

    print(f"# riesztree bench — grid={args.grid}, n_configs={len(cfgs)}, label={args.label}")
    print(f"# python={sys.version.split()[0]}  cwd={os.getcwd()}")

    results: list[Result] = []
    skipped = 0
    cap_hit = False
    for i, cfg in enumerate(cfgs, 1):
        if cap_hit:
            skipped += 1
            continue
        print(f"[{i:>3}/{len(cfgs)}] loss={cfg.loss:<7} n={cfg.n:<7} p={cfg.p:<3} "
              f"depth={str(cfg.max_depth):<5} grow={cfg.growth_policy:<10} ... ", end="", flush=True)
        r = run_one(cfg)
        results.append(r)
        if r.error:
            print(f"ERROR  {r.error}")
        else:
            print(f"fit={r.fit_seconds:>7.2f}s  predict={r.predict_seconds:>6.3f}s  "
                  f"leaves={r.n_leaves:<5} depth={r.max_depth_actual:<3} rmse={r.rmse:.3f}")
            if r.fit_seconds > args.max_fit_seconds:
                print(f"# fit_seconds {r.fit_seconds:.0f}s > cap {args.max_fit_seconds:.0f}s — "
                      f"skipping remaining {len(cfgs) - i} configs")
                cap_hit = True

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            if results:
                fieldnames = list(_result_row(results[0]).keys()) + ["label"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in results:
                    row = _result_row(r)
                    row["label"] = args.label
                    writer.writerow(row)
        print(f"# wrote {len(results)} rows to {args.out}")

    if skipped:
        print(f"# skipped {skipped} configs after wall-time cap")

    return 0


if __name__ == "__main__":
    sys.exit(main())
