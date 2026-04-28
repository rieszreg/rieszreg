"""Numerical regression baselines.

Pins α-RMSE on a fixed DGP + seed + hyperparameters for a handful of
representative estimands. Refactors that silently change quality become
loud test failures here. The goal isn't to enforce SOTA — it's to detect
drift. Tolerances are generous (10% relative); tighten as the codebase
stabilizes.

Baselines live in `baselines.json` next to this file. To regenerate (e.g.
after an intentional algorithmic improvement), run the script's
`compute_baseline(...)` for the relevant key and update the JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rieszboost import ATE, ATT, LocalShift, RieszBooster


BASELINES_PATH = Path(__file__).parent / "baselines.json"
with open(BASELINES_PATH) as f:
    BASELINES = json.load(f)


def _logit(z):
    return 1.0 / (1.0 + np.exp(-z))


def _binary_dgp(n: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    pi = _logit(-0.02 * x - x**2 + 4 * np.log(x + 0.3) + 1.5)
    a = rng.binomial(1, pi)
    return pd.DataFrame({"a": a.astype(float), "x": x}), pi


def _continuous_dgp(n: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 2, n)
    a = rng.normal(x**2 - 1.0, np.sqrt(2.0))
    return pd.DataFrame({"a": a, "x": x}), x


def compute_baseline(key: str) -> float:
    """Refit the booster for `key` and return its α-RMSE on the training data."""
    if key == "ate_binary_n4000_seed42":
        df, pi = _binary_dgp(4000, seed=42)
        booster = RieszBooster(
            estimand=ATE(), n_estimators=300, learning_rate=0.05,
            max_depth=4, random_state=0,
        ).fit(df)
        a = df["a"].values
        truth = a / pi - (1 - a) / (1 - pi)
    elif key == "att_partial_binary_n4000_seed0":
        df, pi = _binary_dgp(4000, seed=0)
        booster = RieszBooster(
            estimand=ATT(),
            n_estimators=2000, early_stopping_rounds=20,
            validation_fraction=0.2, learning_rate=0.05,
            max_depth=3, reg_lambda=10.0, random_state=0,
        ).fit(df)
        a = df["a"].values
        truth = a - (1 - a) * pi / (1 - pi)
    elif key == "lase_partial_continuous_n4000_seed0":
        df, x = _continuous_dgp(4000, seed=0)
        booster = RieszBooster(
            estimand=LocalShift(delta=1.0, threshold=0.0),
            n_estimators=2000, early_stopping_rounds=20,
            validation_fraction=0.2, learning_rate=0.05,
            max_depth=3, reg_lambda=10.0, random_state=0,
        ).fit(df)
        a = df["a"].values
        density_ratio = np.exp((2 * (a - x**2) + 1) / 4)
        truth = (a < 1.0).astype(float) * density_ratio - (a < 0.0).astype(float)
    else:
        raise KeyError(f"No DGP/hyperparameter recipe for baseline key {key!r}")
    return float(np.sqrt(np.mean((booster.predict(df) - truth) ** 2)))


@pytest.mark.parametrize("key", list(BASELINES.keys()))
def test_alpha_rmse_within_tolerance(key: str):
    spec = BASELINES[key]
    expected = float(spec["alpha_rmse"])
    tol = float(spec["tol"])  # relative tolerance
    actual = compute_baseline(key)
    rel_err = abs(actual - expected) / max(expected, 1e-9)
    assert rel_err < tol, (
        f"α-RMSE for {key} drifted: actual={actual:.4f}, expected={expected:.4f}, "
        f"rel_err={rel_err:.3f} (tol={tol:.2f}). If this drift is intentional, "
        f"update {BASELINES_PATH.name}."
    )
