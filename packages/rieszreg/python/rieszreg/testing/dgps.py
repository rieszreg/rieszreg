"""Canonical DGPs for estimator-consistency tests.

Each DGP exposes:
  - `sample(n, rng)` -> pandas DataFrame
  - `true_alpha(z)` -> closed-form α₀ at a row z
  - `feature_keys` matching the corresponding rieszreg estimand factory

Implementation packages run their backend against these DGPs at growing n and
assert the learned α̂ approaches the true α₀.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class DGP:
    name: str
    feature_keys: tuple[str, ...]
    sample: Callable[..., object]            # (n, rng) -> DataFrame
    true_alpha: Callable[..., np.ndarray]    # (df) -> array of α₀ per row
    estimand_factory: str                    # "ATE", "ATT", ...


def _logistic_sampler(sigma_x: float):
    """X ~ N(0, σ²), A ~ Bernoulli(π(x)) with logit π(x) = 0.5 x. Returns
    ``(sample, propensity_of_observed_a)``."""

    def sample(n: int, rng: np.random.Generator):
        import pandas as pd

        x = rng.normal(0.0, sigma_x, size=n)
        a = (rng.uniform(0, 1, size=n) < 1.0 / (1.0 + np.exp(-0.5 * x))).astype(float)
        return pd.DataFrame({"a": a, "x": x})

    def prob_a(df) -> np.ndarray:
        a = np.asarray(df["a"])
        pi = 1.0 / (1.0 + np.exp(-0.5 * np.asarray(df["x"])))
        return a * pi + (1.0 - a) * (1.0 - pi)

    return sample, prob_a


def linear_gaussian_ate(*, sigma_x: float = 1.0) -> DGP:
    """Linear-Gaussian ATE DGP.

    A ~ Bernoulli(π(x)) with logit π(x) = 0.5 x; X ~ N(0, σ²).
    Closed-form Riesz representer for ATE: α₀(a, x) = (2a − 1) / [a · π(x) + (1−a)·(1−π(x))].
    """
    sample, prob_a = _logistic_sampler(sigma_x)
    return DGP(
        name="linear_gaussian_ate",
        feature_keys=("a", "x"),
        sample=sample,
        true_alpha=lambda df: (2.0 * np.asarray(df["a"]) - 1.0) / prob_a(df),
        estimand_factory="ATE",
    )


def logistic_tsm(level: float = 1.0, sigma_x: float = 1.0) -> DGP:
    """Treatment-specific mean DGP. Riesz representer α₀(a, x) = 1[a=level] / π(x|level).

    Useful as a density-ratio estimand: α₀ ≥ 0 everywhere, so KLLoss applies.
    """
    sample, prob_a = _logistic_sampler(sigma_x)
    return DGP(
        name="logistic_tsm",
        feature_keys=("a", "x"),
        sample=sample,
        true_alpha=lambda df: (np.asarray(df["a"]) == level).astype(float) / prob_a(df),
        estimand_factory="TSM",
    )


def assert_consistency(
    fit_predict,
    *,
    dgp: DGP,
    n_grid: tuple[int, ...] = (500, 2000),
    rng_seed: int = 0,
    tol_at_max_n: float = 0.5,
    monotonicity_slack: float = 0.5,
):
    """Assert RMSE of α̂ vs α₀ shrinks across n and lands below `tol_at_max_n`.

    Designed to catch *divergence*, not to benchmark fit quality — single-seed
    RMSEs are noisy, especially for estimators with early stopping that
    plateau quickly. The monotonicity check is therefore lax: it requires
    `rmses[-1] <= rmses[0] * (1 + monotonicity_slack)` (default 50% slack).
    The absolute `tol_at_max_n` is the primary signal.

    Parameters
    ----------
    fit_predict : callable
        Takes (train_df, test_df) → α̂-array of length len(test).
    n_grid : tuple of int
        Sample sizes to evaluate at, in increasing order.
    tol_at_max_n : float
        Maximum acceptable RMSE at the largest n.
    monotonicity_slack : float
        Maximum relative increase from rmses[0] to rmses[-1] (default 0.5).
    """
    rng = np.random.default_rng(rng_seed)
    rmses = []
    for n in n_grid:
        train = dgp.sample(n, rng)
        test = dgp.sample(2 * n, rng)
        alpha_hat = np.asarray(fit_predict(train, test))[: len(test)]
        alpha_true = dgp.true_alpha(test)
        rmses.append(float(np.sqrt(np.mean((alpha_hat - alpha_true) ** 2))))
    # Lax monotonicity: allow noise but flag large divergence.
    if rmses[-1] > rmses[0] * (1.0 + monotonicity_slack) + 1e-6:
        raise AssertionError(
            f"{dgp.name}: RMSE diverged across n_grid={n_grid}; got {rmses} "
            f"(allowed up to {monotonicity_slack:.0%} relative increase)"
        )
    if rmses[-1] > tol_at_max_n:
        raise AssertionError(
            f"{dgp.name}: RMSE at n={n_grid[-1]} is {rmses[-1]:.3f}, "
            f"above tolerance {tol_at_max_n}"
        )
    return rmses
