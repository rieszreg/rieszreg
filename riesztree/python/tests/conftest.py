"""Shared fixtures for riesztree tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def linear_gaussian_ate():
    """Linear-Gaussian ATE DGP with d_x = 5."""
    def make(n: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        x = rng.normal(0.0, 1.0, size=(n, 5))
        logit = 0.7 * x[:, 0] - 0.3 * x[:, 1]
        pi = 1.0 / (1.0 + np.exp(-logit))
        a = (rng.uniform(0, 1, size=n) < pi).astype(float)
        df = pd.DataFrame(x, columns=[f"x{i}" for i in range(5)])
        df.insert(0, "a", a)
        df["_pi"] = pi
        return df

    def truth(df: pd.DataFrame) -> np.ndarray:
        pi = df["_pi"].values
        a = df["a"].values
        prob_a = a * pi + (1.0 - a) * (1.0 - pi)
        return (2.0 * a - 1.0) / prob_a

    return make, truth


@pytest.fixture
def covariate_keys():
    return ("x0", "x1", "x2", "x3", "x4")
