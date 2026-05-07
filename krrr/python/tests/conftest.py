"""Shared fixtures: a small binary-treatment ATE DGP and a continuous-A
DGP for shift / custom-estimand tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def binary_ate_data():
    rng = np.random.default_rng(0)
    n = 300
    x = rng.uniform(0, 1, n)
    pi = 1 / (1 + np.exp(-(-0.02 * x - x ** 2 + 4 * np.log(x + 0.3) + 1.5)))
    a = rng.binomial(1, pi).astype(float)
    df = pd.DataFrame({"a": a, "x": x})
    truth_alpha = a / pi - (1 - a) / (1 - pi)  # ATE Riesz representer
    return df, truth_alpha, pi


@pytest.fixture
def continuous_a_data():
    rng = np.random.default_rng(1)
    n = 200
    x = rng.uniform(0, 1, n)
    a = rng.normal(x, 0.5, n)
    df = pd.DataFrame({"a": a, "x": x})
    return df
