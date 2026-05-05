"""Shared fixtures for forestriesz tests.

Reuses the canonical DGPs from rieszreg.testing.dgps so backend correctness
checks stay aligned with the rieszreg test contract.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rieszreg.testing import dgps


@pytest.fixture
def linear_gaussian_ate_df():
    rng = np.random.default_rng(0)
    return dgps.linear_gaussian_ate().sample(800, rng)


@pytest.fixture
def logistic_tsm_df():
    rng = np.random.default_rng(0)
    return dgps.logistic_tsm(level=1.0).sample(800, rng)


@pytest.fixture
def small_df():
    rng = np.random.default_rng(0)
    n = 50
    return pd.DataFrame(
        {
            "a": (rng.uniform(size=n) > 0.5).astype(float),
            "x": rng.normal(size=n),
        }
    )
