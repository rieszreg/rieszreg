"""Reproduce the dml-tmle/code/R/learners/krrr.R `fit_krrr` reference solution
for the TSM1 estimand to numerical tolerance.

The R reference solves
    (K_WW + n λ I) γ = -(1/(n λ)) K_WM 1
    α̂(W) = K_WW γ + (1/(n λ)) K_WM 1
with k(u, v) = exp(-ρ ||u − v||²) at ρ = 1.

Our framework treats k as a Gaussian with length scale ls = 1/√(2ρ).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from krrr import Gaussian, KernelRieszRegressor, TSM


def _r_reference(W, M, lam, rho=1.0):
    n = W.shape[0]
    K_WW = np.exp(-rho * cdist(W, W, "sqeuclidean"))
    K_WM = np.exp(-rho * cdist(W, M, "sqeuclidean"))
    b = 1.0 / (n * lam)
    rhs = -b * (K_WM @ np.ones(n))
    lhs = K_WW + n * lam * np.eye(n)
    a = np.linalg.solve(lhs, rhs)
    return K_WW @ a + b * (K_WM @ np.ones(n))


def test_tsm1_matches_dml_tmle_reference():
    rng = np.random.default_rng(7)
    n = 60
    x = rng.uniform(0, 1, n)
    a = rng.binomial(1, 0.5, n).astype(float)

    W = np.column_stack([a, x])
    M = np.column_stack([np.ones(n), x])
    lam = 0.05
    alpha_R = _r_reference(W, M, lam)

    df = pd.DataFrame({"a": a, "x": x})
    krr = KernelRieszRegressor(
        estimand=TSM(1, "a", ("x",)),
        kernel=Gaussian(length_scale=1.0 / np.sqrt(2.0)),  # matches ρ=1
        lambda_grid=[lam],
        solver="direct",
        validation_fraction=0.0,
        init=0.0,    # match the R reference's implicit zero baseline
    ).fit(df)
    alpha_K = krr.predict(df)

    np.testing.assert_allclose(alpha_K, alpha_R, atol=1e-8, rtol=1e-8)
