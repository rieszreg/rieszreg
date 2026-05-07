"""Estimator-consistency tests using rieszreg.testing.dgps.

On analytically tractable DGPs, with proper tuning and growing n, the learned
α̂ should approach the true α₀. Each implementation package runs the shared
canonical DGPs against its own backend in CI; this is the krrr suite.

Tolerances are conservative — the goal is to catch regressions where α̂
diverges from α₀, not to benchmark fit quality.
"""

from __future__ import annotations

import numpy as np

from rieszreg import ATE
from rieszreg.testing import dgps

from krrr import Gaussian, KernelRieszRegressor


def test_linear_gaussian_ate_consistency():
    """RMSE shrinks as n grows on the linear-Gaussian ATE DGP."""
    dgp = dgps.linear_gaussian_ate()

    def fit_predict(train, test):
        krr = KernelRieszRegressor(
            estimand=ATE(),
            kernel=Gaussian(length_scale="median"),
            lambda_grid=np.logspace(-3, 0, 11),
            solver="direct",
            validation_fraction=0.25,
            random_state=0,
        ).fit(train)
        return krr.predict(test)

    rmses = dgps.assert_consistency(
        fit_predict, dgp=dgp, n_grid=(400, 1500), tol_at_max_n=2.0,
    )
    # RMSE at n=1500 must be at least as small as at n=400 (modulo noise).
    assert rmses[-1] <= rmses[0] + 0.1
