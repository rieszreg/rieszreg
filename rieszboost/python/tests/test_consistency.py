"""Estimator-consistency tests using rieszreg.testing.dgps.

On analytically tractable DGPs, with proper tuning and growing n, the learned
α̂ should approach the true α₀. Each implementation package runs the shared
canonical DGPs against its own backend in CI; this is the rieszboost suite.

Tolerances are conservative — the goal is to catch regressions where α̂
diverges from α₀, not to benchmark fit quality.
"""

from __future__ import annotations

from rieszreg import ATE
from rieszreg.testing import dgps

from rieszboost import RieszBooster


def test_linear_gaussian_ate_consistency():
    """RMSE shrinks as n grows on the linear-Gaussian ATE DGP."""
    dgp = dgps.linear_gaussian_ate()

    def fit_predict(train, test):
        booster = RieszBooster(
            estimand=ATE(),
            n_estimators=600,
            learning_rate=0.05,
            max_depth=3,
            reg_lambda=1.0,
            early_stopping_rounds=20,
            validation_fraction=0.2,
            random_state=0,
        ).fit(train)
        return booster.predict(test)

    rmses = dgps.assert_consistency(
        fit_predict, dgp=dgp, n_grid=(500, 2000), tol_at_max_n=2.0,
    )
    # RMSE at n=2000 must be at least as small as at n=500 (modulo noise).
    assert rmses[-1] <= rmses[0] + 0.1
