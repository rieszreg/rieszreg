"""Direct, Nyström-CG and RFF should agree on small problems where all
three are exact (or nearly so).
"""

from __future__ import annotations

import numpy as np
import pytest

from krrr import ATE, Gaussian, KernelRieszRegressor


def _fit(df, solver, **kwargs):
    return KernelRieszRegressor(
        estimand=ATE("a", ("x",)),
        kernel=Gaussian(length_scale=0.5),
        lambda_grid=[1e-2],
        solver=solver,
        validation_fraction=0.0,
        random_state=0,
        **kwargs,
    ).fit(df)


def test_direct_vs_nystrom_cg(binary_ate_data):
    df, _, _ = binary_ate_data
    a_direct = _fit(df, "direct").predict(df)
    a_nys = _fit(df, "nystrom_cg", n_landmarks=200).predict(df)
    # With landmarks = full support, Nyström becomes exact KRR up to CG tol.
    rmse = float(np.sqrt(np.mean((a_direct - a_nys) ** 2)))
    assert rmse < 1e-3, f"Direct vs Nyström-CG RMSE = {rmse}"


def test_direct_vs_rff_correlate(binary_ate_data):
    df, _, _ = binary_ate_data
    a_direct = _fit(df, "direct").predict(df)
    a_rff = _fit(df, "rff", n_features=4096).predict(df)
    # RFF is approximate; at D=4096 the correlation should be high but
    # exact equality won't hold.
    corr = float(np.corrcoef(a_direct, a_rff)[0, 1])
    assert corr > 0.95, f"Direct vs RFF correlation = {corr}"
