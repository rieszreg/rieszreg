"""OutcomeRegNormSq parity vs. sklearn KernelRidge.

`m(α)(z, y) = α(x) · y` has Riesz representer μ_0(x) = E[Y | X=x]; under the
squared Bregman-Riesz loss the augmented dataset reduces to (X, is_orig=1,
pdc=-y), and the kernel-ridge normal equation becomes (K + nλI) γ = y. That
is sklearn `KernelRidge(alpha = n·λ)` with the same kernel.

Pearson > 0.99 is the gate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from krrr import Gaussian, KernelRieszRegressor
from rieszreg import OutcomeRegNormSq
from rieszreg.testing.parity import compare


def _regression_df(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-2.0, 2.0, n)
    x1 = rng.normal(0.0, 1.0, n)
    y = np.sin(x0) + 0.5 * x1 + rng.normal(0.0, 0.3, n)
    df = pd.DataFrame({"x0": x0, "x1": x1})
    return df, y


def test_kernel_ridge_outcome_regression_parity_with_sklearn():
    """krrr with `OutcomeRegNormSq` and a single-λ grid should match sklearn's
    `KernelRidge` at α = n·λ on the same data and Gaussian kernel."""
    from sklearn.kernel_ridge import KernelRidge

    df, y = _regression_df(n=200, seed=0)
    X = df.values
    n = len(y)

    length_scale = 1.0
    lam = 1e-3

    riesz = KernelRieszRegressor(
        estimand=OutcomeRegNormSq(covariates=("x0", "x1")),
        kernel=Gaussian(length_scale=length_scale),
        lambda_grid=[lam],
        solver="direct",
        validation_fraction=0.0,
        keep_path=False,
        random_state=0,
    ).fit(df, y)

    # K(x,y) = exp(-||x-y||²/(2·ℓ²)) ⇒ sklearn RBF gamma = 1/(2·ℓ²).
    gamma = 1.0 / (2.0 * length_scale ** 2)
    ref = KernelRidge(kernel="rbf", gamma=gamma, alpha=lam * n).fit(X, y)

    rep = compare(riesz.predict(df), ref.predict(X))
    assert rep.pearson > 0.99, rep.summary()
