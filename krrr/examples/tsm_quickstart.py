"""Quickstart: treatment-specific mean (TSM) on a synthetic DGP.

m(z, α) = α(level, x). The Riesz representer is the inverse density ratio
1{a = level} / π(level | x). We fit it directly with a kernel ridge solve.

Run from the repo root: `.venv/bin/python examples/tsm_quickstart.py`
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from krrr import TSM, Gaussian, KernelRieszRegressor, diagnose_kernel


def main() -> None:
    rng = np.random.default_rng(0)
    n = 1500
    x = rng.uniform(0, 1, n)
    pi = 1 / (1 + np.exp(-(-0.02 * x - x ** 2 + 4 * np.log(x + 0.3) + 1.5)))
    a = rng.binomial(1, pi).astype(float)
    df = pd.DataFrame({"a": a, "x": x})

    krr = KernelRieszRegressor(
        estimand=TSM(level=1, treatment="a", covariates=("x",)),
        kernel=Gaussian(length_scale="median"),
        lambda_grid=np.logspace(-4, 0, 25),
        solver="auto",
        validation_fraction=0.25,
        random_state=0,
    )
    krr.fit(df)
    alpha_hat = krr.predict(df)
    truth = (a == 1).astype(float) / pi

    print(f"selected lambda      : {krr.lambda_:.4g}")
    print(f"alpha_hat range      : [{alpha_hat.min():.3f}, {alpha_hat.max():.3f}]")
    print(f"truth     range      : [{truth.min():.3f}, {truth.max():.3f}]")
    print(f"correlation w/ truth : {np.corrcoef(alpha_hat, truth)[0, 1]:.3f}")
    print()
    print(diagnose_kernel(krr, df).summary())


if __name__ == "__main__":
    main()
