"""Quickstart: additive shift effect on a continuous-treatment DGP.

m(z, α) = α(a + δ, x) − α(a, x). With a continuous treatment, no closed-form
inverse-propensity weight applies; the kernel solver fits α₀ directly.

Run from the repo root: `.venv/bin/python examples/additive_shift_quickstart.py`
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from krrr import AdditiveShift, Gaussian, KernelRieszRegressor, diagnose_kernel


def main() -> None:
    rng = np.random.default_rng(0)
    n = 1500
    x = rng.uniform(0, 1, n)
    a = rng.normal(loc=0.5 * x, scale=0.4, size=n)
    df = pd.DataFrame({"a": a, "x": x})

    delta = 0.2
    krr = KernelRieszRegressor(
        estimand=AdditiveShift(delta=delta, treatment="a", covariates=("x",)),
        kernel=Gaussian(length_scale="median"),
        lambda_grid=np.logspace(-4, 0, 25),
        solver="auto",
        validation_fraction=0.25,
        random_state=0,
    )
    krr.fit(df)
    alpha_hat = krr.predict(df)

    print(f"shift delta         : {delta}")
    print(f"selected lambda     : {krr.lambda_:.4g}")
    print(f"alpha_hat range     : [{alpha_hat.min():.3f}, {alpha_hat.max():.3f}]")
    print(f"alpha_hat mean      : {alpha_hat.mean():.3f}  (≈ d/dδ E[α(a+δ,x)] at δ=0)")
    print()
    print(diagnose_kernel(krr, df).summary())


if __name__ == "__main__":
    main()
