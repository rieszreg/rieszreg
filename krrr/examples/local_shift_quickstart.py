"""Quickstart: local additive shift effect (LASE, partial-parameter).

m(z, α) = 1{a < threshold} · (α(a + δ, x) − α(a, x)). Full LASE divides by
P(A < threshold) and is not a Riesz functional — combine the partial α̂ with
a delta-method EIF downstream.

Run from the repo root: `.venv/bin/python examples/local_shift_quickstart.py`
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from krrr import Gaussian, KernelRieszRegressor, LocalShift, diagnose_kernel


def main() -> None:
    rng = np.random.default_rng(0)
    n = 1500
    x = rng.uniform(0, 1, n)
    a = rng.normal(loc=0.5 * x, scale=0.4, size=n)
    df = pd.DataFrame({"a": a, "x": x})

    delta = 0.2
    threshold = 0.5
    krr = KernelRieszRegressor(
        estimand=LocalShift(
            delta=delta,
            threshold=threshold,
            treatment="a",
            covariates=("x",),
        ),
        kernel=Gaussian(length_scale="median"),
        lambda_grid=np.logspace(-4, 0, 25),
        solver="auto",
        validation_fraction=0.25,
        random_state=0,
    )
    krr.fit(df)
    alpha_hat = krr.predict(df)

    p_below = float((df["a"] < threshold).mean())
    print(f"shift delta          : {delta}")
    print(f"threshold            : {threshold}")
    print(f"P(A < threshold)     : {p_below:.3f}  (denominator for full LASE)")
    print(f"selected lambda      : {krr.lambda_:.4g}")
    print(f"alpha_hat range      : [{alpha_hat.min():.3f}, {alpha_hat.max():.3f}]")
    print()
    print(diagnose_kernel(krr, df).summary())


if __name__ == "__main__":
    main()
