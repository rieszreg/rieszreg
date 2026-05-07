"""Quickstart: stochastic intervention via pre-sampled Monte Carlo treatments.

m(z, α) = (1/K) Σₖ α(a'ₖ, x). Each row carries K samples a'ₖ drawn from the
intervention density. Pre-sample once before fit; the augmentation engine
reads the samples through `extra_keys=("shift_samples",)`.

Run from the repo root: `.venv/bin/python examples/stochastic_intervention_quickstart.py`
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from krrr import (
    Gaussian,
    KernelRieszRegressor,
    StochasticIntervention,
    diagnose_kernel,
)


def main() -> None:
    rng = np.random.default_rng(0)
    n = 1000
    x = rng.uniform(0, 1, n)
    a = rng.normal(loc=0.5 * x, scale=0.4, size=n)
    df = pd.DataFrame({"a": a, "x": x})

    # Modified-treatment policy: a' ~ Normal(a + 0.1, 0.2), 20 samples per row.
    # Each row's samples depend ONLY on that row's data, so cross-fitting splits
    # don't leak across folds.
    df["shift_samples"] = [rng.normal(a_i + 0.1, 0.2, size=20) for a_i in df["a"]]

    krr = KernelRieszRegressor(
        estimand=StochasticIntervention(
            samples_key="shift_samples",
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

    print(f"selected lambda      : {krr.lambda_:.4g}")
    print(f"K (samples / row)    : 20")
    print(f"alpha_hat range      : [{alpha_hat.min():.3f}, {alpha_hat.max():.3f}]")
    print(f"alpha_hat mean       : {alpha_hat.mean():.3f}")
    print()
    print(diagnose_kernel(krr, df).summary())


if __name__ == "__main__":
    main()
