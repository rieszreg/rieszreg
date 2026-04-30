"""Quickstart: TSM (treatment-specific mean) on a synthetic DGP.

Run from the repo root: ``.venv/bin/python riesznet/examples/tsm_quickstart.py``

Demonstrates KLLoss for the density-ratio target α₀(a, x) = 1{a=level}/π(a|x).
The exp link keeps predictions positive without any clamping at predict time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from riesznet import KLLoss, RieszNet, TSM


def main() -> None:
    rng = np.random.default_rng(0)
    n = 1500
    x = rng.uniform(0, 1, n)
    pi = 1 / (1 + np.exp(-(0.5 * x - 0.3)))
    a = rng.binomial(1, pi).astype(float)
    df = pd.DataFrame({"a": a, "x": x})

    rn = RieszNet(
        estimand=TSM(level=1, treatment="a", covariates=("x",)),
        loss=KLLoss(max_eta=10.0),
        hidden_sizes=(64, 64),
        learning_rate=5e-3,
        epochs=400,
        validation_fraction=0.2,
        early_stopping_rounds=30,
        random_state=0,
    )
    rn.fit(df)
    alpha_hat = rn.predict(df)
    truth = (a == 1).astype(float) / pi

    print(f"alpha_hat range      : [{alpha_hat.min():.3f}, {alpha_hat.max():.3f}]")
    print(f"truth     range      : [{truth.min():.3f}, {truth.max():.3f}]")
    print(f"correlation w/ truth : {np.corrcoef(alpha_hat, truth)[0, 1]:.3f}")
    print(f"RMSE vs truth        : {float(np.sqrt(np.mean((alpha_hat - truth) ** 2))):.3f}")
    print(f"best_iteration_      : {rn.best_iteration_}")
    print()
    print(rn.diagnose(df).summary())


if __name__ == "__main__":
    main()
