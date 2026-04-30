"""Quickstart: ATE on a synthetic binary-treatment DGP.

Run from the repo root: ``.venv/bin/python riesznet/examples/ate_quickstart.py``

Demonstrates the simple-MLP convenience class on the inverse-propensity
representer for ATE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from riesznet import ATE, RieszNet


def main() -> None:
    rng = np.random.default_rng(0)
    n = 1500
    x = rng.uniform(0, 1, n)
    pi = 1 / (1 + np.exp(-(-0.02 * x - x ** 2 + 4 * np.log(x + 0.3) + 1.5)))
    a = rng.binomial(1, pi).astype(float)
    df = pd.DataFrame({"a": a, "x": x})

    estimand = ATE(treatment="a", covariates=("x",))
    rn = RieszNet(
        estimand=estimand,
        hidden_sizes=(64, 64),
        learning_rate=5e-3,
        epochs=400,
        validation_fraction=0.2,
        early_stopping_rounds=30,
        random_state=0,
    )
    rn.fit(df)
    alpha_hat = rn.predict(df)
    truth = a / pi - (1 - a) / (1 - pi)

    print(f"alpha_hat range      : [{alpha_hat.min():.3f}, {alpha_hat.max():.3f}]")
    print(f"truth     range      : [{truth.min():.3f}, {truth.max():.3f}]")
    print(f"correlation w/ truth : {np.corrcoef(alpha_hat, truth)[0, 1]:.3f}")
    print(f"RMSE vs truth        : {float(np.sqrt(np.mean((alpha_hat - truth) ** 2))):.3f}")
    print(f"best_iteration_      : {rn.best_iteration_}")
    print(f"best_score_ (-loss)  : {-rn.best_score_:.4f}")
    print()
    print(rn.diagnose(df).summary())


if __name__ == "__main__":
    main()
