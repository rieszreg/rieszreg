"""Quickstart: AdditiveShift(delta=0.5) on continuous treatment data.

Demonstrates the augmentation-style splitter handles shift-style
estimands that have no canonical sieve, without any extra configuration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from riesztree import AdditiveShift, RieszTreeRegressor


def main() -> None:
    rng = np.random.default_rng(0)
    n = 1500
    x = rng.normal(0, 1, n)
    a = rng.normal(0.5 * x, 1.0)   # continuous treatment
    df = pd.DataFrame({"a": a, "x": x})

    est = RieszTreeRegressor(
        estimand=AdditiveShift(delta=0.5, treatment="a", covariates=("x",)),
        max_depth=4,
        random_state=0,
    )
    est.fit(df)
    alpha_hat = est.predict(df)

    print(f"AdditiveShift(delta=0.5)")
    print(f"alpha_hat range : [{alpha_hat.min():.3f}, {alpha_hat.max():.3f}]")
    print(f"alpha_hat mean  : {alpha_hat.mean():.3f}")


if __name__ == "__main__":
    main()
