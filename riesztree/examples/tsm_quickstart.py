"""Quickstart: TSM(level=1) with KLLoss for non-negative density-ratio estimation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from riesztree import KLLoss, RieszTreeRegressor, TSM


def main() -> None:
    rng = np.random.default_rng(0)
    n = 1500
    x = rng.uniform(0, 1, n)
    pi = 1 / (1 + np.exp(-(0.5 * x - 0.3)))
    a = rng.binomial(1, pi).astype(float)
    df = pd.DataFrame({"a": a, "x": x})

    est = RieszTreeRegressor(
        estimand=TSM(level=1.0, treatment="a", covariates=("x",)),
        loss=KLLoss(),
        max_depth=4,
        random_state=0,
    )
    est.fit(df)
    alpha_hat = est.predict(df)
    truth = (a == 1).astype(float) / pi

    print(f"TSM(level=1) with KLLoss → α ≥ 0 enforced by the link's α-domain")
    print(f"all alpha_hat >= 0 : {bool((alpha_hat >= 0).all())}")
    print(f"alpha_hat range   : [{alpha_hat.min():.3f}, {alpha_hat.max():.3f}]")
    print(f"truth     range   : [{truth.min():.3f}, {truth.max():.3f}]")
    print(f"correlation       : {np.corrcoef(alpha_hat, truth)[0, 1]:.3f}")


if __name__ == "__main__":
    main()
