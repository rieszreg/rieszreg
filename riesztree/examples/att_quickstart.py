"""Quickstart: ATT (treatment effect on the treated) representer."""
from __future__ import annotations

import numpy as np
import pandas as pd

from riesztree import ATT, RieszTreeRegressor


def main() -> None:
    rng = np.random.default_rng(0)
    n = 1500
    x = rng.uniform(0, 1, n)
    pi = 1 / (1 + np.exp(-(0.4 * x - 0.6)))
    a = rng.binomial(1, pi).astype(float)
    df = pd.DataFrame({"a": a, "x": x})

    est = RieszTreeRegressor(
        estimand=ATT(treatment="a", covariates=("x",)),
        max_depth=4,
        random_state=0,
    )
    est.fit(df)
    alpha_hat = est.predict(df)

    print(f"ATT representer (partial, see DESIGN.md §1.1)")
    print(f"alpha_hat range : [{alpha_hat.min():.3f}, {alpha_hat.max():.3f}]")
    print(f"alpha_hat mean  : {alpha_hat.mean():.3f}")


if __name__ == "__main__":
    main()
