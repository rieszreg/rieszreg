"""Quickstart: AdditiveShift on a synthetic continuous-treatment DGP.

Run from the repo root: ``.venv/bin/python packages/forestriesz/examples/aug_quickstart.py``

Demonstrates the augmentation-style forest backend. ``AdditiveShift`` has
no canonical list of basis functions of the data, so the moment-style
``ForestRieszRegressor`` would require the user to construct one by hand.
``AugForestRieszRegressor`` works on the augmented dataset directly and
needs no per-estimand configuration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forestriesz import (
    AdditiveShift,
    AugForestRieszRegressor,
)


def main() -> None:
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.normal(size=n)
    a = rng.normal(0.5 * x, 1.0)
    df = pd.DataFrame({"a": a, "x": x})

    est = AugForestRieszRegressor(
        estimand=AdditiveShift(delta=0.5, treatment="a", covariates=("x",)),
        n_estimators=200,
        min_samples_leaf=10,
        random_state=0,
    )
    est.fit(df)
    alpha_hat = est.predict(df)

    print(f"alpha_hat range : [{alpha_hat.min():.3f}, {alpha_hat.max():.3f}]")
    print(f"alpha_hat mean  : {alpha_hat.mean():.3f}")
    print(f"alpha_hat std   : {alpha_hat.std():.3f}")


if __name__ == "__main__":
    main()
