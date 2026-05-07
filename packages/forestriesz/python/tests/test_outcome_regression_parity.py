"""OutcomeRegNormSq parity vs. sklearn RandomForestRegressor.

`m(α)(z, y) = α(x) · y` has Riesz representer μ_0(x) = E[Y | X=x]; under the
moment-style ForestRieszBackend with a constant sieve basis, the per-leaf
solve reduces to θ_ℓ = mean(y_i) over leaf ℓ — the standard regression-forest
prediction rule.

Pearson > 0.99 is the gate. Forest construction differs (EconML BaseGRF vs
sklearn RandomForestRegressor split criterion / honest split / subsampling),
so exact agreement isn't expected; only directional parity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forestriesz import ForestRieszRegressor
from rieszreg import OutcomeRegNormSq
from rieszreg.testing.parity import compare


def _regression_df(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-2.0, 2.0, n)
    x1 = rng.normal(0.0, 1.0, n)
    y = np.sin(x0) + 0.5 * x1 + rng.normal(0.0, 0.3, n)
    df = pd.DataFrame({"x0": x0, "x1": x1})
    return df, y


def test_forestriesz_outcome_regression_parity_with_random_forest():
    from sklearn.ensemble import RandomForestRegressor

    df, y = _regression_df(n=400, seed=0)

    riesz = ForestRieszRegressor(
        estimand=OutcomeRegNormSq(covariates=("x0", "x1")),
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=5,
        min_samples_split=10,
        random_state=0,
    ).fit(df, y)

    ref = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=5,
        min_samples_split=10,
        random_state=0,
    ).fit(df.values, y)

    rep = compare(riesz.predict(df), ref.predict(df.values))
    assert rep.pearson > 0.99, rep.summary()
