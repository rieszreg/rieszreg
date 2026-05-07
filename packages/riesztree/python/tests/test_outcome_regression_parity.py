"""OutcomeRegNormSq parity vs. sklearn DecisionTreeRegressor.

`m(α)(z, y) = α(x) · y` has Riesz representer μ_0(x) = E[Y | X=x]; under the
augmentation-style RieszTreeBackend the augmented dataset is (X, is_orig=1,
pdc=-y), and the per-leaf closed-form α* = -C/D reduces to the leaf mean of
y — the standard regression-tree prediction rule.

Pearson > 0.99 is the gate. Tree-construction details differ (split criterion
gain formula, tie-breaking, growth policy), so exact agreement is not
expected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rieszreg import OutcomeRegNormSq
from rieszreg.testing.parity import compare
from riesztree import RieszTreeRegressor


def _regression_df(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-2.0, 2.0, n)
    x1 = rng.normal(0.0, 1.0, n)
    y = np.sin(x0) + 0.5 * x1 + rng.normal(0.0, 0.3, n)
    df = pd.DataFrame({"x0": x0, "x1": x1})
    return df, y


def test_riesztree_outcome_regression_parity_with_decision_tree():
    from sklearn.tree import DecisionTreeRegressor

    df, y = _regression_df(n=400, seed=0)

    common = dict(
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        max_leaf_nodes=63,
        random_state=0,
    )

    riesz = RieszTreeRegressor(
        estimand=OutcomeRegNormSq(covariates=("x0", "x1")),
        **common,
    ).fit(df, y)

    ref = DecisionTreeRegressor(**common).fit(df.values, y)

    rep = compare(riesz.predict(df), ref.predict(df.values))
    assert rep.pearson > 0.99, rep.summary()
