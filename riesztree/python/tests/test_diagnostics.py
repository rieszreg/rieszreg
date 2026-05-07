"""Tree-specific diagnostics."""
from __future__ import annotations

import numpy as np

from riesztree import ATE, RieszTreeRegressor, diagnose_tree


def test_diagnose_tree_extras(linear_gaussian_ate, covariate_keys):
    make, _ = linear_gaussian_ate
    df = make(800, seed=0)
    est = RieszTreeRegressor(
        estimand=ATE(treatment="a", covariates=covariate_keys),
        max_depth=4,
    ).fit(df)
    d = diagnose_tree(est, df)
    assert d.n == 800
    assert d.n_leaves >= 1
    assert d.max_depth_actual >= 0
    assert isinstance(d.feature_importances, np.ndarray)
    assert d.feature_importances.shape == (len(("a",) + covariate_keys),)
    # importances sum to 1 when at least one split fired.
    if d.n_leaves > 1:
        assert np.isclose(d.feature_importances.sum(), 1.0)
