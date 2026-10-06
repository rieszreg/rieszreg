"""Deep max_depth values: the compiled drivers' node cap must not overflow."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riesztree import ATE, RieszTreeRegressor


@pytest.mark.parametrize("splitter", ["exact", "hist"])
@pytest.mark.parametrize("max_depth", [29, 30, 31])
def test_deep_max_depth_fits(splitter, max_depth):
    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(size=n)
    df = pd.DataFrame({"a": rng.binomial(1, 0.5, n), "x": x})
    est = RieszTreeRegressor(
        estimand=ATE(treatment="a", covariates=["x"]),
        max_depth=max_depth, min_samples_leaf=1, splitter=splitter,
    ).fit(df)
    assert np.isfinite(est.predict(df)).all()
