"""RieszTreeBackend holds rows out only when early stopping will use them,
and early stopping works on the backend alone."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rieszreg import RieszEstimator
from riesztree import ATE, RieszTreeBackend, RieszTreeRegressor


def _df(n=400):
    rng = np.random.default_rng(0)
    return pd.DataFrame({"a": rng.binomial(1, 0.5, n), "x": rng.normal(size=n)})


def test_no_holdout_without_early_stopping():
    df = _df()
    backend = RieszTreeBackend(validation_fraction=0.3)
    assert backend.holdout_fraction() == 0.0
    est = RieszEstimator(estimand=ATE(), backend=backend).fit(df)
    every_row = RieszEstimator(estimand=ATE(), backend=RieszTreeBackend(validation_fraction=0.0)).fit(df)
    np.testing.assert_array_equal(est.predict(df), every_row.predict(df))


def test_early_stopping_on_the_backend_alone_holds_out_the_default_fraction():
    backend = RieszTreeBackend(early_stopping_rounds=5)
    assert backend.holdout_fraction() == 0.1
    est = RieszEstimator(estimand=ATE(), backend=backend).fit(_df())
    assert est.best_score_ is not None  # scored on the held-out rows


def test_regressor_and_backend_agree():
    df = _df()
    kw = dict(early_stopping_rounds=5, validation_fraction=0.2, max_depth=6)
    a = RieszTreeRegressor(estimand=ATE(), **kw).fit(df).predict(df)
    b = RieszEstimator(estimand=ATE(), backend=RieszTreeBackend(**kw)).fit(df).predict(df)
    np.testing.assert_array_equal(a, b)
