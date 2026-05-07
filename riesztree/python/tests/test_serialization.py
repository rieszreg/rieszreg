"""Save/load round-trip across estimands and losses."""
from __future__ import annotations

import tempfile

import numpy as np
import pytest

from rieszreg import KLLoss
from riesztree import (
    ATE,
    AdditiveShift,
    ATT,
    LocalShift,
    RieszTreeRegressor,
    TSM,
)


@pytest.mark.parametrize("estimand_factory", [
    lambda kw: ATE(treatment="a", covariates=kw["covariates"]),
    lambda kw: ATT(treatment="a", covariates=kw["covariates"]),
    lambda kw: TSM(level=1.0, treatment="a", covariates=kw["covariates"]),
    lambda kw: AdditiveShift(delta=0.5, treatment="a", covariates=kw["covariates"]),
    lambda kw: LocalShift(delta=0.5, threshold=0.0, treatment="a", covariates=kw["covariates"]),
])
def test_save_load_roundtrip_per_estimand(estimand_factory, linear_gaussian_ate, covariate_keys):
    make, _ = linear_gaussian_ate
    df = make(400, seed=0)
    estimand = estimand_factory({"covariates": covariate_keys})
    est = RieszTreeRegressor(estimand=estimand, max_depth=4).fit(df)
    a_before = est.predict(df)
    with tempfile.TemporaryDirectory() as tmp:
        est.save(tmp)
        loaded = RieszTreeRegressor.load(tmp)
        a_after = loaded.predict(df)
    assert np.allclose(a_before, a_after, atol=1e-12)


def test_save_load_kl_loss_tsm(linear_gaussian_ate, covariate_keys):
    make, _ = linear_gaussian_ate
    df = make(500, seed=0)
    est = RieszTreeRegressor(
        estimand=TSM(level=1.0, treatment="a", covariates=covariate_keys),
        loss=KLLoss(),
        max_depth=4,
    ).fit(df)
    a_before = est.predict(df)
    with tempfile.TemporaryDirectory() as tmp:
        est.save(tmp)
        loaded = RieszTreeRegressor.load(tmp)
        a_after = loaded.predict(df)
    assert np.allclose(a_before, a_after, atol=1e-12)
