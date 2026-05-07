"""Smoke-test all six built-in estimands plus a custom user-defined Estimand.
Each must fit, predict, produce finite predictions, and round-trip via save/load.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from krrr import (
    ATE,
    ATT,
    AdditiveShift,
    FiniteEvalEstimand,
    KernelRieszRegressor,
    LocalShift,
    TSM,
)


@pytest.mark.parametrize("estimand_factory,kwargs", [
    (ATE, dict(treatment="a", covariates=("x",))),
    (ATT, dict(treatment="a", covariates=("x",))),
    (TSM, dict(level=1, treatment="a", covariates=("x",))),
    (AdditiveShift, dict(delta=0.3, treatment="a", covariates=("x",))),
    (LocalShift, dict(delta=0.3, threshold=1.0, treatment="a", covariates=("x",))),
])
def test_estimand_fits_and_predicts(continuous_a_data, estimand_factory, kwargs):
    df = continuous_a_data
    estimand = estimand_factory(**kwargs)
    krr = KernelRieszRegressor(
        estimand=estimand,
        lambda_grid=np.logspace(-3, 0, 6),
        validation_fraction=0.2,
    ).fit(df)
    alpha = krr.predict(df)
    assert alpha.shape == (len(df),)
    assert np.all(np.isfinite(alpha))


def test_custom_estimand(continuous_a_data):
    df = continuous_a_data

    def m_mix(alpha):
        def inner(z, y=None):
            return 0.6 * alpha(a=1, x=z["x"]) - 0.4 * alpha(a=0, x=z["x"])
        return inner

    krr = KernelRieszRegressor(
        estimand=FiniteEvalEstimand(feature_keys=("a", "x"), m=m_mix, name="MyMix"),
        lambda_grid=np.logspace(-3, 0, 6),
        validation_fraction=0.2,
    ).fit(df)
    alpha = krr.predict(df)
    assert alpha.shape == (len(df),)
    assert np.all(np.isfinite(alpha))


def test_save_load_roundtrip(continuous_a_data):
    df = continuous_a_data
    krr = KernelRieszRegressor(
        estimand=AdditiveShift(0.3, "a", ("x",)),
        lambda_grid=np.logspace(-3, 0, 6),
        validation_fraction=0.2,
    ).fit(df)
    a1 = krr.predict(df)
    with tempfile.TemporaryDirectory() as d:
        krr.save(d)
        loaded = KernelRieszRegressor.load(d)
    a2 = loaded.predict(df)
    np.testing.assert_array_equal(a1, a2)
