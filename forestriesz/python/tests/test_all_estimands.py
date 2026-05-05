"""Smoke test on each built-in estimand.

Difference-style estimands (ATE, ATT, AdditiveShift, LocalShift) require a
sieve — locally constant gives an identically zero moment. Single-point
estimands (TSM) work either way; we use locally constant for them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forestriesz import (
    ATE,
    ATT,
    AdditiveShift,
    ForestRieszRegressor,
    LocalShift,
    TSM,
    default_riesz_features,
)


@pytest.fixture
def df_binary():
    rng = np.random.default_rng(0)
    n = 500
    x = rng.normal(size=n)
    pi = 1.0 / (1.0 + np.exp(-0.5 * x))
    a = (rng.uniform(size=n) < pi).astype(float)
    return pd.DataFrame({"a": a, "x": x})


@pytest.fixture
def df_continuous():
    rng = np.random.default_rng(0)
    n = 500
    x = rng.normal(size=n)
    a = rng.normal(0.5 * x, 1.0)  # continuous treatment
    return pd.DataFrame({"a": a, "x": x})


def _make(estimand, force_constant=False):
    """Build a regressor; default uses the auto sieve. Pass force_constant=True
    to opt into the locally constant fit (which is degenerate for built-ins
    and triggers the row-constant check)."""
    return ForestRieszRegressor(
        estimand=estimand,
        riesz_feature_fns=None if force_constant else "auto",
        n_estimators=30,
        min_samples_leaf=15,
        random_state=0,
    )


def test_ate_auto(df_binary):
    est = _make(ATE())     # auto picks the [1{T=0}, 1{T=1}] sieve
    est.fit(df_binary)
    pred = est.predict(df_binary)
    assert pred.shape == (len(df_binary),)
    assert np.all(np.isfinite(pred))


def test_att_auto(df_binary):
    est = _make(ATT())
    est.fit(df_binary)
    pred = est.predict(df_binary)
    assert pred.shape == (len(df_binary),)
    assert np.all(np.isfinite(pred))


def test_tsm_auto(df_binary):
    est = _make(TSM(level=1))
    est.fit(df_binary)
    pred = est.predict(df_binary)
    assert pred.shape == (len(df_binary),)
    assert np.all(np.isfinite(pred))


def test_additive_shift_constant_raises(df_continuous):
    # Custom estimand with no default sieve → auto falls back to constant →
    # row-constant check fires.
    est = _make(AdditiveShift(delta=0.5))
    with pytest.raises(ValueError, match="row-constant"):
        est.fit(df_continuous)


def test_local_shift_constant_raises(df_continuous):
    est = _make(LocalShift(delta=0.5, threshold=0.0))
    with pytest.raises(ValueError, match="row-constant"):
        est.fit(df_continuous)


