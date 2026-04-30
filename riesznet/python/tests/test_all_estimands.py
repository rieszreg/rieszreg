"""Smoke test on each built-in estimand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riesznet import (
    ATE,
    ATT,
    AdditiveShift,
    LocalShift,
    RieszNet,
    StochasticIntervention,
    TSM,
)


@pytest.fixture
def df_binary():
    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(size=n)
    pi = 1.0 / (1.0 + np.exp(-0.5 * x))
    a = (rng.uniform(size=n) < pi).astype(float)
    return pd.DataFrame({"a": a, "x": x})


@pytest.fixture
def df_continuous():
    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(size=n)
    a = rng.normal(0.5 * x, 1.0)
    return pd.DataFrame({"a": a, "x": x})


@pytest.fixture
def df_stochastic():
    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(size=n)
    pi = 1.0 / (1.0 + np.exp(-0.5 * x))
    a = (rng.uniform(size=n) < pi).astype(float)
    shift_samples = [list(rng.normal(size=3)) for _ in range(n)]
    return pd.DataFrame({"a": a, "x": x, "shift_samples": shift_samples})


def _make(estimand, **overrides):
    kwargs = dict(
        estimand=estimand,
        hidden_sizes=(16, 16),
        epochs=20,
        random_state=0,
    )
    kwargs.update(overrides)
    return RieszNet(**kwargs)


def _check(est, df):
    est.fit(df)
    pred = est.predict(df)
    assert pred.shape == (len(df),)
    assert np.all(np.isfinite(pred))


def test_ate(df_binary):
    _check(_make(ATE()), df_binary)


def test_att(df_binary):
    _check(_make(ATT()), df_binary)


def test_tsm(df_binary):
    _check(_make(TSM(level=1)), df_binary)


def test_additive_shift(df_continuous):
    _check(_make(AdditiveShift(delta=0.5)), df_continuous)


def test_local_shift(df_continuous):
    _check(_make(LocalShift(delta=0.5, threshold=0.0)), df_continuous)


def test_stochastic_intervention(df_stochastic):
    _check(_make(StochasticIntervention(samples_key="shift_samples")), df_stochastic)
