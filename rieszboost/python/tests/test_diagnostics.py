import numpy as np
import pandas as pd
import pytest

import rieszboost
from rieszboost import RieszBooster
from rieszboost.diagnostics import diagnose


def test_diagnose_from_array():
    rng = np.random.default_rng(0)
    alpha = rng.normal(0, 1, size=500)
    d = diagnose(alpha)
    assert d.n == 500
    assert d.min < d.max
    assert pytest.approx(d.rms, rel=1e-3) == float(np.sqrt(np.mean(alpha**2)))
    assert d.warnings == []


def test_diagnose_warns_on_extreme_outlier():
    rng = np.random.default_rng(1)
    alpha = np.concatenate([rng.normal(0, 1, 999), [500.0]])
    d = diagnose(alpha, extreme_threshold=30.0)
    assert d.n_extreme == 1
    assert any("max |alpha_hat|" in w for w in d.warnings)


def test_diagnose_warns_on_many_extremes():
    alpha = np.concatenate([np.zeros(900), np.full(100, 50.0)])
    d = diagnose(alpha, extreme_threshold=30.0, extreme_fraction_warn=0.01)
    assert d.extreme_fraction == 0.1
    assert any("near-positivity" in w for w in d.warnings)


def test_diagnose_summary_renders_without_error():
    s = diagnose(np.linspace(-2, 2, 100)).summary()
    assert "RMS magnitude" in s
    assert "extreme rows" in s


def test_diagnose_with_booster_includes_riesz_loss():
    rng = np.random.default_rng(0)
    n = 500
    x = rng.uniform(0, 1, n)
    pi = 1 / (1 + np.exp(-(-0.02 * x - x**2 + 4 * np.log(x + 0.3) + 1.5)))
    a = rng.binomial(1, pi)
    df = pd.DataFrame({"a": a.astype(float), "x": x.astype(float)})
    booster = RieszBooster(estimand=rieszboost.ATE(), n_estimators=20).fit(df)
    d = diagnose(booster=booster, X=df)
    assert d.riesz_loss is not None
    assert d.n == n


def test_booster_diagnose_method():
    rng = np.random.default_rng(0)
    n = 200
    x = rng.uniform(0, 1, n)
    a = rng.binomial(1, 0.5, n)
    df = pd.DataFrame({"a": a.astype(float), "x": x.astype(float)})
    booster = RieszBooster(estimand=rieszboost.ATE(), n_estimators=10).fit(df)
    d = booster.diagnose(df)
    assert d.n == n


def test_diagnose_requires_alpha_or_booster():
    with pytest.raises(ValueError):
        diagnose()
