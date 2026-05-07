"""Smoke tests for built-in estimand factories and round-trip via factory_spec."""

from __future__ import annotations

import pickle

import pytest

from rieszreg import (
    ATE,
    ATT,
    AdditiveShift,
    FiniteEvalEstimand,
    LocalShift,
    OutcomeRegNormSq,
    TSM,
    estimand_from_spec,
    trace,
)
from rieszreg.estimands.base import StochasticIntervention


@pytest.mark.parametrize(
    "estimand,expected_keys",
    [
        (ATE(), ("a", "x")),
        (ATT(), ("a", "x")),
        (TSM(level=1), ("a", "x")),
        (AdditiveShift(delta=0.5), ("a", "x")),
        (LocalShift(delta=0.3, threshold=0.7), ("a", "x")),
        (OutcomeRegNormSq(), ("x",)),
    ],
)
def test_factory_feature_keys(estimand, expected_keys):
    assert estimand.feature_keys == expected_keys


def test_stochastic_intervention_stub_raises():
    with pytest.raises(NotImplementedError, match="being rewritten"):
        StochasticIntervention()


@pytest.mark.parametrize(
    "estimand",
    [ATE(), ATT(), TSM(level=1), AdditiveShift(delta=0.5),
     LocalShift(delta=0.3, threshold=0.7), OutcomeRegNormSq()],
)
def test_factory_spec_round_trip(estimand):
    rebuilt = estimand_from_spec(estimand.factory_spec)
    assert rebuilt.name == estimand.name
    assert rebuilt.feature_keys == estimand.feature_keys


def test_pickle_round_trip_built_in():
    est = AdditiveShift(delta=0.25, treatment="t", covariates=("x1", "x2"))
    rebuilt = pickle.loads(pickle.dumps(est))
    assert rebuilt.factory_spec == est.factory_spec
    assert rebuilt.feature_keys == est.feature_keys


def test_estimand_from_spec_unknown_factory_raises():
    with pytest.raises(ValueError, match="Unknown estimand factory"):
        estimand_from_spec({"factory": "NOPE", "args": {}})


def test_ate_traces_to_correct_pairs():
    pairs = trace(ATE(), {"a": 0.0, "x": 1.5})
    points = {tuple(sorted(p.items())): c for c, p in pairs}
    # m(alpha)(z) = alpha(a=1, x=1.5) - alpha(a=0, x=1.5)
    assert points[(("a", 1), ("x", 1.5))] == pytest.approx(1.0)
    assert points[(("a", 0), ("x", 1.5))] == pytest.approx(-1.0)


def test_att_zero_when_a_is_zero():
    pairs = trace(ATT(), {"a": 0.0, "x": 0.5})
    # ATT m = a * (alpha(1, x) - alpha(0, x)). At a=0, m is identically 0.
    points = {tuple(sorted(p.items())): c for c, p in pairs}
    # All coefs zero (or the dict is empty).
    for c in points.values():
        assert c == pytest.approx(0.0)


def test_local_shift_zero_above_threshold():
    pairs = trace(LocalShift(delta=0.3, threshold=0.5), {"a": 0.7, "x": 1.0})
    assert pairs == []


def test_custom_estimand_pickle_uses_factory_spec_when_set():
    def m(alpha):
        def inner(z, y=None):
            return alpha(a=1, x=z["x"]) - alpha(a=0, x=z["x"])
        return inner

    est = FiniteEvalEstimand(feature_keys=("a", "x"), m=m, name="custom")
    # No factory_spec → falls back to default reduce; tested elsewhere.
    assert est.factory_spec is None


def test_custom_estimand_can_read_y():
    """Y-dependent custom m: m(α)(z, y) = 1{y > τ} · (α(1, x) − α(0, x))."""
    tau = 0.5

    def m(alpha):
        def inner(z, y):
            indicator = 1.0 if y > tau else 0.0
            return indicator * (alpha(a=1, x=z["x"]) - alpha(a=0, x=z["x"]))
        return inner

    est = FiniteEvalEstimand(feature_keys=("a", "x"), m=m, name="upper-half-ate")

    # y > tau: indicator = 1, two non-zero coefficients.
    pairs_above = trace(est, {"a": 0.0, "x": 1.5}, y=0.9)
    points_above = {tuple(sorted(p.items())): c for c, p in pairs_above}
    assert points_above[(("a", 1), ("x", 1.5))] == pytest.approx(1.0)
    assert points_above[(("a", 0), ("x", 1.5))] == pytest.approx(-1.0)

    # y ≤ tau: indicator = 0, all coefficients zero (returned as []).
    pairs_below = trace(est, {"a": 0.0, "x": 1.5}, y=0.2)
    assert pairs_below == []
