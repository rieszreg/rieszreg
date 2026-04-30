"""Property-based tests via hypothesis.

The point of these tests is to assert *invariants* of the library — properties
that should hold for any reasonable input — rather than specific numeric
outcomes. Hypothesis generates inputs and shrinks failing cases automatically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rieszboost import (
    ATE,
    AdditiveShift,
    ATT,
    BernoulliLoss,
    BoundedSquaredLoss,
    Estimand,
    KLLoss,
    LocalShift,
    RieszBooster,
    SquaredLoss,
    StochasticIntervention,
    TSM,
)
from rieszboost.augmentation import build_augmented
from rieszboost.estimand import estimand_from_spec
from rieszboost.losses import loss_from_spec
from rieszboost.tracer import trace


# ---------- Tracer / Estimand invariants ----------

@given(
    c1=st.floats(min_value=-10, max_value=10, allow_nan=False),
    c2=st.floats(min_value=-10, max_value=10, allow_nan=False),
    x_val=st.floats(min_value=0, max_value=1, allow_nan=False),
)
def test_tracer_is_linear_in_m(c1: float, c2: float, x_val: float):
    """For any scalars c1, c2 and any two linear m's,
        trace(c1·m1 + c2·m2) == c1·trace(m1) + c2·trace(m2)
    term-by-term."""
    def m1(alpha):
        def inner(z):
            return alpha(a=1, x=z["x"]) - alpha(a=0, x=z["x"])
        return inner

    def m2(alpha):
        def inner(z):
            return alpha(a=1, x=z["x"])  # ATE-like and TSM-like
        return inner

    def m_combined(alpha):
        def inner(z):
            return c1 * m1(alpha)(z) + c2 * m2(alpha)(z)
        return inner

    z = {"a": 0, "x": x_val}
    pairs = trace(m_combined, z)
    by_a = {p["a"]: c for c, p in pairs}
    # Expected: {1: c1 + c2, 0: -c1}, modulo float-equality jitter.
    assert by_a.get(1, 0.0) == pytest.approx(c1 + c2, abs=1e-10)
    assert by_a.get(0, 0.0) == pytest.approx(-c1, abs=1e-10)


@given(
    n=st.integers(min_value=10, max_value=50),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_augmentation_invariant_under_row_permutation(n: int, seed: int):
    """Shuffling rows produces the same multiset of augmented (a, b) coefficients."""
    rng = np.random.default_rng(seed)
    rows = [{"a": float(rng.integers(0, 2)), "x": float(rng.uniform())} for _ in range(n)]

    aug1 = build_augmented(rows, ATE())
    perm = list(rng.permutation(n))
    aug2 = build_augmented([rows[i] for i in perm], ATE())

    # Compare the multiset of (a, b) coefficients — independent of order.
    sig1 = sorted(zip(aug1.a.tolist(), aug1.b.tolist()))
    sig2 = sorted(zip(aug2.a.tolist(), aug2.b.tolist()))
    assert sig1 == sig2


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=10, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_fit_is_reproducible_given_random_state(seed: int):
    """Same data + same random_state → bitwise-identical predictions."""
    rng = np.random.default_rng(seed)
    n = 200
    df = pd.DataFrame({"a": rng.binomial(1, 0.5, n).astype(float),
                       "x": rng.uniform(0, 1, n)})
    common = dict(estimand=ATE(), n_estimators=20, learning_rate=0.1, random_state=42)
    p1 = RieszBooster(**common).fit(df).predict(df)
    p2 = RieszBooster(**common).fit(df).predict(df)
    np.testing.assert_array_equal(p1, p2)


@given(
    delta=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False),
    threshold=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False),
    level=st.integers(min_value=0, max_value=10),
)
def test_estimand_factory_spec_round_trip(delta: float, threshold: float, level: int):
    """Every built-in factory_spec must round-trip through estimand_from_spec."""
    for est in (
        ATE(),
        ATT(),
        TSM(level=level),
        AdditiveShift(delta=delta),
        LocalShift(delta=delta, threshold=threshold),
        StochasticIntervention(),
    ):
        roundtrip = estimand_from_spec(est.factory_spec)
        assert roundtrip.feature_keys == est.feature_keys
        assert roundtrip.extra_keys == est.extra_keys
        assert roundtrip.factory_spec == est.factory_spec


# ---------- LossSpec link round-trip ----------

@given(
    eta=st.floats(min_value=-25.0, max_value=25.0, allow_nan=False, allow_infinity=False),
)
def test_squared_loss_link_is_identity(eta: float):
    loss = SquaredLoss()
    assert loss.alpha_to_eta(loss.link_to_alpha(eta)) == pytest.approx(eta, abs=1e-12)


@given(
    eta=st.floats(min_value=-25.0, max_value=25.0, allow_nan=False, allow_infinity=False),
)
def test_kl_loss_link_round_trip(eta: float):
    loss = KLLoss(max_eta=50.0)
    alpha = loss.link_to_alpha(eta)
    assert alpha > 0
    assert loss.alpha_to_eta(alpha) == pytest.approx(eta, abs=1e-9)


@given(
    eta=st.floats(min_value=-15.0, max_value=15.0, allow_nan=False, allow_infinity=False),
)
def test_bernoulli_link_round_trip(eta: float):
    loss = BernoulliLoss(max_abs_eta=30.0)
    alpha = loss.link_to_alpha(eta)
    assert 0 < alpha < 1
    assert loss.alpha_to_eta(alpha) == pytest.approx(eta, abs=1e-9)


@given(
    eta=st.floats(min_value=-15.0, max_value=15.0, allow_nan=False, allow_infinity=False),
    lo=st.floats(min_value=-100.0, max_value=0.0, allow_nan=False),
    width=st.floats(min_value=0.5, max_value=100.0, allow_nan=False),
)
def test_bounded_squared_link_round_trip(eta: float, lo: float, width: float):
    hi = lo + width
    loss = BoundedSquaredLoss(lo=lo, hi=hi, max_abs_eta=30.0)
    alpha = loss.link_to_alpha(eta)
    assert lo < alpha < hi
    assert loss.alpha_to_eta(alpha) == pytest.approx(eta, abs=1e-7)


@given(
    eta=st.floats(min_value=-25.0, max_value=25.0, allow_nan=False, allow_infinity=False),
)
def test_loss_spec_round_trip_through_to_spec(eta: float):
    """Every loss reconstructed from its to_spec() should produce identical
    link mappings."""
    losses = [
        SquaredLoss(),
        KLLoss(max_eta=40.0),
        BernoulliLoss(max_abs_eta=25.0),
        BoundedSquaredLoss(lo=-3.0, hi=3.0),
    ]
    for original in losses:
        reconstructed = loss_from_spec(original.to_spec())
        assert type(reconstructed) is type(original)
        np.testing.assert_array_almost_equal(
            np.asarray(original.link_to_alpha(eta)),
            np.asarray(reconstructed.link_to_alpha(eta)),
        )


# ---------- Loss-spec analytic gradient/hessian sanity ----------

@given(
    eta=st.floats(min_value=-3.0, max_value=3.0, allow_nan=False),
    a_val=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    b_val=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False),
)
@settings(max_examples=30)
def test_squared_loss_gradient_matches_finite_diff(eta: float, a_val: float, b_val: float):
    """Analytic gradient should match numerical finite-difference of loss_row(α(η))."""
    loss = SquaredLoss()
    a = np.array([a_val])
    b = np.array([b_val])
    e = np.array([eta])
    h = 1e-5
    L_plus = loss.loss_row(a, b, loss.link_to_alpha(e + h))[0]
    L_minus = loss.loss_row(a, b, loss.link_to_alpha(e - h))[0]
    fd = (L_plus - L_minus) / (2 * h)
    analytic = loss.gradient(a, b, e)[0]
    assert analytic == pytest.approx(fd, abs=1e-3)
