"""Smoke tests for Bregman-Riesz `Loss` implementations."""

from __future__ import annotations

import numpy as np
import pytest

from rieszreg import (
    BernoulliLoss,
    BoundedSquaredLoss,
    KLLoss,
    Loss,
    SquaredLoss,
    loss_from_spec,
)


@pytest.mark.parametrize(
    "loss",
    [SquaredLoss(), KLLoss(), BernoulliLoss(), BoundedSquaredLoss(lo=0.0, hi=1.0)],
)
def test_to_from_spec_round_trip(loss: Loss):
    rebuilt = loss_from_spec(loss.to_spec())
    assert type(rebuilt) is type(loss)
    assert rebuilt.name == loss.name


def test_squared_link_is_identity():
    loss = SquaredLoss()
    eta = np.array([-1.0, 0.0, 2.0])
    assert np.allclose(loss.link_to_alpha(eta), eta)
    assert np.allclose(loss.alpha_to_eta(eta), eta)


def test_squared_gradient_and_hessian_match_finite_diff():
    loss = SquaredLoss()
    is_original = np.array([1.0, 0.0, 1.0])
    pdc = np.array([0.0, -1.0, -0.75])
    eta = np.array([0.5, 0.5, 0.5])
    h = 1e-6
    grad_analytic = loss.aug_grad_eta(is_original, pdc, eta)
    loss_plus = loss.aug_loss_alpha(is_original, pdc, loss.link_to_alpha(eta + h))
    loss_minus = loss.aug_loss_alpha(is_original, pdc, loss.link_to_alpha(eta - h))
    grad_numeric = (loss_plus - loss_minus) / (2 * h)
    np.testing.assert_allclose(grad_analytic, grad_numeric, rtol=1e-4)
    # Hessian under squared loss is 2D (floored).
    hess = loss.aug_hess_eta(is_original, pdc, eta, hessian_floor=0.0)
    np.testing.assert_allclose(hess, 2.0 * is_original)


def test_kl_link_keeps_alpha_positive():
    loss = KLLoss(max_eta=10.0)
    eta = np.array([-5.0, 0.0, 8.0])
    alpha = loss.link_to_alpha(eta)
    assert np.all(alpha > 0)


def test_bernoulli_link_in_unit_interval():
    loss = BernoulliLoss(max_abs_eta=20.0)
    eta = np.linspace(-30, 30, 21)
    alpha = loss.link_to_alpha(eta)
    assert np.all((alpha > 0) & (alpha < 1))


def test_bounded_squared_clipping():
    loss = BoundedSquaredLoss(lo=2.0, hi=8.0)
    eta = np.array([-50.0, 0.0, 50.0])
    alpha = loss.link_to_alpha(eta)
    assert alpha.min() > 2.0
    assert alpha.max() < 8.0
    # m̄ outside the interval is clipped to the interior.
    assert loss.best_constant_init(0.0) > 2.0    # below lo → floored
    assert loss.best_constant_init(100.0) < 8.0  # above hi → ceilinged
    # m̄ inside the interval round-trips.
    assert loss.best_constant_init(5.0) == pytest.approx(5.0)


def test_bounded_squared_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="must be"):
        BoundedSquaredLoss(lo=5.0, hi=3.0)


def test_loss_from_spec_unknown_raises():
    with pytest.raises(ValueError, match="Unknown loss spec type"):
        loss_from_spec({"type": "Nope"})


def test_inline_loss_squared_matches_squared_built_in():
    """`Loss(potential=t**2, link='identity')` should match `SquaredLoss` end-to-end."""
    from rieszreg import Loss

    inline = Loss(
        potential=lambda t: t**2,
        potential_deriv=lambda t: 2.0 * t,
        link="identity",
    )
    built_in = SquaredLoss()

    is_original = np.array([1.0, 0.0, 1.0])
    pdc = np.array([0.0, -1.0, 0.25])
    eta = np.array([0.5, 0.5, 0.5])

    np.testing.assert_allclose(
        inline.aug_grad_eta(is_original, pdc, eta),
        built_in.aug_grad_eta(is_original, pdc, eta),
        rtol=1e-4,
    )


def test_inline_loss_kl_matches_kl_built_in():
    """`Loss(potential=t·log(t), link='exp')` should match `KLLoss` (up to numerical
    derivative tolerance) on a density-ratio-compatible (D, C)."""
    from rieszreg import Loss

    inline = Loss(
        potential=lambda t: t * np.log(np.maximum(t, 1e-12)),
        potential_deriv=lambda t: np.log(np.maximum(t, 1e-12)) + 1.0,
        link="exp",
    )
    built_in = KLLoss()

    is_original = np.array([1.0, 0.0, 0.0])
    pdc = np.array([0.0, -1.0, -0.5])
    eta = np.array([0.1, 0.2, -0.3])

    np.testing.assert_allclose(
        inline.aug_grad_eta(is_original, pdc, eta),
        built_in.aug_grad_eta(is_original, pdc, eta),
        rtol=1e-4,
    )


def test_subclass_loss_pattern():
    """Subclassing `Loss` (mirroring the built-in style) should work
    end-to-end with the orchestrator helpers."""
    from rieszreg import Loss

    class MyLoss(Loss):
        name = "mine"

        def potential(self, alpha):
            return alpha * np.log(alpha)

        def potential_deriv(self, alpha):
            return np.log(alpha) + 1.0

        def link_to_alpha(self, eta):
            return np.exp(np.clip(eta, -50.0, 50.0))

        def alpha_to_eta(self, alpha):
            return np.log(alpha)

    loss = MyLoss()
    is_original = np.array([1.0, 0.0])
    pdc = np.array([0.0, -1.0])
    alpha = np.array([1.5, 1.5])
    # Augmented loss in α-space matches D · h_tilde(α) + C · h'(α).
    expected = (
        is_original * (alpha * loss.potential_deriv(alpha) - loss.potential(alpha))
        + pdc * loss.potential_deriv(alpha)
    )
    np.testing.assert_allclose(loss.aug_loss_alpha(is_original, pdc, alpha), expected)
