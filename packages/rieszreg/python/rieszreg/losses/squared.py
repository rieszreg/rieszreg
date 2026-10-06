"""Squared Riesz loss: h(t) = t², identity link."""

from __future__ import annotations

import numpy as np

from .base import Loss


class SquaredLoss(Loss):
    """h(t) = t². h_tilde(t) = t · h'(t) - h(t) = t². Identity link η = α.

    Augmented-row loss in α-space: D · α² + 2C · α.
    Gradient (η-space) = 2D · η + 2C. Hessian = 2D (floored).
    """

    name = "squared"

    def potential(self, alpha):
        return alpha ** 2

    def potential_deriv(self, alpha):
        return 2.0 * alpha

    def tilde_potential(self, alpha):
        return alpha ** 2

    def link_to_alpha(self, eta):
        return eta

    def alpha_to_eta(self, alpha):
        return alpha

    def aug_grad_eta(self, is_original, potential_deriv_coef, eta):
        return 2.0 * is_original * eta + 2.0 * potential_deriv_coef

    def aug_hess_eta(self, is_original, potential_deriv_coef, eta, hessian_floor):
        del potential_deriv_coef, eta
        return np.maximum(2.0 * is_original, hessian_floor)

    def curvature_eta(self, eta):
        return np.full(np.shape(eta), 2.0)

    def best_constant_init(self, m_bar: float) -> float:
        return float(m_bar)

    def to_spec(self) -> dict:
        return {"type": "SquaredLoss", "args": {}}
