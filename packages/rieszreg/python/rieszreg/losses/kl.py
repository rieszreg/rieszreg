"""KL Bregman-Riesz loss: h(t) = t · log(t), exp link, density-ratio estimands."""

from __future__ import annotations

import numpy as np

from .base import Loss


class KLLoss(Loss):
    """h(t) = t · log(t). h'(t) = log(t) + 1. h_tilde(t) = t.
    Exp link: α = exp(η) so α > 0 always.

    Augmented-row loss in α-space: D · α + C · log(α) (constant offset dropped).
    With η = log(α), gradient (η) = D · exp(η) + C = D · α + C.
    Hessian (η) = D · exp(η) = D · α (floored).

    Pair with density-ratio estimands (TSM, IPSI). For ATE-style estimands,
    whose α takes negative values, fit raises.
    """

    name = "kl"
    alpha_domain = (0.0, np.inf)

    def __init__(self, max_eta: float = 50.0):
        super().__init__()
        # Clip η before exp() to avoid overflow when the backend takes a big update step.
        self.max_eta = float(max_eta)

    def _clip(self, eta):
        return np.clip(eta, -self.max_eta, self.max_eta)

    def _safe_alpha(self, alpha):
        return np.maximum(alpha, np.exp(-self.max_eta))

    def potential(self, alpha):
        a = self._safe_alpha(alpha)
        return a * np.log(a)

    def potential_deriv(self, alpha):
        a = self._safe_alpha(alpha)
        return np.log(a) + 1.0

    def tilde_potential(self, alpha):
        # h_tilde(t) = t · h'(t) − h(t) = t · (log t + 1) − t · log t = t.
        return alpha

    def link_to_alpha(self, eta):
        return np.exp(self._clip(eta))

    def alpha_to_eta(self, alpha):
        if np.any(np.asarray(alpha) <= 0):
            raise ValueError("KLLoss requires positive alpha for init.")
        return np.log(alpha)

    def aug_grad_eta(self, is_original, potential_deriv_coef, eta):
        alpha = np.exp(self._clip(eta))
        return is_original * alpha + potential_deriv_coef

    def aug_hess_eta(self, is_original, potential_deriv_coef, eta, hessian_floor):
        del potential_deriv_coef
        alpha = np.exp(self._clip(eta))
        return np.maximum(is_original * alpha, hessian_floor)

    def curvature_eta(self, eta):
        return np.exp(self._clip(eta))  # h''(α) · α² = α

    def best_constant_init(self, m_bar: float) -> float:
        # Exp link: α > 0. Floor m̄ at the smallest α the link can represent.
        eps = float(np.exp(-self.max_eta))
        return max(float(m_bar), eps)

    def to_spec(self) -> dict:
        return {"type": "KLLoss", "args": {"max_eta": self.max_eta}}
