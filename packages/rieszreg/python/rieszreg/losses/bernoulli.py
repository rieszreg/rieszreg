"""Bernoulli Bregman-Riesz loss: binary entropy h, sigmoid link."""

from __future__ import annotations

import numpy as np

from .base import Loss


class BernoulliLoss(Loss):
    """h(t) = t · log(t) + (1 − t) · log(1 − t)  (binary entropy on (0, 1)).
    h'(t) = log(t) − log(1 − t) = logit(t). h_tilde(t) = -log(1 − t).
    Sigmoid link: α = σ(η) ∈ (0, 1).

    Augmented-row loss in η:
        l(η) = D · softplus(η) + C · η      (since softplus(η) = -log(1 - σ(η)))
    Gradient (η) = D · α + C,  Hessian (η) = D · α (1 − α).

    Use when α₀ is known to lie in (0, 1) by problem structure (e.g. trimmed
    propensity-score-style representers). If true α₀ exceeds 1, the sigmoid
    saturates and the fit plateaus near 1 — there is no warning, just a poor
    fit. Validate that your representer is bounded before reaching for this.
    """

    name = "bernoulli"

    def __init__(self, max_abs_eta: float = 30.0):
        super().__init__()
        # Clip η before σ to avoid 0/1 saturation that ruins the gradient.
        self.max_abs_eta = float(max_abs_eta)

    def _clip(self, eta):
        return np.clip(eta, -self.max_abs_eta, self.max_abs_eta)

    def _safe_alpha(self, alpha):
        eps = np.exp(-self.max_abs_eta)
        return np.clip(alpha, eps, 1.0 - eps)

    def potential(self, alpha):
        a = self._safe_alpha(alpha)
        return a * np.log(a) + (1.0 - a) * np.log(1.0 - a)

    def potential_deriv(self, alpha):
        a = self._safe_alpha(alpha)
        return np.log(a / (1.0 - a))

    def tilde_potential(self, alpha):
        # h_tilde(t) = t · h'(t) − h(t) = t · logit(t) − [t log t + (1-t) log(1-t)] = -log(1-t)
        a = self._safe_alpha(alpha)
        return -np.log(1.0 - a)

    def link_to_alpha(self, eta):
        eta = self._clip(eta)
        return 1.0 / (1.0 + np.exp(-eta))

    def alpha_to_eta(self, alpha):
        a = np.asarray(alpha)
        if np.any((a <= 0) | (a >= 1)):
            raise ValueError("BernoulliLoss requires alpha in (0, 1) for init.")
        return np.log(alpha / (1.0 - alpha))

    def aug_grad_eta(self, is_original, potential_deriv_coef, eta):
        alpha = self.link_to_alpha(eta)
        return is_original * alpha + potential_deriv_coef

    def aug_hess_eta(self, is_original, potential_deriv_coef, eta, hessian_floor):
        del potential_deriv_coef
        alpha = self.link_to_alpha(eta)
        return np.maximum(is_original * alpha * (1.0 - alpha), hessian_floor)

    def best_constant_init(self, m_bar: float) -> float:
        # Sigmoid link: α ∈ (0, 1). Clip m̄ to the interior.
        eps = float(np.exp(-self.max_abs_eta))
        return float(np.clip(m_bar, eps, 1.0 - eps))

    def to_spec(self) -> dict:
        return {"type": "BernoulliLoss", "args": {"max_abs_eta": self.max_abs_eta}}
