"""Squared loss with sigmoid-scaled link forcing α ∈ (lo, hi)."""

from __future__ import annotations

import numpy as np

from .base import Loss


class BoundedSquaredLoss(Loss):
    """Squared-loss potential in α-space, but with a sigmoid-scaled link
    forcing α ∈ (lo, hi). Useful when you have a hard prior bound on the
    representer (e.g. trimmed propensity 1/π̂ ∈ [1, 1/ε]).

    h(α) = α². h_tilde(α) = α². h'(α) = 2α.
    Augmented-row loss in α: D · α² + 2C · α (same shape as `SquaredLoss`).

    Link: α = lo + R · σ(η),  where R = hi − lo.
    Gradient (η) = 2(D · α + C) · R · σ(η)·(1 − σ(η)).
    Hessian  (η) = 2D · (R · σ(1−σ))² + 2(D · α + C) · R · σ(1−σ)(1 − 2σ).

    Strictly speaking this is squared loss with a saturating reparameterization,
    not a *new* Bregman divergence. Predictions stay in (lo, hi) by
    construction; if true α₀ is outside that range the fit saturates.
    """

    name = "bounded_squared"

    def __init__(self, lo: float, hi: float, max_abs_eta: float = 30.0):
        super().__init__()
        if not (lo < hi):
            raise ValueError(f"lo ({lo}) must be < hi ({hi}).")
        self.lo = float(lo)
        self.hi = float(hi)
        self.max_abs_eta = float(max_abs_eta)

    def _R(self):
        return self.hi - self.lo

    def _sigma(self, eta):
        eta_c = np.clip(eta, -self.max_abs_eta, self.max_abs_eta)
        return 1.0 / (1.0 + np.exp(-eta_c))

    def potential(self, alpha):
        return alpha ** 2

    def potential_deriv(self, alpha):
        return 2.0 * alpha

    def tilde_potential(self, alpha):
        return alpha ** 2

    def link_to_alpha(self, eta):
        return self.lo + self._R() * self._sigma(eta)

    def alpha_to_eta(self, alpha):
        u = (alpha - self.lo) / self._R()
        if np.any((np.asarray(u) <= 0) | (np.asarray(u) >= 1)):
            raise ValueError(
                f"BoundedSquaredLoss requires alpha in ({self.lo}, {self.hi})."
            )
        return np.log(u / (1.0 - u))

    def aug_grad_eta(self, is_original, potential_deriv_coef, eta):
        sigma = self._sigma(eta)
        alpha = self.lo + self._R() * sigma
        # d/dη loss = 2(D·α + C) · dα/dη ;  dα/dη = R σ (1−σ).
        return (
            2.0 * (is_original * alpha + potential_deriv_coef)
            * self._R() * sigma * (1.0 - sigma)
        )

    def aug_hess_eta(self, is_original, potential_deriv_coef, eta, hessian_floor):
        sigma = self._sigma(eta)
        alpha = self.lo + self._R() * sigma
        R = self._R()
        d_alpha = R * sigma * (1.0 - sigma)
        d2_alpha = R * sigma * (1.0 - sigma) * (1.0 - 2.0 * sigma)
        # d²/dη² loss = d²L/dα² · (dα/dη)² + dL/dα · d²α/dη²
        h = (
            2.0 * is_original * d_alpha**2
            + 2.0 * (is_original * alpha + potential_deriv_coef) * d2_alpha
        )
        return np.maximum(h, hessian_floor)

    def curvature_eta(self, eta):
        sigma = self._sigma(eta)
        return 2.0 * (self._R() * sigma * (1.0 - sigma)) ** 2

    def best_constant_init(self, m_bar: float) -> float:
        # Scaled-sigmoid link: α ∈ (lo, hi). Clip m̄ to the interior.
        eps = (self.hi - self.lo) * float(np.exp(-self.max_abs_eta))
        return float(np.clip(m_bar, self.lo + eps, self.hi - eps))

    def to_spec(self) -> dict:
        return {
            "type": "BoundedSquaredLoss",
            "args": {"lo": self.lo, "hi": self.hi, "max_abs_eta": self.max_abs_eta},
        }
