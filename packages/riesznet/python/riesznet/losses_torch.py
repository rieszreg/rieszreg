"""Autograd-friendly Bregman-Riesz loss in PyTorch.

The per-row Riesz loss of original row ``i`` sums the augmented-row terms
that ``Estimand.augment`` emits for it (``origin_index == i``):

    L_i = Σ_r [ D_r · h̃(α(z_r)) + C_r · h'(α(z_r)) ]

the same formula as ``Loss.aug_loss_alpha``. The network produces ``η``;
``h̃`` and ``h'`` are computed directly in torch as functions of ``η`` so
autograd matches the analytic gradients in the loss spec.

| Loss spec type        | ``h̃(α(η))``              | ``h'(α(η))``         |
|-----------------------|---------------------------|----------------------|
| ``SquaredLoss``       | ``η²``                    | ``2η``               |
| ``KLLoss``            | ``exp(η)`` (clamped)      | ``η + 1`` (clamped)  |
| ``BernoulliLoss``     | ``softplus(η)`` (clamped) | ``η`` (clamped)      |
| ``BoundedSquaredLoss``| ``α²``, α=lo+R·σ(η)       | ``2α``               |

η is clamped per the loss spec's own ``max_eta`` / ``max_abs_eta`` for
numerical stability, matching the analytic backends.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from rieszreg.losses import BernoulliLoss, BoundedSquaredLoss, KLLoss, Loss, SquaredLoss


def _terms(loss_spec: Loss) -> tuple[Callable, Callable]:
    """``(h̃∘link, h'∘link)`` as torch functions of η, resolved once per loss."""
    if isinstance(loss_spec, SquaredLoss):
        return (lambda eta: eta * eta), (lambda eta: 2.0 * eta)
    if isinstance(loss_spec, KLLoss):
        b = float(loss_spec.max_eta)
        return (
            lambda eta: torch.exp(torch.clamp(eta, -b, b)),
            lambda eta: torch.clamp(eta, -b, b) + 1.0,
        )
    if isinstance(loss_spec, BernoulliLoss):
        b = float(loss_spec.max_abs_eta)
        return (
            lambda eta: F.softplus(torch.clamp(eta, -b, b)),
            lambda eta: torch.clamp(eta, -b, b),
        )
    if isinstance(loss_spec, BoundedSquaredLoss):
        lo, R, b = float(loss_spec.lo), float(loss_spec.hi - loss_spec.lo), float(loss_spec.max_abs_eta)

        def alpha(eta):
            return lo + R * torch.sigmoid(torch.clamp(eta, -b, b))

        return (lambda eta: alpha(eta) ** 2), (lambda eta: 2.0 * alpha(eta))
    raise NotImplementedError(
        f"riesznet does not support {type(loss_spec).__name__}. Supported "
        "losses: SquaredLoss, KLLoss, BernoulliLoss, BoundedSquaredLoss."
    )


class TorchRieszLoss:
    """Per-row Bregman-Riesz loss for one ``Loss`` spec (type resolved once)."""

    def __init__(self, loss_spec: Loss):
        self.tilde_potential, self.potential_deriv = _terms(loss_spec)

    def per_row(
        self,
        eta: torch.Tensor,
        is_original: torch.Tensor,
        potential_deriv_coef: torch.Tensor,
        origin: torch.Tensor,
        n_rows: int,
    ) -> torch.Tensor:
        """``(n_rows,)`` tensor: Σ over each row's augmented terms of
        ``D · h̃(α(η)) + C · h'(α(η))``, grouped by ``origin`` in ``[0, n_rows)``."""
        terms = (
            is_original * self.tilde_potential(eta)
            + potential_deriv_coef * self.potential_deriv(eta)
        )
        out = torch.zeros(n_rows, device=eta.device, dtype=eta.dtype)
        return out.scatter_add_(0, origin, terms)
