"""Autograd-friendly Bregman-Riesz loss in PyTorch.

For each original row ``z_i`` the per-row Riesz loss is

    L_i = ψ(α(x_i)) − Σ_j coef_j · φ'(α(point_j))

where ``(coef_j, point_j)`` come from ``rieszreg.trace(estimand, z_i)``. The
neural network produces ``η``; the loss spec's ``link_to_alpha`` produces
``α``; ``ψ`` and ``φ'`` are computed directly in torch so autograd matches the
analytic gradients in the loss spec module-for-module.

The four built-in losses are:

| Loss spec type        | ``ψ(α(η))``               | ``φ'(α(η))``         |
|-----------------------|---------------------------|----------------------|
| ``SquaredLoss``       | ``η²``                    | ``2η``               |
| ``KLLoss``            | ``exp(η)`` (clamped)      | ``η`` (clamped)      |
| ``BernoulliLoss``     | ``softplus(η)`` (clamped) | ``η`` (clamped)      |
| ``BoundedSquaredLoss``| ``α²``, α=lo+R·σ(η)       | ``2α``               |

η is clamped per the loss spec's own ``max_eta`` / ``max_abs_eta`` for
numerical stability, matching the analytic backends.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rieszreg.losses import LossSpec


def _spec_type(loss_spec: LossSpec) -> str:
    return loss_spec.to_spec()["type"]


def is_supported(loss_spec: LossSpec) -> bool:
    return _spec_type(loss_spec) in {
        "SquaredLoss",
        "KLLoss",
        "BernoulliLoss",
        "BoundedSquaredLoss",
    }


def validate_supported(loss_spec: LossSpec) -> None:
    if not is_supported(loss_spec):
        raise NotImplementedError(
            f"riesznet does not support loss type {_spec_type(loss_spec)!r}. "
            "Supported losses: SquaredLoss, KLLoss, BernoulliLoss, "
            "BoundedSquaredLoss."
        )


def _clamp_for_loss(loss_spec: LossSpec, eta: torch.Tensor) -> torch.Tensor:
    """Apply the loss spec's η clamp (matches numpy backends' numerical safety)."""
    t = _spec_type(loss_spec)
    if t == "KLLoss":
        bound = float(loss_spec.max_eta)
        return torch.clamp(eta, -bound, bound)
    if t in ("BernoulliLoss", "BoundedSquaredLoss"):
        bound = float(loss_spec.max_abs_eta)
        return torch.clamp(eta, -bound, bound)
    return eta  # SquaredLoss has no clamp


def psi_alpha(loss_spec: LossSpec, eta: torch.Tensor) -> torch.Tensor:
    """ψ(α(η)) — the convex generator evaluated at the link of η."""
    t = _spec_type(loss_spec)
    if t == "SquaredLoss":
        return eta * eta
    if t == "KLLoss":
        eta_c = _clamp_for_loss(loss_spec, eta)
        return torch.exp(eta_c)
    if t == "BernoulliLoss":
        eta_c = _clamp_for_loss(loss_spec, eta)
        return F.softplus(eta_c)
    if t == "BoundedSquaredLoss":
        lo = float(loss_spec.lo)
        hi = float(loss_spec.hi)
        eta_c = _clamp_for_loss(loss_spec, eta)
        sigma = torch.sigmoid(eta_c)
        alpha = lo + (hi - lo) * sigma
        return alpha * alpha
    raise NotImplementedError(t)


def phi_prime_alpha(loss_spec: LossSpec, eta: torch.Tensor) -> torch.Tensor:
    """φ'(α(η)) — the derivative of the convex generator, in η-space."""
    t = _spec_type(loss_spec)
    if t == "SquaredLoss":
        return 2.0 * eta
    if t == "KLLoss":
        # φ(t) = t log t, φ'(t) = log t + 1; the augmentation absorbs the
        # constant 1 (so the linear term in the augmented loss is (b/2)·log α
        # not (b/2)·(log α + 1)). Mirror that convention here so the analytic
        # `loss.gradient(a, b, eta)` parity check passes.
        return _clamp_for_loss(loss_spec, eta)
    if t == "BernoulliLoss":
        # φ'(α) = log(α/(1-α)) = η, with the loss spec's clamp.
        return _clamp_for_loss(loss_spec, eta)
    if t == "BoundedSquaredLoss":
        lo = float(loss_spec.lo)
        hi = float(loss_spec.hi)
        eta_c = _clamp_for_loss(loss_spec, eta)
        sigma = torch.sigmoid(eta_c)
        alpha = lo + (hi - lo) * sigma
        return 2.0 * alpha
    raise NotImplementedError(t)


def per_row_riesz_loss(
    loss_spec: LossSpec,
    eta_orig: torch.Tensor,
    eta_pts: torch.Tensor,
    coefs_pts: torch.Tensor,
    pt_to_row: torch.Tensor,
    n_rows: int,
) -> torch.Tensor:
    """Per-row Bregman-Riesz loss.

    Returns a ``(n_rows,)`` tensor whose ``i``-th entry is

        L_i = ψ(α(η_orig_i)) − Σ_{j: pt_to_row[j]==i} coefs_pts[j] · φ'(α(η_pts[j]))

    Parameters
    ----------
    loss_spec : LossSpec
        The Bregman loss to use. Validated by ``validate_supported``.
    eta_orig : torch.Tensor of shape (n_rows,)
        Network output η at each original row's feature point.
    eta_pts : torch.Tensor of shape (N,)
        Network output η at each trace evaluation point (concatenated across
        all rows in the batch).
    coefs_pts : torch.Tensor of shape (N,)
        Trace coefficients matching ``eta_pts``.
    pt_to_row : torch.LongTensor of shape (N,)
        For each trace point, the index in ``[0, n_rows)`` of the original
        row it belongs to.
    n_rows : int
        Number of original rows (length of ``eta_orig``).
    """
    psi = psi_alpha(loss_spec, eta_orig)                     # (n_rows,)
    phip = phi_prime_alpha(loss_spec, eta_pts)               # (N,)
    moment = torch.zeros(n_rows, device=eta_orig.device, dtype=eta_orig.dtype)
    moment.scatter_add_(0, pt_to_row, coefs_pts * phip)
    return psi - moment
