"""Autograd-vs-analytic gradient parity for all four built-in Bregman losses.

For each loss, we sample random ``(eta, coefs, pt_to_row)``, compute the
per-row Riesz loss in torch, and compare the gradient that autograd produces
to the analytic gradient computed via the loss spec's ``gradient(a, b, eta)``.

If this passes, autograd is computing exactly the same gradient that
augmentation-style backends use — the contract is satisfied for every
Bregman loss the meta-package supports.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rieszreg import (
    BernoulliLoss,
    BoundedSquaredLoss,
    KLLoss,
    SquaredLoss,
)

from riesznet.losses_torch import per_row_riesz_loss


def _build_problem(loss, n_rows=8, k_per_row=3, seed=0):
    """Random eta_orig + per-row trace coefficients/points + their concatenation
    as ``(coefs, pt_to_row)``. Coefs are non-negative for losses that require it."""
    rng = np.random.default_rng(seed)
    spec_type = loss.to_spec()["type"]

    eta_orig_np = rng.normal(0.0, 1.0, size=n_rows)
    if spec_type == "BernoulliLoss":
        eta_orig_np *= 0.5  # keep sigmoid out of saturation

    n_pts = n_rows * k_per_row
    eta_pts_np = rng.normal(0.0, 1.0, size=n_pts)
    if spec_type == "BernoulliLoss":
        eta_pts_np *= 0.5

    if spec_type in ("KLLoss", "BernoulliLoss"):
        coefs_np = rng.uniform(0.0, 1.0, size=n_pts)
    else:
        coefs_np = rng.normal(0.0, 1.0, size=n_pts)
    pt_to_row_np = np.repeat(np.arange(n_rows, dtype=np.int64), k_per_row)

    return eta_orig_np, eta_pts_np, coefs_np, pt_to_row_np


def _torch_grad_per_row(loss, eta_orig_np, eta_pts_np, coefs_np, pt_to_row_np):
    eta_orig = torch.tensor(eta_orig_np, dtype=torch.float64, requires_grad=True)
    eta_pts = torch.tensor(eta_pts_np, dtype=torch.float64, requires_grad=True)
    coefs = torch.as_tensor(coefs_np, dtype=torch.float64)
    pt_to_row = torch.as_tensor(pt_to_row_np, dtype=torch.long)
    n_rows = eta_orig.shape[0]
    per_row = per_row_riesz_loss(loss, eta_orig, eta_pts, coefs, pt_to_row, n_rows)
    per_row.sum().backward()
    return eta_orig.grad.detach().numpy(), eta_pts.grad.detach().numpy()


def _analytic_grad(loss, eta_orig_np, eta_pts_np, coefs_np, pt_to_row_np):
    """Compute analytic dL/d eta_orig and dL/d eta_pts using the loss spec.

    The augmented loss is ``D · ψ(α) + C · φ'(α)``. Original rows are
    ``(D=1, C=0)`` so dL_i/dη_orig_i = aug_grad_eta(1, 0, eta_orig_i). Trace
    points are ``(D=0, C=-coef)`` so dL_i/dη_pts_j = aug_grad_eta(0, -coef_j,
    eta_pts_j) for the row j belongs to.
    """
    grad_orig = loss.aug_grad_eta(np.ones_like(eta_orig_np), np.zeros_like(eta_orig_np), eta_orig_np)
    grad_pts = loss.aug_grad_eta(np.zeros_like(eta_pts_np), -coefs_np, eta_pts_np)
    return grad_orig, grad_pts


@pytest.mark.parametrize(
    "loss",
    [
        SquaredLoss(),
        KLLoss(max_eta=10.0),
        BernoulliLoss(max_abs_eta=10.0),
        BoundedSquaredLoss(lo=0.1, hi=5.0, max_abs_eta=10.0),
    ],
    ids=["squared", "kl", "bernoulli", "bounded_squared"],
)
def test_autograd_matches_analytic_gradient(loss):
    eta_orig_np, eta_pts_np, coefs_np, pt_to_row_np = _build_problem(loss, seed=42)

    g_orig_t, g_pts_t = _torch_grad_per_row(
        loss, eta_orig_np, eta_pts_np, coefs_np, pt_to_row_np
    )
    g_orig_a, g_pts_a = _analytic_grad(
        loss, eta_orig_np, eta_pts_np, coefs_np, pt_to_row_np
    )

    np.testing.assert_allclose(g_orig_t, g_orig_a, atol=1e-8, rtol=1e-7)
    np.testing.assert_allclose(g_pts_t, g_pts_a, atol=1e-8, rtol=1e-7)


def test_squared_loss_per_row_value_matches_explicit_form():
    """Sanity-check that per_row_riesz_loss matches α² − 2 m(z, α) for squared."""
    loss = SquaredLoss()
    rng = np.random.default_rng(0)
    n_rows = 5
    eta_orig = rng.normal(size=n_rows)
    coefs = rng.normal(size=n_rows * 2)
    pt_to_row = np.repeat(np.arange(n_rows), 2)
    eta_pts = rng.normal(size=n_rows * 2)

    eta_orig_t = torch.as_tensor(eta_orig, dtype=torch.float64)
    eta_pts_t = torch.as_tensor(eta_pts, dtype=torch.float64)
    coefs_t = torch.as_tensor(coefs, dtype=torch.float64)
    p2r_t = torch.as_tensor(pt_to_row, dtype=torch.long)
    per_row = per_row_riesz_loss(loss, eta_orig_t, eta_pts_t, coefs_t, p2r_t, n_rows)
    per_row_np = per_row.numpy()

    # Direct computation: L_i = α(x_i)² − 2 Σ coef · α(point) (squared loss in α=η)
    expected = eta_orig**2
    for j, i in enumerate(pt_to_row):
        expected[i] -= 2.0 * coefs[j] * eta_pts[j]
    np.testing.assert_allclose(per_row_np, expected, atol=1e-12)
