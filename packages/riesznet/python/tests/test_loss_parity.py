"""Torch per-row loss vs the numpy ``Loss`` spec for all four Bregman losses.

For random augmented rows ``(η, D, C, origin)``, the torch per-row Riesz loss
must match ``Loss.aug_loss_alpha`` summed per origin row (values) and
``Loss.aug_grad_eta`` (autograd gradients). If both pass, the neural backend
minimizes exactly the objective the augmentation-style backends use.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rieszreg import BernoulliLoss, BoundedSquaredLoss, KLLoss, SquaredLoss

from riesznet.losses_torch import TorchRieszLoss

LOSSES = [
    SquaredLoss(),
    KLLoss(max_eta=10.0),
    BernoulliLoss(max_abs_eta=10.0),
    BoundedSquaredLoss(lo=0.1, hi=5.0, max_abs_eta=10.0),
]
IDS = ["squared", "kl", "bernoulli", "bounded_squared"]


def _problem(loss, n_rows=8, k_per_row=3, seed=0):
    rng = np.random.default_rng(seed)
    n = n_rows * k_per_row
    eta = rng.normal(0.0, 0.5, size=n)
    origin = np.repeat(np.arange(n_rows), k_per_row)
    D = np.tile([1.0] + [0.0] * (k_per_row - 1), n_rows)
    if isinstance(loss, (KLLoss, BernoulliLoss)):
        C = -rng.uniform(0.0, 1.0, size=n)
    else:
        C = rng.normal(size=n)
    return eta, D, C, origin, n_rows


@pytest.mark.parametrize("loss", LOSSES, ids=IDS)
def test_torch_loss_matches_numpy_values_and_gradients(loss):
    eta, D, C, origin, n_rows = _problem(loss, seed=42)
    eta_t = torch.tensor(eta, dtype=torch.float64, requires_grad=True)
    per_row = TorchRieszLoss(loss).per_row(
        eta_t,
        torch.as_tensor(D),
        torch.as_tensor(C),
        torch.as_tensor(origin),
        n_rows,
    )
    per_row.sum().backward()

    expected = np.bincount(
        origin, loss.aug_loss_alpha(D, C, loss.link_to_alpha(eta)), minlength=n_rows
    )
    np.testing.assert_allclose(per_row.detach().numpy(), expected, atol=1e-8, rtol=1e-7)
    np.testing.assert_allclose(
        eta_t.grad.numpy(), loss.aug_grad_eta(D, C, eta), atol=1e-8, rtol=1e-6
    )
