"""Top-level module / optimizer factories used by the convenience class.

These are top-level callables (not closures) so ``functools.partial`` over them
round-trips through ``TorchPredictor.save`` and ``.load`` via the qualname
metadata path. Users who want a custom architecture should define their own
top-level factory in their own module and pass it to ``TorchBackend``.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "elu": nn.ELU,
    "silu": nn.SiLU,
    "leaky_relu": nn.LeakyReLU,
}


def build_mlp(
    input_dim: int,
    *,
    hidden_sizes: tuple[int, ...] = (64, 64),
    activation: str = "relu",
    dropout: float = 0.0,
) -> nn.Module:
    """Construct an MLP that maps ``(batch, input_dim)`` to ``(batch, 1)`` (η).

    Parameters
    ----------
    input_dim : int
        Number of input features. Supplied by ``TorchBackend`` at fit time.
    hidden_sizes : tuple of int, default (64, 64)
        Hidden-layer widths. Empty tuple means a linear model.
    activation : {"relu", "tanh", "gelu", "elu", "silu", "leaky_relu"}, default "relu"
        Nonlinearity between hidden layers.
    dropout : float, default 0.0
        Dropout probability applied after each activation. ``0`` disables.
    """
    if activation not in _ACTIVATIONS:
        raise ValueError(
            f"Unknown activation {activation!r}. Choose from {sorted(_ACTIVATIONS)} "
            "or pass a custom module_factory to TorchBackend."
        )
    act_cls = _ACTIVATIONS[activation]

    layers: list[nn.Module] = []
    prev = int(input_dim)
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, int(h)))
        layers.append(act_cls())
        if dropout > 0:
            layers.append(nn.Dropout(float(dropout)))
        prev = int(h)
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def build_adam(
    params: Iterable[torch.nn.Parameter],
    *,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """Adam optimizer with the given LR and weight decay."""
    return torch.optim.Adam(list(params), lr=float(lr), weight_decay=float(weight_decay))
