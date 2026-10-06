"""Bregman-Riesz losses."""

from .base import Loss
from .bernoulli import BernoulliLoss
from .bounded_squared import BoundedSquaredLoss
from .kl import KLLoss
from .squared import SquaredLoss

__all__ = [
    "BernoulliLoss",
    "BoundedSquaredLoss",
    "KLLoss",
    "Loss",
    "SquaredLoss",
    "loss_from_spec",
]


def loss_from_spec(spec: dict) -> Loss:
    """Reconstruct a Loss from its `to_spec()` dict."""
    classes = {c.__name__: c for c in (SquaredLoss, KLLoss, BernoulliLoss, BoundedSquaredLoss)}
    if spec["type"] not in classes:
        raise ValueError(f"Unknown loss spec type: {spec['type']!r}")
    return classes[spec["type"]](**spec.get("args", {}))
