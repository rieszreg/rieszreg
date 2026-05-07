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
    cls_name = spec["type"]
    args = spec.get("args", {})
    if cls_name == "SquaredLoss":
        return SquaredLoss(**args)
    if cls_name == "KLLoss":
        return KLLoss(**args)
    if cls_name == "BernoulliLoss":
        return BernoulliLoss(**args)
    if cls_name == "BoundedSquaredLoss":
        return BoundedSquaredLoss(**args)
    raise ValueError(f"Unknown loss spec type: {cls_name!r}")
