"""Re-exports from `rieszreg.losses`. Canonical home is rieszreg."""

from rieszreg.losses import (  # noqa: F401
    BernoulliLoss,
    BoundedSquaredLoss,
    KLLoss,
    Loss,
    SquaredLoss,
    loss_from_spec,
)

__all__ = [
    "BernoulliLoss",
    "BoundedSquaredLoss",
    "KLLoss",
    "Loss",
    "SquaredLoss",
    "loss_from_spec",
]
