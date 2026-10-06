"""Backend protocol and predictor-loader registry.

Concrete backends are provided by implementation packages (rieszboost, krrr).
"""

from .base import (
    Backend,
    FitResult,
    HoldoutBackend,
    MomentBackend,
    Predictor,
    holdout_fraction,
    load_predictor,
    register_predictor_loader,
)

__all__ = [
    "Backend",
    "FitResult",
    "HoldoutBackend",
    "MomentBackend",
    "holdout_fraction",
    "Predictor",
    "load_predictor",
    "register_predictor_loader",
]
