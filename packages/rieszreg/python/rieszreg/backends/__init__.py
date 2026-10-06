"""Backend protocol and predictor-loader registry.

Concrete backends are provided by implementation packages (rieszboost, krrr).
"""

from .base import (
    Backend,
    ColumnBackend,
    FitResult,
    HoldoutBackend,
    MomentBackend,
    Predictor,
    bind_columns,
    holdout_fraction,
    load_predictor,
    register_predictor_loader,
    resolve_column_positions,
)

__all__ = [
    "Backend",
    "ColumnBackend",
    "FitResult",
    "HoldoutBackend",
    "MomentBackend",
    "bind_columns",
    "holdout_fraction",
    "Predictor",
    "load_predictor",
    "register_predictor_loader",
    "resolve_column_positions",
]
