"""rieszboost: gradient boosting for Riesz representers.

Public API: a sklearn-compatible `RieszBooster` estimator with swappable
estimand / loss / backend objects baked in at construction.

    from rieszboost import RieszBooster, ATE, SquaredLoss
    booster = RieszBooster(estimand=ATE(), n_estimators=200, learning_rate=0.05)
    booster.fit(Z)
    alpha_hat = booster.predict(Z)
"""

from .augmentation import AugmentedDataset
from .diagnostics import Diagnostics, diagnose
from .estimand import (
    ATE,
    ATT,
    AdditiveShift,
    Estimand,
    LocalShift,
    TSM,
)
from .losses import BernoulliLoss, BoundedSquaredLoss, KLLoss, Loss, SquaredLoss
from .tracer import LinearForm, Tracer, trace

__all__ = [
    "ATE",
    "ATT",
    "AdditiveShift",
    "AugmentedDataset",
    "BernoulliLoss",
    "BoundedSquaredLoss",
    "Diagnostics",
    "Estimand",
    "KLLoss",
    "LinearForm",
    "LocalShift",
    "Loss",
    "RieszBooster",
    "SklearnBackend",
    "SquaredLoss",
    "TSM",
    "Tracer",
    "XGBoostBackend",
    "diagnose",
    "trace",
]


_LAZY = {
    "RieszBooster": ("estimator", "RieszBooster"),
    "XGBoostBackend": ("backends", "XGBoostBackend"),
    "SklearnBackend": ("backends", "SklearnBackend"),
}


def __getattr__(name):
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        from importlib import import_module
        return getattr(import_module(f"{__name__}.{mod_name}"), attr)
    raise AttributeError(f"module 'rieszboost' has no attribute {name!r}")
