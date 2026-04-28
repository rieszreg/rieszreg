from .estimands import ATE, ATT, TSM, AdditiveShift, StochasticIntervention
from .tracer import LinearForm, Tracer, trace

from .losses import KLLoss, LossSpec, SquaredLoss

__all__ = [
    "ATE",
    "ATT",
    "AdditiveShift",
    "CrossFitResult",
    "Diagnostics",
    "GeneralRieszBooster",
    "KLLoss",
    "LinearForm",
    "LossSpec",
    "RieszBooster",
    "SquaredLoss",
    "StochasticIntervention",
    "TSM",
    "Tracer",
    "build_augmented",
    "crossfit",
    "diagnose",
    "fit",
    "general_fit",
    "trace",
]


_LAZY = {
    "RieszBooster": ("engine", "RieszBooster"),
    "build_augmented": ("engine", "build_augmented"),
    "fit": ("engine", "fit"),
    "crossfit": ("crossfit", "crossfit"),
    "CrossFitResult": ("crossfit", "CrossFitResult"),
    "diagnose": ("diagnostics", "diagnose"),
    "Diagnostics": ("diagnostics", "Diagnostics"),
    "general_fit": ("general", "general_fit"),
    "GeneralRieszBooster": ("general", "GeneralRieszBooster"),
}


def __getattr__(name):
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        from importlib import import_module
        return getattr(import_module(f"{__name__}.{mod_name}"), attr)
    raise AttributeError(f"module 'rieszboost' has no attribute {name!r}")
