from .estimands import ATE, TSM, AdditiveShift
from .tracer import LinearForm, Tracer, trace

__all__ = [
    "ATE",
    "AdditiveShift",
    "LinearForm",
    "RieszBooster",
    "TSM",
    "Tracer",
    "build_augmented",
    "fit",
    "trace",
]


def __getattr__(name):
    if name in {"RieszBooster", "build_augmented", "fit"}:
        from . import engine
        return getattr(engine, name)
    raise AttributeError(f"module 'rieszboost' has no attribute {name!r}")
