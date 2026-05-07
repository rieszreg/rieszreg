"""riesznet: neural-network backend for the rieszreg meta-package.

Implements the moment-style ``MomentBackend.fit_rows`` entry point with a
PyTorch training loop. Trains the Riesz representer α(x) of a linear
functional θ(P) = E[m(Z, g₀)] by minimizing the per-row Bregman-Riesz loss

    L_i = ψ(α(x_i)) − Σ_j coef_j · φ'(α(point_j))

where ``(coef_j, point_j)`` come from ``rieszreg.trace(estimand, row_i)``.

Importing ``riesznet.backend`` (which happens lazily on first attribute
access here, e.g. ``riesznet.RieszNet``) registers the predictor loader
for ``rieszreg.RieszEstimator.load`` to round-trip ``"riesznet"`` predictors.

    from riesznet import RieszNet
    from rieszreg import ATE

    est = RieszNet(estimand=ATE(), epochs=200, learning_rate=1e-3)
    est.fit(df)
    alpha_hat = est.predict(df)
"""

from __future__ import annotations

# Re-export the rieszreg primitives users will reach for here. (rieszreg
# does not pull torch on import.)
from rieszreg import (
    ATE,
    ATT,
    AdditiveShift,
    BernoulliLoss,
    BoundedSquaredLoss,
    Diagnostics,
    Estimand,
    KLLoss,
    LinearForm,
    LocalShift,
    Loss,
    SquaredLoss,
    TSM,
    Tracer,
    trace,
)

__all__ = [
    # Local
    "RieszNet",
    "TorchBackend",
    "TorchPredictor",
    "build_mlp",
    "build_adam",
    # Re-exports from rieszreg
    "ATE",
    "ATT",
    "AdditiveShift",
    "BernoulliLoss",
    "BoundedSquaredLoss",
    "Diagnostics",
    "Estimand",
    "KLLoss",
    "LinearForm",
    "LocalShift",
    "Loss",
    "SquaredLoss",
    "TSM",
    "Tracer",
    "trace",
]


# Defer torch import until a torch-using symbol is actually accessed. Mirrors
# rieszboost's __init__.py. Two consequences:
#   1. `import riesznet` alone does not load torch / libomp, so users who
#      `import riesznet` next to `import rieszboost` for symbol access only
#      do not trigger the multi-libomp-in-one-process condition.
#   2. The "riesznet" predictor loader registers on first access of any
#      lazy symbol (which imports `riesznet.backend`); a fresh `import
#      riesznet` is no longer enough to register the loader.
_LAZY = {
    "RieszNet": ("estimator", "RieszNet"),
    "TorchBackend": ("backend", "TorchBackend"),
    "TorchPredictor": ("backend", "TorchPredictor"),
    "build_mlp": ("modules", "build_mlp"),
    "build_adam": ("modules", "build_adam"),
}


def __getattr__(name):
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        from importlib import import_module
        return getattr(import_module(f"{__name__}.{mod_name}"), attr)
    raise AttributeError(f"module 'riesznet' has no attribute {name!r}")
