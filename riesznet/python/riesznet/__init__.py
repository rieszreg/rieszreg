"""riesznet: neural-network backend for the rieszreg meta-package.

Implements the moment-style ``MomentBackend.fit_rows`` entry point with a
PyTorch training loop. Trains the Riesz representer α(x) of a linear
functional θ(P) = E[m(Z, g₀)] by minimizing the per-row Bregman-Riesz loss

    L_i = ψ(α(x_i)) − Σ_j coef_j · φ'(α(point_j))

where ``(coef_j, point_j)`` come from ``rieszreg.trace(estimand, row_i)``.

Importing this module registers the predictor loader for
``rieszreg.RieszEstimator.load`` to round-trip ``"riesznet"`` predictors.

    from riesznet import RieszNet
    from rieszreg import ATE

    est = RieszNet(estimand=ATE(), epochs=200, learning_rate=1e-3)
    est.fit(df)
    alpha_hat = est.predict(df)
"""

from __future__ import annotations

from .backend import TorchBackend, TorchPredictor
from .estimator import RieszNet
from .modules import build_adam, build_mlp

# Re-export the rieszreg primitives users will reach for here.
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
    LossSpec,
    SquaredLoss,
    StochasticIntervention,
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
    "LossSpec",
    "SquaredLoss",
    "StochasticIntervention",
    "TSM",
    "Tracer",
    "trace",
]
