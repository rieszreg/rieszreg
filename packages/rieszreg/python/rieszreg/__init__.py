"""rieszreg: shared abstractions for the Riesz-regression package family.

Learner packages (rieszboost, krrr, forestriesz, riesztree, riesznet)
depend on rieszreg for the estimands, Bregman-Riesz losses, augmentation
engine, Backend Protocols, diagnostics, and the sklearn-compatible
`RieszEstimator`. Most users import from a learner package, which
re-exports the user-facing names listed in `rieszreg.user_api`:

    from rieszboost import RieszBooster, ATE
    est = RieszBooster(estimand=ATE(treatment="treated"))
    est.fit(Z)                   # Z: treatment column + covariates
    alpha_hat = est.predict(Z)
"""

# Mirror sklearn's `sklearn/__init__.py`: when xgboost (rieszboost) and torch
# (riesznet) are loaded into the same process, macOS dyld can map two distinct
# libomp copies and abort. `setdefault` so any value the user already exported
# wins. See https://github.com/joblib/threadpoolctl/blob/master/multiple_openmp.md
# for background; the runtime threadpool-deadlock warning lives in `_omp.py`.
import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
_os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
del _os

from .augmentation import AugmentedDataset
from .backends import (
    Backend,
    FitResult,
    HoldoutBackend,
    MomentBackend,
    Predictor,
    load_predictor,
    register_predictor_loader,
)
from .diagnostics import Diagnostics, diagnose
from .estimands import (
    ATE,
    ATT,
    AdditiveShift,
    Estimand,
    FiniteEvalEstimand,
    LinearForm,
    LocalShift,
    OutcomeRegNormSq,
    TSM,
    Tracer,
    estimand_from_spec,
    trace,
)
from .estimator import RieszEstimator
from .losses import (
    BernoulliLoss,
    BoundedSquaredLoss,
    KLLoss,
    Loss,
    SquaredLoss,
    loss_from_spec,
)
from .scoring import riesz_scorer

__all__ = [
    "ATE",
    "ATT",
    "AdditiveShift",
    "AugmentedDataset",
    "Backend",
    "BernoulliLoss",
    "BoundedSquaredLoss",
    "Diagnostics",
    "Estimand",
    "FiniteEvalEstimand",
    "FitResult",
    "HoldoutBackend",
    "KLLoss",
    "LinearForm",
    "LocalShift",
    "Loss",
    "MomentBackend",
    "OutcomeRegNormSq",
    "Predictor",
    "RieszEstimator",
    "SquaredLoss",
    "TSM",
    "Tracer",
    "diagnose",
    "estimand_from_spec",
    "load_predictor",
    "loss_from_spec",
    "register_predictor_loader",
    "riesz_scorer",
    "trace",
]
