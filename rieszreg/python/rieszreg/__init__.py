"""rieszreg: shared abstractions for the Riesz-regression package family.

Implementation packages (rieszboost, krrr, ...) depend on rieszreg for the
estimand machinery, Bregman-Riesz losses, augmentation engine, Backend
Protocol, base diagnostics, and the sklearn-compatible `RieszEstimator`
orchestrator.

    from rieszreg import RieszEstimator, ATE, SquaredLoss
    from rieszboost.backends import XGBoostBackend
    est = RieszEstimator(estimand=ATE(), loss=SquaredLoss(), backend=XGBoostBackend())
    est.fit(Z)
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

from .augmentation import (
    AugmentedDataset,
    aug_grad_eta,
    aug_hess_eta,
    aug_loss_alpha,
    aug_loss_eta,
    build_augmented,
)
from .backends import (
    Backend,
    FitResult,
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
    LossSpec,
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
    "aug_grad_eta",
    "aug_hess_eta",
    "aug_loss_alpha",
    "aug_loss_eta",
    "BernoulliLoss",
    "BoundedSquaredLoss",
    "Diagnostics",
    "Estimand",
    "FiniteEvalEstimand",
    "FitResult",
    "KLLoss",
    "LinearForm",
    "LocalShift",
    "Loss",
    "LossSpec",
    "MomentBackend",
    "Predictor",
    "RieszEstimator",
    "SquaredLoss",
    "TSM",
    "Tracer",
    "build_augmented",
    "diagnose",
    "estimand_from_spec",
    "load_predictor",
    "loss_from_spec",
    "register_predictor_loader",
    "riesz_scorer",
    "trace",
]
