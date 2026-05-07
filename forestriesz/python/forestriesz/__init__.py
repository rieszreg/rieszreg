"""forestriesz: random-forest backend for the rieszreg meta-package.

Implements the locally constant and locally linear ForestRiesz estimators of
Chernozhukov, Newey, Quintas-Martínez, Syrgkanis (ICML 2022) on top of
EconML's GRF infrastructure, plugged into rieszreg via the ``MomentBackend``
entry point so each tree fits on the n original rows (no augmentation
blow-up).

Importing this module registers the predictor loader for
``rieszreg.RieszEstimator.load`` to round-trip ``"forestriesz"`` predictors.
"""

from __future__ import annotations

# Re-export the rieszreg primitives users will reach for here.
from rieszreg import (
    ATE,
    ATT,
    AdditiveShift,
    Estimand,
    LocalShift,
    Loss,
    SquaredLoss,
    TSM,
)

from .aug_backend import AugForestRieszBackend
from .aug_estimator import AugForestRieszRegressor
from .aug_predictor import AugForestPredictor
from .backend import ForestRieszBackend
from .diagnostics import ForestDiagnostics, diagnose_forest
from .estimator import ForestRieszRegressor
from .feature_fns import default_riesz_features, default_split_feature_indices
from .predictor import ForestPredictor

__all__ = [
    "ATE",
    "ATT",
    "AdditiveShift",
    "AugForestPredictor",
    "AugForestRieszBackend",
    "AugForestRieszRegressor",
    "Estimand",
    "ForestDiagnostics",
    "ForestPredictor",
    "ForestRieszBackend",
    "ForestRieszRegressor",
    "LocalShift",
    "Loss",
    "SquaredLoss",
    "TSM",
    "default_riesz_features",
    "default_split_feature_indices",
    "diagnose_forest",
]
