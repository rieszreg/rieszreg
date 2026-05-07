"""riesztree: single-tree backend for the rieszreg meta-package.

A single decision tree fit by greedy splits on the augmented Bregman-Riesz
loss. Each leaf stores the closed-form per-leaf optimum
``α_ℓ* = -C_ℓ / D_ℓ`` (loss-aware: projected to the loss's α-domain), so
the same tree structure works for ``SquaredLoss``, ``KLLoss``,
``BernoulliLoss``, and ``BoundedSquaredLoss``.

Importing this module registers the predictor loader for
``rieszreg.RieszEstimator.load`` to round-trip ``"riesztree"`` predictors.
"""

from __future__ import annotations

from rieszreg import (
    ATE,
    ATT,
    AdditiveShift,
    BernoulliLoss,
    BoundedSquaredLoss,
    Estimand,
    KLLoss,
    LocalShift,
    Loss,
    SquaredLoss,
    TSM,
)

from .backend import RieszTreeBackend
from .diagnostics import TreeDiagnostics, diagnose_tree
from .estimator import RieszTreeRegressor
from .predictor import RieszTreePredictor
from .pruning import cost_complexity_prune, cost_complexity_pruning_path
from .splitter import make_leaf_solvers
from .tree import Node, feature_importance, max_depth, n_leaves

__all__ = [
    "ATE",
    "ATT",
    "AdditiveShift",
    "BernoulliLoss",
    "BoundedSquaredLoss",
    "Estimand",
    "KLLoss",
    "LocalShift",
    "Loss",
    "Node",
    "RieszTreeBackend",
    "RieszTreePredictor",
    "RieszTreeRegressor",
    "SquaredLoss",
    "TSM",
    "TreeDiagnostics",
    "cost_complexity_prune",
    "cost_complexity_pruning_path",
    "diagnose_tree",
    "feature_importance",
    "make_leaf_solvers",
    "max_depth",
    "n_leaves",
]
