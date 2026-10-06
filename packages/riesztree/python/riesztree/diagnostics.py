"""TreeDiagnostics — extends rieszreg.Diagnostics with tree-specific extras.

Adds ``n_leaves``, ``max_depth_actual``, ``mean_leaf_size``, and
``feature_importances`` (per-feature normalised split-gain).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rieszreg import Diagnostics, diagnose

from .tree import feature_importance, max_depth, n_leaves


@dataclass
class TreeDiagnostics(Diagnostics):
    n_leaves: int = 0
    max_depth_actual: int = 0
    mean_leaf_size: float = float("nan")
    feature_importances: np.ndarray = field(default_factory=lambda: np.zeros(0))


def diagnose_tree(estimator, Z, **kwargs) -> TreeDiagnostics:
    """Run base diagnostics and tack on tree-specific summaries."""
    base = diagnose(estimator=estimator, Z=Z, **kwargs)

    tree = getattr(estimator.predictor_, "tree", None)
    nl = 0
    md = 0
    mls = float("nan")
    fi = np.zeros(0)
    if tree is not None:
        nl = n_leaves(tree)
        md = max_depth(tree)
        # mean leaf size = mean of leaf.n_orig.
        sizes = []
        def _walk(n):
            if n.is_leaf:
                sizes.append(n.n_orig)
            else:
                _walk(n.left)
                _walk(n.right)
        _walk(tree)
        if sizes:
            mls = float(np.mean(sizes))
        fi = feature_importance(tree, len(estimator.estimand_.feature_keys))

    return TreeDiagnostics(
        **vars(base),
        n_leaves=nl,
        max_depth_actual=md,
        mean_leaf_size=mls,
        feature_importances=fi,
    )
