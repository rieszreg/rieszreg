"""ForestDiagnostics — extends rieszreg.Diagnostics with forest-specific extras.

Reports per-feature importance and mean leaf size in addition to the base
diagnostics (RMS, quantiles, extreme-α̂ warnings, held-out Riesz loss).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rieszreg import Diagnostics, diagnose


@dataclass
class ForestDiagnostics(Diagnostics):
    feature_importances: np.ndarray = field(default_factory=lambda: np.zeros(0))
    mean_leaf_size: float = float("nan")
    n_leaves_mean: float = float("nan")


def diagnose_forest(estimator, X, **kwargs) -> ForestDiagnostics:
    """Run base diagnostics and tack on forest-specific summaries.

    Falls back gracefully when extras can't be computed (e.g. predictor was
    loaded without OOB info or feature_importances unavailable).
    """
    base = diagnose(estimator=estimator, X=X, **kwargs)

    forest = getattr(estimator.predictor_, "forest", None)
    importances = np.zeros(0)
    mean_leaf_size = float("nan")
    n_leaves_mean = float("nan")
    if forest is not None:
        try:
            importances = np.asarray(forest.feature_importances())
        except Exception:
            pass
        trees = list(getattr(forest, "estimators_", []) or [])
        leaf_counts = []
        sample_counts = []
        for t in trees:
            try:
                leaf_counts.append(int(np.sum(t.tree_.children_left == -1)))
                sample_counts.append(int(t.tree_.n_node_samples[0]))
            except Exception:
                continue
        if leaf_counts:
            n_leaves_mean = float(np.mean(leaf_counts))
            total_leaves = sum(leaf_counts)
            if total_leaves > 0:
                mean_leaf_size = sum(sample_counts) / float(total_leaves)

    return ForestDiagnostics(
        n=base.n,
        rms=base.rms,
        mean=base.mean,
        min=base.min,
        max=base.max,
        abs_quantiles=base.abs_quantiles,
        n_extreme=base.n_extreme,
        extreme_fraction=base.extreme_fraction,
        extreme_threshold=base.extreme_threshold,
        riesz_loss=base.riesz_loss,
        warnings=base.warnings,
        feature_importances=importances,
        mean_leaf_size=mean_leaf_size,
        n_leaves_mean=n_leaves_mean,
    )
