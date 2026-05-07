"""sklearn scorer factory for Bregman-Riesz losses.

`RieszEstimator.score()` uses the canonical squared yardstick for cross-
estimator comparability (analog of R² for regressors). When a different
yardstick is wanted — e.g. KL on a density-ratio problem — pass
`scoring=riesz_scorer(loss=KLLoss())` to any sklearn CV utility.
"""

from __future__ import annotations

import numpy as np

from .estimator import _features_from_Z
from .losses import Loss, SquaredLoss


def riesz_scorer(loss: Loss | None = None):
    """Return an sklearn-compatible scorer (`(estimator, Z, y=None) -> float`).

    Parameters
    ----------
    loss : Loss or None, default=None
        Yardstick loss to evaluate on the held-out fold. If `None`, defaults
        to `SquaredLoss()` (matches `RieszEstimator.score`).

    Notes
    -----
    The fitted estimator's own link maps backend output η to α; the yardstick
    `loss` is then evaluated on that α with the held-out augmented (a, b)
    coefficients. The yardstick must accept the estimator's α: `SquaredLoss`
    has unrestricted α-domain, while `KLLoss` requires α > 0,
    `BernoulliLoss` requires α ∈ (0, 1), and `BoundedSquaredLoss(lo, hi)`
    requires α ∈ (lo, hi).
    """
    yardstick = loss if loss is not None else SquaredLoss()

    def _scorer(estimator, Z, y=None) -> float:
        if not hasattr(estimator, "predictor_"):
            raise RuntimeError(
                f"{type(estimator).__name__} is not fitted yet."
            )
        feats = _features_from_Z(Z, estimator.estimand)
        aug = estimator.estimand.augment(feats)
        eta = estimator.predictor_.predict_eta(aug.features)
        alpha_hat = estimator.loss_.link_to_alpha(eta)
        return -float(
            np.sum(yardstick.aug_loss_alpha(aug.is_original, aug.potential_deriv_coef, alpha_hat))
            / aug.n_rows
        )

    return _scorer
