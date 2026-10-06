"""AugForestRieszRegressor — sklearn wrapper for the augmentation-style forest.

Hyperparameters mirror :class:`sklearn.ensemble.RandomForestRegressor`
where the augmented Bregman-Riesz setting allows. Works on every estimand
without per-estimand configuration.
"""

from __future__ import annotations

from typing import Sequence

from rieszreg import Estimand, Loss, RieszEstimator

from .aug_backend import AugForestRieszBackend


class AugForestRieszRegressor(RieszEstimator):
    """Random-forest Riesz regression: the recommended forest learner.
    Works on every estimand and every built-in loss with no extra setup.
    (:class:`ForestRieszRegressor` is the published ForestRiesz method,
    kept for comparison.)

    An ensemble of single-tree Riesz regressors fit on the augmented dataset
    of evaluation points with weights ``(D_r, C_r)`` that ``Estimand.augment``
    produces. Estimand-agnostic — works on every built-in estimand and any
    custom ``Estimand`` with no per-estimand configuration.

    Parameters
    ----------
    estimand : rieszreg.Estimand
        What to estimate, e.g. ``ATE(treatment="treated")``. Also names the
        treatment and covariate columns (default: every non-treatment column).
    loss : rieszreg.Loss, default=None
        Resolves to ``SquaredLoss()`` if ``None``. All four built-in
        Bregman losses (``SquaredLoss``, ``KLLoss``, ``BernoulliLoss``,
        ``BoundedSquaredLoss``) are supported via riesztree's loss-aware
        splitter.
    n_estimators, max_depth, min_samples_split, min_samples_leaf,
    min_weight_fraction_leaf, max_features, max_leaf_nodes,
    min_impurity_decrease, ccp_alpha, bootstrap, max_samples, n_jobs,
    verbose, splitter, max_bins, categorical_features
        See :class:`AugForestRieszBackend`. Defaults match
        :class:`sklearn.ensemble.RandomForestRegressor` where the augmented
        Bregman-Riesz setting allows, except ``min_samples_leaf=50``: leaves
        of a few rows give very noisy α̂.
    init : float or None
        Accepted for API parity with the other learners. Leaves store the
        loss-optimal α directly, so it has no effect on the forest.
    random_state : int, default=0
        Seeds the bootstrap draws, per-split feature subsampling and the
        histogram bin subsample.
    """

    def __init__(
        self,
        estimand: Estimand,
        loss: Loss | None = None,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 50,
        min_weight_fraction_leaf: float = 0.0,
        max_features: int | float | str | None = 1.0,
        max_leaf_nodes: int | None = None,
        min_impurity_decrease: float = 0.0,
        ccp_alpha: float = 0.0,
        bootstrap: bool = True,
        max_samples: int | float | None = None,
        n_jobs: int | None = None,
        verbose: int = 0,
        splitter: str = "exact",
        max_bins: int = 255,
        categorical_features: Sequence[int] | None = None,
        init: float | None = None,
        random_state: int = 0,
    ):
        super().__init__(
            estimand=estimand,
            backend=None,
            loss=loss,
            init=init,
            random_state=random_state,
        )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.splitter = splitter
        self.max_bins = max_bins
        self.categorical_features = categorical_features

    def _resolved_backend(self) -> AugForestRieszBackend:
        cat = (
            tuple(int(i) for i in self.categorical_features)
            if self.categorical_features is not None
            else None
        )
        return AugForestRieszBackend(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            ccp_alpha=self.ccp_alpha,
            bootstrap=self.bootstrap,
            max_samples=self.max_samples,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            splitter=self.splitter,
            max_bins=self.max_bins,
            categorical_features=cat,
        )
