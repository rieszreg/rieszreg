"""AugForestRieszRegressor — sklearn wrapper for the augmentation-style forest.

Hyperparameters mirror :class:`sklearn.ensemble.RandomForestRegressor`
where the augmented Bregman-Riesz setting allows. Works on every estimand
without per-estimand configuration.
"""

from __future__ import annotations

from typing import Sequence

from rieszreg import Estimand, Loss, RieszEstimator, SquaredLoss

from .aug_backend import AugForestRieszBackend


class AugForestRieszRegressor(RieszEstimator):
    """Augmentation-style random-forest Riesz regression.

    An ensemble of single-tree Riesz regressors fit on the augmented dataset
    of evaluation points with weights ``(D_r, C_r)`` that ``Estimand.augment``
    produces. Estimand-agnostic — works on every built-in estimand and any
    custom ``Estimand`` with no per-estimand configuration.

    Parameters
    ----------
    estimand : rieszreg.Estimand
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
        Bregman-Riesz setting allows.
    init : float or None
        α-space initialization, threaded through to ``RieszEstimator``.
    random_state : int, default=0
    """

    def __init__(
        self,
        estimand: Estimand,
        loss: Loss | None = None,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
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

    def _resolved_loss(self) -> Loss:
        return self.loss if self.loss is not None else SquaredLoss()

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
            random_state=self.random_state,
            verbose=self.verbose,
            splitter=self.splitter,
            max_bins=self.max_bins,
            categorical_features=cat,
        )

    def _save_hyperparameters(self) -> dict:
        base = super()._save_hyperparameters()
        base.update(
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
            categorical_features=(
                list(int(i) for i in self.categorical_features)
                if self.categorical_features is not None
                else None
            ),
        )
        return base

    @classmethod
    def _construct_for_load(
        cls, *, estimand, loss, hyperparameters: dict
    ) -> "AugForestRieszRegressor":
        cat = hyperparameters.get("categorical_features")
        return cls(
            estimand=estimand,
            loss=loss,
            n_estimators=hyperparameters.get("n_estimators", 100),
            max_depth=hyperparameters.get("max_depth"),
            min_samples_split=hyperparameters.get("min_samples_split", 2),
            min_samples_leaf=hyperparameters.get("min_samples_leaf", 1),
            min_weight_fraction_leaf=hyperparameters.get(
                "min_weight_fraction_leaf", 0.0
            ),
            max_features=hyperparameters.get("max_features", 1.0),
            max_leaf_nodes=hyperparameters.get("max_leaf_nodes"),
            min_impurity_decrease=hyperparameters.get("min_impurity_decrease", 0.0),
            ccp_alpha=hyperparameters.get("ccp_alpha", 0.0),
            bootstrap=hyperparameters.get("bootstrap", True),
            max_samples=hyperparameters.get("max_samples"),
            n_jobs=hyperparameters.get("n_jobs"),
            verbose=hyperparameters.get("verbose", 0),
            splitter=hyperparameters.get("splitter", "exact"),
            max_bins=hyperparameters.get("max_bins", 255),
            categorical_features=tuple(int(i) for i in cat) if cat else None,
            init=hyperparameters.get("init"),
            random_state=hyperparameters.get("random_state", 0),
        )
