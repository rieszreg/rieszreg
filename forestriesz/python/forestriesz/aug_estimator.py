"""AugForestRieszRegressor — sklearn wrapper for the augmentation-style forest.

Sister to ``ForestRieszRegressor``. The two share GRF hyperparameters but
differ in which Backend Protocol they implement: ``ForestRieszRegressor`` uses
``MomentBackend.fit_rows`` (faster, requires sieve for built-in estimands,
supports ``predict_interval``); ``AugForestRieszRegressor`` uses
``Backend.fit_augmented`` (slower by ~k×, fully estimand-agnostic, no CIs in v1).
"""

from __future__ import annotations

from typing import Callable

from rieszreg import Estimand, LossSpec, RieszEstimator, SquaredLoss

from .aug_backend import AugForestRieszBackend


class AugForestRieszRegressor(RieszEstimator):
    """Augmentation-style random-forest Riesz regression.

    Trains on the M = k·n augmented dataset that ``rieszreg.build_augmented``
    produces. No sieve required — even with the constant basis the per-augmented-row
    Jacobian and moment vary across rows, so the forest can split usefully.

    Parameters
    ----------
    estimand : rieszreg.Estimand
    riesz_feature_fns : list of callables, optional
        Optional sieve. The default ``None`` uses the constant basis (which
        works here because augmented rows already vary in J / A).
    n_estimators, max_depth, min_samples_split, min_samples_leaf,
    max_features, max_samples, min_balancedness_tol, honest, inference,
    fit_intercept, subforest_size, l2, n_jobs, verbose
        Forwarded to EconML's ``BaseGRF``.
    loss : rieszreg.LossSpec, default=SquaredLoss()
    init : float, "m1", or None
    random_state : int, default=0
    """

    def __init__(
        self,
        estimand: Estimand,
        riesz_feature_fns: list[Callable] | None = None,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        min_weight_fraction_leaf: float = 0.0,
        min_var_fraction_leaf: float | None = None,
        max_features: object = "auto",
        min_impurity_decrease: float = 0.0,
        max_samples: float = 0.45,
        min_balancedness_tol: float = 0.45,
        honest: bool = False,
        inference: bool = False,
        fit_intercept: bool = True,
        subforest_size: int = 4,
        l2: float = 0.01,
        n_jobs: int = -1,
        verbose: int = 0,
        loss: LossSpec | None = None,
        init: float | str | None = None,
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
        self.riesz_feature_fns = riesz_feature_fns
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.min_var_fraction_leaf = min_var_fraction_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.max_samples = max_samples
        self.min_balancedness_tol = min_balancedness_tol
        self.honest = honest
        self.inference = inference
        self.fit_intercept = fit_intercept
        self.subforest_size = subforest_size
        self.l2 = l2
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _resolved_loss(self) -> LossSpec:
        return self.loss if self.loss is not None else SquaredLoss()

    def _resolved_backend(self) -> AugForestRieszBackend:
        return AugForestRieszBackend(
            riesz_feature_fns=self.riesz_feature_fns,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            min_var_fraction_leaf=self.min_var_fraction_leaf,
            max_features=self.max_features,
            min_impurity_decrease=self.min_impurity_decrease,
            max_samples=self.max_samples,
            min_balancedness_tol=self.min_balancedness_tol,
            honest=self.honest,
            inference=self.inference,
            fit_intercept=self.fit_intercept,
            subforest_size=self.subforest_size,
            l2=self.l2,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
        )

    def _save_hyperparameters(self) -> dict:
        base = super()._save_hyperparameters()
        base.update(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            min_var_fraction_leaf=self.min_var_fraction_leaf,
            max_features=self.max_features,
            min_impurity_decrease=self.min_impurity_decrease,
            max_samples=self.max_samples,
            min_balancedness_tol=self.min_balancedness_tol,
            honest=self.honest,
            inference=self.inference,
            fit_intercept=self.fit_intercept,
            subforest_size=self.subforest_size,
            l2=self.l2,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            has_sieve=self.riesz_feature_fns is not None,
        )
        return base

    @classmethod
    def _construct_for_load(
        cls, *, estimand, loss, hyperparameters: dict
    ) -> "AugForestRieszRegressor":
        kwargs = {
            k: hyperparameters[k]
            for k in (
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "min_var_fraction_leaf",
                "max_features",
                "min_impurity_decrease",
                "max_samples",
                "min_balancedness_tol",
                "honest",
                "inference",
                "fit_intercept",
                "subforest_size",
                "l2",
                "n_jobs",
                "verbose",
            )
            if k in hyperparameters
        }
        return cls(
            estimand=estimand,
            riesz_feature_fns=None,
            n_estimators=hyperparameters.get("n_estimators", 100),
            loss=loss,
            init=hyperparameters.get("init"),
            random_state=hyperparameters.get("random_state", 0),
            **kwargs,
        )
