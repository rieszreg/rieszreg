"""ForestRieszRegressor — sklearn-compatible random-forest Riesz regressor.

Subclass of ``rieszreg.RieszEstimator`` that defaults the backend to
``ForestRieszBackend`` and surfaces forest-specific hyperparameters as
constructor args. Composes with ``GridSearchCV``, ``cross_val_predict``,
``Pipeline``.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from rieszreg import Estimand, LossSpec, RieszEstimator, SquaredLoss

from .backend import ForestRieszBackend


class ForestRieszRegressor(RieszEstimator):
    """Random-forest Riesz regression for the representer α₀ of a linear
    functional θ(P) = E[m(Z, g₀)].

    Reuses rieszreg's estimand machinery and Bregman-loss framework; swaps in
    a GRF-based moment backend for the actual fit.

    Parameters
    ----------
    estimand : rieszreg.Estimand
        Carries ``feature_keys``, ``extra_keys``, and m(z, alpha).
    riesz_feature_fns : list of callables, "auto", or None
        Sieve basis ``[φ_1, …, φ_p]`` for the locally linear flavor. Each
        callable takes a feature matrix ``(n, n_features)`` (columns ordered
        by ``estimand.feature_keys``) and returns a ``(n,)`` array. The
        default ``"auto"`` resolves to ``default_riesz_features(estimand)``
        for built-in estimands (treatment indicators for ATE/ATT/TSM); custom
        estimands fall back to a constant basis. Pass ``None`` to force the
        constant basis (the degeneracy check will likely raise — built-in
        estimands have row-constant moments under a constant basis).
    split_feature_indices : sequence of int, optional
        Which feature columns the forest splits on. None auto-selects: with a
        treatment-indexed sieve the splitter sees covariates only; otherwise
        all features.
    n_estimators, max_depth, min_samples_split, min_samples_leaf,
    min_weight_fraction_leaf, min_var_fraction_leaf, max_features,
    min_impurity_decrease, max_samples, min_balancedness_tol, honest,
    inference, fit_intercept, subforest_size, n_jobs, verbose
        Forest hyperparameters forwarded to EconML's ``BaseGRF``. ``honest``
        defaults to False (cross-fitting works without honesty); enable it
        plus ``inference=True`` to use ``predict_interval``.
    l2 : float
        Ridge added to the per-leaf Jacobian for numerical stability.
    loss : rieszreg.LossSpec, default=SquaredLoss()
        Currently only ``SquaredLoss`` is supported.
    init : float, "m1", or None
        α-space initialization. None defers to ``loss.default_init_alpha()``.
    random_state : int, default=0
    """

    def __init__(
        self,
        estimand: Estimand,
        riesz_feature_fns: list[Callable] | str | None = "auto",
        split_feature_indices: Sequence[int] | None = None,
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
            backend=None,                # built lazily in _resolved_backend
            loss=loss,
            init=init,
            random_state=random_state,
        )
        self.n_estimators = n_estimators
        self.riesz_feature_fns = riesz_feature_fns
        self.split_feature_indices = split_feature_indices
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

    # ---- defaults / backend construction ----

    def _resolved_loss(self) -> LossSpec:
        return self.loss if self.loss is not None else SquaredLoss()

    def _resolved_backend(self) -> ForestRieszBackend:
        return ForestRieszBackend(
            riesz_feature_fns=self.riesz_feature_fns,
            split_feature_indices=self.split_feature_indices,
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

    # ---- inference passthroughs ----

    def predict_interval(
        self, X, *, alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (lb, ub) arrays for α(X) at confidence 1 - alpha.

        Requires ``honest=True`` and ``inference=True`` at fit. Locally
        constant only in v1; sieve case raises NotImplementedError.
        """
        if not hasattr(self, "predictor_"):
            raise RuntimeError(
                f"{type(self).__name__} is not fitted yet. Call .fit() first."
            )
        from .estimator import ForestRieszRegressor  # noqa: F401 (self-ref ok)
        from rieszreg.estimator import _features_from_rows, _rows_from_X

        rows = _rows_from_X(X, self.estimand)
        feats = _features_from_rows(rows, self.estimand)
        return self.predictor_.predict_interval(feats, alpha=alpha)

    # ---- save/load ----

    def _save_hyperparameters(self) -> dict:
        base = super()._save_hyperparameters()
        base.update(
            split_feature_indices=(
                list(self.split_feature_indices)
                if self.split_feature_indices is not None
                else None
            ),
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
    ) -> "ForestRieszRegressor":
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
        split_idx = hyperparameters.get("split_feature_indices")
        return cls(
            estimand=estimand,
            riesz_feature_fns=None,  # patched by caller for sieve estimators
            split_feature_indices=tuple(split_idx) if split_idx else None,
            n_estimators=hyperparameters.get("n_estimators", 100),
            loss=loss,
            init=hyperparameters.get("init"),
            random_state=hyperparameters.get("random_state", 0),
            **kwargs,
        )

    @classmethod
    def load(
        cls,
        path,
        *,
        estimand=None,
        riesz_feature_fns: list[Callable] | str | None = "auto",
    ) -> "ForestRieszRegressor":
        """Load a fitted ForestRieszRegressor.

        Defaults to ``riesz_feature_fns="auto"`` which re-resolves the sieve
        from the estimand metadata for built-in estimands — that's enough to
        round-trip every save produced by a default-args fit. For custom
        sieves you supplied at fit time, repass the same list here so the
        predictor can evaluate the basis (callables aren't pickled).
        """
        from .feature_fns import default_riesz_features

        instance = super().load(path, estimand=estimand)
        sieve = riesz_feature_fns
        if sieve == "auto":
            sieve = default_riesz_features(instance.estimand)
        if sieve is not None:
            instance.riesz_feature_fns = sieve
            instance.predictor_.riesz_feature_fns = sieve
        return instance
