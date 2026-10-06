"""ForestRieszRegressor — sklearn-compatible random-forest Riesz regressor.

Subclass of ``rieszreg.RieszEstimator`` that defaults the backend to
``ForestRieszBackend`` and surfaces forest-specific hyperparameters as
constructor args. Composes with ``GridSearchCV``, ``cross_val_predict``,
``Pipeline``.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from rieszreg import Estimand, Loss, RieszEstimator

from .backend import ForestRieszBackend


class ForestRieszRegressor(RieszEstimator):
    """Reference implementation of ForestRiesz (Chernozhukov, Newey,
    Quintas-Martínez & Syrgkanis, ICML 2022), kept for comparison with the
    published method. For general use prefer :class:`AugForestRieszRegressor`,
    which supports every estimand and every loss.

    Fits the representer α₀ with a generalized random forest (EconML's GRF)
    on the n original rows. Supports ``ATE``, ``ATT`` and ``TSM`` out of the
    box (other estimands need a custom ``riesz_feature_fns`` basis), only
    ``SquaredLoss``, and offers honest confidence intervals via
    :meth:`predict_interval`.

    Parameters
    ----------
    estimand : rieszreg.Estimand
        What to estimate, e.g. ``ATE(treatment="treated")``. Also names the
        treatment and covariate columns (default: every non-treatment column).
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
        Forest hyperparameters forwarded to EconML's ``BaseGRF``, with the
        defaults of the published ForestRiesz implementation (which is why
        they differ from ``AugForestRieszRegressor``'s sklearn-style ones). ``honest``
        defaults to False (cross-fitting works without honesty); enable it
        plus ``inference=True`` to use ``predict_interval``.
    l2 : float, default=0.0
        Ridge added to the per-leaf Jacobian: each leaf solves
        ``(mean J + l2 · I) θ = mean A``. For TSM this gives
        ``θ = 1 / (P̂(A = level | leaf) + l2)``. With ``l2 > 0`` the forest
        can grow leaves with no treated (or no control) rows, whose
        ``θ ≈ mean A / l2`` gives very large α̂ for new rows landing there.
        Leave it at 0 unless you have a reason to ridge the leaf solve.
    loss : rieszreg.Loss, default=SquaredLoss()
        Currently only ``SquaredLoss`` is supported.
    init : float or None
        Accepted for API parity with the other learners. Each leaf solves
        for α directly, so it has no effect on the forest.
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
        l2: float = 0.0,
        n_jobs: int = -1,
        verbose: int = 0,
        loss: Loss | None = None,
        init: float | None = None,
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
            verbose=self.verbose,
        )

    def diagnose(self, Z, y=None, **kwargs):
        """Base diagnostics plus forest extras: feature importances, mean
        leaf size and mean number of leaves per tree."""
        from .diagnostics import diagnose_forest
        return diagnose_forest(self, Z, y=y, **kwargs)

    # ---- inference passthroughs ----

    def predict_interval(
        self, Z, *, alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (lb, ub) arrays for α(Z) at confidence 1 - alpha.

        Requires ``honest=True`` and ``inference=True`` at fit. Locally
        constant only in v1; sieve case raises NotImplementedError.
        """
        feats = self._features(Z)
        return self.predictor_.predict_interval(feats, alpha=alpha)

    # ---- save/load ----

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
        if riesz_feature_fns != "auto":
            instance.riesz_feature_fns = riesz_feature_fns
        sieve = instance.riesz_feature_fns
        instance.predictor_.riesz_feature_fns = (
            default_riesz_features(instance.estimand) if sieve == "auto" else sieve
        )
        return instance
