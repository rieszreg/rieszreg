"""ForestRieszBackend — implements `rieszreg.MomentBackend`.

Consumes the original-row feature matrix + the estimand (the moment-style
entry point), computes per-row moments from the augmented data, packs them
as a linear-moment problem for EconML's ``BaseGRF``, and returns a
``FitResult`` whose predictor is a ``ForestPredictor``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from rieszreg import AugmentedDataset, Estimand, FitResult, Loss, SquaredLoss

from ._grf import _RieszGRF
from .feature_fns import default_riesz_features, default_split_feature_indices
from .predictor import ForestPredictor


def _eval_phi(features: np.ndarray, phi_fns: Sequence[Callable]) -> np.ndarray:
    """Stack vectorized basis evaluations into an (n, p) matrix."""
    return np.column_stack([np.asarray(fn(features), dtype=float) for fn in phi_fns])


def _per_row_moments(aug: AugmentedDataset, phi_fns: Sequence[Callable]) -> np.ndarray:
    """``A[i, j] = m(W_i; φ_j) = Σ_k coef_k · φ_j(point_k)``.

    Each augmented row carries ``C_r = −Σ coef`` at its point, so the moment
    is ``−Σ_{r: origin_r = i} C_r · φ_j(z_r)``, one ``bincount`` per basis.
    """
    phi_aug = _eval_phi(aug.features, phi_fns)
    w = -aug.potential_deriv_coef[:, None] * phi_aug
    return np.column_stack([
        np.bincount(aug.origin_index, weights=w[:, j], minlength=aug.n_rows)
        for j in range(phi_aug.shape[1])
    ])


@dataclass
class ForestRieszBackend:
    """Random-forest Riesz regression backend.

    Wraps EconML's ``BaseGRF`` with the linear-moment criterion. Implements
    ``MomentBackend.fit_rows``: the forest is grown on the n original rows,
    with per-row moments computed from the augmented data.

    Parameters
    ----------
    riesz_feature_fns
        Basis ``φ_1, …, φ_p`` for the locally linear sieve (each callable
        takes a feature matrix ``(n, n_features)`` and returns ``(n,)``). The
        default ``"auto"`` resolves to ``default_riesz_features(estimand)``
        for built-in estimands (treatment indicators for ATE/ATT/TSM); custom
        estimands fall back to a constant basis. Pass an explicit list to
        override; pass ``None`` to force the constant basis (rarely useful —
        all built-in estimands give a row-constant moment in that case and
        the degeneracy check will raise).
    split_feature_indices
        Which feature columns the forest splits on. When ``None``, a default
        is chosen from the estimand and sieve (covariates only when a
        treatment-indexed sieve is supplied; otherwise all features).
    n_estimators, max_depth, min_samples_split, min_samples_leaf,
    min_weight_fraction_leaf, min_var_fraction_leaf, max_features,
    min_impurity_decrease, max_samples, min_balancedness_tol, honest,
    inference, fit_intercept, subforest_size, n_jobs, random_state, verbose
        Forwarded to ``econml.grf._base_grf.BaseGRF``. ``honest`` defaults to
        False because cross-fitting (``cross_val_predict``) does not require
        honesty; flip to True when you want ``predict_interval``.
    l2
        Ridge added to the per-leaf Jacobian: each leaf solves
        ``(mean J + l2 · I) θ = mean A``. Default 0; see
        ``ForestRieszRegressor`` for why ``l2 > 0`` can hurt.
    """

    riesz_feature_fns: list[Callable] | str | None = "auto"
    split_feature_indices: Sequence[int] | None = None
    n_estimators: int = 100
    max_depth: int | None = None
    min_samples_split: int = 10
    min_samples_leaf: int = 5
    min_weight_fraction_leaf: float = 0.0
    min_var_fraction_leaf: float | None = None
    max_features: object = "auto"
    min_impurity_decrease: float = 0.0
    max_samples: float = 0.45
    min_balancedness_tol: float = 0.45
    honest: bool = False
    inference: bool = False
    fit_intercept: bool = True
    subforest_size: int = 4
    l2: float = 0.0
    n_jobs: int = -1
    verbose: int = 0

    def fit_rows(
        self,
        X_train: np.ndarray,
        X_valid: np.ndarray | None,
        estimand: Estimand,
        loss: Loss,
        *,
        aug_train: AugmentedDataset,
        aug_valid: AugmentedDataset | None,
        base_score: float,
        random_state: int,
    ) -> FitResult:
        if not isinstance(loss, SquaredLoss):
            raise NotImplementedError(
                f"ForestRieszBackend (moment-style) currently supports "
                f"SquaredLoss only (got {type(loss).__name__}). For non-"
                "quadratic Bregman losses, use AugForestRieszRegressor "
                "instead — it ships a per-leaf Newton iteration on the "
                "augmented loss that handles KLLoss, BernoulliLoss, and "
                "BoundedSquaredLoss. The moment-style equivalent needs "
                "different per-leaf gradients than the loss API exposes; "
                "planned for v3."
            )
        del base_score  # each leaf solves for α directly

        # Sieve: "auto" => default_riesz_features(estimand) when one exists;
        # None (or no default) => constant basis.
        sieve = (
            default_riesz_features(estimand)
            if self.riesz_feature_fns == "auto"
            else self.riesz_feature_fns
        )
        phi_fns = sieve if sieve else [lambda f: np.ones(len(f))]
        p = len(phi_fns)
        n_train = X_train.shape[0]

        # Per-row basis values φ(W_i) and moments A[i, j] = m(W_i; φ_j).
        phi_W = _eval_phi(X_train, phi_fns)
        A = _per_row_moments(aug_train, phi_fns)

        # Pack T = [vec(J) | A] per row, with J = φφ' (symmetric, so flat
        # order is immaterial). y is a dummy scalar zero column — EconML's
        # LinearMomentGRFCriterion requires scalar y.
        JJ = np.einsum("ij,ik->ijk", phi_W, phi_W).reshape(n_train, p * p)
        T_pack = np.ascontiguousarray(np.column_stack([JJ, A]))
        y_pack = np.zeros((n_train, 1), dtype=float)

        # Degeneracy: for built-in estimands the per-row moments under a
        # constant basis don't depend on W, so both A and J are identical
        # across rows and splits learn nothing. The natural fix is the sieve.
        if A.size > 0:
            if np.allclose(JJ - JJ[0:1], 0.0, atol=1e-12) and np.allclose(A - A[0:1], 0.0, atol=1e-12):
                if default_riesz_features(estimand) is None:
                    raise ValueError(
                        f"ForestRieszRegressor (the reference ForestRiesz "
                        f"implementation) can't fit {estimand.name} without a "
                        "basis of treatment features. Use "
                        "AugForestRieszRegressor, which works for every "
                        "estimand with no extra setup, or pass your own "
                        "riesz_feature_fns=[...] list."
                    )
                raise ValueError(
                    f"ForestRieszRegressor can't learn α for {estimand.name} "
                    "with riesz_feature_fns=None: every row gives the forest "
                    "the same data. Keep the default riesz_feature_fns='auto', "
                    "or use AugForestRieszRegressor."
                )

        split_idx = self.split_feature_indices
        if split_idx is None:
            split_idx = default_split_feature_indices(estimand, self.riesz_feature_fns)
        split_idx = tuple(int(i) for i in split_idx)

        forest = _RieszGRF(
            n_outputs_riesz=p,
            l2=self.l2,
            n_estimators=self.n_estimators,
            criterion="mse",
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
            n_jobs=self.n_jobs,
            random_state=random_state,
            verbose=self.verbose,
            warm_start=False,
        )
        forest.fit(X_train[:, list(split_idx)], T_pack, y_pack)

        predictor = ForestPredictor(
            forest=forest,
            loss=loss,
            # Always store the resolved sieve, never the "auto" sentinel.
            riesz_feature_fns=sieve if sieve else None,
            split_feature_indices=split_idx,
        )

        val_score = (
            aug_valid.mean_loss(loss, predictor.predict_alpha(aug_valid.features))
            if aug_valid is not None else None
        )
        return FitResult(predictor=predictor, best_score=val_score)
