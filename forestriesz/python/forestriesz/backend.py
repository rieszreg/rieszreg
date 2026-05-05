"""ForestRieszBackend — implements `rieszreg.MomentBackend`.

Consumes raw rows + the estimand directly (the moment-style entry point),
computes per-row moments via `rieszreg.trace`, packs them as a linear-moment
problem for EconML's `BaseGRF`, and returns a `FitResult` whose predictor is a
`ForestPredictor`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from rieszreg import (
    AugmentedDataset,
    Estimand,
    FiniteEvalEstimand,
    FitResult,
    LossSpec,
    SquaredLoss,
    aug_loss_alpha,
    build_augmented,
    trace,
)

from ._grf import _RieszGRF
from .feature_fns import default_split_feature_indices
from .predictor import ForestPredictor


def _materialize_features(
    rows: list[dict[str, Any]], feature_keys: Sequence[str]
) -> np.ndarray:
    return np.array([[r[k] for k in feature_keys] for r in rows], dtype=float)


def _eval_phi(
    features: np.ndarray, phi_fns: Sequence[Callable]
) -> np.ndarray:
    """Stack vectorized basis evaluations into an (n, p) matrix."""
    return np.column_stack([np.asarray(fn(features), dtype=float) for fn in phi_fns])


def _compute_per_row_moments(
    rows: list[dict[str, Any]],
    estimand: Estimand,
    phi_fns: Sequence[Callable],
    feature_keys: Sequence[str],
    ys: list | None = None,
) -> np.ndarray:
    """Compute A[i, j] = m(W_i; phi_j) = sum over (coef, point) in trace(W_i, y_i)
    of coef * phi_j(point), for each original row i and basis j. ``ys`` is the
    sklearn-style per-row outcome; pass ``None`` when the estimand's m doesn't
    depend on Y."""
    n = len(rows)
    p = len(phi_fns)
    if n == 0:
        return np.zeros((0, p))
    A = np.zeros((n, p))
    for i, row in enumerate(rows):
        y_i = ys[i] if ys is not None else None
        for coef, point in trace(estimand, row, y_i):
            point_arr = np.array([[point[k] for k in feature_keys]], dtype=float)
            phi_at_point = np.array([float(fn(point_arr)[0]) for fn in phi_fns])
            A[i] += coef * phi_at_point
    return A


def _holdout_riesz_loss(
    rows_valid: list[dict[str, Any]],
    estimand: Estimand,
    predictor: ForestPredictor,
    loss: LossSpec,
    ys_valid: list | None = None,
) -> float:
    """Mean per-original-row Riesz loss on the validation rows.

    Uses build_augmented + loss_row to share the formula with the rest of the
    framework, so val scores are comparable across backends. ``ys_valid``
    threads the per-row outcome through to ``m(alpha)(z, y)`` for
    Y-dependent estimands.
    """
    if not rows_valid:
        return float("nan")
    aug = build_augmented(rows_valid, estimand, ys_valid)
    eta = predictor.predict_eta(aug.features)
    alpha = loss.link_to_alpha(eta)
    return float(
        np.sum(aug_loss_alpha(loss, aug.is_original, aug.potential_deriv_coef, alpha))
        / aug.n_rows
    )


@dataclass
class ForestRieszBackend:
    """Random-forest Riesz regression backend.

    Wraps EconML's ``BaseGRF`` with the linear-moment criterion. Implements
    ``MomentBackend.fit_rows`` so it consumes raw rows and uses
    ``rieszreg.trace`` to evaluate per-row moments directly — no augmented
    dataset blow-up.

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
        Ridge added to the per-leaf Jacobian for numerical stability.
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
    l2: float = 0.01
    n_jobs: int = -1
    random_state: int = 0
    verbose: int = 0

    def fit_rows(
        self,
        rows_train: list[dict[str, Any]],
        rows_valid: list[dict[str, Any]] | None,
        estimand: Estimand,
        loss: LossSpec,
        *,
        base_score: float,
        random_state: int,
        hyperparams: dict[str, Any],
        ys_train: list | None = None,
        ys_valid: list | None = None,
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
        del hyperparams

        seed = random_state if random_state is not None else self.random_state
        feature_keys = estimand.feature_keys

        # 1. Materialize feature matrix.
        features = _materialize_features(rows_train, feature_keys)

        # 2. Resolve sieve. "auto" => default_riesz_features(estimand) when one
        # exists; otherwise constant. None => force constant.
        from .feature_fns import default_riesz_features

        if self.riesz_feature_fns == "auto":
            sieve = default_riesz_features(estimand)
        else:
            sieve = self.riesz_feature_fns
        phi_fns = sieve if sieve else [lambda f: np.ones(len(f))]
        p = len(phi_fns)

        # 3. Per-row basis values φ(W_i).
        phi_W = _eval_phi(features, phi_fns)             # (n, p)

        # 4. Per-row moment A[i, j] = m(W_i; φ_j).
        A = _compute_per_row_moments(rows_train, estimand, phi_fns, feature_keys, ys_train)

        # 5. Fold base_score into A so the predictor returns base_score + leaf θ·φ.
        if base_score != 0.0:
            A = A - base_score * phi_W

        # 6. Pack T = [vec(J) | A] per row, with J = φφ' (symmetric, so flat
        # order is immaterial). y is a dummy scalar zero column — EconML's
        # LinearMomentGRFCriterion requires scalar y.
        n_train = len(rows_train)
        JJ = np.einsum("ij,ik->ijk", phi_W, phi_W).reshape(n_train, p * p)
        T_pack = np.ascontiguousarray(np.column_stack([JJ, A]))
        y_pack = np.zeros((n_train, 1), dtype=float)

        # 6b. Detect degeneracy. For all built-in estimands the trace returns
        # a fixed set of (coef, point) pairs that don't depend on W, so under
        # a constant basis both A and J = φφ' are identical across rows and
        # the forest cannot learn anything from splits. The natural fix is the
        # sieve.
        if A.size > 0 and base_score == 0.0:
            j_row_constant = bool(np.allclose(JJ - JJ[0:1], 0.0, atol=1e-12))
            a_row_constant = bool(np.allclose(A - A[0:1], 0.0, atol=1e-12))
            if j_row_constant and a_row_constant:
                sieve_hint = (
                    "riesz_feature_fns='auto' (the default)"
                    if default_riesz_features(estimand) is not None
                    else "a custom riesz_feature_fns list capturing the "
                    "treatment / intervention structure of your estimand"
                )
                raise ValueError(
                    f"Per-row moment A and Jacobian J are both row-constant "
                    f"under the current basis for estimand {estimand.name!r}. "
                    "The forest cannot learn α from row-constant moment data. "
                    "This typically means the locally constant basis is being "
                    "used with a built-in estimand whose moment doesn't depend "
                    f"on W. Pass {sieve_hint} to use the locally linear sieve."
                )

        # 7. Choose split features.
        split_idx = self.split_feature_indices
        if split_idx is None:
            split_idx = default_split_feature_indices(estimand, self.riesz_feature_fns)
        split_idx = tuple(int(i) for i in split_idx)
        X_split = features[:, list(split_idx)]

        # 8. Fit forest.
        forest = _RieszGRF(
            n_outputs_riesz=p,
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
            random_state=seed,
            verbose=self.verbose,
            warm_start=False,
        )
        forest.fit(X_split, T_pack, y_pack)

        predictor = ForestPredictor(
            forest=forest,
            loss=loss,
            base_score=base_score,
            # Always store the resolved sieve, never the "auto" sentinel.
            riesz_feature_fns=sieve if sieve else None,
            feature_keys=tuple(feature_keys),
            split_feature_indices=split_idx,
        )

        val_score = None
        if rows_valid:
            val_score = _holdout_riesz_loss(rows_valid, estimand, predictor, loss, ys_valid)

        return FitResult(
            predictor=predictor,
            best_iteration=None,
            best_score=val_score,
            history=None,
        )
