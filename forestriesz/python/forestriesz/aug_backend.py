"""AugForestRieszBackend — augmentation-style forest backend.

Alternative to ``ForestRieszBackend``. Implements ``rieszreg.Backend`` (the
augmentation-style entry point) instead of ``MomentBackend``: the orchestrator
hands it the precomputed ``AugmentedDataset`` and the backend trains the GRF
on the M = k·n augmented evaluation points directly.

Why a second backend? The moment-style ``ForestRieszBackend`` needs a sieve
that captures the moment's dependence on W — for built-in estimands without a
canonical sieve (AdditiveShift, LocalShift, custom user moments) it raises a
row-constant degeneracy error. The augmentation-style path sidesteps the
sieve question entirely: even with a constant basis, J_k = 2 a_k and
A_k = -b_k vary across augmented rows (originals have a=1, b=0;
counterfactual evaluation points have a=0, b≠0), so the forest can split
usefully on the full feature space without estimand-specific configuration.

Trade-offs vs ``ForestRieszBackend``:
  + Estimand-agnostic: works on any built-in or custom Estimand.
  + No sieve choice; splitter sees the full feature space.
  - Forest training set is ~k× larger (typically 2-3×).
  - Default GRF variance ignores within-block correlation across augmented
    rows from the same original W, so ``predict_interval`` is not exposed
    here. Use cluster-robust variance with origin_index as cluster id if you
    need CIs (planned for v2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from rieszreg import (
    AugmentedDataset,
    FitResult,
    LossSpec,
    SquaredLoss,
)

from ._grf import _RieszGRF
from .aug_predictor import AugForestPredictor


def _eval_phi(features: np.ndarray, phi_fns: Sequence[Callable]) -> np.ndarray:
    return np.column_stack(
        [np.asarray(fn(features), dtype=float) for fn in phi_fns]
    )


def _holdout_riesz_loss(
    aug_valid: AugmentedDataset, predictor: AugForestPredictor, loss: LossSpec
) -> float:
    eta = predictor.predict_eta(aug_valid.features)
    alpha = loss.link_to_alpha(eta)
    return float(np.sum(loss.loss_row(aug_valid.a, aug_valid.b, alpha)) / aug_valid.n_rows)


@dataclass
class AugForestRieszBackend:
    """Augmentation-style random-forest Riesz backend.

    Parameters
    ----------
    riesz_feature_fns
        Optional sieve basis. Each callable maps an ``(n, n_features)`` matrix
        to ``(n,)``. ``None`` (the default) uses the constant basis, which is
        non-degenerate here because J / A vary across augmented rows.
    n_estimators, max_depth, ... (every BaseGRF knob, same defaults as
    ForestRieszBackend).
    """

    riesz_feature_fns: list[Callable] | None = None
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

    def fit_augmented(
        self,
        aug_train: AugmentedDataset,
        aug_valid: AugmentedDataset | None,
        loss: LossSpec,
        *,
        base_score: float,
        random_state: int,
        hyperparams: dict[str, Any],
    ) -> FitResult:
        del hyperparams

        # All four built-in losses are supported. For non-squared Bregman
        # losses we still fit the tree structure under the squared MSE
        # criterion (splits that maximize variance reduction in -B/(2A) also
        # separate the monotonically-related Bregman optima well), then
        # post-hoc replace each leaf value with the Bregman per-leaf optimum.
        loss.validate_coefficients(aug_train.b)

        seed = random_state if random_state is not None else self.random_state

        # 1. Resolve sieve. Default = constant basis.
        phi_fns = self.riesz_feature_fns or [lambda f: np.ones(len(f))]
        p = len(phi_fns)

        # 2. Per-augmented-row basis.
        phi = _eval_phi(aug_train.features, phi_fns)        # (M, p)
        a = aug_train.a                                      # (M,)
        b = aug_train.b                                      # (M,)

        # 3. Fold base_score into the linear coefficient (same trick as krrr).
        if base_score != 0.0:
            b = b + 2.0 * a * base_score

        # 4. Per-augmented-row J = 2 a φ φ', A = -b φ.
        # J shape (M, p, p) flattened to (M, p²); A shape (M, p).
        JJ = (2.0 * a)[:, None, None] * np.einsum("ij,ik->ijk", phi, phi)
        JJ_flat = JJ.reshape(-1, p * p)
        A = -(b[:, None] * phi)                              # (M, p)

        # 5. Pack into the EconML T slot (LinearMomentGRFCriterion wants scalar y).
        T_pack = np.ascontiguousarray(np.column_stack([JJ_flat, A]))
        y_pack = np.zeros((aug_train.features.shape[0], 1), dtype=float)

        # 6. Fit forest. Splitter sees the full augmented feature space.
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
        forest.fit(aug_train.features, T_pack, y_pack)

        # 7. Bregman post-processing. For squared loss the EconML leaf solve
        # already gives the right θ. For other losses walk every (tree, leaf)
        # and replace the stored θ with the Bregman per-leaf optimum from a
        # Newton iteration on the leaf's rows. The Newton uses the original
        # (un-base-score-shifted) b because it evaluates gradients at
        # η = θ · φ + base_score directly.
        leaf_eta_table = None
        if not isinstance(loss, SquaredLoss):
            from ._leaf_solver import compute_leaf_eta_table

            leaf_eta_table = compute_leaf_eta_table(
                forest=forest,
                X_aug=aug_train.features,
                a_aug=a,
                b_aug=aug_train.b,            # original b, not the squared-fold-in
                phi_aug=phi,
                loss=loss,
                base_score=base_score,
            )

        predictor = AugForestPredictor(
            forest=forest,
            loss=loss,
            base_score=base_score,
            riesz_feature_fns=self.riesz_feature_fns,
            leaf_eta_table=leaf_eta_table,
        )

        val_score = None
        if aug_valid is not None:
            val_score = _holdout_riesz_loss(aug_valid, predictor, loss)

        return FitResult(
            predictor=predictor,
            best_iteration=None,
            best_score=val_score,
            history=None,
        )
