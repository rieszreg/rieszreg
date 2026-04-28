"""Sklearn backend: first-order gradient boosting (Friedman 2001) on the
augmented dataset with any sklearn-compatible base learner.

Per round: residual = -gradient (in η space), fit base learner, line-search
the optimal step, update F. Slower than xgboost but works with KernelRidge,
MLPs, custom regressors, anything with .fit(X, y) / .predict(X).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..augmentation import AugmentedDataset
from ..losses import LossSpec
from .base import FitResult, Predictor


@dataclass
class SklearnPredictor:
    learners: list
    steps: list[float]
    base_score: float
    loss: LossSpec
    best_iteration: int | None = None

    def _end(self) -> int:
        if self.best_iteration is not None:
            return self.best_iteration + 1
        return len(self.learners)

    def predict_eta(self, features: np.ndarray) -> np.ndarray:
        X = np.asarray(features, dtype=float)
        eta = np.full(X.shape[0], self.base_score)
        end = self._end()
        for h, step in zip(self.learners[:end], self.steps[:end]):
            eta = eta + step * np.asarray(h.predict(X))
        return eta

    def predict_alpha(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(self.loss.link_to_alpha(self.predict_eta(features)))


def _line_search(loss: LossSpec, a: np.ndarray, b: np.ndarray, F: np.ndarray, h: np.ndarray) -> float:
    """Closed-form γ minimizing E[loss_row(a, b, α(F + γ h))] under the
    second-order quadratic surrogate (correct for SquaredLoss; approximate but
    well-behaved for general convex losses)."""
    grad_F = loss.gradient(a, b, F)
    hess_F = loss.hessian(a, b, F, hessian_floor=1e-6)
    # numerator = -∇·h, denom = h·H·h (diagonal-Hessian surrogate)
    num = -float(np.sum(h * grad_F))
    denom = float(np.sum(h * h * hess_F))
    if denom <= 1e-12:
        return 0.0
    return num / denom


@dataclass
class SklearnBackend:
    """Friedman gradient boosting backend. `base_learner_factory()` is a
    zero-arg callable returning a fresh sklearn-compatible regressor."""

    base_learner_factory: Callable[[], Any]

    def fit_augmented(
        self,
        aug_train: AugmentedDataset,
        aug_valid: AugmentedDataset | None,
        loss: LossSpec,
        *,
        n_estimators: int,
        learning_rate: float,
        base_score: float,
        early_stopping_rounds: int | None,
        random_state: int,
        hyperparams: dict[str, Any],
    ) -> FitResult:
        # SklearnBackend has no extra hyperparams it consumes; warn-silently
        # discard anything passed (xgboost-only params don't apply here).
        del random_state, hyperparams

        loss.validate_coefficients(aug_train.b)
        if aug_valid is not None:
            loss.validate_coefficients(aug_valid.b)

        a, b = aug_train.a, aug_train.b
        F_train = np.full(aug_train.features.shape[0], base_score)

        have_valid = aug_valid is not None
        if early_stopping_rounds is not None and not have_valid:
            raise ValueError(
                "early_stopping_rounds requires validation data — pass "
                "`validation_fraction>0` or `eval_set=...` to RieszBooster."
            )
        if have_valid:
            F_val = np.full(aug_valid.features.shape[0], base_score)

        learners: list = []
        steps: list[float] = []
        history: list[float] = []
        best_score = float("inf")
        best_iter: int | None = None
        no_improve = 0

        for it in range(n_estimators):
            grad_train = loss.gradient(a, b, F_train)
            residual = -grad_train

            learner = self.base_learner_factory()
            learner.fit(aug_train.features, residual)
            h_train = np.asarray(learner.predict(aug_train.features))

            gamma = _line_search(loss, a, b, F_train, h_train)
            step = learning_rate * gamma
            F_train = F_train + step * h_train

            learners.append(learner)
            steps.append(step)

            if have_valid:
                h_val = np.asarray(learner.predict(aug_valid.features))
                F_val = F_val + step * h_val
                alpha_val = loss.link_to_alpha(F_val)
                val_loss = float(
                    np.sum(loss.loss_row(aug_valid.a, aug_valid.b, alpha_val))
                    / aug_valid.n_rows
                )
                history.append(val_loss)
                if val_loss < best_score - 1e-12:
                    best_score = val_loss
                    best_iter = it
                    no_improve = 0
                else:
                    no_improve += 1
                if early_stopping_rounds is not None and no_improve >= early_stopping_rounds:
                    break

        predictor = SklearnPredictor(
            learners=learners,
            steps=steps,
            base_score=base_score,
            loss=loss,
            best_iteration=best_iter,
        )
        return FitResult(
            predictor=predictor,
            best_iteration=best_iter,
            best_score=best_score if best_iter is not None else None,
            history=history,
        )
