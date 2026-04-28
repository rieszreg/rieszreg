"""xgboost backend: data augmentation + custom objective.

Per-row gradient and hessian are sourced from the LossSpec, so this backend
works for any LossSpec (squared, KL, …) so long as the loss/link can produce
finite grad/hess values from the predicted η. xgboost boosts in η-space; the
predictor applies `loss.link_to_alpha` to convert to α space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import xgboost as xgb

from ..augmentation import AugmentedDataset
from ..losses import LossSpec
from .base import FitResult, Predictor


@dataclass
class XGBoostPredictor:
    booster: xgb.Booster
    base_score: float
    loss: LossSpec
    best_iteration: int | None = None

    def _iter_range(self):
        if self.best_iteration is not None:
            return (0, self.best_iteration + 1)
        return None

    def predict_eta(self, features: np.ndarray) -> np.ndarray:
        dmat = xgb.DMatrix(np.asarray(features, dtype=float))
        rng = self._iter_range()
        if rng is not None:
            return self.booster.predict(dmat, iteration_range=rng)
        return self.booster.predict(dmat)

    def predict_alpha(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(self.loss.link_to_alpha(self.predict_eta(features)))


def _make_objective(
    a: np.ndarray,
    b: np.ndarray,
    loss: LossSpec,
    hessian_floor: float,
    gradient_only: bool,
):
    def obj(preds: np.ndarray, dtrain) -> tuple[np.ndarray, np.ndarray]:
        del dtrain
        grad = loss.gradient(a, b, preds)
        if gradient_only:
            hess = np.ones_like(grad)
        else:
            hess = loss.hessian(a, b, preds, hessian_floor)
        return grad, hess
    return obj


def _make_metric(
    a_val: np.ndarray, b_val: np.ndarray, n_val_rows: int, loss: LossSpec
):
    def metric(predt: np.ndarray, dval) -> tuple[str, float]:
        del dval
        alpha = loss.link_to_alpha(predt)
        per_row = loss.loss_row(a_val, b_val, alpha)
        return "riesz_loss", float(np.sum(per_row) / n_val_rows)
    return metric


@dataclass
class XGBoostBackend:
    """Default backend. Construct with hessian_floor / gradient_only knobs;
    other xgboost passthrough params (max_depth, reg_lambda, subsample) come
    via `hyperparams` from RieszBooster.
    """

    hessian_floor: float = 2.0
    gradient_only: bool = False

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
        loss.validate_coefficients(aug_train.b)
        if aug_valid is not None:
            loss.validate_coefficients(aug_valid.b)

        dtrain = xgb.DMatrix(aug_train.features)

        params = {
            "learning_rate": learning_rate,
            "base_score": base_score,
            "seed": random_state,
            "disable_default_eval_metric": 1,
            **hyperparams,
        }

        evals: list[tuple] = []
        custom_metric = None
        if aug_valid is not None:
            dvalid = xgb.DMatrix(aug_valid.features)
            evals = [(dvalid, "valid")]
            custom_metric = _make_metric(
                aug_valid.a, aug_valid.b, aug_valid.n_rows, loss
            )
        elif early_stopping_rounds is not None:
            raise ValueError(
                "early_stopping_rounds was set but no validation data was "
                "provided. Pass `validation_fraction>0` or `eval_set=...` "
                "to RieszBooster."
            )

        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            obj=_make_objective(
                aug_train.a, aug_train.b, loss,
                hessian_floor=self.hessian_floor,
                gradient_only=self.gradient_only,
            ),
            evals=evals,
            custom_metric=custom_metric,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )
        best_iteration = getattr(booster, "best_iteration", None)
        best_score = getattr(booster, "best_score", None)

        predictor = XGBoostPredictor(
            booster=booster,
            base_score=base_score,
            loss=loss,
            best_iteration=best_iteration,
        )
        return FitResult(
            predictor=predictor,
            best_iteration=best_iteration,
            best_score=float(best_score) if best_score is not None else None,
        )
