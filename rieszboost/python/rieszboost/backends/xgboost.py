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

from rieszreg.augmentation import (
    AugmentedDataset,
    aug_grad_eta,
    aug_hess_eta,
    aug_loss_alpha,
)
from rieszreg.backends.base import FitResult, Predictor, register_predictor_loader
from rieszreg.losses import LossSpec


@dataclass
class XGBoostPredictor:
    booster: xgb.Booster
    base_score: float
    loss: LossSpec
    best_iteration: int | None = None

    kind = "xgboost"

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

    def save(self, dir_path):
        """Save booster as JSON (xgboost native format) inside dir_path."""
        from pathlib import Path
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(dir_path / "booster.ubj"))

    @classmethod
    def load(cls, dir_path, base_score: float, loss: LossSpec,
             best_iteration: int | None) -> "XGBoostPredictor":
        from pathlib import Path
        bst = xgb.Booster()
        bst.load_model(str(Path(dir_path) / "booster.ubj"))
        return cls(booster=bst, base_score=base_score, loss=loss,
                   best_iteration=best_iteration)


def _make_objective(
    is_original: np.ndarray,
    potential_deriv_coef: np.ndarray,
    loss: LossSpec,
    hessian_floor: float,
    gradient_only: bool,
):
    def obj(preds: np.ndarray, dtrain) -> tuple[np.ndarray, np.ndarray]:
        del dtrain
        grad = aug_grad_eta(loss, is_original, potential_deriv_coef, preds)
        if gradient_only:
            hess = np.ones_like(grad)
        else:
            hess = aug_hess_eta(loss, is_original, potential_deriv_coef, preds, hessian_floor)
        return grad, hess
    return obj


def _make_metric(
    is_original_val: np.ndarray,
    potential_deriv_coef_val: np.ndarray,
    n_val_rows: int,
    loss: LossSpec,
):
    def metric(predt: np.ndarray, dval) -> tuple[str, float]:
        del dval
        alpha = loss.link_to_alpha(predt)
        per_row = aug_loss_alpha(loss, is_original_val, potential_deriv_coef_val, alpha)
        return "riesz_loss", float(np.sum(per_row) / n_val_rows)
    return metric


@dataclass
class XGBoostBackend:
    """Default backend. Construct with the boosting-loop knobs (n_estimators,
    learning_rate, early_stopping_rounds) plus stability tweaks
    (hessian_floor, gradient_only). Other xgboost passthrough params
    (max_depth, reg_lambda, subsample) come via ``hyperparams`` from
    RieszBooster.
    """

    n_estimators: int = 200
    learning_rate: float = 0.05
    early_stopping_rounds: int | None = None
    validation_fraction: float = 0.0
    hessian_floor: float = 2.0
    gradient_only: bool = False

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
        dtrain = xgb.DMatrix(aug_train.features)

        params = {
            "learning_rate": self.learning_rate,
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
                aug_valid.is_original,
                aug_valid.potential_deriv_coef,
                aug_valid.n_rows,
                loss,
            )
        elif self.early_stopping_rounds is not None:
            raise ValueError(
                "early_stopping_rounds was set but no validation data was "
                "provided. Pass `validation_fraction>0` or `eval_set=...` "
                "to RieszBooster."
            )

        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            obj=_make_objective(
                aug_train.is_original,
                aug_train.potential_deriv_coef,
                loss,
                hessian_floor=self.hessian_floor,
                gradient_only=self.gradient_only,
            ),
            evals=evals,
            custom_metric=custom_metric,
            early_stopping_rounds=self.early_stopping_rounds,
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


register_predictor_loader("xgboost", XGBoostPredictor.load)
