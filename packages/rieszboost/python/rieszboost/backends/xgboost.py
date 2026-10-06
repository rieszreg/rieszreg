"""xgboost backend: data augmentation + custom objective.

Per-row gradient and hessian are sourced from the Loss, so this backend
works for any Loss (squared, KL, …) so long as the loss/link can produce
finite grad/hess values from the predicted η. xgboost boosts in η-space; the
predictor applies `loss.link_to_alpha` to convert to α space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator

from rieszreg.augmentation import AugmentedDataset
from rieszreg.backends.base import FitResult, register_predictor_loader
from rieszreg.losses import Loss


@dataclass
class XGBoostPredictor:
    booster: xgb.Booster
    base_score: float
    loss: Loss
    best_iteration: int | None = None

    kind = "xgboost"

    def _iter_range(self):
        if self.best_iteration is not None:
            return (0, self.best_iteration + 1)
        return None

    def predict_eta(self, features: np.ndarray) -> np.ndarray:
        dmat = xgb.DMatrix(np.asarray(features, dtype=float))
        rng = self._iter_range()
        kw = {} if rng is None else {"iteration_range": rng}
        # xgboost returns float32; apply the link in float64 so saturating
        # links (sigmoid) keep α̂ strictly inside their domain.
        return self.booster.predict(dmat, **kw).astype(np.float64)

    def predict_alpha(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(self.loss.link_to_alpha(self.predict_eta(features)))

    @property
    def n_trees(self) -> int:
        """Trees fitted, including any past the best early-stopping round."""
        return int(self.booster.num_boosted_rounds())

    def predict_eta_path(
        self, features: np.ndarray, n_estimators_grid: Sequence[int]
    ) -> np.ndarray:
        dmat = xgb.DMatrix(np.asarray(features, dtype=float))
        out = np.empty((dmat.num_row(), len(n_estimators_grid)), dtype=float)
        for j, k in enumerate(n_estimators_grid):
            out[:, j] = self.booster.predict(dmat, iteration_range=(0, k))
        return out

    def predict_alpha_path(
        self, features: np.ndarray, n_estimators_grid: Sequence[int]
    ) -> np.ndarray:
        eta = self.predict_eta_path(features, n_estimators_grid)
        return np.asarray(self.loss.link_to_alpha(eta))

    def save(self, dir_path):
        """Save the booster in xgboost's native UBJSON format inside dir_path."""
        from pathlib import Path
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(dir_path / "booster.ubj"))

    @classmethod
    def load(cls, dir_path, base_score: float, loss: Loss,
             best_iteration: int | None) -> "XGBoostPredictor":
        from pathlib import Path
        bst = xgb.Booster()
        bst.load_model(str(Path(dir_path) / "booster.ubj"))
        return cls(booster=bst, base_score=base_score, loss=loss,
                   best_iteration=best_iteration)


def _make_objective(
    aug: AugmentedDataset,
    loss: Loss,
    hessian_floor: float | str,
    gradient_only: bool,
    subsample: float,
    seed: int,
):
    is_original, potential_deriv_coef = aug.is_original, aug.potential_deriv_coef
    rng = np.random.default_rng(seed)

    def obj(preds: np.ndarray, dtrain) -> tuple[np.ndarray, np.ndarray]:
        del dtrain
        # xgboost passes float32 margins; a saturated link rounds to its bound.
        grad, hess = _grad_hess(preds.astype(np.float64))
        if subsample < 1.0:
            # Called once per round: draw individuals, keep all their
            # augmented rows. Zeroed rows drop out of every split and leaf,
            # which is how xgboost's own `subsample` removes rows.
            keep = (rng.random(aug.n_rows) < subsample)[aug.origin_index]
            grad, hess = grad * keep, hess * keep
        return grad, hess

    def _grad_hess(preds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        grad = loss.aug_grad_eta(is_original, potential_deriv_coef, preds)
        if gradient_only:
            hess = np.ones_like(grad)
        elif hessian_floor == "auto":
            # Floor each row at an observed row's curvature at its current
            # prediction (1e-6 backstop for α̂ near the link's boundary).
            floor = np.maximum(loss.curvature_eta(preds), 1e-6)
            hess = loss.aug_hess_eta(is_original, potential_deriv_coef, preds, floor)
        else:
            hess = loss.aug_hess_eta(is_original, potential_deriv_coef, preds, hessian_floor)
        return grad, hess
    return obj


def _make_metric(aug_valid: AugmentedDataset, loss: Loss):
    def metric(predt: np.ndarray, dval) -> tuple[str, float]:
        del dval
        return "riesz_loss", aug_valid.mean_loss(loss, loss.link_to_alpha(predt.astype(np.float64)))
    return metric


@dataclass(repr=False)
class XGBoostBackend(BaseEstimator):
    """Default backend. Construct with the boosting-loop knobs (n_estimators,
    learning_rate, early_stopping_rounds), the xgboost tree params
    (max_depth, reg_lambda, subsample) and stability tweaks (hessian_floor,
    gradient_only).

    Has sklearn's ``get_params`` / ``set_params``, so ``GridSearchCV`` can
    tune any field through ``backend__<field>`` on the estimator.

    ``hessian_floor`` is the lower bound on each row's Hessian. Counterfactual
    rows have a true Hessian of 0 (except under ``BoundedSquaredLoss``, whose
    link adds a term), so without a floor xgboost's Newton leaf step
    ``-G / (H + λ)`` blows up. ``"auto"`` (default) floors each row at
    ``loss.curvature_eta(η̂)``, the Gauss-Newton curvature of an observed row
    at the current prediction: 2 for ``SquaredLoss``, α̂ for ``KLLoss``,
    α̂(1 − α̂) for ``BernoulliLoss``. A float sets a fixed floor.

    ``subsample`` is the fraction of individuals (original rows) drawn each
    round. An individual's augmented rows are kept or dropped together.

    ``n_jobs`` is the number of xgboost threads; ``None`` uses all cores.
    """

    n_estimators: int = 200
    learning_rate: float = 0.05
    max_depth: int = 4
    reg_lambda: float = 1.0
    subsample: float = 1.0
    early_stopping_rounds: int | None = None
    validation_fraction: float = 0.0
    hessian_floor: float | str = "auto"
    gradient_only: bool = False
    n_jobs: int | None = None

    def fit_augmented(
        self,
        aug_train: AugmentedDataset,
        aug_valid: AugmentedDataset | None,
        loss: Loss,
        *,
        base_score: float,
        random_state: int,
    ) -> FitResult:
        # xgboost no longer sees `subsample` (the objective applies it), so
        # check its range here.
        if not 0.0 < self.subsample <= 1.0:
            raise ValueError(f"subsample must be in (0, 1]; got {self.subsample!r}.")
        dtrain = xgb.DMatrix(aug_train.features)

        params = {
            "learning_rate": self.learning_rate,
            "base_score": base_score,
            "seed": random_state,
            "disable_default_eval_metric": 1,
            "max_depth": self.max_depth,
            "reg_lambda": self.reg_lambda,
        }
        if self.n_jobs is not None:
            params["nthread"] = self.n_jobs

        # The held-out metric only drives early stopping; without it, skip the
        # per-round evaluation entirely.
        evals: list[tuple] = []
        custom_metric = None
        if self.early_stopping_rounds is not None and aug_valid is not None:
            evals = [(xgb.DMatrix(aug_valid.features), "valid")]
            custom_metric = _make_metric(aug_valid, loss)
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
                aug_train,
                loss,
                hessian_floor=self.hessian_floor,
                gradient_only=self.gradient_only,
                subsample=self.subsample,
                seed=random_state,
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
