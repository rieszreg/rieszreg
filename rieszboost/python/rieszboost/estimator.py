"""rieszboost's user-facing convenience class. Subclass of `rieszreg.RieszEstimator`
that defaults the backend to `XGBoostBackend` and surfaces xgboost-specific
hyperparameters (`max_depth`, `reg_lambda`, `subsample`) as constructor args.

Designed to compose with `sklearn.model_selection.GridSearchCV`,
`cross_val_predict`, `clone`, etc.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from rieszreg.estimands.base import Estimand
from rieszreg.estimator import RieszEstimator, _features_from_rows, _rows_from_X
from rieszreg.losses import LossSpec

from .backends import Backend, XGBoostBackend


class RieszBooster(RieszEstimator):
    """Gradient-boosted estimator for the Riesz representer α₀ of a linear
    functional. ngboost / sklearn-style object-oriented API.

    Parameters
    ----------
    estimand : Estimand
        Carries `feature_keys` and the `m(alpha)(z, y)` operator. Required.
    backend : Backend, default=XGBoostBackend()
        Where the actual tree fitting happens. Swap to `SklearnBackend(...)`
        to use a non-tree base learner (KernelRidge, MLPs, etc.).
    loss : LossSpec, default=SquaredLoss()
        The Bregman-Riesz loss to minimize. `KLLoss()` is the alternative.
    n_estimators : int, default=200
    learning_rate : float, default=0.05
    max_depth : int, default=4
    reg_lambda : float, default=1.0
    subsample : float, default=1.0
    early_stopping_rounds : int or None
        If set, requires either `validation_fraction>0` or `eval_set=...` at
        fit time.
    validation_fraction : float, default=0.0
        Fraction of training data held out internally for early stopping.
    init : float, "m1", or None
        α-space initialization. None defers to `loss.default_init_alpha()`.
    random_state : int, default=0
    """

    def __init__(
        self,
        estimand: Estimand,
        backend: Backend | None = None,
        loss: LossSpec | None = None,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        reg_lambda: float = 1.0,
        subsample: float = 1.0,
        early_stopping_rounds: int | None = None,
        validation_fraction: float = 0.0,
        init: float | str | None = None,
        random_state: int = 0,
    ):
        super().__init__(
            estimand=estimand,
            backend=backend,
            loss=loss,
            init=init,
            random_state=random_state,
        )
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.subsample = subsample

    def predict_path(
        self, X, n_estimators_grid: Sequence[int]
    ) -> np.ndarray:
        """Predict α̂ at every tree count in `n_estimators_grid` from one fit.

        Returns an array of shape ``(n_rows, len(n_estimators_grid))`` whose
        column ``j`` is the prediction obtained by truncating the booster to
        ``n_estimators_grid[j]`` trees. xgboost's ``iteration_range`` makes
        column ``j`` bit-equal to a fresh fit with ``n_estimators=
        n_estimators_grid[j]`` (same training data, same seed).

        Each grid entry must satisfy ``1 ≤ k ≤ booster.num_boosted_rounds()``.
        """
        if not hasattr(self, "predictor_"):
            raise RuntimeError(
                f"{type(self).__name__} is not fitted yet. Call .fit() first."
            )
        rows = _rows_from_X(X, self.estimand)
        feats = _features_from_rows(rows, self.estimand)
        return self.predictor_.predict_alpha_path(feats, n_estimators_grid)

    def _resolved_backend(self) -> Backend:
        if self.backend is not None:
            return self.backend
        return XGBoostBackend(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            early_stopping_rounds=self.early_stopping_rounds,
            validation_fraction=self.validation_fraction,
        )

    def _backend_hyperparams(self) -> dict:
        return {
            "max_depth": self.max_depth,
            "reg_lambda": self.reg_lambda,
            "subsample": self.subsample,
        }

    def _save_hyperparameters(self) -> dict:
        base = super()._save_hyperparameters()
        base.update(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            early_stopping_rounds=self.early_stopping_rounds,
            validation_fraction=self.validation_fraction,
            max_depth=self.max_depth,
            reg_lambda=self.reg_lambda,
            subsample=self.subsample,
        )
        return base

    @classmethod
    def _construct_for_load(cls, *, estimand, loss, hyperparameters: dict) -> "RieszBooster":
        return cls(
            estimand=estimand,
            loss=loss,
            n_estimators=hyperparameters.get("n_estimators", 200),
            learning_rate=hyperparameters.get("learning_rate", 0.05),
            max_depth=hyperparameters.get("max_depth", 4),
            reg_lambda=hyperparameters.get("reg_lambda", 1.0),
            subsample=hyperparameters.get("subsample", 1.0),
            early_stopping_rounds=hyperparameters.get("early_stopping_rounds"),
            validation_fraction=hyperparameters.get("validation_fraction", 0.0),
            init=hyperparameters.get("init"),
            random_state=hyperparameters.get("random_state", 0),
        )
