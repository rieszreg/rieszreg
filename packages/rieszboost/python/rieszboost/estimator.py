"""rieszboost's user-facing convenience class. Subclass of `rieszreg.RieszEstimator`
that defaults the backend to `XGBoostBackend` and surfaces xgboost-specific
hyperparameters (`max_depth`, `reg_lambda`, `subsample`) as constructor args.

Designed to compose with `sklearn.model_selection.GridSearchCV`,
`cross_val_predict`, `clone`, etc.
"""

from __future__ import annotations

import inspect
from typing import Sequence

import numpy as np

from rieszreg.estimands.base import Estimand
from rieszreg.estimator import RieszEstimator
from rieszreg.losses import Loss

from .backends import Backend, XGBoostBackend


class RieszBooster(RieszEstimator):
    """Gradient-boosted estimator for the Riesz representer α₀ of a linear
    functional. ngboost / sklearn-style object-oriented API.

    Parameters
    ----------
    estimand : Estimand
        What to estimate, e.g. ``ATE(treatment="treated")``. Also names the
        treatment and covariate columns (default: every non-treatment column).
    backend : Backend, default=XGBoostBackend()
        Where the actual tree fitting happens. Swap to `SklearnBackend(...)`
        to use a non-tree base learner (KernelRidge, MLPs, etc.).
    loss : Loss, default=SquaredLoss()
        The Bregman-Riesz loss to minimize. `KLLoss()` is the alternative.
    n_estimators : int, default=200
    learning_rate : float, default=0.05
    max_depth : int, default=4
    reg_lambda : float, default=1.0
    subsample : float, default=1.0
    early_stopping_rounds : int or None, default=None
        Stop adding trees once the held-out loss hasn't improved for this
        many rounds. ``None`` fits all ``n_estimators`` trees.
    validation_fraction : float, default=0.1
        Fraction of the training rows held out for early stopping. Only used
        when ``early_stopping_rounds`` is set and no ``eval_set`` is passed
        to ``fit``.
    init : float or None
        α-space starting value. ``None`` (default) starts from the constant
        that minimizes the Riesz loss.
    random_state : int, default=0
    """

    def __init__(
        self,
        estimand: Estimand,
        backend: Backend | None = None,
        loss: Loss | None = None,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        reg_lambda: float = 1.0,
        subsample: float = 1.0,
        early_stopping_rounds: int | None = None,
        validation_fraction: float = 0.1,
        init: float | None = None,
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
        self, Z, n_estimators_grid: Sequence[int]
    ) -> np.ndarray:
        """Predict α̂ at every tree count in `n_estimators_grid` from one fit.

        Returns an array of shape ``(n_rows, len(n_estimators_grid))`` whose
        column ``j`` is the prediction obtained by truncating the booster to
        ``n_estimators_grid[j]`` trees. xgboost's ``iteration_range`` makes
        column ``j`` bit-equal to a fresh fit with ``n_estimators=
        n_estimators_grid[j]`` (same training data, same seed).

        Each grid entry must satisfy ``1 ≤ k ≤ booster.num_boosted_rounds()``.
        """
        feats = self._features(Z)
        return self.predictor_.predict_alpha_path(feats, n_estimators_grid)

    # Knobs that only take effect when RieszBooster builds the backend
    # (backend=None); an explicit backend is used as-is.
    _BACKEND_PARAMS = (
        "n_estimators", "learning_rate", "max_depth", "reg_lambda", "subsample",
        "early_stopping_rounds", "validation_fraction",
    )

    def _resolved_backend(self) -> Backend:
        if self.backend is not None:
            defaults = inspect.signature(RieszBooster.__init__).parameters
            overridden = [n for n in self._BACKEND_PARAMS if getattr(self, n) != defaults[n].default]
            if overridden:
                raise ValueError(
                    f"RieszBooster(backend={self.backend!r}) uses the backend as "
                    f"given, so the non-default {overridden} would be ignored. "
                    "Set them on the backend object instead, e.g. "
                    "XGBoostBackend(n_estimators=..., max_depth=...), or on the "
                    "base learner of a SklearnBackend."
                )
            return self.backend
        return XGBoostBackend(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            reg_lambda=self.reg_lambda,
            subsample=self.subsample,
            early_stopping_rounds=self.early_stopping_rounds,
            validation_fraction=(
                self.validation_fraction if self.early_stopping_rounds is not None else 0.0
            ),
        )
