"""Backend protocol: the swappable component that consumes the per-row data
and a Loss and produces a fitted Predictor.

Two entry points are supported. A backend implements at most one:

  * ``fit_augmented`` — for learners whose loss decomposes naturally over
    augmented evaluation points (kernel ridge, gradient boosting). Receives an
    ``AugmentedDataset`` of (a, b) coefficients at concrete evaluation points.
    Implementations: ``KernelRidgeBackend`` (krrr), ``XGBoostBackend`` /
    ``SklearnBackend`` (rieszboost).
  * ``fit_rows`` — for learners whose loss decomposes per original sample row
    (random forests, neural nets). Receives raw ``rows`` plus the ``Estimand``
    so the backend can compute per-row moments via ``rieszreg.trace`` directly.
    Implementations: ``ForestRieszBackend`` (forestriesz).

The ``RieszEstimator`` orchestrator dispatches by looking for ``fit_rows``
first; if absent, it builds the augmented dataset and calls ``fit_augmented``.

Concrete backends live in implementation packages (rieszboost, krrr,
forestriesz, ...).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from ..augmentation import AugmentedDataset
from ..losses import Loss

if TYPE_CHECKING:  # pragma: no cover - type-only import
    from ..estimands.base import Estimand


class Predictor(Protocol):
    """Output of `Backend.fit_augmented`. RieszEstimator delegates to this for
    prediction. Implementations should apply the loss spec's link in
    `predict_alpha` so callers see α̂, not raw η.

    Attributes
    ----------
    kind : str
        Short identifier (e.g. "xgboost", "sklearn", "kernel-ridge") used by
        the registry-based load path.
    """

    kind: str

    def predict_eta(self, features: np.ndarray) -> np.ndarray: ...
    def predict_alpha(self, features: np.ndarray) -> np.ndarray: ...

    def save(self, dir_path) -> None:
        """Write the binary payload (e.g. native model file, joblib pickle,
        torch state_dict) into `dir_path`. Metadata (loss, estimand spec,
        hyperparameters) is written by the orchestrator estimator separately."""
        ...


@dataclass
class FitResult:
    predictor: Predictor
    best_iteration: int | None = None
    best_score: float | None = None
    history: list[float] | None = None


class Backend(Protocol):
    """Augmentation-style backend Protocol.

    Implementers consume a precomputed ``AugmentedDataset`` of (a, b)
    coefficients at evaluation points. The orchestrator builds the augmented
    dataset by tracing the estimand on each input row before calling.

    Method kwargs are universal: data, ``base_score``, ``random_state``,
    and ``hyperparams`` (a dict for backend-specific passthrough). All
    learner-specific knobs (``n_estimators``, ``learning_rate``,
    ``early_stopping_rounds``, kernel choice, …) live on the concrete
    backend's constructor — see DESIGN.md §A.1.
    """

    def fit_augmented(
        self,
        aug_train: AugmentedDataset,
        aug_valid: AugmentedDataset | None,
        loss: Loss,
        *,
        base_score: float,
        random_state: int,
        hyperparams: dict[str, Any],
    ) -> FitResult:
        ...


class MomentBackend(Protocol):
    """Moment-style backend Protocol.

    Alternative to ``Backend`` for learners that consume raw rows + the
    estimand directly. Useful for random forests and neural nets where each
    sample row contributes an independent loss term — these learners benefit
    from per-row moment evaluation rather than the augmented (a, b) view.

    Same calling convention as ``Backend.fit_augmented``: data, ``base_score``,
    ``random_state``, and ``hyperparams``. ``ys_train`` / ``ys_valid`` carry
    the per-row outcome (sklearn-style) for estimands whose ``m`` reads it;
    they are ``None`` otherwise. Learner-specific knobs live on the concrete
    backend.
    """

    def fit_rows(
        self,
        rows_train: list[dict[str, Any]],
        rows_valid: list[dict[str, Any]] | None,
        estimand: "Estimand",
        loss: Loss,
        *,
        base_score: float,
        random_state: int,
        hyperparams: dict[str, Any],
        ys_train: list | None = None,
        ys_valid: list | None = None,
    ) -> FitResult:
        ...


# ----- Predictor loader registry (used by RieszEstimator.load) -----

_PREDICTOR_LOADERS: dict[str, Any] = {}


def register_predictor_loader(kind: str, loader) -> None:
    """Register a loader callable for a predictor kind.

    Implementation packages call this at import time:

        register_predictor_loader("xgboost", XGBoostPredictor.load)

    The loader signature is `(dir_path, base_score, loss, best_iteration) -> Predictor`.
    """
    _PREDICTOR_LOADERS[kind] = loader


def load_predictor(kind: str, dir_path, *, base_score, loss, best_iteration):
    """Look up a registered loader and instantiate the predictor."""
    if kind not in _PREDICTOR_LOADERS:
        raise ValueError(
            f"No loader registered for predictor kind {kind!r}. "
            f"Import a learner package (e.g. `import rieszboost`, `import krrr`, "
            f"`import forestriesz`, `import riesznet`) "
            f"before calling .load(...). Registered kinds: {sorted(_PREDICTOR_LOADERS)}."
        )
    return _PREDICTOR_LOADERS[kind](
        dir_path, base_score=base_score, loss=loss, best_iteration=best_iteration
    )
