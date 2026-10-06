"""Backend protocol: the swappable component that consumes the per-row data
and a Loss and produces a fitted Predictor.

Two entry points are supported. A backend implements one; if it has both,
the orchestrator calls ``fit_augmented``:

  * ``fit_augmented`` — for learners that fit directly on the augmented
    evaluation points (kernel ridge, gradient boosting, trees, neural nets).
    Receives an ``AugmentedDataset`` of (a, b) coefficients at concrete
    evaluation points. Implementations: ``KernelRidgeBackend`` (krrr),
    ``XGBoostBackend`` / ``SklearnBackend`` (rieszboost), ``RieszTreeBackend``
    (riesztree), ``AugForestRieszBackend`` (forestriesz), ``TorchBackend``
    (riesznet).
  * ``fit_rows`` — for learners that fit on the original sample rows and use
    the augmented data only to form per-row moments. Receives the
    original-row feature matrix, the ``Estimand`` and the augmented data
    (grouped back to rows by ``origin_index``). Implementation:
    ``ForestRieszBackend`` (forestriesz).

The ``RieszEstimator`` orchestrator builds the augmented dataset in both
cases; it calls ``fit_rows`` when a backend defines only that method and
``fit_augmented`` otherwise.

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
    dataset with ``estimand.augment`` before calling.

    Method kwargs are universal: data, ``base_score`` and ``random_state``.
    All learner-specific knobs (``n_estimators``, ``learning_rate``,
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
    ) -> FitResult:
        ...


class MomentBackend(Protocol):
    """Moment-style backend Protocol.

    Alternative to ``Backend`` for learners that fit on the original rows
    and read per-row moments off the augmented data (random forests solving
    a local moment equation). ``X_train`` / ``X_valid`` are
    ``(n, len(estimand.feature_keys))`` float arrays with columns in
    ``feature_keys`` order; ``aug_train`` / ``aug_valid`` are their
    augmentations (``estimand.augment``, outcome included). The validation
    arguments are ``None`` without a validation set. ``base_score`` and
    ``random_state`` are as in ``Backend.fit_augmented``; learner-specific
    knobs live on the concrete backend.
    """

    def fit_rows(
        self,
        X_train: np.ndarray,
        X_valid: np.ndarray | None,
        estimand: "Estimand",
        loss: Loss,
        *,
        aug_train: AugmentedDataset,
        aug_valid: AugmentedDataset | None,
        base_score: float,
        random_state: int,
    ) -> FitResult:
        ...


# ----- Predictor loader registry (used by RieszEstimator.load) -----

_PREDICTOR_LOADERS: dict[str, Any] = {}


def register_predictor_loader(kind: str, loader) -> None:
    """Register a loader for a predictor kind.

    ``loader`` is a callable ``(dir_path, base_score, loss, best_iteration)
    -> Predictor``, or a ``"module:attr"`` string naming one. Packages with
    heavy lazily-imported dependencies register the string from their
    ``__init__`` so ``RieszEstimator.load`` works after a plain
    ``import <pkg>`` without importing the dependency up front:

        register_predictor_loader("riesznet", "riesznet.backend:TorchPredictor.load")
    """
    _PREDICTOR_LOADERS[kind] = loader


def load_predictor(kind: str, dir_path, *, base_score, loss, best_iteration):
    """Look up a registered loader and instantiate the predictor."""
    if kind not in _PREDICTOR_LOADERS:
        raise ValueError(
            f"No loader registered for predictor kind {kind!r}. "
            f"Import the learner package that fit this model (e.g. `import "
            f"rieszboost`) before calling .load(...). Registered kinds: "
            f"{sorted(_PREDICTOR_LOADERS)}."
        )
    loader = _PREDICTOR_LOADERS[kind]
    if isinstance(loader, str):
        from importlib import import_module
        mod_name, attr = loader.split(":")
        obj = import_module(mod_name)
        for part in attr.split("."):
            obj = getattr(obj, part)
        loader = obj
    return loader(
        dir_path, base_score=base_score, loss=loss, best_iteration=best_iteration
    )
