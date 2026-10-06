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

A backend that uses held-out rows (early stopping, λ selection) also
implements ``HoldoutBackend.holdout_fraction``. The orchestrator splits off
that fraction of rows before augmentation and passes them as the
validation data.

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


class HoldoutBackend(Protocol):
    """Optional capability of a ``Backend`` or ``MomentBackend``: held-out rows.

    ``holdout_fraction()`` returns the fraction of training rows to hold out
    for this fit, or 0 when the fit won't use held-out rows (for example,
    early stopping is off). The orchestrator splits the rows before
    augmentation and passes the held-out part as ``aug_valid``. An
    ``eval_set`` passed to ``fit`` replaces the split.
    """

    def holdout_fraction(self) -> float:
        ...


def holdout_fraction(backend) -> float:
    """The fraction of training rows the orchestrator holds out for ``backend``.

    ``backend.holdout_fraction()`` when the backend defines it, otherwise 0:
    a backend without the method gets no holdout.
    """
    method = getattr(backend, "holdout_fraction", None)
    return float(method()) if callable(method) else 0.0


class ColumnBackend(Protocol):
    """Optional capability of a ``Backend`` or ``MomentBackend``: settings
    that name input columns (e.g. ``categorical_features``).

    ``bind_columns(feature_keys)`` returns a copy of the backend with those
    names resolved to positions in ``feature_keys``, the bound estimand's
    column order. The orchestrator calls it once per fit, after binding the
    estimand, and fits the copy; the user's backend is left unchanged.
    """

    def bind_columns(self, feature_keys: tuple[str, ...]):
        ...


def bind_columns(backend, feature_keys) -> Any:
    """The backend to fit: ``backend.bind_columns(feature_keys)`` when the
    backend defines it, else ``backend`` itself."""
    method = getattr(backend, "bind_columns", None)
    if callable(method):
        return method(tuple(feature_keys))
    return backend


def resolve_column_positions(columns, feature_keys, setting: str) -> tuple[int, ...]:
    """Map ``columns`` (names or 0-based positions) to positions in
    ``feature_keys``. With ``feature_keys=None`` (no bound estimand yet),
    only positions are accepted.

    Positions index ``feature_keys``, where the treatment comes first, not the
    columns of the user's data; names avoid that pitfall.
    """
    out = []
    for c in columns or ():
        if isinstance(c, str):
            if feature_keys is None:
                raise ValueError(
                    f"{setting} names column {c!r}, but names are resolved only "
                    "when fitting through RieszEstimator (or a learner class). "
                    "Pass 0-based positions into the estimand's feature_keys "
                    "when calling the backend directly."
                )
            if c not in feature_keys:
                raise ValueError(
                    f"{setting} names column {c!r}, which α̂ doesn't use; its "
                    f"columns are {list(feature_keys)}."
                )
            out.append(feature_keys.index(c))
        else:
            j = int(c)
            if feature_keys is not None and not 0 <= j < len(feature_keys):
                raise ValueError(
                    f"{setting} position {j} is out of range for α̂'s "
                    f"{len(feature_keys)} columns {list(feature_keys)}."
                )
            out.append(j)
    return tuple(out)


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
