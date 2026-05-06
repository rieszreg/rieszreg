"""Sklearn-compatible orchestrator for Riesz representer estimation.

`RieszEstimator` takes (estimand, loss, backend) at construction and implements
the standard sklearn `fit / predict / score` API. Backend-specific
hyperparameters live on subclasses (e.g. `RieszBooster` in `rieszboost` adds
`max_depth`, `reg_lambda`, `subsample`).

Designed to compose with `sklearn.model_selection.GridSearchCV`,
`cross_val_predict`, `clone`, etc. Mirrors ngboost's API style.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from ._omp import warn_if_multi_backend_omp
from .augmentation import aug_loss_alpha, build_augmented
from .backends import Backend, load_predictor
from .estimands.base import Estimand, FiniteEvalEstimand, estimand_from_spec
from .estimands.tracer import trace
from .losses import LossSpec, SquaredLoss, loss_from_spec


def _is_dataframe(Z) -> bool:
    return hasattr(Z, "columns") and hasattr(Z, "iloc")


def _rows_from_Z(Z, estimand: Estimand) -> list[dict]:
    """Convert ndarray or DataFrame Z into a list of row-dicts keyed by
    `estimand.feature_keys`. Ndarray input is interpreted column-by-column in
    `feature_keys` order; DataFrame columns are matched by name."""
    if _is_dataframe(Z):
        cols_needed = list(estimand.feature_keys)
        missing = [c for c in cols_needed if c not in Z.columns]
        if missing:
            raise ValueError(
                f"DataFrame is missing columns required by estimand "
                f"{estimand.name!r}: {missing}"
            )
        rows = []
        for i in range(len(Z)):
            row = {}
            for k in cols_needed:
                v = Z[k].iloc[i]
                row[k] = v
            rows.append(row)
        return rows

    arr = np.asarray(Z)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[1] != len(estimand.feature_keys):
        raise ValueError(
            f"Estimand {estimand.name!r} expects {len(estimand.feature_keys)} "
            f"feature columns ({estimand.feature_keys}), got Z.shape[1]="
            f"{arr.shape[1]}."
        )
    return [
        {k: arr[i, j] for j, k in enumerate(estimand.feature_keys)}
        for i in range(arr.shape[0])
    ]


def _ys_from_y(y, n: int) -> list | None:
    """Coerce `y` (a sklearn-style outcome vector) into a list of per-row
    scalars aligned with the rows. Returns None when `y is None`. Raises if
    the length doesn't match `n`."""
    if y is None:
        return None
    if hasattr(y, "to_numpy"):
        y_arr = y.to_numpy()
    else:
        y_arr = np.asarray(y)
    if y_arr.ndim > 1:
        y_arr = y_arr.reshape(-1)
    if len(y_arr) != n:
        raise ValueError(
            f"len(y)={len(y_arr)} does not match number of rows in Z ({n})."
        )
    return list(y_arr)


def _features_from_rows(rows: Sequence[dict], estimand: Estimand) -> np.ndarray:
    return np.asarray(
        [[row[k] for k in estimand.feature_keys] for row in rows], dtype=float
    )


def _split_Z(Z, y, validation_fraction: float, random_state: int):
    """Split (Z, y) into train/valid by `validation_fraction`. `y=None` is
    threaded through unchanged. Returns `(Z_train, Z_valid, y_train, y_valid)`
    where the validation halves are `None` when `validation_fraction <= 0`."""
    n = len(Z) if _is_dataframe(Z) else len(np.asarray(Z))
    if validation_fraction <= 0:
        return Z, None, y, None
    idx = np.arange(n)
    tr_idx, va_idx = train_test_split(
        idx, test_size=validation_fraction, random_state=random_state
    )
    if _is_dataframe(Z):
        Z_train, Z_valid = Z.iloc[tr_idx], Z.iloc[va_idx]
    else:
        arr = np.asarray(Z)
        Z_train, Z_valid = arr[tr_idx], arr[va_idx]
    if y is None:
        return Z_train, Z_valid, None, None
    if hasattr(y, "iloc"):
        y_train, y_valid = y.iloc[tr_idx], y.iloc[va_idx]
    else:
        y_arr = np.asarray(y)
        y_train, y_valid = y_arr[tr_idx], y_arr[va_idx]
    return Z_train, Z_valid, y_train, y_valid


class RieszEstimator(BaseEstimator):
    """Generic sklearn-compatible orchestrator. Implementation packages
    typically expose a thin subclass with a default `backend` baked in
    (e.g. `RieszBooster` defaults to `XGBoostBackend()`).

    Parameters
    ----------
    estimand : Estimand
        Carries `feature_keys` and the `m(alpha)(z, y)` operator. Required.
    backend : Backend
        Concrete backend implementing the `fit_augmented` (or `fit_rows`)
        Protocol. Required (no default at this level). All learner-specific
        knobs (`n_estimators`, `learning_rate`, `early_stopping_rounds`,
        NN training-loop config, kernel choice, etc.) live on the backend
        constructor — see DESIGN.md §A.1 (the agnostic-orchestrator rule).
    loss : LossSpec, default=None
        The Bregman-Riesz loss to minimize. Defaults to `SquaredLoss()`.
    init : float or None
        α-space initialization. ``None`` (default) sets α to the constant
        that minimizes the empirical Riesz loss — namely ``m̄ = E[m(Z, 1)]``
        on the training rows, projected into the loss's α-domain. Pass an
        explicit float to override (e.g. ``init=0`` for hard-zero start).
    random_state : int, default=0
    """

    def __init__(
        self,
        estimand: Estimand,
        backend: Backend | None = None,
        loss: LossSpec | None = None,
        init: float | str | None = None,
        random_state: int = 0,
    ):
        self.estimand = estimand
        self.backend = backend
        self.loss = loss
        self.init = init
        self.random_state = random_state

    # ---- internal accessors that resolve defaults / hyperparams ----

    def _resolved_backend(self) -> Backend:
        if self.backend is None:
            raise ValueError(
                "RieszEstimator requires a `backend=`. Either pass one explicitly "
                "or use a subclass (e.g. RieszBooster) that bakes in a default."
            )
        return self.backend

    def _resolved_loss(self) -> LossSpec:
        return self.loss if self.loss is not None else SquaredLoss()

    def _backend_hyperparams(self) -> dict:
        """Backend-specific hyperparameters routed via `hyperparams=`. Subclasses
        override to surface their own knobs (max_depth, reg_lambda, etc.)."""
        return {}

    # ---- sklearn API ----

    def fit(self, Z, y=None, eval_set=None, eval_y=None) -> "RieszEstimator":
        """Fit the Riesz representer.

        `Z` is the predictor matrix — treatment column(s) plus covariates
        in `estimand.feature_keys` order. `y` is sklearn-style: a separate
        per-row outcome vector. It is plumbed into the estimand's
        `m(alpha)(z, y)` and into the augmentation / moment backends.
        Built-in factories (`ATE`, `ATT`, `TSM`, `AdditiveShift`,
        `LocalShift`) ignore `y`; pass it anyway when following sklearn
        convention. Custom Y-dependent estimands require `y` to be
        provided.

        `eval_set` is the held-out predictor matrix for early-stopping /
        λ-selection (when the backend uses one); pair it with `eval_y` to
        feed an outcome vector for the same rows.
        """
        warn_if_multi_backend_omp()
        loss = self._resolved_loss()
        backend = self._resolved_backend()

        if not isinstance(self.estimand, FiniteEvalEstimand):
            raise TypeError(
                f"RieszEstimator.fit() requires a FiniteEvalEstimand; got "
                f"{type(self.estimand).__name__}. Use a built-in factory "
                "(ATE, ATT, TSM, ...) or `FiniteEvalEstimand(feature_keys=..., m=...)`."
            )

        # Resolve validation slice. Backends that use a held-out slice for
        # fit-time logic (early stopping, λ selection) expose
        # `validation_fraction` as a constructor attribute; the orchestrator
        # reads it via getattr and performs the split before augmentation.
        val_frac = float(getattr(backend, "validation_fraction", 0.0) or 0.0)
        if eval_set is not None:
            Z_train, Z_valid = Z, eval_set
            y_train, y_valid = y, eval_y
        elif val_frac > 0:
            Z_train, Z_valid, y_train, y_valid = _split_Z(
                Z, y, val_frac, self.random_state
            )
        else:
            Z_train, Z_valid = Z, None
            y_train, y_valid = y, None

        rows_train = _rows_from_Z(Z_train, self.estimand)
        ys_train = _ys_from_y(y_train, len(rows_train))
        rows_valid = (
            _rows_from_Z(Z_valid, self.estimand)
            if Z_valid is not None and len(Z_valid) > 0
            else None
        )
        ys_valid = (
            _ys_from_y(y_valid, len(rows_valid))
            if rows_valid is not None
            else None
        )

        # Resolve init in α-space, then convert to η.
        # Default (init=None): use the constant that minimizes the empirical
        # Riesz loss. For any Bregman loss with strictly convex φ this is
        # m̄ = E[m(Z, 1)] (FOC ψ'(a) = φ''(a)·m̄ collapses to a = m̄ via
        # ψ'(t) = t·φ''(t)). Each loss projects m̄ into its α-domain.
        init_arg = self.init
        if init_arg is None:
            if ys_train is None:
                m_bar = float(np.mean(
                    [sum(c for c, _ in trace(self.estimand, z)) for z in rows_train]
                ))
            else:
                m_bar = float(np.mean(
                    [
                        sum(c for c, _ in trace(self.estimand, z, y_i))
                        for z, y_i in zip(rows_train, ys_train)
                    ]
                ))
            init_alpha = loss.best_constant_init(m_bar)
        elif isinstance(init_arg, (int, float)):
            init_alpha = float(init_arg)
        else:
            raise ValueError(f"init must be float or None; got {init_arg!r}")
        base_score = float(loss.alpha_to_eta(init_alpha))

        common_kwargs = dict(
            base_score=base_score,
            random_state=self.random_state,
            hyperparams=self._backend_hyperparams(),
        )

        # Dispatch: moment-style backends consume rows + estimand directly.
        # Augmentation-style backends receive a precomputed AugmentedDataset.
        # Backends implementing both default to fit_augmented for back-compat.
        uses_moment_path = hasattr(backend, "fit_rows") and not hasattr(backend, "fit_augmented")
        if uses_moment_path:
            result = backend.fit_rows(
                rows_train,
                rows_valid,
                self.estimand,
                loss,
                ys_train=ys_train,
                ys_valid=ys_valid,
                **common_kwargs,
            )
        else:
            aug_train = build_augmented(rows_train, self.estimand, ys_train)
            aug_valid = (
                build_augmented(rows_valid, self.estimand, ys_valid)
                if rows_valid
                else None
            )
            result = backend.fit_augmented(aug_train, aug_valid, loss, **common_kwargs)

        self.predictor_ = result.predictor
        self.best_iteration_ = result.best_iteration
        self.best_score_ = result.best_score
        self.base_score_ = base_score
        self.loss_ = loss
        self.feature_keys_ = self.estimand.feature_keys
        return self

    def predict(self, Z) -> np.ndarray:
        if not hasattr(self, "predictor_"):
            raise RuntimeError(f"{type(self).__name__} is not fitted yet. Call .fit() first.")
        rows = _rows_from_Z(Z, self.estimand)
        feats = _features_from_rows(rows, self.estimand)
        return self.predictor_.predict_alpha(feats)

    def riesz_loss(self, Z, y=None) -> float:
        """Per-row empirical Riesz loss on Z under this estimator's loss.

        Pass `y` when the estimand's `m` reads it (most built-ins do not).
        """
        if not hasattr(self, "predictor_"):
            raise RuntimeError(f"{type(self).__name__} is not fitted yet.")
        rows = _rows_from_Z(Z, self.estimand)
        ys = _ys_from_y(y, len(rows))
        aug = build_augmented(rows, self.estimand, ys)
        eta = self.predictor_.predict_eta(aug.features)
        alpha = self.loss_.link_to_alpha(eta)
        return float(
            np.sum(aug_loss_alpha(self.loss_, aug.is_original, aug.potential_deriv_coef, alpha))
            / aug.n_rows
        )

    def score(self, Z, y=None) -> float:
        """Return negative held-out canonical Riesz loss (squared loss).

        Following sklearn convention (R² for regressors, accuracy for
        classifiers), `score()` evaluates a fixed yardstick — squared Riesz
        loss — independent of the loss the estimator was trained with. This
        makes `cross_val_score` and `GridSearchCV` results comparable across
        estimators fit with different losses (e.g. KL vs squared).

        Pass `scoring=riesz_scorer(loss=...)` to sklearn CV utilities to use a
        different yardstick. `riesz_loss(Z)` remains available as the
        own-loss diagnostic. `y` is plumbed into `m(alpha)(z, y)` for
        Y-dependent custom estimands.
        """
        if not hasattr(self, "predictor_"):
            raise RuntimeError(f"{type(self).__name__} is not fitted yet.")
        rows = _rows_from_Z(Z, self.estimand)
        ys = _ys_from_y(y, len(rows))
        aug = build_augmented(rows, self.estimand, ys)
        eta = self.predictor_.predict_eta(aug.features)
        alpha_hat = self.loss_.link_to_alpha(eta)
        yardstick = SquaredLoss()
        return -float(
            np.sum(aug_loss_alpha(yardstick, aug.is_original, aug.potential_deriv_coef, alpha_hat))
            / aug.n_rows
        )

    def diagnose(self, Z, **kwargs):
        from .diagnostics import diagnose
        return diagnose(estimator=self, Z=Z, **kwargs)

    # ---- serialization ----

    def save(self, path) -> None:
        """Save a fitted estimator to a directory.

        Writes:
          - the predictor's binary payload (whatever format the backend uses —
            native model files, joblib pickles, torch state_dicts, etc.) via
            `predictor.save(dir_path)`
          - `metadata.json` with the loss spec, estimand factory_spec (if
            built-in), feature_keys, base_score, best_iteration_, and the
            estimator's constructor hyperparameters.

        Custom (non-built-in) estimands cannot be auto-reconstructed; the file
        will save fine, but `.load(path)` will require the user to pass
        `estimand=...` explicitly.
        """
        import json
        from pathlib import Path

        if not hasattr(self, "predictor_"):
            raise RuntimeError(
                f"Cannot save unfitted {type(self).__name__}. Call .fit() first."
            )

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.predictor_.save(path)

        metadata = {
            "rieszreg_format_version": 1,
            "predictor_kind": self.predictor_.kind,
            "loss": self.loss_.to_spec(),
            "estimand_factory_spec": self.estimand.factory_spec,  # None if custom
            "feature_keys": list(self.feature_keys_),
            "base_score": self.base_score_,
            "best_iteration": self.best_iteration_,
            "best_score": self.best_score_,
            "estimator_class": type(self).__name__,
            "hyperparameters": self._save_hyperparameters(),
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _save_hyperparameters(self) -> dict:
        """Snapshot of constructor args for round-trip. Subclasses extend."""
        return {
            "init": self.init,
            "random_state": self.random_state,
        }

    @classmethod
    def load(cls, path, *, estimand: Estimand | None = None) -> "RieszEstimator":
        """Load an estimator from a directory written by `save(...)`.

        For custom (non-built-in) estimands, pass `estimand=` to inject the
        original Estimand instance. For built-ins, reconstruction is automatic.

        The predictor binary is loaded via the registered predictor-loader for
        its `kind` (see `rieszreg.backends.register_predictor_loader`).
        Implementation packages register loaders at import time, so importing
        the relevant package (e.g. `import rieszboost`) is enough.
        """
        import json
        from pathlib import Path

        path = Path(path)
        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        loss = loss_from_spec(metadata["loss"])

        if estimand is None:
            spec = metadata.get("estimand_factory_spec")
            if spec is None:
                raise ValueError(
                    f"Saved estimator at {path} has a custom (non-built-in) "
                    "estimand. Pass `estimand=...` explicitly to "
                    f"{cls.__name__}.load(path, estimand=my_estimand)."
                )
            estimand = estimand_from_spec(spec)

        predictor = load_predictor(
            metadata["predictor_kind"],
            path,
            base_score=metadata["base_score"],
            loss=loss,
            best_iteration=metadata.get("best_iteration"),
        )

        hp = metadata.get("hyperparameters", {})
        instance = cls._construct_for_load(estimand=estimand, loss=loss, hyperparameters=hp)
        instance.predictor_ = predictor
        instance.best_iteration_ = metadata.get("best_iteration")
        instance.best_score_ = metadata.get("best_score")
        instance.base_score_ = metadata["base_score"]
        instance.loss_ = loss
        instance.feature_keys_ = tuple(metadata["feature_keys"])
        return instance

    @classmethod
    def _construct_for_load(cls, *, estimand, loss, hyperparameters: dict):
        """Build an unfit instance from saved hyperparameters. Subclasses
        override to consume their additional knobs."""
        return cls(
            estimand=estimand,
            loss=loss,
            init=hyperparameters.get("init"),
            random_state=hyperparameters.get("random_state", 0),
        )
