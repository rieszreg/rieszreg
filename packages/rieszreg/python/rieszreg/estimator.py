"""Sklearn-compatible orchestrator for Riesz representer estimation.

`RieszEstimator` takes (estimand, loss, backend) at construction and implements
the standard sklearn `fit / predict / score` API. Learner-specific
hyperparameters live on subclasses (e.g. `RieszBooster` in `rieszboost` adds
`max_depth`, `reg_lambda`, `subsample`).

Composes with `sklearn.model_selection.GridSearchCV`, `cross_val_predict`,
`clone`, `Pipeline`, etc.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from importlib import import_module
from numbers import Real
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from ._omp import warn_if_multi_backend_omp
from .backends import Backend, load_predictor
from .estimands.base import Estimand, FiniteEvalEstimand, estimand_from_spec
from .losses import Loss, SquaredLoss, loss_from_spec


def _is_dataframe(Z) -> bool:
    return hasattr(Z, "columns") and hasattr(Z, "iloc")


def _n_columns(Z) -> int:
    if _is_dataframe(Z):
        return Z.shape[1]
    arr = np.asarray(Z)
    return 1 if arr.ndim == 1 else arr.shape[1]


def _features_from_Z(Z, estimand: Estimand) -> np.ndarray:
    """DataFrame/ndarray → ``feature_keys``-ordered float ndarray.

    DataFrame columns are matched by name (compared as strings, so integer
    column labels work); ndarray columns are taken in ``feature_keys`` order.
    """
    keys = list(estimand.feature_keys)
    if _is_dataframe(Z):
        position = {str(c): i for i, c in enumerate(Z.columns)}
        missing = [k for k in keys if str(k) not in position]
        if missing:
            raise ValueError(
                f"The data is missing columns {missing} needed by {estimand.name}; "
                f"it has columns {list(Z.columns)}. Tell the estimand your column "
                "names, e.g. ATE(treatment=\"treated\", covariates=[\"age\", \"income\"])."
            )
        return Z.iloc[:, [position[str(k)] for k in keys]].to_numpy(dtype=float)

    arr = np.asarray(Z, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[1] != len(keys):
        raise ValueError(
            f"Estimand {estimand.name!r} expects {len(keys)} feature columns "
            f"({estimand.feature_keys}), got Z.shape[1]={arr.shape[1]}."
        )
    return arr


def _ys_from_y(y, n: int) -> np.ndarray | None:
    """Coerce `y` into a flat float array aligned with the rows (None stays None)."""
    if y is None:
        return None
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if len(y_arr) != n:
        raise ValueError(
            f"len(y)={len(y_arr)} does not match number of rows in Z ({n})."
        )
    return y_arr


def _check_outcome_not_covariate(estimand, feats: np.ndarray, ys) -> None:
    """With ``covariates=None`` every non-treatment column is a covariate,
    including an outcome column left in Z, which biases α̂."""
    for j, key in enumerate(estimand.feature_keys):
        if key == getattr(estimand, "treatment", None):
            continue
        if ys is not None and np.array_equal(feats[:, j], ys):
            raise ValueError(
                f"Column {key!r} is the outcome y, but {estimand.name} is using it "
                "as a covariate (covariates=None uses every non-treatment column). "
                f"Drop it from Z or pass covariates=[...] to {estimand.name}."
            )
        if key.lower() in ("y", "outcome"):
            warnings.warn(
                f"{estimand.name} is using column {key!r} as a covariate "
                "(covariates=None uses every non-treatment column). If it is the "
                f"outcome, drop it from Z or pass covariates=[...] to {estimand.name}.",
                UserWarning,
                stacklevel=3,
            )


def _split_Z(Z, y, validation_fraction: float, random_state):
    """Split (Z, y) into train/valid by `validation_fraction`. `y=None` is
    threaded through unchanged. Returns `(Z_train, Z_valid, y_train, y_valid)`."""
    tr_idx, va_idx = train_test_split(
        np.arange(len(Z)), test_size=validation_fraction, random_state=random_state
    )

    def take(a, idx):
        return a.iloc[idx] if hasattr(a, "iloc") else np.asarray(a)[idx]

    ys = (None, None) if y is None else (take(y, tr_idx), take(y, va_idx))
    return take(Z, tr_idx), take(Z, va_idx), *ys


def _jsonable(value) -> bool:
    try:
        json.dumps(value, default=_json_default)
        return True
    except TypeError:
        return False


def _backend_spec(backend) -> dict | None:
    """``{"module", "qualname", "params"}`` for a dataclass backend whose
    fields are all JSON-serializable, else None (the backend isn't saved)."""
    if not dataclasses.is_dataclass(backend):
        return None
    params = {f.name: getattr(backend, f.name) for f in dataclasses.fields(backend) if f.init}
    cls = type(backend)
    return {"module": cls.__module__, "qualname": cls.__qualname__, "params": params} if _jsonable(params) else None


def _backend_from_spec(spec: dict):
    obj = import_module(spec["module"])
    for part in spec["qualname"].split("."):
        obj = getattr(obj, part)
    return obj(**spec["params"])


def _json_default(obj):
    """JSON fallback for numpy scalars / arrays in saved metadata."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")


class RieszEstimator(BaseEstimator):
    """Estimate the Riesz representer α₀ of a causal estimand.

    Most users start from a learner package's subclass, which picks the
    backend for you: `RieszBooster` (rieszboost), `KernelRieszRegressor`
    (krrr), `ForestRieszRegressor` / `AugForestRieszRegressor` (forestriesz),
    `RieszNet` (riesznet), `RieszTreeRegressor` (riesztree). Use
    `RieszEstimator` directly to pair an estimand with a backend object.

    Parameters
    ----------
    estimand : Estimand
        What you want to estimate, e.g. ``ATE(treatment="treated")``. The
        estimand also names the treatment and covariate columns of ``X``.
    backend : Backend
        The learner that fits α̂ (e.g. ``XGBoostBackend()``). Required here;
        the learner-package subclasses supply one for you.
    loss : Loss, default=None
        The Bregman-Riesz loss to minimize. Defaults to `SquaredLoss()`.
    init : float or None
        α-space initialization. ``None`` (default) sets α to the constant
        that minimizes the empirical Riesz loss — namely ``m̄ = E[m(Z, 1)]``
        on the training rows, projected into the loss's α-domain. Pass an
        explicit float to override (e.g. ``init=0`` for hard-zero start).
    random_state : int, default=0
        Seed for every source of randomness in the fit (validation split,
        subsampling, weight initialization).

    Attributes
    ----------
    estimand_ : Estimand
        The estimand with its columns resolved against the training data.
    n_features_in_ : int
        Number of columns of ``X`` seen at fit.
    feature_names_in_ : ndarray of str
        Column names of ``X`` seen at fit. Defined only when ``X`` was a
        DataFrame. α̂ itself reads ``estimand_.feature_keys``.
    loss_ : Loss
        The loss used for fitting.
    """

    def __init__(
        self,
        estimand: Estimand,
        backend: Backend | None = None,
        loss: Loss | None = None,
        init: float | None = None,
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

    def _resolved_loss(self) -> Loss:
        return self.loss if self.loss is not None else SquaredLoss()

    def _backend_hyperparams(self) -> dict:
        """Backend-specific hyperparameters routed via `hyperparams=`. Subclasses
        override to surface their own knobs (max_depth, reg_lambda, etc.)."""
        return {}

    # ---- sklearn API ----

    def fit(self, Z, y=None, eval_set=None, eval_y=None) -> "RieszEstimator":
        """Fit the Riesz representer.

        Parameters
        ----------
        Z : DataFrame or ndarray of shape (n, p)
            The treatment column plus covariates (sklearn's ``X``). With a
            DataFrame, columns are matched by the names the estimand was
            given; with an ndarray, the treatment is column 0.
        y : array-like of shape (n,), optional
            The outcome. The built-in treatment estimands (``ATE``, ``ATT``,
            ``TSM``, ``AdditiveShift``, ``LocalShift``) do not use it — the
            Riesz representer depends only on treatment and covariates — so
            passing it is harmless. Estimands whose functional reads the
            outcome (``OutcomeRegNormSq``, custom Y-dependent ones) need it.
        eval_set : DataFrame or ndarray, optional
            Held-out rows for early stopping / λ selection, when the learner
            uses one. Overrides the learner's internal ``validation_fraction``.
        eval_y : array-like, optional
            Outcome for the ``eval_set`` rows.
        """
        warn_if_multi_backend_omp()
        loss = self._resolved_loss()
        backend = self._resolved_backend()

        if not isinstance(self.estimand, FiniteEvalEstimand):
            raise TypeError(
                f"estimand must be a built-in estimand (ATE, ATT, TSM, "
                f"AdditiveShift, LocalShift, OutcomeRegNormSq) or a "
                f"FiniteEvalEstimand(feature_keys=..., m=...); got "
                f"{type(self.estimand).__name__}."
            )
        estimand = self.estimand.bind(
            list(Z.columns) if _is_dataframe(Z) else _n_columns(Z)
        )

        # Resolve validation slice. Backends that use a held-out slice for
        # fit-time logic (early stopping, λ selection) expose
        # `validation_fraction` as a constructor attribute; the orchestrator
        # reads it via getattr and performs the split before augmentation.
        val_frac = float(getattr(backend, "validation_fraction", 0.0) or 0.0)
        if eval_set is not None:
            if _is_dataframe(Z) and not _is_dataframe(eval_set):
                warnings.warn(
                    "Z is a DataFrame but eval_set is an array without column "
                    f"names. Its columns are read in the order "
                    f"{list(estimand.feature_keys)}; pass a DataFrame to match "
                    "columns by name.",
                    UserWarning,
                    stacklevel=2,
                )
            Z_train, Z_valid = Z, eval_set
            y_train, y_valid = y, eval_y
        elif val_frac > 0:
            Z_train, Z_valid, y_train, y_valid = _split_Z(
                Z, y, val_frac, self.random_state
            )
        else:
            Z_train, Z_valid = Z, None
            y_train, y_valid = y, None

        feats_train = _features_from_Z(Z_train, estimand)
        ys_train = _ys_from_y(y_train, feats_train.shape[0])
        has_valid = Z_valid is not None and len(Z_valid) > 0
        if has_valid:
            feats_valid = _features_from_Z(Z_valid, estimand)
            ys_valid = _ys_from_y(y_valid, feats_valid.shape[0])
        else:
            feats_valid = ys_valid = None
        if getattr(self.estimand, "covariates", ()) is None:
            _check_outcome_not_covariate(estimand, feats_train, ys_train)
        estimand.check_support(feats_train)
        aug_train = estimand.augment(feats_train, ys=ys_train)
        loss.check_estimand(aug_train, estimand)
        aug_valid = estimand.augment(feats_valid, ys=ys_valid) if has_valid else None

        # init=None: the constant minimizing the empirical Riesz loss. For any
        # Bregman loss with strictly convex h that is m̄ = E[m(Z, 1)], and
        # Σ_r C_r = −Σ_i m(Z_i, 1), so m̄ falls out of the augmentation.
        if self.init is None:
            init_alpha = loss.best_constant_init(aug_train.m_bar)
        elif isinstance(self.init, Real):
            init_alpha = float(self.init)
        else:
            raise ValueError(f"init must be float or None; got {self.init!r}")
        base_score = float(loss.alpha_to_eta(init_alpha))

        common_kwargs = dict(
            base_score=base_score,
            random_state=self.random_state,
            hyperparams=self._backend_hyperparams(),
        )

        # Moment-style backends also take the feature arrays + estimand;
        # everything else (including backends implementing both) gets only
        # the augmented data.
        if hasattr(backend, "fit_rows") and not hasattr(backend, "fit_augmented"):
            result = backend.fit_rows(
                feats_train, feats_valid, estimand, loss,
                aug_train=aug_train, aug_valid=aug_valid, **common_kwargs,
            )
        else:
            result = backend.fit_augmented(aug_train, aug_valid, loss, **common_kwargs)

        self.predictor_ = result.predictor
        self.best_iteration_ = result.best_iteration
        self.best_score_ = result.best_score
        self.base_score_ = base_score
        self.loss_ = loss
        self.estimand_ = estimand
        self._set_input_attributes(Z)
        return self

    def _set_input_attributes(self, Z) -> None:
        self.n_features_in_ = _n_columns(Z)
        if _is_dataframe(Z):
            self.feature_names_in_ = np.asarray([str(c) for c in Z.columns], dtype=object)
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_

    def _features(self, Z) -> np.ndarray:
        """Check the estimator is fitted and pull α̂'s input columns from Z."""
        check_is_fitted(self, "predictor_")
        if hasattr(self, "feature_names_in_") and not _is_dataframe(Z):
            warnings.warn(
                f"{type(self).__name__} was fit on a DataFrame but got an array "
                f"without column names. Array columns are read in the order "
                f"{list(self.estimand_.feature_keys)}; pass a DataFrame to match "
                "columns by name.",
                UserWarning,
                stacklevel=3,
            )
        return _features_from_Z(Z, self.estimand_)

    def predict(self, Z) -> np.ndarray:
        """Return α̂ evaluated at each row of Z, shape ``(n,)``."""
        feats = self._features(Z)
        return self.predictor_.predict_alpha(feats)

    def _loss_on(self, Z, y, loss: Loss) -> float:
        """Mean per-row Riesz loss of the fitted α̂ on (Z, y) under ``loss``."""
        feats = self._features(Z)
        aug = self.estimand_.augment(feats, ys=_ys_from_y(y, feats.shape[0]))
        return aug.mean_loss(loss, self.predictor_.predict_alpha(aug.features))

    def riesz_loss(self, Z, y=None) -> float:
        """Mean per-row Riesz loss on (Z, y) under the loss the estimator was
        trained with. Lower is better. Pass ``y`` only when the estimand's
        functional reads the outcome."""
        return self._loss_on(Z, y, self.loss_)

    def score(self, Z, y=None) -> float:
        """Return negative held-out canonical Riesz loss (squared loss).

        Following sklearn convention (R² for regressors, accuracy for
        classifiers), `score()` evaluates a fixed yardstick — squared Riesz
        loss — independent of the loss the estimator was trained with. This
        makes `cross_val_score` and `GridSearchCV` results comparable across
        estimators fit with different losses (e.g. KL vs squared).

        Pass `scoring=riesz_scorer(loss=...)` to sklearn CV utilities to use a
        different yardstick. `riesz_loss(Z)` is the own-loss diagnostic.
        `y` is plumbed into `m(alpha)(z, y)` for Y-dependent estimands.
        """
        return -self._loss_on(Z, y, SquaredLoss())

    def diagnose(self, Z, y=None, **kwargs):
        """Health checks on α̂ over Z: magnitude, extreme values (a sign of
        poor overlap / near-positivity violations), and held-out Riesz loss.
        Returns a `Diagnostics` object; call ``.summary()`` for a report.
        Pass ``y`` when the estimand's functional reads the outcome. Keyword
        arguments (``extreme_threshold``, ``extreme_fraction_warn``) are
        forwarded to `rieszreg.diagnose`."""
        from .diagnostics import diagnose
        return diagnose(estimator=self, Z=Z, y=y, **kwargs)

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
        check_is_fitted(self, "predictor_")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.predictor_.save(path)

        metadata = {
            "rieszreg_format_version": 1,
            "predictor_kind": self.predictor_.kind,
            "loss": self.loss_.to_spec(),
            "estimand_factory_spec": self.estimand_.factory_spec,  # None if custom
            "feature_keys": list(self.estimand_.feature_keys),
            "base_score": self.base_score_,
            "best_iteration": self.best_iteration_,
            "best_score": self.best_score_,
            "n_features_in": self.n_features_in_,
            "feature_names_in": (
                list(self.feature_names_in_) if hasattr(self, "feature_names_in_") else None
            ),
            "estimator_class": type(self).__name__,
            "hyperparameters": self._save_hyperparameters(),
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=_json_default)

    def _save_hyperparameters(self) -> dict:
        """JSON-serializable constructor args for round-trip. Params that
        can't be written as JSON (callables, custom objects) are skipped and
        come back as their defaults; subclasses add special cases."""
        params = self.get_params(deep=False)
        hp = {k: v for k, v in params.items() if k not in ("estimand", "loss", "backend") and _jsonable(v)}
        backend = _backend_spec(params.get("backend"))
        if backend is not None:
            hp["backend"] = backend
        return hp

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
        # A passed-in built-in like ATE() has covariates=None; resolve it
        # against the columns the model was trained on.
        estimand = estimand.bind(metadata["feature_keys"])
        if tuple(estimand.feature_keys) != tuple(metadata["feature_keys"]):
            raise ValueError(
                f"The estimand passed to load() uses columns "
                f"{list(estimand.feature_keys)}, but the model at {path} was "
                f"fit on {metadata['feature_keys']}."
            )

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
        instance.estimand_ = estimand
        # Models saved before these keys existed: fall back to the estimand.
        instance.n_features_in_ = metadata.get("n_features_in", len(metadata["feature_keys"]))
        if metadata.get("feature_names_in") is not None:
            instance.feature_names_in_ = np.asarray(metadata["feature_names_in"], dtype=object)
        return instance

    @classmethod
    def _construct_for_load(cls, *, estimand, loss, hyperparameters: dict):
        """Build an unfit instance from saved hyperparameters."""
        names = set(cls._get_param_names())
        kwargs = {k: v for k, v in hyperparameters.items() if k in names}
        if "backend" in kwargs:
            kwargs["backend"] = _backend_from_spec(kwargs["backend"])
        return cls(estimand=estimand, loss=loss, **kwargs)
