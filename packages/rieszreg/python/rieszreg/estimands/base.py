"""Estimand base + concrete subclasses.

`Estimand` is the abstract base. Concrete usage goes through `FiniteEvalEstimand`,
the subclass for estimands whose `m` reduces to a finite linear combination of
point evaluations of `alpha` (ATE, ATT, TSM, additive shifts, ...). Every
built-in subclass (`ATE`, `ATT`, `TSM`, `AdditiveShift`, `LocalShift`) inherits
from `FiniteEvalEstimand` and adds a vectorised `augment(features)` override.

Each estimand carries (1) the column names alpha is indexed by (`feature_keys`),
(2) the `m(alpha)(z, y)` operator, and (3) an `augment(features, ys=None)`
method that produces the augmented dataset for the orchestrator. Custom
user estimands instantiate `FiniteEvalEstimand` directly and inherit the
default `augment()` implementation, which traces `m` row-by-row. Built-in
subclasses override `augment()` with vectorised numpy.

`m` is an operator: it takes a candidate function `alpha` and returns a function
of the row `z` and the per-row outcome `y`. The default `augment()` calls
`m(alpha)(z, y)` row-by-row, passing a `Tracer` for `alpha` to extract the
linear-form structure. `Y` flows in sklearn-style: separate from `Z` at every
layer (no outcome column inside the row dict). When the user's `m` doesn't read
`y` (the case for every built-in), the inner closure ignores its second arg.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from ..augmentation import AugmentedDataset


class Estimand:
    """Abstract base class for estimands.

    Do not construct directly — use `FiniteEvalEstimand` for the finite-evaluation
    case (every estimand currently supported by `rieszreg`). Future subclasses
    may handle estimands outside the finite-evaluation algebra (integrals,
    derivatives without a finite-difference reduction, etc.).
    """

    pass


class FiniteEvalEstimand(Estimand):
    """Estimand whose `m(alpha)(z, y)` is a finite linear combination of point
    evaluations of `alpha`. The tracer extracts the (coefficient, point) pairs;
    the augmentation engine uses them to build the augmented dataset.

    Subclasses (`ATE`, `ATT`, `TSM`, `AdditiveShift`, `LocalShift`) override
    `augment()` with vectorised numpy. Custom estimands instantiate this class
    directly and use the default Tracer-based `augment()`.
    """

    name: str = "custom"

    def __init__(
        self,
        *,
        feature_keys: Sequence[str],
        m: Callable[..., Any],
        name: str | None = None,
        factory_spec: dict | None = None,
    ):
        self.feature_keys = tuple(feature_keys)
        self.m = m
        if name is not None:
            self.name = name
        self.factory_spec = factory_spec

    def bind(self, columns) -> "FiniteEvalEstimand":
        """Return the estimand with its input columns resolved against the
        data: ``columns`` is a DataFrame's column names, or an ndarray's
        column count. Custom estimands have fixed ``feature_keys`` and return
        themselves; built-ins with ``covariates=None`` return a copy whose
        covariates are every non-treatment column."""
        return self

    def __eq__(self, other) -> bool:
        if not isinstance(other, FiniteEvalEstimand):
            return NotImplemented
        # Built-in estimands compare by factory_spec — two `ATE()` calls
        # produce different `m` closures but represent the same functional.
        if self.factory_spec is not None or other.factory_spec is not None:
            return self.factory_spec == other.factory_spec
        # Custom estimands fall back to identity-on-`m` plus structural fields.
        return (
            self.feature_keys == other.feature_keys
            and self.name == other.name
            and self.m is other.m
        )

    def __hash__(self) -> int:
        if self.factory_spec is not None:
            import json
            return hash(json.dumps(self.factory_spec, sort_keys=True, default=str))
        return hash((self.feature_keys, self.name, id(self.m)))

    # ---- Augmentation ----

    def _normalise_features(self, features, ys) -> tuple[np.ndarray, int]:
        """Coerce `features` to a contiguous float ndarray and validate ys
        length. Subclass overrides use this to share input handling with the
        base default."""
        features = np.asarray(features, dtype=float)
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        n = features.shape[0]
        if ys is not None and len(ys) != n:
            raise ValueError(
                f"len(ys)={len(ys)} does not match number of rows ({n})."
            )
        return features, n

    def augment(self, features: np.ndarray, ys: Sequence[Any] | None = None) -> AugmentedDataset:
        """Build the augmented dataset by tracing `m(alpha)(z, y)` row-by-row.

        Subclasses override with vectorised emitters. The default implementation
        here is the symbolic Tracer path used by custom estimands.

        `features` is an (n, p) ndarray with columns in `self.feature_keys` order.
        `ys` is the per-row outcome aligned with the rows; pass `None` when
        the estimand's `m` doesn't read y. When provided, its length must
        match `features.shape[0]`.
        """
        from .tracer import trace  # deferred to break the base ↔ tracer cycle
        features, n = self._normalise_features(features, ys)

        keys: list[tuple] = []
        is_orig_list: list[float] = []
        pdc_list: list[float] = []
        origin: list[int] = []

        for i in range(n):
            z_key = tuple(features[i])
            z = dict(zip(self.feature_keys, z_key))
            acc: dict[tuple, list[float]] = {z_key: [1.0, 0.0]}
            for coef, point in trace(self, z, None if ys is None else ys[i]):
                missing = [k for k in self.feature_keys if k not in point]
                if missing:
                    raise ValueError(
                        f"m evaluated alpha at a point missing keys {missing}; "
                        f"all feature_keys {list(self.feature_keys)} must be specified."
                    )
                key = tuple(float(point[k]) for k in self.feature_keys)
                acc.setdefault(key, [0.0, 0.0])[1] -= coef
            for key, (d, c) in acc.items():
                keys.append(key)
                is_orig_list.append(d)
                pdc_list.append(c)
                origin.append(i)

        return AugmentedDataset(
            features=np.array(keys, dtype=float).reshape(len(keys), len(self.feature_keys)),
            is_original=np.asarray(is_orig_list, dtype=float),
            potential_deriv_coef=np.asarray(pdc_list, dtype=float),
            origin_index=np.asarray(origin, dtype=np.int64),
            n_rows=n,
        )


def _rebuild_builtin(cls, spec_args):
    return cls(**spec_args)


# ---------------------------------------------------------------------------
# Built-in subclasses. Each provides:
#   - __init__ forwarding its own args (level, delta, ...) to `_BuiltinEstimand`,
#     which stores treatment / covariate names and seeds `factory_spec`.
#   - `_m(alpha)` building the functional. It reads covariates from whatever
#     keys the row carries, so it works before and after `bind`.
#   - `augment(features, ys=None)` override that emits augmented rows in
#     vectorised numpy. Row order is implementation-defined and not part of
#     the public contract.


class _BuiltinEstimand(FiniteEvalEstimand):
    """Shared plumbing for the built-in estimands.

    ``treatment`` names the treatment column (``None`` for estimands with no
    treatment). ``covariates`` names the covariate columns; ``None`` means
    "every other column of the data", resolved at fit time by :meth:`bind`.
    The treatment is always column 0 of ``feature_keys``; ``augment`` reads it
    positionally, so an unbound estimand augments ndarray input directly.
    """

    def __init__(self, treatment: str | None, covariates, **args):
        if isinstance(covariates, str):
            covariates = (covariates,)
        cov = None if covariates is None else tuple(covariates)
        self.treatment = treatment
        self.covariates = cov
        for k, v in args.items():
            setattr(self, k, v)
        spec_args = dict(args)
        if treatment is not None:
            spec_args["treatment"] = treatment
        spec_args["covariates"] = None if cov is None else list(cov)
        self._spec_args = spec_args
        label = ", ".join(f"{k}={v!r}" for k, v in args.items())
        # Only the registered built-ins round-trip by name through save/load.
        # A user subclass (class MyShift(AdditiveShift)) is saved as custom.
        builtin = _FACTORY_REGISTRY.get(type(self).__name__) is type(self)
        super().__init__(
            feature_keys=() if cov is None else (*self._treatment_keys(), *cov),
            m=self._m,
            name=f"{type(self).__name__}({label})" if label else type(self).__name__,
            factory_spec={"factory": type(self).__name__, "args": spec_args} if builtin else None,
        )

    # Identity is (class, constructor args), so pickling, deepcopy and
    # sklearn.clone preserve subclasses of the built-ins.
    def __eq__(self, other) -> bool:
        if not isinstance(other, _BuiltinEstimand):
            return NotImplemented
        return type(self) is type(other) and self._spec_args == other._spec_args

    def __hash__(self) -> int:
        import json
        return hash((type(self), json.dumps(self._spec_args, sort_keys=True, default=str)))

    def __reduce__(self):
        return (_rebuild_builtin, (type(self), self._spec_args))

    def _treatment_keys(self) -> tuple[str, ...]:
        return () if self.treatment is None else (self.treatment,)

    def _covariate_values(self, z) -> dict:
        return {k: v for k, v in z.items() if k != self.treatment}

    def _binary_treatment(self, a: np.ndarray) -> np.ndarray:
        """Return the treated mask, raising if the treatment isn't coded 0/1."""
        bad = np.setdiff1d(a, (0.0, 1.0))
        if bad.size:
            raise ValueError(
                f"{self.name} needs the treatment column {self.treatment!r} coded "
                f"0/1; found values {bad[:5].tolist()}. Recode it (e.g. treated = 1, "
                "control = 0) before fitting."
            )
        return a == 1.0

    def bind(self, columns) -> "_BuiltinEstimand":
        if self.covariates is not None:
            return self
        t = self._treatment_keys()
        if isinstance(columns, int):
            # ndarray input: treatment is column 0, the rest are covariates.
            cov = [f"x{j}" for j in range(columns - len(t))]
        else:
            columns = [str(c) for c in columns]
            if self.treatment is not None and str(self.treatment) not in columns:
                raise ValueError(
                    f"{self.name} needs a treatment column named {self.treatment!r}, "
                    f"but the data has columns {columns}. Tell the estimand which "
                    f"column is the treatment, e.g. {type(self).__name__}("
                    f"treatment={columns[0]!r})."
                )
            cov = [c for c in columns if c != str(self.treatment)]
        return type(self)(**{**self._spec_args, "covariates": cov})


class ATE(_BuiltinEstimand):
    """Average treatment effect: m(α)(z, y) = α(1, x) − α(0, x).

    Parameters
    ----------
    treatment : str, default="a"
        Name of the binary (0/1) treatment column.
    covariates : sequence of str or None, default=None
        Names of the covariate columns. ``None`` uses every column of the
        data other than ``treatment``. For ndarray input, column 0 is the
        treatment and the remaining columns are the covariates.
    """

    def __init__(self, treatment: str = "a", covariates: Sequence[str] | None = None):
        super().__init__(treatment, covariates)

    def _m(self, alpha):
        def inner(z, y=None):
            x = self._covariate_values(z)
            return alpha(**{self.treatment: 1, **x}) - alpha(**{self.treatment: 0, **x})
        return inner

    def augment(self, features, ys=None):
        features, n = self._normalise_features(features, ys)
        is_treated = self._binary_treatment(features[:, 0])  # treatment is column 0
        aug = np.vstack([features, features])
        aug[:n, 0] = 1.0
        aug[n:, 0] = 0.0
        return AugmentedDataset(
            features=aug,
            is_original=np.concatenate([is_treated, ~is_treated]).astype(float),
            potential_deriv_coef=np.concatenate([np.full(n, -1.0), np.full(n, 1.0)]),
            origin_index=np.tile(np.arange(n, dtype=np.int64), 2),
            n_rows=n,
        )


class ATT(_BuiltinEstimand):
    """ATT *partial-estimand* surface: m(α)(z, y) = a · (α(1, x) − α(0, x)).

    Full ATT divides by P(A=1) and is not a Riesz functional — combine
    α̂_partial with a delta-method EIF (Hubbard 2011) downstream.
    ``treatment`` and ``covariates`` work as in :class:`ATE`.
    """

    def __init__(self, treatment: str = "a", covariates: Sequence[str] | None = None):
        super().__init__(treatment, covariates)

    def _m(self, alpha):
        def inner(z, y=None):
            x = self._covariate_values(z)
            return z[self.treatment] * (
                alpha(**{self.treatment: 1, **x}) - alpha(**{self.treatment: 0, **x})
            )
        return inner

    def augment(self, features, ys=None):
        features, n = self._normalise_features(features, ys)
        a_idx = 0  # treatment is always column 0 of feature_keys
        treated_mask = self._binary_treatment(features[:, a_idx])
        control = features[~treated_mask]
        treated = features[treated_mask]
        treated_1 = treated.copy()
        treated_1[:, a_idx] = 1.0
        treated_0 = treated.copy()
        treated_0[:, a_idx] = 0.0
        return AugmentedDataset(
            features=np.vstack([control, treated_1, treated_0]),
            is_original=np.concatenate([
                np.ones(len(control)),
                np.ones(len(treated)),
                np.zeros(len(treated)),
            ]),
            potential_deriv_coef=np.concatenate([
                np.zeros(len(control)),
                np.full(len(treated), -1.0),
                np.full(len(treated), 1.0),
            ]),
            origin_index=np.concatenate([
                np.where(~treated_mask)[0],
                np.where(treated_mask)[0],
                np.where(treated_mask)[0],
            ]).astype(np.int64),
            n_rows=n,
        )


class TSM(_BuiltinEstimand):
    """Treatment-specific mean: m(α)(z, y) = α(level, x).

    ``level`` is the treatment value to evaluate at. ``treatment`` and
    ``covariates`` work as in :class:`ATE`.
    """

    def __init__(self, level, treatment: str = "a", covariates: Sequence[str] | None = None):
        super().__init__(treatment, covariates, level=level)

    def _m(self, alpha):
        def inner(z, y=None):
            return alpha(**{self.treatment: self.level, **self._covariate_values(z)})
        return inner

    def augment(self, features, ys=None):
        features, n = self._normalise_features(features, ys)
        a_idx = 0  # treatment is always column 0 of feature_keys
        a = features[:, a_idx]
        eq_mask = (a == self.level)
        eq = features[eq_mask]
        neq = features[~eq_mask]
        neq_L = neq.copy()
        neq_L[:, a_idx] = self.level
        return AugmentedDataset(
            features=np.vstack([eq, neq, neq_L]),
            is_original=np.concatenate([
                np.ones(len(eq)),
                np.ones(len(neq)),
                np.zeros(len(neq)),
            ]),
            potential_deriv_coef=np.concatenate([
                np.full(len(eq), -1.0),
                np.zeros(len(neq)),
                np.full(len(neq), -1.0),
            ]),
            origin_index=np.concatenate([
                np.where(eq_mask)[0],
                np.where(~eq_mask)[0],
                np.where(~eq_mask)[0],
            ]).astype(np.int64),
            n_rows=n,
        )


class AdditiveShift(_BuiltinEstimand):
    """Additive shift effect: m(α)(z, y) = α(a + δ, x) − α(a, x).

    ``delta`` is the (non-zero) amount added to a continuous treatment.
    ``treatment`` and ``covariates`` work as in :class:`ATE`.
    """

    def __init__(self, delta: float, treatment: str = "a", covariates: Sequence[str] | None = None):
        if delta == 0:
            raise ValueError("AdditiveShift requires delta != 0 (delta=0 is a degenerate, vacuous estimand).")
        super().__init__(treatment, covariates, delta=delta)

    def _m(self, alpha):
        def inner(z, y=None):
            a, x = z[self.treatment], self._covariate_values(z)
            return alpha(**{self.treatment: a + self.delta, **x}) - alpha(**{self.treatment: a, **x})
        return inner

    def augment(self, features, ys=None):
        features, n = self._normalise_features(features, ys)
        a_idx = 0  # treatment is always column 0 of feature_keys
        original = features
        shifted = features.copy()
        shifted[:, a_idx] = features[:, a_idx] + self.delta
        return AugmentedDataset(
            features=np.vstack([original, shifted]),
            is_original=np.concatenate([np.ones(n), np.zeros(n)]),
            potential_deriv_coef=np.concatenate([np.full(n, 1.0), np.full(n, -1.0)]),
            origin_index=np.tile(np.arange(n, dtype=np.int64), 2),
            n_rows=n,
        )


class LocalShift(_BuiltinEstimand):
    """LASE *partial-estimand* surface: m(α)(z, y) = 1(a < threshold) · (α(a+δ, x) − α(a, x)).

    Full LASE divides by P(A < threshold) and is not a Riesz functional.
    ``treatment`` and ``covariates`` work as in :class:`ATE`.
    """

    def __init__(
        self,
        delta: float,
        threshold: float,
        treatment: str = "a",
        covariates: Sequence[str] | None = None,
    ):
        if delta == 0:
            raise ValueError("LocalShift requires delta != 0 (delta=0 is a degenerate, vacuous estimand).")
        super().__init__(treatment, covariates, delta=delta, threshold=threshold)

    def _m(self, alpha):
        def inner(z, y=None):
            a = z[self.treatment]
            if a >= self.threshold:
                return 0
            x = self._covariate_values(z)
            return alpha(**{self.treatment: a + self.delta, **x}) - alpha(**{self.treatment: a, **x})
        return inner

    def augment(self, features, ys=None):
        features, n = self._normalise_features(features, ys)
        a_idx = 0  # treatment is always column 0 of feature_keys
        a = features[:, a_idx]
        below_mask = (a < self.threshold)
        above = features[~below_mask]
        below = features[below_mask]
        below_shifted = below.copy()
        below_shifted[:, a_idx] = below[:, a_idx] + self.delta
        return AugmentedDataset(
            features=np.vstack([above, below, below_shifted]),
            is_original=np.concatenate([
                np.ones(len(above)),
                np.ones(len(below)),
                np.zeros(len(below)),
            ]),
            potential_deriv_coef=np.concatenate([
                np.zeros(len(above)),
                np.full(len(below), 1.0),
                np.full(len(below), -1.0),
            ]),
            origin_index=np.concatenate([
                np.where(~below_mask)[0],
                np.where(below_mask)[0],
                np.where(below_mask)[0],
            ]).astype(np.int64),
            n_rows=n,
        )


class OutcomeRegNormSq(_BuiltinEstimand):
    """Squared L² norm of the outcome regression: θ_0 = E[μ_0(X)²].

    Moment functional `m(α)(z, y) = α(x) · y` has Riesz representer μ_0(x) =
    E[Y | X=x]. Under the squared Bregman-Riesz loss the empirical objective
    collapses to ∑ (α(x_i) − y_i)², so Riesz training reproduces standard MSE
    regression — a parity check against stock sklearn/xgboost/torch regressors.
    ``covariates=None`` uses every column of the data. Requires ``y`` at fit.
    """

    def __init__(self, covariates: Sequence[str] | None = None):
        super().__init__(None, covariates)

    def _m(self, alpha):
        def inner(z, y):
            return alpha(**z) * y
        return inner

    def augment(self, features, ys=None):
        features, n = self._normalise_features(features, ys)
        if ys is None:
            raise ValueError("OutcomeRegNormSq needs the outcome: call fit(X, y).")
        y = np.asarray(ys, dtype=float)
        return AugmentedDataset(
            features=features,
            is_original=np.ones(n, dtype=float),
            potential_deriv_coef=-y,
            origin_index=np.arange(n, dtype=np.int64),
            n_rows=n,
        )


# Registry for round-tripping. Updated when new built-in subclasses are added.
_FACTORY_REGISTRY: dict[str, type] = {
    "ATE": ATE,
    "ATT": ATT,
    "TSM": TSM,
    "AdditiveShift": AdditiveShift,
    "LocalShift": LocalShift,
    "OutcomeRegNormSq": OutcomeRegNormSq,
}


def estimand_from_spec(spec: dict) -> FiniteEvalEstimand:
    """Reconstruct a FiniteEvalEstimand from its `factory_spec` dict. Only
    built-in subclasses round-trip; custom estimands must be re-passed at load
    time."""
    factory_name = spec["factory"]
    if factory_name not in _FACTORY_REGISTRY:
        raise ValueError(
            f"Unknown estimand factory {factory_name!r}; only built-ins "
            f"({sorted(_FACTORY_REGISTRY)}) are round-trippable. For custom "
            f"estimands, pass `estimand=...` explicitly to .load(...)."
        )
    return _FACTORY_REGISTRY[factory_name](**spec.get("args", {}))
