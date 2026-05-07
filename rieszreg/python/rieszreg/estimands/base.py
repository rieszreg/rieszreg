"""Estimand base + concrete subclasses.

`Estimand` is the abstract base. Concrete usage goes through `FiniteEvalEstimand`,
the subclass for estimands whose `m` reduces to a finite linear combination of
point evaluations of `alpha` (ATE, ATT, TSM, additive shifts, ...). Every
built-in subclass (`ATE`, `ATT`, `TSM`, `AdditiveShift`, `LocalShift`) inherits
from `FiniteEvalEstimand` and adds a vectorised `augment(features)` override
plus a closed-form `m_bar`.

Each estimand carries (1) the column names alpha is indexed by (`feature_keys`),
(2) the `m(alpha)(z, y)` operator, and (3) an `augment(features, ys=None)`
method that produces the augmented dataset for the orchestrator. Custom
user estimands instantiate `FiniteEvalEstimand` directly and inherit the
default `augment()` implementation, which traces `m` row-by-row. Built-in
subclasses override `augment()` with vectorised numpy.

`m` is an operator: it takes a candidate function `alpha` and returns a function
of the row `z` and the per-row outcome `y`. The orchestrator calls
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
    `augment()` with vectorised numpy and set `m_bar` to the closed-form value
    of `E[m(alpha=1)(Z)]`. Custom estimands instantiate this class directly and
    use the default Tracer-based `augment()`.
    """

    name: str = "custom"
    # Closed-form m_bar = E[m(alpha=1)(Z)] for built-ins. None ⇒ compute empirically.
    m_bar: float | None = None

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

    def __call__(self, alpha):
        return self.m(alpha)

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

    def __reduce__(self):
        """Round-trip via the factory_spec for built-in estimands.

        Stock pickle / joblib can't serialize the closure `m` returned by a
        subclass __init__, so we redirect to `estimand_from_spec(...)` on
        unpickle. Custom estimands without a factory_spec fall back to a
        rebuild helper — that requires the user's `m` to be importable /
        picklable.
        """
        if self.factory_spec is not None:
            return (estimand_from_spec, (self.factory_spec,))
        return (
            _rebuild_custom_estimand,
            (self.feature_keys, self.m, self.name),
        )

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

        feats: list[np.ndarray] = []
        is_orig_list: list[float] = []
        pdc_list: list[float] = []
        origin: list[int] = []

        for i in range(n):
            z = {k: features[i, j] for j, k in enumerate(self.feature_keys)}
            y_i = ys[i] if ys is not None else None
            acc: dict[tuple, tuple[float, float]] = {}
            z_key = tuple(z[k] for k in self.feature_keys)
            acc[z_key] = (1.0, 0.0)

            for coef, point in trace(self, z, y_i):
                missing = [k for k in self.feature_keys if k not in point]
                if missing:
                    raise ValueError(
                        f"m evaluated alpha at a point missing keys {missing}; "
                        f"all feature_keys {list(self.feature_keys)} must be specified."
                    )
                key = tuple(point[k] for k in self.feature_keys)
                cur_d, cur_c = acc.get(key, (0.0, 0.0))
                acc[key] = (cur_d, cur_c - coef)

            for key, (d, c) in acc.items():
                feats.append(np.asarray(key, dtype=float))
                is_orig_list.append(d)
                pdc_list.append(c)
                origin.append(i)

        return AugmentedDataset(
            features=np.vstack(feats) if feats else np.zeros((0, len(self.feature_keys))),
            is_original=np.asarray(is_orig_list, dtype=float),
            potential_deriv_coef=np.asarray(pdc_list, dtype=float),
            origin_index=np.asarray(origin, dtype=np.int64),
            n_rows=n,
        )


def _rebuild_custom_estimand(feature_keys, m, name):
    return FiniteEvalEstimand(feature_keys=feature_keys, m=m, name=name)


# ---------------------------------------------------------------------------
# Built-in subclasses. Each provides:
#   - __init__ that stores treatment/covariate names, builds `m`, and seeds
#     `factory_spec` for round-trip.
#   - `augment(features, ys=None)` override that emits augmented rows in
#     vectorised numpy. Row order is implementation-defined and not part of
#     the public contract.
#   - class-level `m_bar` giving the closed-form E[m(alpha=1)(Z)].


class ATE(FiniteEvalEstimand):
    """Average treatment effect: m(α)(z, y) = α(1, x) − α(0, x)."""

    name = "ATE"
    m_bar = 0.0

    def __init__(self, treatment: str = "a", covariates: Sequence[str] = ("x",)):
        cov = tuple(covariates)
        self.treatment = treatment
        self.covariates = cov

        def m(alpha):
            def inner(z, y=None):
                x_kwargs = {k: z[k] for k in cov}
                return alpha(**{treatment: 1, **x_kwargs}) - alpha(**{treatment: 0, **x_kwargs})
            return inner

        super().__init__(
            feature_keys=(treatment, *cov), m=m,
            factory_spec={"factory": "ATE", "args": {"treatment": treatment, "covariates": list(cov)}},
        )

    def augment(self, features, ys=None):
        features, n = self._normalise_features(features, ys)
        a_idx = self.feature_keys.index(self.treatment)
        a = features[:, a_idx]
        treated = features.copy()
        treated[:, a_idx] = 1.0
        control = features.copy()
        control[:, a_idx] = 0.0
        return AugmentedDataset(
            features=np.vstack([treated, control]),
            is_original=np.concatenate([(a == 1.0).astype(float), (a == 0.0).astype(float)]),
            potential_deriv_coef=np.concatenate([np.full(n, -1.0), np.full(n, 1.0)]),
            origin_index=np.tile(np.arange(n, dtype=np.int64), 2),
            n_rows=n,
        )


class ATT(FiniteEvalEstimand):
    """ATT *partial-estimand* surface: m(α)(z, y) = a · (α(1, x) − α(0, x)).

    Full ATT divides by P(A=1) and is not a Riesz functional — combine
    α̂_partial with a delta-method EIF (Hubbard 2011) downstream.
    """

    name = "ATT"
    m_bar = 0.0

    def __init__(self, treatment: str = "a", covariates: Sequence[str] = ("x",)):
        cov = tuple(covariates)
        self.treatment = treatment
        self.covariates = cov

        def m(alpha):
            def inner(z, y=None):
                a = z[treatment]
                x_kwargs = {k: z[k] for k in cov}
                return a * (
                    alpha(**{treatment: 1, **x_kwargs}) - alpha(**{treatment: 0, **x_kwargs})
                )
            return inner

        super().__init__(
            feature_keys=(treatment, *cov), m=m,
            factory_spec={"factory": "ATT", "args": {"treatment": treatment, "covariates": list(cov)}},
        )

    def augment(self, features, ys=None):
        features, n = self._normalise_features(features, ys)
        a_idx = self.feature_keys.index(self.treatment)
        a = features[:, a_idx]
        treated_mask = (a == 1.0)
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


class TSM(FiniteEvalEstimand):
    """Treatment-specific mean: m(α)(z, y) = α(level, x)."""

    m_bar = 1.0

    def __init__(self, level, treatment: str = "a", covariates: Sequence[str] = ("x",)):
        cov = tuple(covariates)
        self.level = level
        self.treatment = treatment
        self.covariates = cov

        def m(alpha):
            def inner(z, y=None):
                x_kwargs = {k: z[k] for k in cov}
                return alpha(**{treatment: level, **x_kwargs})
            return inner

        super().__init__(
            feature_keys=(treatment, *cov), m=m,
            name=f"TSM(level={level!r})",
            factory_spec={"factory": "TSM", "args": {"level": level, "treatment": treatment, "covariates": list(cov)}},
        )

    def augment(self, features, ys=None):
        features, n = self._normalise_features(features, ys)
        a_idx = self.feature_keys.index(self.treatment)
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


class AdditiveShift(FiniteEvalEstimand):
    """Additive shift effect: m(α)(z, y) = α(a + δ, x) − α(a, x)."""

    m_bar = 0.0

    def __init__(self, delta: float, treatment: str = "a", covariates: Sequence[str] = ("x",)):
        if delta == 0:
            raise ValueError("AdditiveShift requires delta != 0 (delta=0 is a degenerate, vacuous estimand).")
        cov = tuple(covariates)
        self.delta = delta
        self.treatment = treatment
        self.covariates = cov

        def m(alpha):
            def inner(z, y=None):
                a = z[treatment]
                x_kwargs = {k: z[k] for k in cov}
                return alpha(**{treatment: a + delta, **x_kwargs}) - alpha(
                    **{treatment: a, **x_kwargs}
                )
            return inner

        super().__init__(
            feature_keys=(treatment, *cov), m=m,
            name=f"AdditiveShift(delta={delta})",
            factory_spec={"factory": "AdditiveShift", "args": {"delta": delta, "treatment": treatment, "covariates": list(cov)}},
        )

    def augment(self, features, ys=None):
        features, n = self._normalise_features(features, ys)
        a_idx = self.feature_keys.index(self.treatment)
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


class LocalShift(FiniteEvalEstimand):
    """LASE *partial-estimand* surface: m(α)(z, y) = 1(a < threshold) · (α(a+δ, x) − α(a, x)).

    Full LASE divides by P(A < threshold) and is not a Riesz functional.
    """

    m_bar = 0.0

    def __init__(
        self,
        delta: float,
        threshold: float,
        treatment: str = "a",
        covariates: Sequence[str] = ("x",),
    ):
        if delta == 0:
            raise ValueError("LocalShift requires delta != 0 (delta=0 is a degenerate, vacuous estimand).")
        cov = tuple(covariates)
        self.delta = delta
        self.threshold = threshold
        self.treatment = treatment
        self.covariates = cov

        def m(alpha):
            def inner(z, y=None):
                a = z[treatment]
                if a >= threshold:
                    return 0
                x_kwargs = {k: z[k] for k in cov}
                return alpha(**{treatment: a + delta, **x_kwargs}) - alpha(
                    **{treatment: a, **x_kwargs}
                )
            return inner

        super().__init__(
            feature_keys=(treatment, *cov), m=m,
            name=f"LocalShift(delta={delta}, threshold={threshold})",
            factory_spec={"factory": "LocalShift", "args": {"delta": delta, "threshold": threshold, "treatment": treatment, "covariates": list(cov)}},
        )

    def augment(self, features, ys=None):
        features, n = self._normalise_features(features, ys)
        a_idx = self.feature_keys.index(self.treatment)
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


def StochasticIntervention(
    samples_key: str = "shift_samples",
    treatment: str = "a",
    covariates: Sequence[str] = ("x",),
) -> FiniteEvalEstimand:
    """Stochastic intervention via Monte Carlo samples per row.

    Currently being rewritten — the previous implementation relied on an
    `extra_keys` payload mechanism that has been removed. A reintroduction
    will land in a follow-up that establishes how per-row samples flow into
    `m(alpha)(z, y)` without the payload-column shortcut.
    """
    raise NotImplementedError(
        "StochasticIntervention is being rewritten; will be re-added in a future PR."
    )


# Registry for round-tripping. Updated when new built-in subclasses are added.
_FACTORY_REGISTRY: dict[str, type] = {
    "ATE": ATE,
    "ATT": ATT,
    "TSM": TSM,
    "AdditiveShift": AdditiveShift,
    "LocalShift": LocalShift,
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
