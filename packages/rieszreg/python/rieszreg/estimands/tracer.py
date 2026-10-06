"""Symbolic tracer that extracts (coefficient, point) terms from a user's m().

The user writes m as an operator: it takes alpha and returns a function of
the row z and the per-row outcome y.

    def m_ate(alpha):
        def inner(z, y=None):
            return alpha(a=1, x=z["x"]) - alpha(a=0, x=z["x"])
        return inner

We pass a `Tracer` in as `alpha`. Each call records a `LinearTerm`; arithmetic
builds a `LinearForm`. The returned LinearForm gives us the exact finite list
of (coefficient, point) pairs we need to build the augmented dataset. Anything
outside the linear-form algebra (integrals, derivatives, non-linear ops)
raises with a clear error pointing at the offending operation.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any

from .base import FiniteEvalEstimand


@dataclass(frozen=True)
class _Point:
    """A point at which alpha is evaluated. Stored as a sorted tuple of
    (key, value) pairs so points are hashable and comparable across rows."""
    items: tuple[tuple[str, Any], ...]

    @classmethod
    def from_kwargs(cls, kwargs: dict[str, Any]) -> "_Point":
        return cls(tuple(sorted(kwargs.items(), key=lambda kv: kv[0])))

    def as_dict(self) -> dict[str, Any]:
        return dict(self.items)


class LinearForm:
    """A finite linear combination of alpha-evaluations: sum_j c_j * alpha(p_j)."""

    __slots__ = ("terms",)

    def __init__(self, terms: dict[_Point, float] | None = None):
        self.terms: dict[_Point, float] = dict(terms) if terms else {}

    @classmethod
    def single(cls, point: _Point, coef: float = 1.0) -> "LinearForm":
        return cls({point: coef})

    def __add__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                return self
            raise TypeError(
                "Cannot add a scalar to a LinearForm — m must be linear in alpha "
                "(no constant offsets)."
            )
        if not isinstance(other, LinearForm):
            return NotImplemented
        out = dict(self.terms)
        for p, c in other.terms.items():
            out[p] = out.get(p, 0.0) + c
        return LinearForm(out)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self + (-1.0) * other

    def __rsub__(self, other):
        return (-1.0) * self + other

    def __neg__(self):
        return (-1.0) * self

    def __mul__(self, scalar):
        if not isinstance(scalar, Real):
            raise TypeError(
                f"LinearForm * {type(scalar).__name__} is not allowed — m must be "
                "linear in alpha (only scalar multiplication)."
            )
        return LinearForm({p: float(scalar) * c for p, c in self.terms.items()})

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        if not isinstance(scalar, Real):
            raise TypeError(
                "LinearForm can only be divided by scalars — m must be linear in alpha."
            )
        return self * (1.0 / scalar)

    def as_pairs(self) -> list[tuple[float, dict[str, Any]]]:
        """Return [(coefficient, point_dict), ...] with zero-coef terms dropped."""
        return [(c, p.as_dict()) for p, c in self.terms.items() if c != 0.0]

    def __repr__(self) -> str:
        if not self.terms:
            return "LinearForm(0)"
        parts = [f"{c:+g}*alpha({dict(p.items)})" for p, c in self.terms.items()]
        return "LinearForm(" + " ".join(parts) + ")"


class Tracer:
    """Stand-in for `alpha` during tracing. The trace step calls
    `m(Tracer())(z, y)`; each `alpha(...)` call inside the user's m records a
    single-term LinearForm, and the user's m composes these via +/-/scalar*."""

    def __call__(self, **kwargs) -> LinearForm:
        return LinearForm.single(_Point.from_kwargs(kwargs), 1.0)


def trace(
    estimand: FiniteEvalEstimand, z, y=None
) -> list[tuple[float, dict[str, Any]]]:
    """Run the estimand's m as `m(Tracer())(z, y)` on a single row and return
    the (coef, point) pair list. Raises if m leaves the linear-form algebra.
    `y` is the per-row outcome; estimands whose m doesn't read y can leave it
    unset."""
    if not isinstance(estimand, FiniteEvalEstimand):
        raise TypeError(
            f"trace() requires a FiniteEvalEstimand; got {type(estimand).__name__}. "
            "Construct your estimand via a built-in factory (ATE, ATT, TSM, ...) "
            "or wrap your m in `FiniteEvalEstimand(feature_keys=..., m=...)`."
        )
    result = estimand.m(Tracer())(z, y)
    if isinstance(result, (int, float)):
        if result == 0:
            return []
        raise TypeError(
            "m returned a non-zero scalar — m must be linear in alpha (the result "
            "must be a LinearForm built from alpha(...) calls)."
        )
    if not isinstance(result, LinearForm):
        raise TypeError(
            f"m returned {type(result).__name__}; expected LinearForm. "
            "If your m involves integrals or derivatives of alpha, sample/discretize "
            "first — the linear-form engine only supports finite linear combinations "
            "of point evaluations."
        )
    return result.as_pairs()
