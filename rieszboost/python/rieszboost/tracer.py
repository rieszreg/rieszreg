"""Symbolic tracer that extracts (coefficient, point) terms from a user's m().

The user writes m opaquely:

    def m_ate(z, alpha):
        return alpha(a=1, x=z["x"]) - alpha(a=0, x=z["x"])

We pass a `Tracer` in as `alpha`. Each call records a `LinearTerm`; arithmetic
builds a `LinearForm`. If the user's m stays inside the linear-form algebra
(only +, -, scalar *, alpha calls), the returned LinearForm gives us the exact
finite list of (coefficient, point) pairs we need to build the augmented
dataset for the fast engine. Anything else (integrals, derivatives, non-linear
ops) raises and signals the dispatcher to fall back to the slow path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
        if not isinstance(scalar, (int, float)):
            raise TypeError(
                f"LinearForm * {type(scalar).__name__} is not allowed — m must be "
                "linear in alpha (only scalar multiplication)."
            )
        return LinearForm({p: float(scalar) * c for p, c in self.terms.items()})

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        if not isinstance(scalar, (int, float)):
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
    """Stand-in for `alpha` during tracing. Each call records a single-term
    LinearForm; the user's m composes these via +/-/scalar*."""

    def __call__(self, **kwargs) -> LinearForm:
        return LinearForm.single(_Point.from_kwargs(kwargs), 1.0)


def trace(m, z) -> list[tuple[float, dict[str, Any]]]:
    """Run the user's m on a single row z with the Tracer as alpha and return
    the (coef, point) pair list. Raises if m leaves the linear-form algebra."""
    result = m(z, Tracer())
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
            "If your m involves integrals or derivatives of alpha, use the slow "
            "engine — the fast engine only supports finite linear combinations of "
            "point evaluations."
        )
    return result.as_pairs()
