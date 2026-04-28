"""Built-in estimand factories. Each returns an opaque m(z, alpha) callable
suitable for the fast engine."""

from __future__ import annotations

from typing import Callable, Sequence


def ATE(treatment: str = "a", covariates: Sequence[str] = ("x",)) -> Callable:
    """Average treatment effect: m(z, alpha) = alpha(a=1, x) - alpha(a=0, x)."""
    cov = tuple(covariates)

    def m(z, alpha):
        x_kwargs = {k: z[k] for k in cov}
        return alpha(**{treatment: 1, **x_kwargs}) - alpha(**{treatment: 0, **x_kwargs})

    return m


def TSM(level, treatment: str = "a", covariates: Sequence[str] = ("x",)) -> Callable:
    """Treatment-specific mean: m(z, alpha) = alpha(a=level, x)."""
    cov = tuple(covariates)

    def m(z, alpha):
        x_kwargs = {k: z[k] for k in cov}
        return alpha(**{treatment: level, **x_kwargs})

    return m


def AdditiveShift(
    delta: float, treatment: str = "a", covariates: Sequence[str] = ("x",)
) -> Callable:
    """Additive shift effect: m(z, alpha) = alpha(a + delta, x) - alpha(a, x)."""
    cov = tuple(covariates)

    def m(z, alpha):
        a = z[treatment]
        x_kwargs = {k: z[k] for k in cov}
        return alpha(**{treatment: a + delta, **x_kwargs}) - alpha(
            **{treatment: a, **x_kwargs}
        )

    return m
