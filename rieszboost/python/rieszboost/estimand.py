"""Estimand: a self-contained description of the linear functional to fit.

An `Estimand` carries (1) the column names alpha is indexed by (`feature_keys`),
(2) per-row payload columns that aren't tree features but are referenced by m
(`extra_keys`, e.g. "shift_samples" for stochastic interventions), and (3) the
opaque m(z, alpha) callable itself.

`RieszBooster` reads `feature_keys` and `extra_keys` off the estimand at fit
time — no need for the user to pass these as separate arguments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence


@dataclass
class Estimand:
    feature_keys: tuple[str, ...]
    m: Callable[..., Any]
    extra_keys: tuple[str, ...] = ()
    name: str = "custom"

    def __call__(self, z, alpha):
        return self.m(z, alpha)


def ATE(treatment: str = "a", covariates: Sequence[str] = ("x",)) -> Estimand:
    """Average treatment effect: m(z, α) = α(1, x) − α(0, x)."""
    cov = tuple(covariates)

    def m(z, alpha):
        x_kwargs = {k: z[k] for k in cov}
        return alpha(**{treatment: 1, **x_kwargs}) - alpha(**{treatment: 0, **x_kwargs})

    return Estimand(feature_keys=(treatment, *cov), m=m, name="ATE")


def ATT(treatment: str = "a", covariates: Sequence[str] = ("x",)) -> Estimand:
    """ATT *partial parameter* m(z, α) = a · (α(1, x) − α(0, x)).

    Full ATT divides by P(A=1) and is not a Riesz functional — combine
    α̂_partial with a delta-method EIF (Hubbard 2011) downstream.
    """
    cov = tuple(covariates)

    def m(z, alpha):
        a = z[treatment]
        x_kwargs = {k: z[k] for k in cov}
        return a * (
            alpha(**{treatment: 1, **x_kwargs}) - alpha(**{treatment: 0, **x_kwargs})
        )

    return Estimand(feature_keys=(treatment, *cov), m=m, name="ATT")


def TSM(level, treatment: str = "a", covariates: Sequence[str] = ("x",)) -> Estimand:
    """Treatment-specific mean: m(z, α) = α(level, x)."""
    cov = tuple(covariates)

    def m(z, alpha):
        x_kwargs = {k: z[k] for k in cov}
        return alpha(**{treatment: level, **x_kwargs})

    return Estimand(feature_keys=(treatment, *cov), m=m, name=f"TSM(level={level!r})")


def AdditiveShift(
    delta: float, treatment: str = "a", covariates: Sequence[str] = ("x",)
) -> Estimand:
    """Additive shift effect: m(z, α) = α(a + δ, x) − α(a, x)."""
    cov = tuple(covariates)

    def m(z, alpha):
        a = z[treatment]
        x_kwargs = {k: z[k] for k in cov}
        return alpha(**{treatment: a + delta, **x_kwargs}) - alpha(
            **{treatment: a, **x_kwargs}
        )

    return Estimand(
        feature_keys=(treatment, *cov), m=m, name=f"AdditiveShift(delta={delta})"
    )


def LocalShift(
    delta: float,
    threshold: float,
    treatment: str = "a",
    covariates: Sequence[str] = ("x",),
) -> Estimand:
    """LASE *partial parameter* m(z, α) = 1(a < threshold) · (α(a+δ, x) − α(a, x)).

    Full LASE divides by P(A < threshold) and is not a Riesz functional.
    """
    cov = tuple(covariates)

    def m(z, alpha):
        a = z[treatment]
        if a >= threshold:
            return 0
        x_kwargs = {k: z[k] for k in cov}
        return alpha(**{treatment: a + delta, **x_kwargs}) - alpha(
            **{treatment: a, **x_kwargs}
        )

    return Estimand(
        feature_keys=(treatment, *cov),
        m=m,
        name=f"LocalShift(delta={delta}, threshold={threshold})",
    )


def StochasticIntervention(
    samples_key: str = "shift_samples",
    treatment: str = "a",
    covariates: Sequence[str] = ("x",),
) -> Estimand:
    """Stochastic intervention via Monte Carlo samples per row.

    Each row carries `z[samples_key]` = sequence of treatment values drawn
    from the intervention density. `m(z, α) = (1/K) Σ_k α(a' = sample_k, x)`.

    Pre-sample once before fit:

        rng = np.random.default_rng(0)
        df["shift_samples"] = [rng.normal(a + delta, sigma, K) for a in df["a"]]
    """
    cov = tuple(covariates)

    def m(z, alpha):
        x_kwargs = {k: z[k] for k in cov}
        samples = z[samples_key]
        K = len(samples)
        if K == 0:
            return 0
        return sum(
            alpha(**{treatment: float(s), **x_kwargs}) for s in samples
        ) / K

    return Estimand(
        feature_keys=(treatment, *cov),
        m=m,
        extra_keys=(samples_key,),
        name=f"StochasticIntervention(samples_key={samples_key!r})",
    )
