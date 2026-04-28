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


def StochasticIntervention(
    samples_key: str = "shift_samples",
    treatment: str = "a",
    covariates: Sequence[str] = ("x",),
) -> Callable:
    """Stochastic intervention via pre-computed Monte Carlo samples.

    The functional is θ = E[∫ μ(a', X) g(a' | A, X) da'] for some intervention
    density g. We approximate the integral by Monte Carlo: each row `z` must
    contain a sequence `z[samples_key]` of treatment values drawn from
    g(· | a, x). The empirical m is

        m(z, alpha) = (1/K) Σ_k alpha(a' = z[samples_key][k], x)

    Pre-sample once before calling `fit(...)`, e.g.:

        rng = np.random.default_rng(0)
        for row in rows:
            row["shift_samples"] = rng.normal(
                row["a"] + delta, sigma, size=n_mc_samples
            )

    Increasing `n_mc_samples` reduces Monte Carlo noise; common choice is
    10–50. Note `feature_keys` should NOT include `samples_key` (it's not
    a tree feature, just a per-row payload).
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

    return m


def ATT(
    p_treated: float, treatment: str = "a", covariates: Sequence[str] = ("x",)
) -> Callable:
    """Average treatment effect on the treated.

    m(z, alpha) = (a / p_treated) * (alpha(1, x) - alpha(0, x)).
    For control rows (a=0) the contribution is zero — those rows contribute
    only the alpha^2 term in the loss. `p_treated` is the marginal P(A=1)
    (estimate as `np.mean(a)` outside).
    """
    cov = tuple(covariates)

    def m(z, alpha):
        a = z[treatment]
        x_kwargs = {k: z[k] for k in cov}
        weight = a / p_treated
        return weight * (
            alpha(**{treatment: 1, **x_kwargs}) - alpha(**{treatment: 0, **x_kwargs})
        )

    return m
