"""`Loss` base class for Bregman-Riesz losses.

A `Loss` describes a Bregman-Riesz loss as a function of `alpha` (and `eta`,
via the link) only. The (is_original, potential_deriv_coef) augmented-row
coefficients live on `AugmentedDataset` (produced by `Estimand.augment`); the
loss's `aug_loss_alpha`, `aug_loss_eta`, `aug_grad_eta`, and `aug_hess_eta`
methods combine those coefficients with the loss's α-space functions to give
per-row loss / gradient / Hessian.

The Bregman-Riesz loss with strictly convex potential `h` (`potential` in
code) has the population form

    L_h(alpha) = const + E[h_tilde(alpha(Z))] - E[m(h' o alpha)(Z, Y)]

with `h_tilde(t) = t · h'(t) - h(t)`; equivalently `h_tilde = h* o h'` where
`h*` is the Legendre transform of `h` (`tilde_potential` in code). On a
finite-evaluation estimand, the empirical loss reduces over the augmented
dataset (D_r, C_r) to

    D_r · h_tilde(alpha(z_r)) + C_r · h'(alpha(z_r))

where `D_r = 1` (`is_original`) for the original-row evaluation and `D_r = 0`,
`C_r = -c_k` (`potential_deriv_coef`) at each counterfactual point of m. Each
Loss defines a **link** mapping eta → alpha (boosting outputs and neural-network
outputs are real-valued; the link maps them into alpha's domain).

Code-side names mirror the math:

- ``potential(alpha)``        ↔  $h(\\alpha)$
- ``potential_deriv(alpha)``  ↔  $h'(\\alpha)$
- ``tilde_potential(alpha)``   ↔  $\\tilde h(\\alpha) = h^*(h'(\\alpha))$

User-facing patterns:

    # Subclass — recommended for performance (analytic overrides):
    class MyLoss(Loss):
        name = "mine"
        def potential(self, alpha):       return alpha * np.log(alpha)
        def potential_deriv(self, alpha): return np.log(alpha) + 1.0
        def link_to_alpha(self, eta):     return np.exp(eta)
        def alpha_to_eta(self, alpha):    return np.log(alpha)

    # Inline — for quick experiments (numerical defaults):
    loss = Loss(potential=lambda a: a * np.log(a),
                potential_deriv=lambda a: np.log(a) + 1.0,
                link="exp")
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np


# Built-in links. Each entry: (link_to_alpha, alpha_to_eta).
_LINKS: dict[
    str,
    tuple[
        Callable[[np.ndarray], np.ndarray],
        Callable[[float | np.ndarray], float | np.ndarray],
    ],
] = {
    "identity": (
        lambda eta: eta,
        lambda alpha: alpha,
    ),
    "exp": (
        lambda eta: np.exp(np.clip(eta, -50.0, 50.0)),
        lambda alpha: np.log(alpha),
    ),
    "sigmoid": (
        lambda eta: 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0))),
        lambda alpha: np.log(alpha / (1.0 - alpha)),
    ),
}

# α-domain per link, used only by `Loss.best_constant_init` defaults.
_LINK_DOMAIN: dict[str, tuple[float, float]] = {
    "identity": (-np.inf, np.inf),
    "exp": (0.0, np.inf),
    "sigmoid": (0.0, 1.0),
}


def _numerical_derivative(fn, x, eps=1e-6):
    return (fn(x + eps) - fn(x - eps)) / (2.0 * eps)


def _try_build_jax_grad(fn):
    """Build a vmapped `jax.grad(fn)` if jax is installed AND `fn` is
    jax-traceable. Returns None on failure."""
    try:
        import jax
    except ImportError:
        return None
    try:
        grad = jax.vmap(jax.grad(lambda x: fn(x)))
        # Smoke-test: trace a tiny array. Failure indicates fn uses non-jax ops.
        _ = grad(np.array([1.0, 2.0]))
        return grad
    except Exception:
        return None


def _autograd_or_numerical(fn, alpha, owner):
    """Compute `fn'(alpha)` via jax-grad when possible, numerical otherwise.
    Caches the chosen strategy on `owner` so we don't re-probe on every call."""
    strategy = getattr(owner, "_grad_strategy", None)
    if strategy is None:
        grad_fn = _try_build_jax_grad(fn)
        if grad_fn is not None:
            owner._jax_grad_fn = grad_fn
            owner._grad_strategy = "jax"
            strategy = "jax"
        else:
            owner._grad_strategy = "numerical"
            strategy = "numerical"
    if strategy == "jax":
        arr = np.atleast_1d(np.asarray(alpha, dtype=float))
        out = np.asarray(owner._jax_grad_fn(arr))
        return out.reshape(np.shape(alpha)) if np.ndim(alpha) > 0 else out[0]
    return _numerical_derivative(fn, alpha)


class Loss:
    """Bregman-Riesz loss base class.

    Subclass and override `potential`, `potential_deriv`, and the link methods
    for an analytic, performance-optimized loss. Or instantiate inline:

        Loss(potential=lambda a: a**2, link="identity")

    Class attributes (override in subclasses):

    - ``name``: short identifier used in error messages and serialization.
    """

    name: str = "loss"

    def __init__(
        self,
        *,
        potential: Callable[[np.ndarray], np.ndarray] | None = None,
        potential_deriv: Callable[[np.ndarray], np.ndarray] | None = None,
        link: Literal["identity", "exp", "sigmoid"] | None = None,
        name: str | None = None,
    ):
        if potential is not None:
            self._potential_inline = potential
        if potential_deriv is not None:
            self._potential_deriv_inline = potential_deriv
        if link is not None:
            if link not in _LINKS:
                raise ValueError(
                    f"Unknown link {link!r}; expected one of {sorted(_LINKS)}."
                )
            link_fwd, link_inv = _LINKS[link]
            self._link_fwd_inline = link_fwd
            self._link_inv_inline = link_inv
        if name is not None:
            self.name = name

    # ---- α-space loss functions ----

    def potential(self, alpha: np.ndarray) -> np.ndarray:
        """Bregman potential `h(α)`. Strictly convex on the link's α-domain."""
        if hasattr(self, "_potential_inline"):
            return self._potential_inline(alpha)
        raise NotImplementedError(
            f"{type(self).__name__} must define potential(alpha) (subclass) or "
            "pass potential= to Loss(...)."
        )

    def potential_deriv(self, alpha: np.ndarray) -> np.ndarray:
        """First derivative `h'(α)`.

        Resolution order:

        1. If `potential_deriv=` was passed to the constructor, use that.
        2. If jax is installed and `potential` is jax-traceable (uses
           `jax.numpy` ops), use `jax.grad`.
        3. Fall back to a central-difference numerical derivative.

        Override in subclasses for performance.
        """
        if hasattr(self, "_potential_deriv_inline"):
            return self._potential_deriv_inline(alpha)
        return _autograd_or_numerical(self.potential, alpha, self)

    def tilde_potential(self, alpha: np.ndarray) -> np.ndarray:
        """`h_tilde(α) = α · h'(α) - h(α)`. Equivalently `h_tilde = h* o h'`
        where `h*` is the Legendre transform of `h`, so this is the value of
        `h*` at the slope `h'(α)`. Computed from `potential` and
        `potential_deriv` by default; override only if a tighter analytic
        form is available.
        """
        return alpha * self.potential_deriv(alpha) - self.potential(alpha)

    # ---- Link (η ↔ α) ----

    def link_to_alpha(self, eta: np.ndarray) -> np.ndarray:
        """Inverse link: convert backend output η to α."""
        if hasattr(self, "_link_fwd_inline"):
            return self._link_fwd_inline(eta)
        return eta  # identity default

    def alpha_to_eta(self, alpha: float | np.ndarray) -> float | np.ndarray:
        """Forward link: convert α to η (used for `init=` translation)."""
        if hasattr(self, "_link_inv_inline"):
            return self._link_inv_inline(alpha)
        return alpha  # identity default

    # ---- Augmented-loss helpers ----
    #
    # These take the augmentation packaging `(is_original, potential_deriv_coef)`
    # — abbreviated D, C in the math — and combine them with the loss's α-space
    # functions. `aug_loss_alpha` is the formula `D · h_tilde(α) + C · h'(α)`;
    # `aug_loss_eta` is the same after applying the link to η. The grad/Hessian
    # methods route through the link via numerical defaults; override in
    # subclasses for an analytic form.

    def aug_loss_alpha(self, is_original, potential_deriv_coef, alpha):
        """Per-row augmented loss in α-space: D · h_tilde(α) + C · h'(α)."""
        return (
            is_original * self.tilde_potential(alpha)
            + potential_deriv_coef * self.potential_deriv(alpha)
        )

    def aug_loss_eta(self, is_original, potential_deriv_coef, eta):
        """Per-row augmented loss in η-space: aug_loss_alpha after the link."""
        return self.aug_loss_alpha(
            is_original, potential_deriv_coef, self.link_to_alpha(eta)
        )

    def aug_grad_eta(self, is_original, potential_deriv_coef, eta):
        """∂[D · h_tilde(α) + C · h'(α)]/∂η."""
        eps = 1e-6
        L_plus = self.aug_loss_eta(is_original, potential_deriv_coef, eta + eps)
        L_minus = self.aug_loss_eta(is_original, potential_deriv_coef, eta - eps)
        return (L_plus - L_minus) / (2.0 * eps)

    def aug_hess_eta(self, is_original, potential_deriv_coef, eta, hessian_floor):
        """∂²[D · h_tilde(α) + C · h'(α)]/∂η² (floored)."""
        eps = 1e-4
        L_plus = self.aug_loss_eta(is_original, potential_deriv_coef, eta + eps)
        L_zero = self.aug_loss_eta(is_original, potential_deriv_coef, eta)
        L_minus = self.aug_loss_eta(is_original, potential_deriv_coef, eta - eps)
        h2 = (L_plus - 2.0 * L_zero + L_minus) / (eps * eps)
        return np.maximum(h2, hessian_floor)

    # ---- Initialization ----

    def best_constant_init(self, m_bar: float) -> float:
        """Loss-minimizing constant α* given m̄ = E[m(α=1)(Z, Y)].

        For any Bregman loss with strictly convex `potential` the empirical
        FOC `tilde_potential'(a) = potential''(a) · m̄` collapses to `a* = m̄`
        via `tilde_potential'(t) = t · potential''(t)`. Implementations project
        `m̄` into the loss's α-domain when its codomain is restricted (KL: α > 0;
        Bernoulli: α ∈ (0, 1); BoundedSquared: α ∈ (lo, hi)).
        """
        # Default: project into the inline link's α-domain if known.
        for link_str, (fwd, _) in _LINKS.items():
            if getattr(self, "_link_fwd_inline", None) is fwd:
                lo, hi = _LINK_DOMAIN[link_str]
                if lo == -np.inf and hi == np.inf:
                    return float(m_bar)
                eps = 1e-9
                return float(np.clip(m_bar, lo + eps, hi - eps))
        return float(m_bar)

    # ---- Round-trip ----

    def to_spec(self) -> dict:
        """JSON-serializable round-trip spec. Built-in subclasses override
        with `{"type": "<name>", "args": {...}}`. Inline-constructed losses
        can't round-trip (the user's `potential` callable doesn't survive)."""
        raise NotImplementedError(
            f"{type(self).__name__} is not save/load round-trippable. Pass the "
            "loss explicitly when reloading. (For inline `Loss(potential=...)`, "
            "the user's callable can't be JSON-serialized.)"
        )

