"""Per-leaf Bregman optimum and split-gain computation.

Given the per-augmented-row coefficients ``(D_r, C_r)`` from
``Estimand.augment`` (where ``D_r`` is ``is_original`` and ``C_r``
is ``potential_deriv_coef``), the constant-per-leaf augmented Bregman loss
in leaf ``ℓ`` is

    L(α_ℓ) = D_ℓ · h̃(α_ℓ) + C_ℓ · h'(α_ℓ),

with ``D_ℓ = Σ_{r∈ℓ} D_r`` and ``C_ℓ = Σ_{r∈ℓ} C_r``. Setting the
derivative to zero and using the identity ``h̃'(t) = t · h''(t)``:

    h''(α_ℓ) · (D_ℓ · α_ℓ + C_ℓ) = 0   ⇒   α_ℓ* = -C_ℓ / D_ℓ,

with the value projected to the loss's α-domain when it falls outside.
The leaf-loss-at-optimum value is ``L(α_ℓ*)``; the per-loss closed forms
live in ``_LEAF_LOSS_DISPATCH`` below.

The splitter sweeps each candidate split position and computes

    gain = L(α_p*) - L(α_ℓ*) - L(α_r*),

picking the largest. ``min_orig_leaf`` constrains the count of original
(``D > 0``) rows in each child.
"""

from __future__ import annotations

import numpy as np

from rieszreg import (
    BernoulliLoss,
    BoundedSquaredLoss,
    KLLoss,
    Loss,
    SquaredLoss,
)


# ---------------------------------------------------------------------------
# Leaf-loss-at-optimum, per Loss subclass.

def _leaf_loss_squared(D: float, C: float) -> float:
    """``h(t) = t²``; α* = -C/D unconditionally; L(α*) = -C²/D."""
    if D <= 0.0:
        return 0.0
    return -C * C / D


def _leaf_loss_kl(D: float, C: float) -> float:
    """``h(t) = t log t - t`` on t > 0; α* = -C/D when C < 0.

    L(α) = D·α + C·log α. At α* = -C/D > 0:
        L(α*) = D·(-C/D) + C·log(-C/D) = -C + C·log(-C/D).

    Boundary cases (α ≥ 0 required by the link domain):
      - C = 0: L(α) = D·α has infimum 0 at α = 0+ (boundary value).
      - C > 0: minimum unbounded below as α → ∞ would be wrong direction;
        actually L'(α) = D + C/α > 0 for D > 0, C > 0, α > 0, so L is
        increasing. Infimum at α = 0+ is C·(-∞) = -∞ — infeasible. Return
        +∞ to disqualify this configuration as a child of a split.
    """
    if D <= 0.0:
        return 0.0
    if C == 0.0:
        return 0.0
    if C > 0.0:
        return float("inf")
    return -C + C * np.log(-C / D)


def _leaf_loss_bernoulli(D: float, C: float) -> float:
    """``h(t) = t log t + (1-t) log(1-t)`` on (0, 1); α* = -C/D when -D < C < 0.

    Derivation: h'(t) = log(t/(1-t)); h̃(t) = -log(1-t).
    L(α) = -D·log(1-α) + C·log(α/(1-α)). Setting L'(α) = 0 gives α = -C/D.
    At α* = -C/D ∈ (0, 1) (i.e. -D < C < 0):
        1 - α* = (D + C) / D
        α*/(1-α*) = -C/(D+C)
        L(α*) = -D·log((D+C)/D) + C·log(-C/(D+C))
              = D·log D - (D+C)·log(D+C) + C·log(-C).

    At C = -D, α* = 1 and L(α*) = 0 (the (D+C)·log(D+C) term is 0·log 0).
    Outside [-D, 0): infeasible — return +∞.
    """
    if D <= 0.0:
        return 0.0
    if C == 0.0:
        return 0.0
    if not (-D <= C < 0.0):
        return float("inf")
    if C == -D:
        return 0.0
    return D * np.log(D) - (D + C) * np.log(D + C) + C * np.log(-C)


def _leaf_loss_bounded_squared(D: float, C: float, lo: float, hi: float) -> float:
    """``h(t) = t²`` on (lo, hi); α* = -C/D projected into [lo, hi].

    Per-leaf loss L(α) = D·α² + 2C·α (as in SquaredLoss). Project
    α* = -C/D into the interval and evaluate.
    """
    if D <= 0.0:
        return 0.0
    a_star = -C / D
    if a_star < lo:
        a_star = lo
    elif a_star > hi:
        a_star = hi
    return D * a_star * a_star + 2.0 * C * a_star


def _leaf_alpha_squared(D: float, C: float) -> float:
    return 0.0 if D <= 0.0 else -C / D


def _leaf_alpha_kl(D: float, C: float) -> float:
    if D <= 0.0:
        return 0.0
    if C >= 0.0:
        return 0.0  # boundary
    return -C / D


def _leaf_alpha_bernoulli(D: float, C: float) -> float:
    if D <= 0.0:
        return 0.5
    a_star = -C / D
    if a_star <= 0.0:
        return 1e-6
    if a_star >= 1.0:
        return 1.0 - 1e-6
    return a_star


def _leaf_alpha_bounded_squared(D: float, C: float, lo: float, hi: float) -> float:
    if D <= 0.0:
        return 0.5 * (lo + hi)
    a_star = -C / D
    if a_star < lo:
        return lo
    if a_star > hi:
        return hi
    return a_star


def make_leaf_solvers(loss: Loss):
    """Return ``(loss_at_optimum, alpha_at_optimum)`` callables for ``loss``.

    Both signatures: ``f(D: float, C: float) -> float``.

    Built-in dispatch covers ``SquaredLoss``, ``KLLoss``,
    ``BernoulliLoss``, ``BoundedSquaredLoss``. Custom losses can plug
    in via :func:`riesztree.fast.register_fast_leaf_solver`, which
    contributes both a Cython-callable leaf-loss kernel (used by
    ``splitter='exact'``) and a Python ``alpha_at_opt`` callable
    (consumed here for the leaf-payload computation). The registry is
    checked first, as in :func:`riesztree.fast._splitter.loss_kind_for`,
    so a registered subclass of a built-in uses the registered solver for
    both split search and leaf values.
    """
    # The registered cfunc is callable from Python too, so it serves the
    # Python paths (pruning, leaf payloads).
    from .fast._splitter import _lookup_user_kernel
    user_entry = _lookup_user_kernel(loss)
    if user_entry is not None:
        return user_entry
    if isinstance(loss, SquaredLoss):
        return _leaf_loss_squared, _leaf_alpha_squared
    if isinstance(loss, KLLoss):
        return _leaf_loss_kl, _leaf_alpha_kl
    if isinstance(loss, BernoulliLoss):
        return _leaf_loss_bernoulli, _leaf_alpha_bernoulli
    if isinstance(loss, BoundedSquaredLoss):
        lo = float(getattr(loss, "lo", 0.0))
        hi = float(getattr(loss, "hi", 1.0))
        loss_fn = lambda D, C, _lo=lo, _hi=hi: _leaf_loss_bounded_squared(D, C, _lo, _hi)
        alpha_fn = lambda D, C, _lo=lo, _hi=hi: _leaf_alpha_bounded_squared(D, C, _lo, _hi)
        return loss_fn, alpha_fn

    raise NotImplementedError(
        f"riesztree has no analytic leaf solver for loss type "
        f"{type(loss).__name__}. Built-in support: SquaredLoss, KLLoss, "
        "BernoulliLoss, BoundedSquaredLoss. To use a custom Loss subclass, "
        "either extend riesztree.splitter.make_leaf_solvers or register "
        "a Cython-callable kernel via "
        "riesztree.fast.register_fast_leaf_solver(MyLoss, leaf_loss_cfunc, alpha_at_opt)."
    )


# ---------------------------------------------------------------------------
# Per-feature split sweep.

def best_split_continuous(
    feature_col: np.ndarray,
    D: np.ndarray,
    C: np.ndarray,
    idx: np.ndarray,
    leaf_loss,
    *,
    min_orig_leaf: int,
):
    """Best gain split on a continuous feature for the rows in ``idx``.

    Pure-Python reference for the Cython sweep in
    :mod:`riesztree.fast._splitter_c` (the parity tests compare the two).
    Returns ``(gain, threshold, left_idx, right_idx)`` or ``None``.
    """
    vals = feature_col[idx]
    order = np.argsort(vals, kind="mergesort")
    sidx = idx[order]
    svals = feature_col[sidx]
    sD = D[sidx]
    sC = C[sidx]
    sorig = (sD > 0).astype(np.int64)

    cum_D = np.cumsum(sD)
    cum_C = np.cumsum(sC)
    cum_orig = np.cumsum(sorig)
    total_D = cum_D[-1]
    total_C = cum_C[-1]
    total_orig = cum_orig[-1]
    parent_loss = leaf_loss(float(total_D), float(total_C))

    best = None
    n = len(svals)
    for k in range(n - 1):
        if svals[k] == svals[k + 1]:
            continue
        D_l = float(cum_D[k])
        C_l = float(cum_C[k])
        n_l = int(cum_orig[k])
        D_r = float(total_D - D_l)
        C_r = float(total_C - C_l)
        n_r = int(total_orig - n_l)
        if n_l < min_orig_leaf or n_r < min_orig_leaf:
            continue
        L_l = leaf_loss(D_l, C_l)
        L_r = leaf_loss(D_r, C_r)
        if not np.isfinite(L_l) or not np.isfinite(L_r):
            continue
        gain = parent_loss - L_l - L_r
        if best is None or gain > best[0]:
            thresh = 0.5 * (svals[k] + svals[k + 1])
            if thresh == svals[k + 1] or not np.isfinite(thresh):
                thresh = svals[k]  # midpoint rounded onto the right value
            best = (gain, float(thresh), sidx[: k + 1].copy(), sidx[k + 1:].copy())
    return best


def best_split_categorical(
    feature_col: np.ndarray,
    D: np.ndarray,
    C: np.ndarray,
    idx: np.ndarray,
    leaf_loss,
    alpha_at_opt,
    *,
    min_orig_leaf: int,
):
    """Best gain split on a categorical feature for rows in ``idx``.

    Implements the standard Breiman et al. trick: order the levels by the
    within-level optimal α (``α*_level = -C_level / D_level`` projected to
    the loss's α-domain), then sweep contiguous splits in that order. The
    ordering theorem holds for any convex per-level objective, including
    the augmented Bregman loss.
    """
    vals = feature_col[idx]
    levels, inverse = np.unique(vals, return_inverse=True)
    if len(levels) < 2:
        return None

    # Per-level (D, C, n_orig) sums.
    n_lev = len(levels)
    D_idx = D[idx]
    D_lev = np.bincount(inverse, weights=D_idx, minlength=n_lev)
    C_lev = np.bincount(inverse, weights=C[idx], minlength=n_lev)
    n_orig_lev = np.bincount(inverse, weights=(D_idx > 0), minlength=n_lev).astype(np.int64)

    # Sort levels by α*_level.
    a_star_lev = np.array(
        [alpha_at_opt(float(d), float(c)) for d, c in zip(D_lev, C_lev)]
    )
    order = np.argsort(a_star_lev, kind="mergesort")

    cum_D = np.cumsum(D_lev[order])
    cum_C = np.cumsum(C_lev[order])
    cum_orig = np.cumsum(n_orig_lev[order])
    total_D = cum_D[-1]
    total_C = cum_C[-1]
    total_orig = cum_orig[-1]
    parent_loss = leaf_loss(float(total_D), float(total_C))

    # Rank of each level in the α*-sorted order; a split at k sends the
    # levels with rank ≤ k left.
    rank = np.empty(n_lev, dtype=np.int64)
    rank[order] = np.arange(n_lev)
    row_rank = rank[inverse]

    best = None
    for k in range(len(order) - 1):
        D_l = float(cum_D[k])
        C_l = float(cum_C[k])
        n_l = int(cum_orig[k])
        D_r = float(total_D - D_l)
        C_r = float(total_C - C_l)
        n_r = int(total_orig - n_l)
        if n_l < min_orig_leaf or n_r < min_orig_leaf:
            continue
        L_l = leaf_loss(D_l, C_l)
        L_r = leaf_loss(D_r, C_r)
        if not np.isfinite(L_l) or not np.isfinite(L_r):
            continue
        gain = parent_loss - L_l - L_r
        if best is None or gain > best[0]:
            best = (gain, k)
    if best is None:
        return None
    gain, k = best
    goes_left = row_rank <= k
    return (
        gain,
        tuple(int(levels[i]) for i in order[: k + 1]),
        idx[goes_left],
        idx[~goes_left],
    )
