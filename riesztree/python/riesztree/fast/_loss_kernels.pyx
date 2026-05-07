# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
"""Cython kernels for the four built-in Bregman-Riesz losses.

Each loss is identified by an integer ``loss_kind`` (the ``LOSS_*``
module constants). The two ``dispatch_*`` functions select among the
inlined kernels via a switch — at -O2 the compiler typically inlines
the chosen branch back into the caller, leaving no per-call dispatch
cost. Phase 5 will add a Numba ``@cfunc`` registration path that
replaces the switch with a function pointer when the user supplies a
custom loss.

Math is identical to ``riesztree/splitter.py``'s ``make_leaf_solvers``
dispatcher; this file is the C-speed mirror used by the Cython splitter.
"""

from libc.math cimport log, INFINITY, NAN, isfinite


# Public loss-kind constants. The Python ``_splitter._loss_kind_for``
# helper picks one based on the LossSpec subclass.
cdef int LOSS_SQUARED = 0
cdef int LOSS_KL = 1
cdef int LOSS_BERNOULLI = 2
cdef int LOSS_BOUNDED_SQUARED = 3

LOSS_SQUARED_PY = LOSS_SQUARED
LOSS_KL_PY = LOSS_KL
LOSS_BERNOULLI_PY = LOSS_BERNOULLI
LOSS_BOUNDED_SQUARED_PY = LOSS_BOUNDED_SQUARED


# ---------------------------------------------------------------------------
# Leaf-loss kernels.

cdef inline double squared_leaf_loss(double D, double C) noexcept nogil:
    if D <= 0.0:
        return 0.0
    return -C * C / D


cdef inline double kl_leaf_loss(double D, double C) noexcept nogil:
    # Boundary cases match riesztree.splitter._leaf_loss_kl:
    #   D <= 0  -> 0
    #   C == 0  -> 0    (boundary value: infimum at α = 0+)
    #   C > 0   -> +inf (infeasible — disqualifies this split)
    # Otherwise the closed-form L(α*) = -C + C·log(-C / D).
    if D <= 0.0:
        return 0.0
    if C == 0.0:
        return 0.0
    if C > 0.0:
        return INFINITY
    return -C + C * log(-C / D)


cdef inline double bernoulli_leaf_loss(double D, double C) noexcept nogil:
    # Boundary cases match riesztree.splitter._leaf_loss_bernoulli:
    #   D <= 0  -> 0
    #   C == 0  -> 0
    #   not in (-D, 0) -> +inf (infeasible)
    # Otherwise L(α*) = D·log D - (D+C)·log(D+C) + C·log(-C).
    if D <= 0.0:
        return 0.0
    if C == 0.0:
        return 0.0
    if not (-D < C < 0.0):
        return INFINITY
    return D * log(D) - (D + C) * log(D + C) + C * log(-C)


cdef inline double bounded_squared_leaf_loss(
    double D, double C, double lo, double hi
) noexcept nogil:
    # Project α* into [lo, hi] then evaluate D·α² + 2C·α.
    if D <= 0.0:
        return 0.0
    cdef double a = -C / D
    if a < lo:
        a = lo
    elif a > hi:
        a = hi
    return D * a * a + 2.0 * C * a


cdef double dispatch_leaf_loss(
    int loss_kind, double D, double C, double bounded_lo, double bounded_hi
) noexcept nogil:
    if loss_kind == LOSS_SQUARED:
        return squared_leaf_loss(D, C)
    elif loss_kind == LOSS_KL:
        return kl_leaf_loss(D, C)
    elif loss_kind == LOSS_BERNOULLI:
        return bernoulli_leaf_loss(D, C)
    elif loss_kind == LOSS_BOUNDED_SQUARED:
        return bounded_squared_leaf_loss(D, C, bounded_lo, bounded_hi)
    else:
        return NAN


# ---------------------------------------------------------------------------
# Alpha-at-opt kernels.

cdef inline double squared_alpha_at_opt(double D, double C) noexcept nogil:
    if D <= 0.0:
        return 0.0
    return -C / D


cdef inline double kl_alpha_at_opt(double D, double C) noexcept nogil:
    # KL: α* = -C/D, defined only when C < 0; if not, return 0 (the leaf
    # would have been disqualified upstream by the +inf leaf-loss).
    if D <= 0.0 or C >= 0.0:
        return 0.0
    return -C / D


cdef inline double bernoulli_alpha_at_opt(double D, double C) noexcept nogil:
    """Match Python ``_leaf_alpha_bernoulli`` exactly: project into
    the open interval (0, 1) using a 1e-6 epsilon. The αs at the
    open boundary are link-domain-invalid so the leaf would predict
    ±∞ in η-space; we clip into the interior instead."""
    if D <= 0.0:
        return 0.5
    cdef double a = -C / D
    if a <= 0.0:
        return 1e-6
    if a >= 1.0:
        return 1.0 - 1e-6
    return a


cdef inline double bounded_squared_alpha_at_opt(
    double D, double C, double lo, double hi
) noexcept nogil:
    """Match Python ``_leaf_alpha_bounded_squared`` exactly: project
    into [lo, hi]; D=0 → midpoint."""
    if D <= 0.0:
        return 0.5 * (lo + hi)
    cdef double a = -C / D
    if a < lo:
        return lo
    if a > hi:
        return hi
    return a


cdef double dispatch_alpha_at_opt(
    int loss_kind, double D, double C, double bounded_lo, double bounded_hi
) noexcept nogil:
    if loss_kind == LOSS_SQUARED:
        return squared_alpha_at_opt(D, C)
    elif loss_kind == LOSS_KL:
        return kl_alpha_at_opt(D, C)
    elif loss_kind == LOSS_BERNOULLI:
        return bernoulli_alpha_at_opt(D, C)
    elif loss_kind == LOSS_BOUNDED_SQUARED:
        return bounded_squared_alpha_at_opt(D, C, bounded_lo, bounded_hi)
    else:
        return NAN


# ---------------------------------------------------------------------------
# Python-side helpers for tests / introspection.

def py_dispatch_leaf_loss(int loss_kind, double D, double C, double lo, double hi):
    return dispatch_leaf_loss(loss_kind, D, C, lo, hi)


def py_dispatch_alpha_at_opt(int loss_kind, double D, double C, double lo, double hi):
    return dispatch_alpha_at_opt(loss_kind, D, C, lo, hi)
