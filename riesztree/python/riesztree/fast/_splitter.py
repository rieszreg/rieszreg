"""Python-side facade for the Cython splitter.

Translates a ``Loss`` instance to the integer ``loss_kind`` used by
the compiled kernels, plus the ``(lo, hi)`` parameters needed by
``BoundedSquaredLoss``. Custom user losses (anything not in the four
built-ins) can plug in via :func:`register_fast_leaf_solver`, which
registers a Numba ``@cfunc`` (or any C-callable with the matching
signature) for that loss subclass; the dispatcher in
:mod:`riesztree.grow` then routes continuous splits through the
user's compiled kernel at C speed. If no kernel is registered the
dispatcher falls back to the pure-Python splitter with a one-time
``UserWarning``.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from rieszreg import (
    BernoulliLoss,
    BoundedSquaredLoss,
    KLLoss,
    Loss,
    SquaredLoss,
)

# Loss-kind constants exported from the compiled extension. Re-exported
# here for tests and for the grow.py dispatcher to use.
try:  # extension imports lazily so a fresh source checkout still loads
    from ._loss_kernels import (
        LOSS_BERNOULLI_PY as LOSS_BERNOULLI,
        LOSS_BOUNDED_SQUARED_PY as LOSS_BOUNDED_SQUARED,
        LOSS_KL_PY as LOSS_KL,
        LOSS_SQUARED_PY as LOSS_SQUARED,
    )
except ImportError:
    LOSS_SQUARED = 0
    LOSS_KL = 1
    LOSS_BERNOULLI = 2
    LOSS_BOUNDED_SQUARED = 3


# ---------------------------------------------------------------------------
# User-loss registry: subclass-of-Loss →
# (leaf_loss_cfunc_address, alpha_at_opt_python_callable).
#
# The leaf-loss is a C-callable address used by the Cython splitter's
# tight loop; the alpha-at-opt is a Python callable used by ``_make_leaf``
# to populate the leaf's α* payload (and by ``cost_complexity_prune``).

_USER_LOSS_REGISTRY: dict[type, tuple[int, callable]] = {}
# Side mapping address → cfunc object so the Python-side
# ``make_leaf_solvers`` can hand a Python-callable wrapper to
# ``_make_leaf`` / pruning. Numba cfuncs are themselves callable
# from Python; storing the object keeps that callable alive.
_USER_CFUNC_OBJECTS: dict[int, callable] = {}


def register_fast_leaf_solver(
    loss_class: type,
    leaf_loss_addr_or_cfunc,
    alpha_at_opt,
) -> None:
    """Plug a custom loss into the Cython splitter.

    Parameters
    ----------
    loss_class
        The user's Loss subclass. The splitter uses MRO to
        match — registering the most specific class wins.
    leaf_loss_addr_or_cfunc
        Either an integer (a raw C-callable address) or an object
        with a ``.address`` attribute (e.g. a Numba ``@cfunc``).
        Signature ``double(double, double)``; called from the Cython
        splitter without the GIL.
    alpha_at_opt
        Python callable ``(D, C) -> alpha_star``. Used by
        ``_make_leaf`` to populate the leaf payload and by pruning to
        compute the would-be-leaf α at internal nodes. A
        Numba ``@cfunc`` instance also works (it's a Python callable
        too); plain ``def`` is fine if perf isn't critical here.

    Worked example::

        import numba
        from rieszreg import Loss
        from riesztree.fast import register_fast_leaf_solver

        class MyLoss(Loss):
            ...

        @numba.cfunc("float64(float64, float64)", cache=True, nopython=True)
        def my_leaf_loss(D, C):
            if D <= 0.0:
                return 0.0
            return -C * C / D    # SquaredLoss equivalent

        def my_alpha(D, C):
            return 0.0 if D <= 0 else -C / D

        register_fast_leaf_solver(MyLoss, my_leaf_loss, my_alpha)
        # Subsequent fits with loss=MyLoss() now use the C-speed kernel.
    """
    if hasattr(leaf_loss_addr_or_cfunc, "address"):
        addr = int(leaf_loss_addr_or_cfunc.address)
        cfunc_obj = leaf_loss_addr_or_cfunc
    else:
        addr = int(leaf_loss_addr_or_cfunc)
        cfunc_obj = None
    if addr <= 0:
        raise ValueError(
            f"register_fast_leaf_solver: invalid C-callable address {addr!r} "
            f"for {loss_class.__name__}; expected a positive integer "
            f"(typically a numba.cfunc's .address)."
        )
    if not callable(alpha_at_opt):
        raise TypeError(
            f"register_fast_leaf_solver: alpha_at_opt for "
            f"{loss_class.__name__} must be callable; got "
            f"{type(alpha_at_opt).__name__}."
        )
    _USER_LOSS_REGISTRY[loss_class] = (addr, alpha_at_opt)
    # Track a Python-callable wrapper for the leaf-loss too, so the
    # Python paths (pruning, leaf-payload computation, holdout-loss)
    # can share the same kernel. cfunc instances are themselves
    # callable from Python; raw addresses lose that ability.
    if cfunc_obj is not None:
        _USER_CFUNC_OBJECTS[addr] = cfunc_obj


def _lookup_user_kernel(loss: Loss) -> tuple[int, callable] | None:
    """Return ``(leaf_loss_addr, alpha_at_opt_callable)`` for the
    closest registered ancestor of ``type(loss)``, or ``None``."""
    for cls in type(loss).__mro__:
        if cls in _USER_LOSS_REGISTRY:
            return _USER_LOSS_REGISTRY[cls]
    return None


def lookup_user_alpha_at_opt(loss: Loss):
    """Public accessor: returns the registered Python alpha_at_opt
    callable for ``loss``, or ``None`` if not registered. Used by
    ``riesztree.splitter.make_leaf_solvers`` to extend its dispatch
    to user-registered losses."""
    found = _lookup_user_kernel(loss)
    if found is None:
        return None
    return found[1]


# Sentinel ``loss_kind`` used by :func:`best_split_continuous_fast` to
# signal "use the user-cfunc entry point". Distinct from the four
# built-in ``LOSS_*`` values.
LOSS_USER_CFUNC = -1


def loss_kind_for(
    loss: Loss,
) -> tuple[int, float, float, int] | None:
    """Map a Loss to ``(loss_kind, bounded_lo, bounded_hi, user_addr)``.

    Returns ``None`` only when the loss is *neither* a built-in *nor*
    in the user registry. For built-ins ``user_addr`` is 0; for
    user-registered losses ``loss_kind`` is :data:`LOSS_USER_CFUNC`
    and ``user_addr`` is the registered C-callable address (the
    other slots are unused).

    The user registry is checked first so that users can override
    built-in behaviour (e.g. plug in a faster KL kernel).
    """
    user_entry = _lookup_user_kernel(loss)
    if user_entry is not None:
        addr, _alpha_fn = user_entry
        return LOSS_USER_CFUNC, np.nan, np.nan, int(addr)
    if isinstance(loss, SquaredLoss):
        return LOSS_SQUARED, np.nan, np.nan, 0
    if isinstance(loss, KLLoss):
        return LOSS_KL, np.nan, np.nan, 0
    if isinstance(loss, BernoulliLoss):
        return LOSS_BERNOULLI, np.nan, np.nan, 0
    if isinstance(loss, BoundedSquaredLoss):
        return LOSS_BOUNDED_SQUARED, float(loss.lo), float(loss.hi), 0
    return None


def accumulate_hist(
    X_binned: np.ndarray,
    D: np.ndarray,
    C: np.ndarray,
    idx: np.ndarray,
    candidate_features: np.ndarray,
    max_bins: int,
):
    """Build per-feature histograms over the rows in ``idx``.

    Returns ``(hD, hC, hO, total_D, total_C, total_orig)``. See
    :mod:`riesztree.fast._splitter_hist.accumulate_hist_c` for the
    contract. Used by the parent-minus-sibling path in
    :mod:`riesztree.grow`.
    """
    from . import _splitter_hist  # type: ignore[attr-defined]
    candidate_features = np.ascontiguousarray(candidate_features, dtype=np.int32)
    return _splitter_hist.accumulate_hist_c(
        X_binned, D, C, idx, candidate_features, int(max_bins),
    )


def find_best_split_in_hist(
    hD: np.ndarray,
    hC: np.ndarray,
    hO: np.ndarray,
    total_D: float,
    total_C: float,
    total_orig: int,
    candidate_features: np.ndarray,
    n_bins_per_feature: np.ndarray,
    *,
    loss_kind: int,
    bounded_lo: float,
    bounded_hi: float,
    min_orig_leaf: int,
):
    """Find the best split given pre-built histograms.

    Returns ``(best_feat, best_bin, gain)`` or ``None``.
    """
    from . import _splitter_hist  # type: ignore[attr-defined]
    candidate_features = np.ascontiguousarray(candidate_features, dtype=np.int32)
    n_bins_per_feature = np.ascontiguousarray(n_bins_per_feature, dtype=np.int32)
    return _splitter_hist.find_best_split_in_hist_c(
        hD, hC, hO, float(total_D), float(total_C), int(total_orig),
        candidate_features, n_bins_per_feature,
        int(loss_kind), float(bounded_lo), float(bounded_hi),
        int(min_orig_leaf),
    )


def partition_idx_by_bin(X_binned: np.ndarray, idx: np.ndarray, best_feat: int, best_bin: int):
    """Partition ``idx`` into (left, right) on ``X_binned[:, best_feat] <= best_bin``."""
    from . import _splitter_hist  # type: ignore[attr-defined]
    return _splitter_hist.partition_idx_by_bin_c(X_binned, idx, int(best_feat), int(best_bin))


def best_split_at_hist(
    X_binned: np.ndarray,
    D: np.ndarray,
    C: np.ndarray,
    idx: np.ndarray,
    *,
    bin_thresholds: list,
    n_bins_per_feature: np.ndarray,
    candidate_features: np.ndarray,
    loss_kind: int,
    bounded_lo: float,
    bounded_hi: float,
    min_orig_leaf: int,
    max_bins: int,
):
    """Cython histogram-based best-split sweep across all candidate features.

    Returns ``(best_feat, gain, threshold, left_idx, right_idx)`` or
    ``None`` — same shape as the per-feature exact splitter, except
    that ``best_feat`` is the global winner across the candidate
    features (the per-feature dispatch loop happens inside the Cython
    kernel for less overhead).
    """
    from . import _splitter_hist  # type: ignore[attr-defined]

    return _splitter_hist.best_split_continuous_hist(
        X_binned, D, C, idx,
        n_bins_per_feature, candidate_features, bin_thresholds,
        int(loss_kind), float(bounded_lo), float(bounded_hi),
        int(min_orig_leaf), int(max_bins),
    )


def best_split_continuous_random(
    feature_col: np.ndarray,
    D: np.ndarray,
    C: np.ndarray,
    idx: np.ndarray,
    *,
    loss_kind: int,
    bounded_lo: float,
    bounded_hi: float,
    min_orig_leaf: int,
    rng: np.random.Generator,
):
    """Random-threshold split (sklearn ``splitter='random'``).

    Draws one uniform threshold in ``[col_min, col_max]`` from ``rng``
    and evaluates the gain at that single point in Cython. Returns
    ``None`` when the column is constant (no valid threshold exists).
    """
    if idx.size < 2:
        return None
    feature_col = np.ascontiguousarray(feature_col, dtype=np.float64)
    D = np.ascontiguousarray(D, dtype=np.float64)
    C = np.ascontiguousarray(C, dtype=np.float64)
    idx = np.ascontiguousarray(idx, dtype=np.int64)

    leaf_vals = feature_col[idx]
    lo = float(leaf_vals.min())
    hi = float(leaf_vals.max())
    if lo == hi:
        return None  # constant column — no split possible
    threshold = float(rng.uniform(lo, hi))

    from . import _splitter_c  # type: ignore[attr-defined]
    return _splitter_c.best_split_continuous_random_c(
        feature_col, D, C, idx,
        int(loss_kind), float(bounded_lo), float(bounded_hi),
        int(min_orig_leaf), threshold,
    )


def best_split_continuous_fast(
    feature_col: np.ndarray,
    D: np.ndarray,
    C: np.ndarray,
    idx: np.ndarray,
    *,
    loss_kind: int,
    bounded_lo: float,
    bounded_hi: float,
    min_orig_leaf: int,
    user_cfunc_addr: int = 0,
):
    """Fast Cython best-split sweep on a continuous feature.

    Routes between the built-in dispatcher (one of the four
    ``LOSS_*`` integer kinds) and the user-cfunc entry point
    (``LOSS_USER_CFUNC``, with ``user_cfunc_addr`` carrying the
    function address). Falls back to the pure-Python
    ``best_split_continuous`` if the compiled extension hasn't been
    built.
    """
    try:
        from . import _splitter_c  # type: ignore[attr-defined]
    except ImportError:
        from ..splitter import best_split_continuous

        # Build a Python leaf_loss closure compatible with the Python splitter.
        from ._loss_kernels import py_dispatch_leaf_loss

        def _leaf_loss(D_, C_):
            return py_dispatch_leaf_loss(loss_kind, D_, C_, bounded_lo, bounded_hi)

        return best_split_continuous(
            feature_col, D, C, idx, _leaf_loss, min_orig_leaf=min_orig_leaf
        )

    feature_col = np.ascontiguousarray(feature_col, dtype=np.float64)
    D = np.ascontiguousarray(D, dtype=np.float64)
    C = np.ascontiguousarray(C, dtype=np.float64)
    idx = np.ascontiguousarray(idx, dtype=np.int64)

    if loss_kind == LOSS_USER_CFUNC:
        return _splitter_c.best_split_continuous_user_c(
            feature_col, D, C, idx,
            int(user_cfunc_addr),
            int(min_orig_leaf),
        )
    return _splitter_c.best_split_continuous_c(
        feature_col, D, C, idx,
        int(loss_kind), float(bounded_lo), float(bounded_hi),
        int(min_orig_leaf),
    )


_WARNED_FALLBACK: set[type] = set()


def warn_python_fallback(loss: Loss) -> None:
    """Emit a one-time UserWarning when the user's loss isn't in the four
    built-ins and growth has to use the Python splitter."""
    cls = type(loss)
    if cls in _WARNED_FALLBACK:
        return
    _WARNED_FALLBACK.add(cls)
    warnings.warn(
        f"Loss {cls.__name__!r} is not among the riesztree fast-splitter "
        f"built-ins (Squared/KL/Bernoulli/BoundedSquared); falling back "
        f"to the pure-Python splitter. Phase 5 will add a registration "
        f"hook for Numba @cfunc kernels.",
        UserWarning,
        stacklevel=3,
    )
