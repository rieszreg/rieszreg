# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
"""Cython histogram-based best-split sweep.

Per-leaf algorithm:
  1. Accumulate a histogram of (sum_D, sum_C, count_orig) per bin
     for each feature, over the rows in the leaf. O(n_leaf · p).
  2. For each feature, sweep ``n_bins - 1`` candidate split positions,
     computing left/right (D, C, orig_count) via running cumulatives
     and evaluating the gain via the loss-aware leaf-loss kernel.
     O(p · max_bins).
  3. Track the best split across features.

This eliminates the per-leaf ``np.argsort`` that dominates the
exact splitter's cost for large ``n_leaf``. Bin width is set by the
:class:`riesztree.fast._binner.BinMapper` at fit start (default 255
quantile bins per feature).

Approximation: the best split lies on a bin boundary, not on a
distinct value. With 255 bins on quantile-sampled data the gap to the
exact split is typically negligible for split-ranking; the leaf α*
itself is unaffected by binning (it depends only on (D_leaf, C_leaf),
both summed exactly within the leaf).
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport INFINITY, isfinite

from ._loss_kernels cimport dispatch_leaf_loss

ctypedef cnp.float64_t f64
ctypedef cnp.int64_t i64
ctypedef cnp.uint8_t u8


def best_split_continuous_hist(
    cnp.ndarray[u8, ndim=2, mode="c"] X_binned,    # (n_aug, p)
    cnp.ndarray[f64, ndim=1] D,                   # (n_aug,)
    cnp.ndarray[f64, ndim=1] C,                   # (n_aug,)
    cnp.ndarray[i64, ndim=1] idx,                 # leaf row indices
    cnp.ndarray[cnp.int32_t, ndim=1] n_bins_per_feature,
    cnp.ndarray[cnp.int32_t, ndim=1] candidate_features,
    list bin_thresholds,                          # per-feature thresholds (Python list of f64 arrays)
    int loss_kind,
    double bounded_lo,
    double bounded_hi,
    int min_orig_leaf,
    int max_bins,
):
    """Return ``(best_feat, gain, threshold, left_idx, right_idx)`` or ``None``.

    Unlike the exact splitter (which returns split components per feature
    and is dispatched in Python per feature), this kernel walks every
    candidate feature internally and returns the global best — so the
    per-feature dispatch overhead also disappears.
    """
    cdef Py_ssize_t n = idx.shape[0]
    if n < 2:
        return None

    cdef Py_ssize_t n_features_to_try = candidate_features.shape[0]
    if n_features_to_try == 0:
        return None

    cdef Py_ssize_t i, j_idx, j, b
    cdef i64 row
    cdef u8 bin_idx
    cdef i64[::1] idx_v = idx
    cdef u8[:, ::1] X_binned_v = X_binned
    cdef f64[::1] D_v = D
    cdef f64[::1] C_v = C
    cdef cnp.int32_t[::1] n_bins_v = n_bins_per_feature
    cdef cnp.int32_t[::1] cand_v = candidate_features

    # Allocate per-feature histograms inline; reused across features.
    cdef cnp.ndarray[f64, ndim=1] hist_D = np.empty(max_bins, dtype=np.float64)
    cdef cnp.ndarray[f64, ndim=1] hist_C = np.empty(max_bins, dtype=np.float64)
    cdef cnp.ndarray[i64, ndim=1] hist_orig = np.empty(max_bins, dtype=np.int64)
    cdef f64[::1] hD = hist_D
    cdef f64[::1] hC = hist_C
    cdef i64[::1] hO = hist_orig

    # First pass: accumulate the leaf totals once. We can derive parent
    # (D, C, orig) directly without re-iterating per feature.
    cdef double total_D = 0.0
    cdef double total_C = 0.0
    cdef i64 total_orig = 0
    for i in range(n):
        row = idx_v[i]
        total_D += D_v[row]
        total_C += C_v[row]
        if D_v[row] > 0.0:
            total_orig += 1

    cdef double parent_loss = dispatch_leaf_loss(
        loss_kind, total_D, total_C, bounded_lo, bounded_hi
    )

    cdef double best_gain = -INFINITY
    cdef Py_ssize_t best_feat = -1
    cdef Py_ssize_t best_bin = -1
    cdef double D_l, C_l, D_r, C_r, L_l, L_r, gain
    cdef i64 n_l, n_r
    cdef cnp.int32_t nb

    for j_idx in range(n_features_to_try):
        j = cand_v[j_idx]
        nb = n_bins_v[j]
        if nb < 2:
            continue

        # Zero the histogram slice we'll touch.
        for b in range(nb):
            hD[b] = 0.0
            hC[b] = 0.0
            hO[b] = 0
        # Accumulate.
        for i in range(n):
            row = idx_v[i]
            bin_idx = X_binned_v[row, j]
            hD[bin_idx] += D_v[row]
            hC[bin_idx] += C_v[row]
            if D_v[row] > 0.0:
                hO[bin_idx] += 1

        # Sweep candidate splits: split = "left includes bins [0..b]".
        # Running cumulatives reuse the histogram in-place.
        D_l = 0.0
        C_l = 0.0
        n_l = 0
        for b in range(nb - 1):
            D_l += hD[b]
            C_l += hC[b]
            n_l += hO[b]
            n_r = total_orig - n_l
            if n_l < min_orig_leaf or n_r < min_orig_leaf:
                continue
            D_r = total_D - D_l
            C_r = total_C - C_l
            L_l = dispatch_leaf_loss(loss_kind, D_l, C_l, bounded_lo, bounded_hi)
            L_r = dispatch_leaf_loss(loss_kind, D_r, C_r, bounded_lo, bounded_hi)
            if not isfinite(L_l) or not isfinite(L_r):
                continue
            gain = parent_loss - L_l - L_r
            if gain > best_gain:
                best_gain = gain
                best_feat = j
                best_bin = b

    if best_feat < 0:
        return None

    # Reconstruct threshold from bin boundary. Threshold is the
    # right-inclusive boundary of bin best_bin.
    cdef cnp.ndarray[f64, ndim=1] thr = bin_thresholds[best_feat]
    if thr.size == 0:
        # Single-bin feature — should never reach this branch
        # because we skip nb < 2 above. Defensive fallback.
        return None
    cdef double threshold = thr[best_bin]

    # Build left/right index arrays via a single pass.
    cdef cnp.ndarray[i64, ndim=1] left_idx = np.empty(n, dtype=np.int64)
    cdef cnp.ndarray[i64, ndim=1] right_idx = np.empty(n, dtype=np.int64)
    cdef Py_ssize_t k_l = 0, k_r = 0
    for i in range(n):
        row = idx_v[i]
        if X_binned_v[row, best_feat] <= best_bin:
            left_idx[k_l] = row
            k_l += 1
        else:
            right_idx[k_r] = row
            k_r += 1
    return (
        int(best_feat),
        float(best_gain),
        float(threshold),
        np.asarray(left_idx[:k_l]),
        np.asarray(right_idx[:k_r]),
    )


# ---------------------------------------------------------------------------
# Parent-minus-sibling primitives. Split the monolithic kernel into
# accumulate + find-best so the Python driver can maintain per-leaf
# histograms and compute the larger child's histogram by subtraction
# (parent - smaller_child) instead of re-accumulating it from rows.

def accumulate_hist_c(
    cnp.ndarray[u8, ndim=2, mode="c"] X_binned,
    cnp.ndarray[f64, ndim=1] D,
    cnp.ndarray[f64, ndim=1] C,
    cnp.ndarray[i64, ndim=1] idx,
    cnp.ndarray[cnp.int32_t, ndim=1] candidate_features,
    int max_bins,
):
    """Build per-feature histograms over the rows in ``idx``.

    Returns ``(hD, hC, hO, total_D, total_C, total_orig)``:
      hD : float64[n_candidate, max_bins]   — per-feature sum of D
      hC : float64[n_candidate, max_bins]   — per-feature sum of C
      hO : int64[n_candidate, max_bins]     — per-feature count of original rows
      total_D, total_C : double             — leaf-wide aggregates
      total_orig : int64                     — leaf-wide original-row count

    The candidate-feature column index ``j_idx`` (0..n_candidate) maps
    back to the actual feature index via ``candidate_features[j_idx]``.
    """
    cdef Py_ssize_t n = idx.shape[0]
    cdef Py_ssize_t n_features = candidate_features.shape[0]
    cdef cnp.ndarray[f64, ndim=2, mode="c"] hD = np.zeros((n_features, max_bins), dtype=np.float64)
    cdef cnp.ndarray[f64, ndim=2, mode="c"] hC = np.zeros((n_features, max_bins), dtype=np.float64)
    cdef cnp.ndarray[i64, ndim=2, mode="c"] hO = np.zeros((n_features, max_bins), dtype=np.int64)
    cdef f64[:, ::1] hD_v = hD
    cdef f64[:, ::1] hC_v = hC
    cdef i64[:, ::1] hO_v = hO
    cdef u8[:, ::1] X_binned_v = X_binned
    cdef f64[::1] D_v = D
    cdef f64[::1] C_v = C
    cdef i64[::1] idx_v = idx
    cdef cnp.int32_t[::1] cand_v = candidate_features

    cdef Py_ssize_t i, j_idx
    cdef i64 row, j
    cdef u8 bin_idx
    cdef double total_D = 0.0
    cdef double total_C = 0.0
    cdef i64 total_orig = 0
    cdef int is_orig

    for i in range(n):
        row = idx_v[i]
        total_D += D_v[row]
        total_C += C_v[row]
        is_orig = 1 if D_v[row] > 0.0 else 0
        total_orig += is_orig
        for j_idx in range(n_features):
            j = cand_v[j_idx]
            bin_idx = X_binned_v[row, j]
            hD_v[j_idx, bin_idx] += D_v[row]
            hC_v[j_idx, bin_idx] += C_v[row]
            hO_v[j_idx, bin_idx] += is_orig

    return hD, hC, hO, total_D, total_C, total_orig


def find_best_split_in_hist_c(
    cnp.ndarray[f64, ndim=2, mode="c"] hD,
    cnp.ndarray[f64, ndim=2, mode="c"] hC,
    cnp.ndarray[i64, ndim=2, mode="c"] hO,
    double total_D,
    double total_C,
    i64 total_orig,
    cnp.ndarray[cnp.int32_t, ndim=1] candidate_features,
    cnp.ndarray[cnp.int32_t, ndim=1] n_bins_per_feature,
    int loss_kind,
    double bounded_lo,
    double bounded_hi,
    int min_orig_leaf,
):
    """Find the best split given pre-built per-feature histograms.

    Returns ``(best_feat, best_bin, gain)`` or ``None`` when no
    above-threshold split exists.
    """
    cdef Py_ssize_t n_features = candidate_features.shape[0]
    if n_features == 0:
        return None

    cdef f64[:, ::1] hD_v = hD
    cdef f64[:, ::1] hC_v = hC
    cdef i64[:, ::1] hO_v = hO
    cdef cnp.int32_t[::1] cand_v = candidate_features
    cdef cnp.int32_t[::1] n_bins_v = n_bins_per_feature

    cdef double parent_loss = dispatch_leaf_loss(
        loss_kind, total_D, total_C, bounded_lo, bounded_hi
    )
    cdef double best_gain = -INFINITY
    cdef Py_ssize_t best_feat = -1
    cdef Py_ssize_t best_bin = -1
    cdef double D_l, C_l, D_r, C_r, L_l, L_r, gain
    cdef i64 n_l, n_r
    cdef cnp.int32_t nb
    cdef Py_ssize_t j_idx, b
    cdef i64 j

    for j_idx in range(n_features):
        j = cand_v[j_idx]
        nb = n_bins_v[j]
        if nb < 2:
            continue
        D_l = 0.0
        C_l = 0.0
        n_l = 0
        for b in range(nb - 1):
            D_l += hD_v[j_idx, b]
            C_l += hC_v[j_idx, b]
            n_l += hO_v[j_idx, b]
            n_r = total_orig - n_l
            if n_l < min_orig_leaf or n_r < min_orig_leaf:
                continue
            D_r = total_D - D_l
            C_r = total_C - C_l
            L_l = dispatch_leaf_loss(loss_kind, D_l, C_l, bounded_lo, bounded_hi)
            L_r = dispatch_leaf_loss(loss_kind, D_r, C_r, bounded_lo, bounded_hi)
            if not isfinite(L_l) or not isfinite(L_r):
                continue
            gain = parent_loss - L_l - L_r
            if gain > best_gain:
                best_gain = gain
                best_feat = j
                best_bin = b

    if best_feat < 0:
        return None
    return int(best_feat), int(best_bin), float(best_gain)


def partition_idx_by_bin_c(
    cnp.ndarray[u8, ndim=2, mode="c"] X_binned,
    cnp.ndarray[i64, ndim=1] idx,
    int best_feat,
    int best_bin,
):
    """Partition ``idx`` on the bin boundary: left = rows with
    ``X_binned[row, best_feat] <= best_bin``."""
    cdef Py_ssize_t n = idx.shape[0]
    cdef cnp.ndarray[i64, ndim=1] left_idx = np.empty(n, dtype=np.int64)
    cdef cnp.ndarray[i64, ndim=1] right_idx = np.empty(n, dtype=np.int64)
    cdef u8[:, ::1] X_binned_v = X_binned
    cdef i64[::1] idx_v = idx
    cdef i64[::1] left_v = left_idx
    cdef i64[::1] right_v = right_idx
    cdef Py_ssize_t i, k_l = 0, k_r = 0
    cdef i64 row
    for i in range(n):
        row = idx_v[i]
        if X_binned_v[row, best_feat] <= best_bin:
            left_v[k_l] = row
            k_l += 1
        else:
            right_v[k_r] = row
            k_r += 1
    return np.asarray(left_idx[:k_l]), np.asarray(right_idx[:k_r])
