# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
"""Iterative depthwise grow loop for the exact (per-leaf-sort) splitter.

Mirrors :func:`riesztree.fast._grow_c.grow_depthwise_hist_c` (the
histogram-path Cython driver) but operates on raw continuous features
instead of pre-binned features. Per leaf, for each candidate feature,
the active rows are sorted by feature value via a small in-Cython
quicksort and swept by the same gain formula the existing Cython
exact splitter already uses.

Eliminates two costs in the existing Python-driven exact path:

  1. The Python-facade overhead of
     :func:`riesztree.fast._splitter.best_split_continuous_fast` per
     (leaf × feature) — currently ~5 numpy roundtrips per call.

  2. The redundant ``np.ascontiguousarray(features[:, j], ...)`` per
     call, which copies a non-contiguous column view per
     (leaf × feature). Profiles show this alone is ~45% of fit time at
     unlimited depth.

Same scope guards as :func:`grow_depthwise_hist_c`: ``splitter='exact'``,
``growth_policy='depthwise'``, no categoricals, no max_features
subsampling, no early stopping, no validation set, built-in (or
user-registered) loss. Other configurations keep the existing Python
recursion in :mod:`riesztree.grow`.
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport INFINITY, isfinite
from libc.stdlib cimport malloc, free

from ._loss_kernels cimport dispatch_leaf_loss, dispatch_alpha_at_opt

ctypedef cnp.float64_t f64
ctypedef cnp.int64_t i64
ctypedef cnp.int32_t i32


# ---------------------------------------------------------------------------
# In-place quicksort on (value, index) pairs.
#
# Both arrays are reordered in lockstep so that ``svals`` ends up
# non-decreasing and ``sidx`` carries the corresponding row indices.
# Used per (leaf × feature) to compute cumulative sums in feature-sorted
# order. Median-of-three pivot + insertion-sort cutoff for small
# partitions; standard textbook quicksort otherwise. Average O(n log n);
# good enough at our typical leaf sizes (a few rows up to a few thousand).

cdef inline void _insertion_sort_pair(
    f64[::1] vals, i64[::1] idx, Py_ssize_t lo, Py_ssize_t hi,
) noexcept nogil:
    cdef Py_ssize_t i, j
    cdef double v
    cdef i64 k
    for i in range(lo + 1, hi + 1):
        v = vals[i]
        k = idx[i]
        j = i - 1
        while j >= lo and vals[j] > v:
            vals[j + 1] = vals[j]
            idx[j + 1] = idx[j]
            j -= 1
        vals[j + 1] = v
        idx[j + 1] = k


cdef inline void _swap_pair(
    f64[::1] vals, i64[::1] idx, Py_ssize_t a, Py_ssize_t b,
) noexcept nogil:
    cdef double tv = vals[a]
    cdef i64 ti = idx[a]
    vals[a] = vals[b]
    idx[a] = idx[b]
    vals[b] = tv
    idx[b] = ti


cdef void _quicksort_pair(
    f64[::1] vals, i64[::1] idx, Py_ssize_t lo, Py_ssize_t hi,
) noexcept nogil:
    """Quicksort vals[lo:hi+1] (inclusive), keeping idx in lockstep."""
    cdef Py_ssize_t mid
    cdef double pivot
    cdef Py_ssize_t i, j
    while hi - lo > 16:
        # Median-of-three pivot selection.
        mid = lo + ((hi - lo) >> 1)
        if vals[mid] < vals[lo]:
            _swap_pair(vals, idx, lo, mid)
        if vals[hi] < vals[lo]:
            _swap_pair(vals, idx, lo, hi)
        if vals[hi] < vals[mid]:
            _swap_pair(vals, idx, mid, hi)
        # Move pivot to position hi-1; partition.
        _swap_pair(vals, idx, mid, hi - 1)
        pivot = vals[hi - 1]
        i = lo
        j = hi - 1
        while True:
            i += 1
            while vals[i] < pivot:
                i += 1
            j -= 1
            while vals[j] > pivot:
                j -= 1
            if i >= j:
                break
            _swap_pair(vals, idx, i, j)
        # Restore pivot.
        _swap_pair(vals, idx, i, hi - 1)
        # Recurse on smaller side, iterate on the larger.
        if i - lo < hi - i:
            _quicksort_pair(vals, idx, lo, i - 1)
            lo = i + 1
        else:
            _quicksort_pair(vals, idx, i + 1, hi)
            hi = i - 1
    _insertion_sort_pair(vals, idx, lo, hi)


# ---------------------------------------------------------------------------
# In-place partition of the contiguous index slice on (feature, threshold).

cdef inline Py_ssize_t _partition_inplace_continuous(
    i64[::1] idx_buf,
    Py_ssize_t start,
    Py_ssize_t end,
    f64[:, ::1] features,
    int best_feat,
    double threshold,
) noexcept nogil:
    """Two-pointer partition of ``idx_buf[start:end]``: rows with
    ``features[row, best_feat] <= threshold`` go first. Returns the
    boundary ``mid`` such that ``idx_buf[start:mid]`` is the left child
    and ``idx_buf[mid:end]`` is the right child."""
    cdef Py_ssize_t lo = start
    cdef Py_ssize_t hi = end - 1
    cdef i64 tmp, row
    while lo <= hi:
        row = idx_buf[lo]
        if features[row, best_feat] <= threshold:
            lo += 1
        else:
            tmp = idx_buf[hi]
            idx_buf[hi] = row
            idx_buf[lo] = tmp
            hi -= 1
    return lo


# ---------------------------------------------------------------------------
# Best-split sweep across all features for a single leaf.
#
# Returns (best_feat, best_threshold, best_gain) via out parameters.
# Sets best_feat = -1 if no valid split exists.

cdef void _best_split_at_leaf(
    f64[:, ::1] features,
    f64[::1] D,
    f64[::1] C,
    i64[::1] idx_buf,
    Py_ssize_t start,
    Py_ssize_t end,
    int n_features,
    int loss_kind,
    double bounded_lo,
    double bounded_hi,
    int min_orig_leaf,
    f64[::1] svals,    # scratch, size >= n_leaf
    i64[::1] sidx,     # scratch
    f64[::1] cum_D,    # scratch
    f64[::1] cum_C,    # scratch
    i64[::1] cum_orig, # scratch
    int* best_feat_out,
    double* best_threshold_out,
    double* best_gain_out,
) noexcept nogil:
    """Best split across all features for the leaf rows ``idx_buf[start:end]``.

    Per-feature ``total_D`` / ``total_C`` / ``parent_loss`` are recomputed
    from the sort-order cumulative sums (``cum_*[n_leaf - 1]``) — this
    matches the per-feature float order used by the legacy Python and
    Cython splitters, preserving byte-equivalent gain comparisons across
    paths even on losses sensitive to float-precision (notably KLLoss,
    whose log() amplifies tiny differences in D / C).
    """
    cdef Py_ssize_t n_leaf = end - start
    cdef Py_ssize_t i, k
    cdef i64 row
    cdef int j
    cdef double total_D, total_C, parent_loss
    cdef i64 total_orig
    cdef double D_l, C_l, D_r, C_r, L_l, L_r, gain
    cdef i64 n_l, n_r
    cdef Py_ssize_t feat_best_k
    cdef double feat_best_gain, feat_best_thr
    cdef double best_gain_overall = -INFINITY
    cdef int best_feat = -1
    cdef double best_threshold = 0.0

    for j in range(n_features):
        # Extract leaf rows for this feature.
        for i in range(n_leaf):
            row = idx_buf[start + i]
            svals[i] = features[row, j]
            sidx[i] = row

        # In-place sort by feature value.
        _quicksort_pair(svals, sidx, 0, n_leaf - 1)

        # Cumulative sums in sorted order.
        D_l = 0.0
        C_l = 0.0
        n_l = 0
        for k in range(n_leaf):
            row = sidx[k]
            D_l += D[row]
            C_l += C[row]
            if D[row] > 0.0:
                n_l += 1
            cum_D[k] = D_l
            cum_C[k] = C_l
            cum_orig[k] = n_l

        # Per-feature parent totals derived from the sorted cumsum, to
        # match the float-precision contract of the legacy splitters.
        total_D = cum_D[n_leaf - 1]
        total_C = cum_C[n_leaf - 1]
        total_orig = cum_orig[n_leaf - 1]
        parent_loss = dispatch_leaf_loss(
            loss_kind, total_D, total_C, bounded_lo, bounded_hi
        )

        # Sweep candidate split positions.
        feat_best_gain = -INFINITY
        feat_best_k = -1
        feat_best_thr = 0.0
        for k in range(n_leaf - 1):
            if svals[k] == svals[k + 1]:
                continue
            D_l = cum_D[k]
            C_l = cum_C[k]
            n_l = cum_orig[k]
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
            if gain > feat_best_gain:
                feat_best_gain = gain
                feat_best_k = k
                feat_best_thr = 0.5 * (svals[k] + svals[k + 1])

        if feat_best_k >= 0 and feat_best_gain > best_gain_overall:
            best_gain_overall = feat_best_gain
            best_feat = j
            best_threshold = feat_best_thr

    best_feat_out[0] = best_feat
    best_threshold_out[0] = best_threshold
    best_gain_out[0] = best_gain_overall


# ---------------------------------------------------------------------------
# Driver entry point.

def grow_depthwise_exact_c(
    cnp.ndarray[f64, ndim=2, mode="c"] features,
    cnp.ndarray[f64, ndim=1] D,
    cnp.ndarray[f64, ndim=1] C,
    int max_depth,
    int min_samples_split,
    int min_orig_leaf,
    double min_impurity_decrease,
    int loss_kind,
    double bounded_lo,
    double bounded_hi,
):
    """Iterative depthwise grow on the exact (per-leaf-sort) splitter.

    Returns a :class:`riesztree.fast._grow_c.GrowableFlatTree` with one
    slot per node. The Python ``Node`` tree is rebuilt once at fit-end
    via :func:`riesztree.fast._tree.node_from_growable_flat_tree`.
    """
    from ._grow_c import GrowableFlatTree

    cdef Py_ssize_t n_aug = D.shape[0]
    cdef int n_features = features.shape[1]

    # Worst-case node-count cap. Same formula as
    # :func:`grow_depthwise_hist_c`. ``2 * n_aug + 1`` is a safe upper
    # bound at unlimited depth (every leaf has ≥ 1 row).
    cdef Py_ssize_t max_nodes_cap
    if max_depth >= 31:
        max_nodes_cap = max(2 * n_aug + 1, 1024)
    else:
        max_nodes_cap = (1 << (max_depth + 1)) + 1
        if max_nodes_cap < 1024:
            max_nodes_cap = 1024

    cdef object tree = GrowableFlatTree(max_nodes_cap)
    cdef i32[::1] tree_feature = tree.feature
    cdef f64[::1] tree_threshold = tree.threshold
    cdef i32[::1] tree_left = tree.left
    cdef i32[::1] tree_right = tree.right
    cdef f64[::1] tree_value = tree.value
    cdef f64[::1] tree_D_sum = tree.D_sum
    cdef f64[::1] tree_C_sum = tree.C_sum
    cdef i64[::1] tree_n_orig = tree.n_orig
    cdef f64[::1] tree_gain_v = tree.gain
    cdef i32[::1] tree_depth = tree.depth

    cdef cnp.ndarray[i64, ndim=1] idx_buf_arr = np.arange(n_aug, dtype=np.int64)
    cdef i64[::1] idx_v = idx_buf_arr
    cdef f64[:, ::1] X_v = features
    cdef f64[::1] D_v = D
    cdef f64[::1] C_v = C

    # Per-feature scratch buffers, allocated once at fit start and
    # reused across all (leaf × feature) sort+sweeps. Sized to the
    # max possible leaf (n_aug at the root); tighter bounds aren't
    # worth the complexity.
    cdef cnp.ndarray[f64, ndim=1] svals_arr = np.empty(n_aug, dtype=np.float64)
    cdef cnp.ndarray[i64, ndim=1] sidx_arr = np.empty(n_aug, dtype=np.int64)
    cdef cnp.ndarray[f64, ndim=1] cum_D_arr = np.empty(n_aug, dtype=np.float64)
    cdef cnp.ndarray[f64, ndim=1] cum_C_arr = np.empty(n_aug, dtype=np.float64)
    cdef cnp.ndarray[i64, ndim=1] cum_orig_arr = np.empty(n_aug, dtype=np.int64)
    cdef f64[::1] svals_v = svals_arr
    cdef i64[::1] sidx_v = sidx_arr
    cdef f64[::1] cum_D_v = cum_D_arr
    cdef f64[::1] cum_C_v = cum_C_arr
    cdef i64[::1] cum_orig_v = cum_orig_arr

    # Per-loop scalars.
    cdef Py_ssize_t i
    cdef i64 row
    cdef double root_D = 0.0
    cdef double root_C = 0.0
    cdef i64 root_orig = 0
    cdef double root_alpha
    cdef Py_ssize_t root_idx
    cdef Py_ssize_t node_idx, start, end
    cdef int depth_v, best_feat
    cdef double best_threshold, best_gain
    cdef Py_ssize_t mid, left_idx, right_idx, n_left, n_right
    cdef double left_D, right_D, left_C, right_C, left_alpha, right_alpha
    cdef i64 left_orig, right_orig
    cdef Py_ssize_t k_iter

    # Bootstrap the root.
    for i in range(n_aug):
        row = idx_v[i]
        root_D += D_v[row]
        root_C += C_v[row]
        if D_v[row] > 0.0:
            root_orig += 1
    root_alpha = dispatch_alpha_at_opt(loss_kind, root_D, root_C, bounded_lo, bounded_hi)

    root_idx = tree.n_nodes_used
    if root_idx >= tree.max_nodes:
        raise RuntimeError(
            f"GrowableFlatTree exhausted at {tree.max_nodes} nodes."
        )
    tree_feature[root_idx] = -1
    tree_value[root_idx] = root_alpha
    tree_D_sum[root_idx] = root_D
    tree_C_sum[root_idx] = root_C
    tree_n_orig[root_idx] = root_orig
    tree_depth[root_idx] = 0
    tree.n_nodes_used += 1

    # Worklist entries: (node_idx, start, end, depth).
    cdef list worklist = [(root_idx, 0, n_aug, 0)]
    cdef object item

    while worklist:
        item = worklist.pop()
        node_idx = item[0]
        start = item[1]
        end = item[2]
        depth_v = item[3]

        # Stopping conditions (mirror Python ``_recurse``).
        if depth_v >= max_depth:
            continue
        if tree_n_orig[node_idx] < min_samples_split:
            continue
        if (end - start) < 2:
            continue

        _best_split_at_leaf(
            X_v, D_v, C_v, idx_v, start, end, n_features,
            loss_kind, bounded_lo, bounded_hi, min_orig_leaf,
            svals_v, sidx_v, cum_D_v, cum_C_v, cum_orig_v,
            &best_feat, &best_threshold, &best_gain,
        )
        if best_feat < 0:
            continue
        if best_gain <= min_impurity_decrease:
            continue

        # Partition idx_buf[start:end] in place on (best_feat, best_threshold).
        mid = _partition_inplace_continuous(
            idx_v, start, end, X_v, best_feat, best_threshold,
        )
        n_left = mid - start
        n_right = end - mid
        if n_left == 0 or n_right == 0:
            continue

        # Compute child leaf payloads via direct sum over the partitioned
        # slices. Sum left and right independently rather than deriving
        # ``right = total - left`` to avoid amplifying float-precision
        # drift in deep subtrees (matters for KLLoss's log-amplified gain).
        left_D = 0.0
        left_C = 0.0
        left_orig = 0
        for k_iter in range(start, mid):
            row = idx_v[k_iter]
            left_D += D_v[row]
            left_C += C_v[row]
            if D_v[row] > 0.0:
                left_orig += 1
        right_D = 0.0
        right_C = 0.0
        right_orig = 0
        for k_iter in range(mid, end):
            row = idx_v[k_iter]
            right_D += D_v[row]
            right_C += C_v[row]
            if D_v[row] > 0.0:
                right_orig += 1

        left_alpha = dispatch_alpha_at_opt(
            loss_kind, left_D, left_C, bounded_lo, bounded_hi
        )
        right_alpha = dispatch_alpha_at_opt(
            loss_kind, right_D, right_C, bounded_lo, bounded_hi
        )

        # Append left and right child leaves to the tree.
        left_idx = tree.n_nodes_used
        if left_idx >= tree.max_nodes:
            raise RuntimeError(
                f"GrowableFlatTree exhausted at {tree.max_nodes} nodes; "
                "increase the per-fit cap."
            )
        tree_feature[left_idx] = -1
        tree_value[left_idx] = left_alpha
        tree_D_sum[left_idx] = left_D
        tree_C_sum[left_idx] = left_C
        tree_n_orig[left_idx] = left_orig
        tree_depth[left_idx] = depth_v + 1
        tree.n_nodes_used += 1

        right_idx = tree.n_nodes_used
        if right_idx >= tree.max_nodes:
            raise RuntimeError(
                f"GrowableFlatTree exhausted at {tree.max_nodes} nodes; "
                "increase the per-fit cap."
            )
        tree_feature[right_idx] = -1
        tree_value[right_idx] = right_alpha
        tree_D_sum[right_idx] = right_D
        tree_C_sum[right_idx] = right_C
        tree_n_orig[right_idx] = right_orig
        tree_depth[right_idx] = depth_v + 1
        tree.n_nodes_used += 1

        # Convert the parent slot from leaf to internal.
        tree_feature[node_idx] = best_feat
        tree_threshold[node_idx] = best_threshold
        tree_left[node_idx] = <i32>left_idx
        tree_right[node_idx] = <i32>right_idx
        tree_gain_v[node_idx] = best_gain

        # Push children. Append right first so .pop() processes left first
        # (matches the existing Python recursion's DFS-left-first order).
        worklist.append((right_idx, mid, end, depth_v + 1))
        worklist.append((left_idx, start, mid, depth_v + 1))

    return tree
