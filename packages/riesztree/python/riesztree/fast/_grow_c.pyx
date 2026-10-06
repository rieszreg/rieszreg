# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
"""Iterative depthwise grow loop (Cython).

Replaces the Python ``_recurse`` driver in :mod:`riesztree.grow` with a
Cython worklist loop that:

  * Maintains an **in-place index buffer** (one int64 array of length
    ``n_aug``). Each node references a contiguous slice
    ``[start, end)``; splits partition the slice in place via a
    two-pointer swap. Eliminates the per-split numpy allocation of
    ``left_idx`` / ``right_idx``.

  * Writes node entries directly into a pre-allocated **flat-array
    tree** (:class:`GrowableFlatTree`). Eliminates per-split Python
    ``Node`` allocation in the hot loop. The Python ``Node`` tree is
    rebuilt **once** at fit-end via :func:`node_tree_from_growable`
    for backward-compat with diagnostics, pruning, and serialisation.

  * Drives the worklist iteratively (no Python recursion). Stack
    depth limit no longer caps tree depth.

Scope of v1: ``splitter='hist'`` + ``growth_policy='depthwise'`` +
no categorical features + no early stopping + no max_features
subsampling + built-in or user-registered loss. Other configurations
keep the existing Python recursion in :mod:`riesztree.grow`. The
dispatcher in ``grow.py`` decides at fit start which path to take.
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport INFINITY, isfinite

from ._loss_kernels cimport dispatch_leaf_loss, dispatch_alpha_at_opt

ctypedef cnp.float64_t f64
ctypedef cnp.int64_t i64
ctypedef cnp.int32_t i32
ctypedef cnp.uint8_t u8


cdef class GrowableFlatTree:
    """Pre-allocated parallel-array tree for the iterative grow driver.

    Each node uses one slot across the parallel arrays. Internal nodes
    fill ``feature``, ``threshold``, ``left``, ``right``, ``gain``;
    leaves fill ``value``, ``D_sum``, ``C_sum``, ``n_orig`` and have
    ``feature[idx] == -1``. The same node is initialised as a leaf and
    later converted to internal via :meth:`convert_to_internal`.
    """

    cdef public:
        cnp.ndarray feature
        cnp.ndarray threshold
        cnp.ndarray left
        cnp.ndarray right
        cnp.ndarray value
        cnp.ndarray D_sum
        cnp.ndarray C_sum
        cnp.ndarray n_orig
        cnp.ndarray gain
        cnp.ndarray depth
        Py_ssize_t n_nodes_used
        Py_ssize_t max_nodes

    def __init__(self, Py_ssize_t max_nodes):
        self.max_nodes = max_nodes
        self.feature = np.full(max_nodes, -1, dtype=np.int32)
        self.threshold = np.full(max_nodes, np.nan, dtype=np.float64)
        self.left = np.full(max_nodes, -1, dtype=np.int32)
        self.right = np.full(max_nodes, -1, dtype=np.int32)
        self.value = np.zeros(max_nodes, dtype=np.float64)
        self.D_sum = np.zeros(max_nodes, dtype=np.float64)
        self.C_sum = np.zeros(max_nodes, dtype=np.float64)
        self.n_orig = np.zeros(max_nodes, dtype=np.int64)
        self.gain = np.zeros(max_nodes, dtype=np.float64)
        self.depth = np.zeros(max_nodes, dtype=np.int32)
        self.n_nodes_used = 0


cdef inline Py_ssize_t _add_leaf(
    GrowableFlatTree tree,
    double D, double C, i64 n_orig_count, double alpha, int depth_v,
):
    cdef Py_ssize_t idx = tree.n_nodes_used
    if idx >= tree.max_nodes:
        raise RuntimeError(
            f"GrowableFlatTree exhausted at {tree.max_nodes} nodes; "
            "increase the per-fit cap (typically 2 * max_leaf_nodes + 1)."
        )
    cdef i32[::1] feat_v = tree.feature
    cdef f64[::1] val_v = tree.value
    cdef f64[::1] D_v = tree.D_sum
    cdef f64[::1] C_v = tree.C_sum
    cdef i64[::1] n_v = tree.n_orig
    cdef i32[::1] depth_arr = tree.depth
    feat_v[idx] = -1
    val_v[idx] = alpha
    D_v[idx] = D
    C_v[idx] = C
    n_v[idx] = n_orig_count
    depth_arr[idx] = depth_v
    tree.n_nodes_used += 1
    return idx


cdef inline void _convert_to_internal(
    GrowableFlatTree tree, Py_ssize_t idx,
    int feat, double thr, double gain_v,
    Py_ssize_t left_idx, Py_ssize_t right_idx,
):
    cdef i32[::1] feat_v = tree.feature
    cdef f64[::1] thr_v = tree.threshold
    cdef i32[::1] left_v = tree.left
    cdef i32[::1] right_v = tree.right
    cdef f64[::1] gain_arr = tree.gain
    feat_v[idx] = feat
    thr_v[idx] = thr
    left_v[idx] = <i32>left_idx
    right_v[idx] = <i32>right_idx
    gain_arr[idx] = gain_v


cdef inline Py_ssize_t _partition_inplace(
    i64[::1] idx_buf,
    Py_ssize_t start,
    Py_ssize_t end,
    u8[:, ::1] X_binned,
    int best_feat,
    int best_bin,
) noexcept nogil:
    """Two-pointer in-place partition of ``idx_buf[start:end]``.

    Rearranges so that all rows with ``X_binned[row, best_feat] <= best_bin``
    come first. Returns the boundary ``mid`` such that
    ``idx_buf[start:mid]`` is the left child and ``idx_buf[mid:end]`` is
    the right child.
    """
    cdef Py_ssize_t lo = start
    cdef Py_ssize_t hi = end - 1
    cdef i64 tmp
    cdef i64 row
    while lo <= hi:
        row = idx_buf[lo]
        if X_binned[row, best_feat] <= best_bin:
            lo += 1
        else:
            tmp = idx_buf[hi]
            idx_buf[hi] = row
            idx_buf[lo] = tmp
            hi -= 1
    return lo


cdef void _accumulate_hist_slice(
    u8[:, ::1] X_binned,
    f64[::1] D,
    f64[::1] C,
    i64[::1] idx_buf,
    Py_ssize_t start,
    Py_ssize_t end,
    cnp.int32_t[::1] candidate_features,
    f64[:, ::1] hD,
    f64[:, ::1] hC,
    i64[:, ::1] hO,
    double* total_D_out,
    double* total_C_out,
    i64* total_orig_out,
) noexcept nogil:
    """Accumulate per-feature histograms over ``idx_buf[start:end]``
    into pre-zeroed ``(hD, hC, hO)`` of shape ``(n_features, max_bins)``."""
    cdef Py_ssize_t i, j_idx
    cdef i64 row, j
    cdef int n_features = candidate_features.shape[0]
    cdef u8 bin_idx
    cdef double total_D = 0.0
    cdef double total_C = 0.0
    cdef i64 total_orig = 0
    cdef int is_orig

    for i in range(start, end):
        row = idx_buf[i]
        total_D += D[row]
        total_C += C[row]
        is_orig = 1 if D[row] > 0.0 else 0
        total_orig += is_orig
        for j_idx in range(n_features):
            j = candidate_features[j_idx]
            bin_idx = X_binned[row, j]
            hD[j_idx, bin_idx] += D[row]
            hC[j_idx, bin_idx] += C[row]
            hO[j_idx, bin_idx] += is_orig

    total_D_out[0] = total_D
    total_C_out[0] = total_C
    total_orig_out[0] = total_orig


cdef _find_best_split(
    f64[:, ::1] hD_v,
    f64[:, ::1] hC_v,
    i64[:, ::1] hO_v,
    double total_D,
    double total_C,
    i64 total_orig,
    cnp.int32_t[::1] cand_v,
    cnp.int32_t[::1] nbins_v,
    int loss_kind,
    double bounded_lo,
    double bounded_hi,
    int min_orig_leaf,
):
    """Return ``(best_feat, best_bin, gain)`` or ``None``."""
    cdef double parent_loss = dispatch_leaf_loss(loss_kind, total_D, total_C, bounded_lo, bounded_hi)
    cdef double best_gain = -INFINITY
    cdef Py_ssize_t best_feat = -1
    cdef Py_ssize_t best_bin = -1
    cdef double D_l, C_l, D_r, C_r, L_l, L_r, gain
    cdef i64 n_l, n_r
    cdef cnp.int32_t nb
    cdef Py_ssize_t j_idx, b
    cdef i64 j
    cdef Py_ssize_t n_features = cand_v.shape[0]

    for j_idx in range(n_features):
        j = cand_v[j_idx]
        nb = nbins_v[j]
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


cdef int _take_slot(
    list free_slots, list hD_pool, list hC_pool, list hO_pool,
    list slot_D, list slot_C, list slot_orig,
):
    """Pop a zeroed histogram slot, growing the pool when it's empty."""
    cdef int slot_id
    if free_slots:
        slot_id = free_slots.pop()
        hD_pool[slot_id].fill(0.0)
        hC_pool[slot_id].fill(0.0)
        hO_pool[slot_id].fill(0)
        return slot_id
    hD_pool.append(np.zeros_like(hD_pool[0]))
    hC_pool.append(np.zeros_like(hC_pool[0]))
    hO_pool.append(np.zeros_like(hO_pool[0]))
    slot_D.append(0.0)
    slot_C.append(0.0)
    slot_orig.append(0)
    return len(hD_pool) - 1


def grow_depthwise_hist_c(
    cnp.ndarray[u8, ndim=2, mode="c"] X_binned,
    cnp.ndarray[f64, ndim=1] D,
    cnp.ndarray[f64, ndim=1] C,
    cnp.ndarray[cnp.int32_t, ndim=1] n_bins_per_feature,
    list bin_thresholds,
    int max_bins,
    int max_depth,
    int min_samples_split,
    int min_orig_leaf,
    double min_impurity_decrease,
    int loss_kind,
    double bounded_lo,
    double bounded_hi,
):
    """Iterative depthwise grow on the histogram path. See module docstring."""
    cdef Py_ssize_t n_aug = D.shape[0]
    cdef int n_features = X_binned.shape[1]

    # Node cap: a depth-d binary tree has at most 2^(d+1) - 1 nodes, and
    # every leaf is non-empty, so there are at most 2 * n_aug - 1 nodes.
    cdef Py_ssize_t max_nodes_cap = 2 * n_aug + 1
    # Shift in Py_ssize_t: a C int overflows at max_depth = 30.
    if max_depth < 31 and ((<Py_ssize_t>1) << (max_depth + 1)) + 1 < max_nodes_cap:
        max_nodes_cap = ((<Py_ssize_t>1) << (max_depth + 1)) + 1

    cdef GrowableFlatTree tree = GrowableFlatTree(max_nodes_cap)

    cdef cnp.ndarray[i64, ndim=1] idx_buf_arr = np.arange(n_aug, dtype=np.int64)
    cdef i64[::1] idx_v = idx_buf_arr
    cdef u8[:, ::1] X_v = X_binned
    cdef f64[::1] D_v = D
    cdef f64[::1] C_v = C

    cdef cnp.ndarray[cnp.int32_t, ndim=1] candidate_features = np.arange(n_features, dtype=np.int32)
    cdef cnp.int32_t[::1] cand_v = candidate_features
    cdef cnp.int32_t[::1] nbins_v = n_bins_per_feature

    # Histogram buffer pool. In DFS the number of leaf histograms in
    # flight is bounded by the depth of the current path plus one PMS
    # temporary. Slots are recycled; the pool grows on demand when an
    # unbalanced tree gets deeper than the initial allocation.
    cdef int pool_size = (max_depth if max_depth < 32 else 32) + 2
    cdef list hD_pool = [np.zeros((n_features, max_bins), dtype=np.float64) for _ in range(pool_size)]
    cdef list hC_pool = [np.zeros((n_features, max_bins), dtype=np.float64) for _ in range(pool_size)]
    cdef list hO_pool = [np.zeros((n_features, max_bins), dtype=np.int64) for _ in range(pool_size)]
    cdef list free_slots = list(range(pool_size))

    # Per-slot scalar totals (sum_D, sum_C, sum_orig at the time of last
    # accumulation into that slot). Indexed by slot id.
    cdef list slot_D = [0.0] * pool_size
    cdef list slot_C = [0.0] * pool_size
    cdef list slot_orig = [0] * pool_size

    # Per-loop variables (cdef must be at function scope in Cython).
    cdef Py_ssize_t i
    cdef i64 row
    cdef double root_D = 0.0
    cdef double root_C = 0.0
    cdef i64 root_orig = 0
    cdef double root_alpha
    cdef Py_ssize_t root_idx
    cdef Py_ssize_t node_idx, start, end
    cdef int depth_v, best_feat, best_bin
    cdef double gain_v, threshold_v
    cdef double total_D_local, total_C_local
    cdef i64 total_orig_local
    cdef Py_ssize_t mid, n_left, n_right, n_smaller
    cdef Py_ssize_t left_idx, right_idx
    cdef double left_D, right_D, left_C, right_C, left_alpha, right_alpha
    cdef i64 left_orig, right_orig
    cdef double sm_tD, sm_tC
    cdef i64 sm_tO
    cdef Py_ssize_t feat_idx_in_hist
    cdef Py_ssize_t b
    cdef bint pms_worth_it
    cdef int slot_id, smaller_slot_id, larger_slot_id
    cdef int parent_slot
    cdef f64[:, ::1] hD_v, hC_v
    cdef i64[:, ::1] hO_v

    # Bootstrap the root.
    for i in range(n_aug):
        row = idx_v[i]
        root_D += D_v[row]
        root_C += C_v[row]
        if D_v[row] > 0.0:
            root_orig += 1
    root_alpha = dispatch_alpha_at_opt(loss_kind, root_D, root_C, bounded_lo, bounded_hi)
    root_idx = _add_leaf(tree, root_D, root_C, root_orig, root_alpha, 0)

    # Worklist entries: (node_idx, start, end, depth, slot_id_or_-1).
    # slot_id == -1 means "no parent_hist, allocate fresh" (root + PMS-skip).
    cdef list worklist = [(root_idx, 0, n_aug, 0, -1)]
    cdef i64[::1] tree_n_orig = tree.n_orig
    cdef object item

    while worklist:
        item = worklist.pop()
        node_idx = item[0]
        start = item[1]
        end = item[2]
        depth_v = item[3]
        slot_id = item[4]

        # Stopping conditions.
        if depth_v >= max_depth:
            if slot_id >= 0:
                free_slots.append(slot_id)
            continue
        if tree_n_orig[node_idx] < min_samples_split:
            if slot_id >= 0:
                free_slots.append(slot_id)
            continue

        # Get / build histogram for this node into a pool slot.
        if slot_id < 0:
            slot_id = _take_slot(free_slots, hD_pool, hC_pool, hO_pool, slot_D, slot_C, slot_orig)
            hD_v = hD_pool[slot_id]
            hC_v = hC_pool[slot_id]
            hO_v = hO_pool[slot_id]
            _accumulate_hist_slice(
                X_v, D_v, C_v, idx_v, start, end,
                cand_v, hD_v, hC_v, hO_v,
                &total_D_local, &total_C_local, &total_orig_local,
            )
            slot_D[slot_id] = total_D_local
            slot_C[slot_id] = total_C_local
            slot_orig[slot_id] = total_orig_local
        else:
            hD_v = hD_pool[slot_id]
            hC_v = hC_pool[slot_id]
            hO_v = hO_pool[slot_id]
            total_D_local = slot_D[slot_id]
            total_C_local = slot_C[slot_id]
            total_orig_local = slot_orig[slot_id]

        # Find best split.
        result = _find_best_split(
            hD_v, hC_v, hO_v,
            total_D_local, total_C_local, total_orig_local,
            cand_v, nbins_v, loss_kind, bounded_lo, bounded_hi, min_orig_leaf,
        )
        if result is None:
            free_slots.append(slot_id)
            continue
        best_feat = result[0]
        best_bin = result[1]
        gain_v = result[2]
        if gain_v <= min_impurity_decrease:
            free_slots.append(slot_id)
            continue

        # Partition idx_buf[start:end] in place on (best_feat, best_bin).
        mid = _partition_inplace(idx_v, start, end, X_v, best_feat, best_bin)
        n_left = mid - start
        n_right = end - mid
        if n_left == 0 or n_right == 0:
            free_slots.append(slot_id)
            continue

        # Compute child leaf payloads (D, C, n_orig) by summing the
        # histogram bins on the chosen feature.
        feat_idx_in_hist = -1
        for b in range(n_features):
            if cand_v[b] == best_feat:
                feat_idx_in_hist = b
                break
        left_D = 0.0
        left_C = 0.0
        left_orig = 0
        for b in range(best_bin + 1):
            left_D += hD_v[feat_idx_in_hist, b]
            left_C += hC_v[feat_idx_in_hist, b]
            left_orig += hO_v[feat_idx_in_hist, b]
        right_D = total_D_local - left_D
        right_C = total_C_local - left_C
        right_orig = total_orig_local - left_orig

        left_alpha = dispatch_alpha_at_opt(loss_kind, left_D, left_C, bounded_lo, bounded_hi)
        right_alpha = dispatch_alpha_at_opt(loss_kind, right_D, right_C, bounded_lo, bounded_hi)
        threshold_v = float(bin_thresholds[best_feat][best_bin])

        left_idx = _add_leaf(tree, left_D, left_C, left_orig, left_alpha, depth_v + 1)
        right_idx = _add_leaf(tree, right_D, right_C, right_orig, right_alpha, depth_v + 1)
        _convert_to_internal(tree, node_idx, best_feat, threshold_v, gain_v, left_idx, right_idx)

        # Decide whether to use PMS for the children's histograms.
        n_smaller = n_left if n_left <= n_right else n_right
        pms_worth_it = (n_smaller * 4) > (n_features * max_bins)

        # parent_slot is the current slot_id; it'll either be reused for the
        # larger child (PMS) or freed (PMS-skip).
        parent_slot = slot_id

        if not pms_worth_it:
            # Free parent's slot, allocate two fresh slots for the children.
            free_slots.append(parent_slot)
            # Children use slot_id == -1 to signal "accumulate fresh on pop".
            worklist.append((right_idx, mid, end, depth_v + 1, -1))
            worklist.append((left_idx, start, mid, depth_v + 1, -1))
        elif n_left <= n_right:
            # Smaller = left. Get a new slot, accumulate left, then subtract
            # in place from parent_slot to get right's hist.
            smaller_slot_id = _take_slot(free_slots, hD_pool, hC_pool, hO_pool, slot_D, slot_C, slot_orig)
            _accumulate_hist_slice(
                X_v, D_v, C_v, idx_v, start, mid,
                cand_v,
                hD_pool[smaller_slot_id], hC_pool[smaller_slot_id], hO_pool[smaller_slot_id],
                &sm_tD, &sm_tC, &sm_tO,
            )
            slot_D[smaller_slot_id] = sm_tD
            slot_C[smaller_slot_id] = sm_tC
            slot_orig[smaller_slot_id] = sm_tO
            # Subtract in place: parent_slot now holds larger (= right) child's hist.
            np.subtract(hD_pool[parent_slot], hD_pool[smaller_slot_id], out=hD_pool[parent_slot])
            np.subtract(hC_pool[parent_slot], hC_pool[smaller_slot_id], out=hC_pool[parent_slot])
            np.subtract(hO_pool[parent_slot], hO_pool[smaller_slot_id], out=hO_pool[parent_slot])
            slot_D[parent_slot] = total_D_local - sm_tD
            slot_C[parent_slot] = total_C_local - sm_tC
            slot_orig[parent_slot] = total_orig_local - sm_tO
            worklist.append((right_idx, mid, end, depth_v + 1, parent_slot))
            worklist.append((left_idx, start, mid, depth_v + 1, smaller_slot_id))
        else:
            # Smaller = right; mirror.
            smaller_slot_id = _take_slot(free_slots, hD_pool, hC_pool, hO_pool, slot_D, slot_C, slot_orig)
            _accumulate_hist_slice(
                X_v, D_v, C_v, idx_v, mid, end,
                cand_v,
                hD_pool[smaller_slot_id], hC_pool[smaller_slot_id], hO_pool[smaller_slot_id],
                &sm_tD, &sm_tC, &sm_tO,
            )
            slot_D[smaller_slot_id] = sm_tD
            slot_C[smaller_slot_id] = sm_tC
            slot_orig[smaller_slot_id] = sm_tO
            np.subtract(hD_pool[parent_slot], hD_pool[smaller_slot_id], out=hD_pool[parent_slot])
            np.subtract(hC_pool[parent_slot], hC_pool[smaller_slot_id], out=hC_pool[parent_slot])
            np.subtract(hO_pool[parent_slot], hO_pool[smaller_slot_id], out=hO_pool[parent_slot])
            slot_D[parent_slot] = total_D_local - sm_tD
            slot_C[parent_slot] = total_C_local - sm_tC
            slot_orig[parent_slot] = total_orig_local - sm_tO
            worklist.append((right_idx, mid, end, depth_v + 1, smaller_slot_id))
            worklist.append((left_idx, start, mid, depth_v + 1, parent_slot))

    return tree
