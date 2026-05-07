"""Tree growth: depthwise (default) and leafwise (best-first) strategies.

Both strategies share the same per-node split-finding routine
(``best_split_at``); they differ only in the order in which leaves are
considered for splitting.

Validation early stopping: when ``early_stopping_rounds`` is set, growth
tracks the held-out augmented-loss of the partial tree at each split. If
the held-out loss has not strictly improved for that many consecutive
splits, growth halts (depthwise) or pops no further leaves (leafwise).
"""

from __future__ import annotations

import heapq
import itertools
import math
from typing import Sequence

import numpy as np

from .fast._splitter import (
    accumulate_hist,
    best_split_at_hist,
    best_split_continuous_fast,
    best_split_continuous_random,
    find_best_split_in_hist,
    loss_kind_for,
    partition_idx_by_bin,
    warn_python_fallback,
)
from .splitter import (
    best_split_categorical,
    best_split_continuous,
    make_leaf_solvers,
)
from .tree import Node


# ---------------------------------------------------------------------------
# Hyperparameter resolution helpers (sklearn parity).

def _resolve_max_features(max_features, n_features: int) -> int:
    """Resolve ``max_features`` to an int in ``[1, n_features]``.

    Mirrors :class:`sklearn.tree.DecisionTreeRegressor`. Accepts:

    - ``None`` (or ``"all"``): all features.
    - ``int``: exact count, clipped to ``[1, n_features]``.
    - ``float`` in ``(0, 1]``: fraction of features (rounded up, ≥ 1).
    - ``"sqrt"``: ``max(1, ⌊√n_features⌋)``.
    - ``"log2"``: ``max(1, ⌊log2(n_features)⌋)``.
    """
    if max_features is None or max_features == "all":
        return n_features
    if isinstance(max_features, str):
        if max_features == "sqrt":
            return max(1, int(math.isqrt(n_features)))
        if max_features == "log2":
            return max(1, int(math.log2(max(n_features, 1))))
        raise ValueError(
            f"max_features={max_features!r}; expected None, 'all', 'sqrt', "
            "'log2', an int, or a float in (0, 1]."
        )
    if isinstance(max_features, float):
        if not (0.0 < max_features <= 1.0):
            raise ValueError(
                f"max_features={max_features!r}; float must be in (0, 1]."
            )
        return max(1, int(math.ceil(max_features * n_features)))
    if isinstance(max_features, (int, np.integer)):
        if max_features < 1:
            raise ValueError(f"max_features={max_features!r}; int must be ≥ 1.")
        return min(int(max_features), n_features)
    raise TypeError(
        f"max_features={max_features!r} ({type(max_features).__name__}); "
        "expected None, str, int, or float."
    )


def _effective_min_orig_leaf(
    min_samples_leaf: int,
    min_weight_fraction_leaf: float,
    n_orig_total: int,
) -> int:
    """sklearn parity: leaves must satisfy both `min_samples_leaf` and
    `ceil(min_weight_fraction_leaf * n_orig_total)`. With unit weights
    (the only case riesztree currently supports) the weighted sample count
    equals the original-row count, so this reduces to a max() over the two."""
    if min_weight_fraction_leaf <= 0.0:
        return int(min_samples_leaf)
    floor_weighted = int(math.ceil(min_weight_fraction_leaf * max(n_orig_total, 0)))
    return int(max(min_samples_leaf, floor_weighted))


def _make_leaf(
    D: np.ndarray,
    C: np.ndarray,
    idx: np.ndarray,
    leaf_loss,
    alpha_at_opt,
    *,
    depth: int,
) -> Node:
    D_sum = float(D[idx].sum())
    C_sum = float(C[idx].sum())
    n_orig = int((D[idx] > 0).sum())
    return Node(
        is_leaf=True,
        D=D_sum,
        C=C_sum,
        n_orig=n_orig,
        n_aug=int(idx.size),
        depth=depth,
        alpha=alpha_at_opt(D_sum, C_sum),
        leaf_loss_value=leaf_loss(D_sum, C_sum),
    )


def best_split_at(
    features: np.ndarray,
    D: np.ndarray,
    C: np.ndarray,
    idx: np.ndarray,
    *,
    leaf_loss,
    alpha_at_opt,
    min_orig_leaf: int,
    categorical_features: Sequence[int],
    feature_indices: Sequence[int] | None = None,
    fast_loss_kind: int | None = None,
    bounded_lo: float = float("nan"),
    bounded_hi: float = float("nan"),
    user_cfunc_addr: int = 0,
    hist_payload: dict | None = None,
    random_splitter_rng: np.random.Generator | None = None,
):
    """Best (gain, feature_index, split_payload) across all considered features.

    If ``feature_indices`` is given, only those columns are evaluated
    (sklearn-style ``max_features`` subsampling, drawn by the caller).

    Three split-finding paths:

    - ``hist_payload`` is non-None → Cython histogram sweep on
      pre-binned continuous features. The kernel returns the global
      best across all candidate continuous features; categorical
      splits still go through the Python path and the higher-gain
      result wins.
    - Otherwise, ``fast_loss_kind`` non-None → Cython exact sweep
      per continuous feature.
    - Otherwise → pure-Python splitter (categoricals always; also
      continuous when ``splitter='python'`` or for losses outside
      the four built-ins).
    """
    cat_set = set(categorical_features) if categorical_features else set()
    cols_all = (
        list(range(features.shape[1]))
        if feature_indices is None
        else [int(j) for j in feature_indices]
    )
    cont_cols = [j for j in cols_all if j not in cat_set]
    cat_cols = [j for j in cols_all if j in cat_set]

    best = None
    best_feat = None

    # Continuous features.
    if hist_payload is not None and cont_cols:
        # Histogram path: one Cython call returns the best across all
        # candidate continuous features. Returns
        # (best_feat, gain, threshold, left_idx, right_idx) or None.
        result = best_split_at_hist(
            hist_payload["X_binned"], D, C, idx,
            bin_thresholds=hist_payload["bin_thresholds"],
            n_bins_per_feature=hist_payload["n_bins_per_feature"],
            candidate_features=np.asarray(cont_cols, dtype=np.int32),
            loss_kind=fast_loss_kind,
            bounded_lo=bounded_lo, bounded_hi=bounded_hi,
            min_orig_leaf=min_orig_leaf,
            max_bins=hist_payload["max_bins"],
        )
        if result is not None:
            feat_h, gain_h, thr_h, l_idx, r_idx = result
            best = (gain_h, thr_h, l_idx, r_idx)
            best_feat = feat_h
    elif cont_cols:
        for j in cont_cols:
            if random_splitter_rng is not None and fast_loss_kind is not None:
                cand = best_split_continuous_random(
                    features[:, j], D, C, idx,
                    loss_kind=fast_loss_kind,
                    bounded_lo=bounded_lo, bounded_hi=bounded_hi,
                    min_orig_leaf=min_orig_leaf,
                    rng=random_splitter_rng,
                )
            elif fast_loss_kind is not None:
                cand = best_split_continuous_fast(
                    features[:, j], D, C, idx,
                    loss_kind=fast_loss_kind,
                    bounded_lo=bounded_lo, bounded_hi=bounded_hi,
                    min_orig_leaf=min_orig_leaf,
                    user_cfunc_addr=user_cfunc_addr,
                )
            else:
                cand = best_split_continuous(
                    features[:, j], D, C, idx, leaf_loss,
                    min_orig_leaf=min_orig_leaf,
                )
            if cand is None:
                continue
            if best is None or cand[0] > best[0]:
                best = cand
                best_feat = j

    # Categorical features always go through the Python path.
    for j in cat_cols:
        cand = best_split_categorical(
            features[:, j], D, C, idx, leaf_loss, alpha_at_opt,
            min_orig_leaf=min_orig_leaf,
        )
        if cand is None:
            continue
        if best is None or cand[0] > best[0]:
            best = cand
            best_feat = j

    if best is None:
        return None
    return best_feat, best


_VALID_SPLITTERS = ("exact", "hist", "random", "python")
_PYTHON_SPLITTER_WARNED = False


def _resolve_fast_loss_args(
    splitter: str, loss
) -> tuple[int | None, float, float, int]:
    """Decide whether the Cython splitter is usable for this fit.

    Returns ``(loss_kind, bounded_lo, bounded_hi, user_cfunc_addr)``.
    ``loss_kind`` is ``None`` when the user explicitly chose
    ``splitter='python'`` or when ``loss`` is neither a built-in nor
    a user-registered fast solver (which triggers a one-time
    UserWarning). For built-ins ``user_cfunc_addr`` is 0; for
    user-registered losses ``loss_kind`` is ``LOSS_USER_CFUNC`` and
    ``user_cfunc_addr`` carries the registered C-callable address.
    """
    if splitter == "python":
        # Phase 10: deprecate explicit ``splitter='python'`` selection.
        # The Cython exact / hist / random paths are byte-equivalent (up
        # to floating-point) and significantly faster. This path is
        # scheduled for removal in v0.0.3.
        global _PYTHON_SPLITTER_WARNED
        if not _PYTHON_SPLITTER_WARNED:
            import warnings as _warnings
            _warnings.warn(
                "splitter='python' is deprecated and will be removed in "
                "riesztree v0.0.3. Use splitter='exact' (the default) or "
                "splitter='hist' for the Cython-backed paths; both are "
                "byte-equivalent or numerically very close to the legacy "
                "Python splitter on the four built-in losses. For custom "
                "Loss subclasses, register a Cython kernel via "
                "riesztree.fast.register_fast_leaf_solver(...) instead of "
                "relying on the python fallback.",
                DeprecationWarning,
                stacklevel=4,
            )
            _PYTHON_SPLITTER_WARNED = True
        return None, float("nan"), float("nan"), 0
    if splitter not in ("exact", "hist", "random"):
        raise ValueError(
            f"splitter={splitter!r}; expected one of {_VALID_SPLITTERS}."
        )
    resolved = loss_kind_for(loss)
    if resolved is None:
        warn_python_fallback(loss)
        return None, float("nan"), float("nan"), 0
    kind, lo, hi, user_addr = resolved
    return int(kind), float(lo), float(hi), int(user_addr)


def _split_node_into_children(
    node: Node,
    best_feat: int,
    best_split,
    *,
    is_categorical: bool,
    features: np.ndarray,
    D: np.ndarray,
    C: np.ndarray,
    leaf_loss,
    alpha_at_opt,
):
    """Mutate ``node`` from leaf to internal; return (left_idx, right_idx)."""
    gain = float(best_split[0])
    if is_categorical:
        _, left_levels, left_idx, right_idx = best_split
        node.split_kind = "categorical"
        node.split_left_levels = left_levels
        node.split_threshold = None
    else:
        _, threshold, left_idx, right_idx = best_split
        node.split_kind = "continuous"
        node.split_threshold = threshold
        node.split_left_levels = None
    node.is_leaf = False
    node.split_feature = best_feat
    node.split_gain = gain
    node.left = _make_leaf(
        D, C, left_idx, leaf_loss, alpha_at_opt, depth=node.depth + 1
    )
    node.right = _make_leaf(
        D, C, right_idx, leaf_loss, alpha_at_opt, depth=node.depth + 1
    )
    return left_idx, right_idx


# ---------------------------------------------------------------------------
# Holdout-loss bookkeeping for early stopping.

def _holdout_loss(
    root: Node,
    aug_valid_features: np.ndarray | None,
    aug_valid_D: np.ndarray | None,
    aug_valid_C: np.ndarray | None,
    loss,
) -> float:
    """Held-out augmented Bregman loss for early-stopping monitoring.

    Builds a flat-array companion of ``root`` and walks it via the
    Cython ``predict_alpha`` kernel — much faster than the Python
    Node tree-walk that this used to call (``riesztree.tree.predict_array``).
    Phase 9: this path is invoked once per accepted split when early
    stopping is enabled; the speedup compounds over a full fit.
    """
    if aug_valid_features is None or len(aug_valid_features) == 0:
        return float("nan")
    from .fast import flat_tree_from_node, predict_alpha as _flat_predict
    flat = flat_tree_from_node(root)
    alpha_hat = _flat_predict(flat, aug_valid_features)
    # Augmented α-space loss, summed and normalised by n_orig in valid set.
    n_orig_valid = float((aug_valid_D > 0).sum())
    if n_orig_valid <= 0:
        return float("nan")
    return float(
        np.sum(
            aug_valid_D * loss.tilde_potential(alpha_hat)
            + aug_valid_C * loss.potential_deriv(alpha_hat)
        )
        / n_orig_valid
    )


# ---------------------------------------------------------------------------
# Depthwise (recursive) growth.

def grow_depthwise(
    features: np.ndarray,
    D: np.ndarray,
    C: np.ndarray,
    loss,
    *,
    max_depth: int,
    min_samples_split: int,
    min_orig_leaf: int,
    categorical_features: Sequence[int] = (),
    aug_valid: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    early_stopping_rounds: int | None = None,
    max_features=None,
    min_impurity_decrease: float = 0.0,
    min_weight_fraction_leaf: float = 0.0,
    random_state: int = 0,
    splitter: str = "exact",
    hist_payload: dict | None = None,
) -> Node:
    """Greedy recursive depth-first growth.

    Early stopping (if ``early_stopping_rounds`` is set) is approximate
    here: we stop *expanding any further nodes* once the held-out loss
    fails to strictly improve for that many consecutive accepted splits,
    counting splits in DFS order. For most grids this is the same trees
    leafwise growth would produce, but it interacts oddly with
    depth-first ordering. Use ``growth_policy="leafwise"`` for textbook
    early stopping behaviour.

    sklearn-parity hyperparameters
    ------------------------------
    ``max_features``
        Sub-sample candidate features at each split (drawn from
        ``random_state``). See :func:`_resolve_max_features`.
    ``min_impurity_decrease``
        Reject splits with gain ≤ this threshold. Replaces the v0.0.1
        hard-coded ``1e-12``; the default is now ``0.0`` (sklearn).
    ``min_weight_fraction_leaf``
        With unit weights, leaves must contain at least
        ``ceil(min_weight_fraction_leaf * n_original_total)`` original
        rows. Combined with ``min_orig_leaf`` via ``max(...)``.
    """
    leaf_loss, alpha_at_opt = make_leaf_solvers(loss)
    fast_loss_kind, bounded_lo, bounded_hi, user_cfunc_addr = _resolve_fast_loss_args(splitter, loss)

    valid_features = D_v = C_v = None
    if aug_valid is not None:
        valid_features, D_v, C_v = aug_valid

    n_features = features.shape[1]
    n_features_to_consider = _resolve_max_features(max_features, n_features)
    rng = np.random.default_rng(random_state)
    # Random splitter draws one threshold per feature per leaf from a
    # dedicated stream (independent of the max_features subsample stream).
    random_splitter_rng = (
        np.random.default_rng(random_state + 1) if splitter == "random" else None
    )
    n_orig_total = int((D > 0).sum())
    eff_min_orig_leaf = _effective_min_orig_leaf(
        min_orig_leaf, min_weight_fraction_leaf, n_orig_total
    )

    root = _make_leaf(
        D, C, np.arange(features.shape[0]), leaf_loss, alpha_at_opt, depth=0
    )

    # Mutable counter passed by closure via list to support early-stop.
    es_state = {
        "best_loss": _holdout_loss(root, valid_features, D_v, C_v, loss),
        "rounds_since_improve": 0,
        "stop": False,
        "root": root,    # consumed by the PMS path's holdout-loss recompute
    }

    # Parent-minus-sibling (PMS) eligibility: the histogram path can amortise
    # half its accumulation cost by computing the larger child's histogram via
    # ``parent_hist - smaller_child_hist`` instead of from rows. We only enable
    # it when no categorical features and no max_features subsampling are in
    # play — those would change the candidate-feature set per leaf, breaking
    # the "subtract from parent" invariant.
    pms_eligible = (
        splitter == "hist"
        and hist_payload is not None
        and not categorical_features
        and n_features_to_consider == n_features
        and fast_loss_kind is not None
    )
    # Iterative-grow Cython driver eligibility: PMS-eligible AND no early
    # stopping (the Cython driver doesn't yet support the held-out-loss
    # callback). When eligible, we skip the Python recursion entirely and
    # let the Cython worklist drive growth into a pre-allocated flat-array
    # tree, which we then convert back to a Node tree at fit-end.
    cython_iterative_eligible = (
        pms_eligible
        and early_stopping_rounds is None
        and aug_valid is None
    )
    if cython_iterative_eligible:
        from .fast._grow_c import grow_depthwise_hist_c
        from .fast._tree import node_from_growable_flat_tree
        growable = grow_depthwise_hist_c(
            hist_payload["X_binned"], D, C,
            np.ascontiguousarray(hist_payload["n_bins_per_feature"], dtype=np.int32),
            list(hist_payload["bin_thresholds"]),
            int(hist_payload["max_bins"]),
            int(max_depth),
            int(min_samples_split),
            int(eff_min_orig_leaf),
            float(min_impurity_decrease),
            int(fast_loss_kind),
            float(bounded_lo),
            float(bounded_hi),
        )
        return node_from_growable_flat_tree(growable, loss=loss)

    if pms_eligible:
        _recurse_pms_depthwise(
            root, np.arange(features.shape[0]),
            features=features, D=D, C=C, hist_payload=hist_payload,
            leaf_loss=leaf_loss, alpha_at_opt=alpha_at_opt,
            fast_loss_kind=fast_loss_kind,
            bounded_lo=bounded_lo, bounded_hi=bounded_hi,
            min_samples_split=min_samples_split,
            eff_min_orig_leaf=eff_min_orig_leaf,
            min_impurity_decrease=min_impurity_decrease,
            max_depth=max_depth,
            es_state=es_state,
            valid_features=valid_features, D_v=D_v, C_v=C_v, loss=loss,
            early_stopping_rounds=early_stopping_rounds,
            parent_hist=None,
        )
        return root

    def _recurse(node: Node, idx: np.ndarray) -> None:
        if es_state["stop"]:
            return
        if node.depth >= max_depth:
            return
        if node.n_orig < min_samples_split:
            return
        if n_features_to_consider < n_features:
            feat_idx = rng.choice(
                n_features, size=n_features_to_consider, replace=False
            )
        else:
            feat_idx = None
        best = best_split_at(
            features, D, C, idx,
            leaf_loss=leaf_loss, alpha_at_opt=alpha_at_opt,
            min_orig_leaf=eff_min_orig_leaf,
            categorical_features=categorical_features,
            feature_indices=feat_idx,
            fast_loss_kind=fast_loss_kind,
            bounded_lo=bounded_lo, bounded_hi=bounded_hi,
            user_cfunc_addr=user_cfunc_addr,
            hist_payload=hist_payload,
            random_splitter_rng=random_splitter_rng,
        )
        if best is None:
            return
        best_feat, best_split = best
        if best_split[0] <= min_impurity_decrease:
            return
        is_cat = best_feat in (set(categorical_features) if categorical_features else set())
        left_idx, right_idx = _split_node_into_children(
            node, best_feat, best_split,
            is_categorical=is_cat,
            features=features, D=D, C=C,
            leaf_loss=leaf_loss, alpha_at_opt=alpha_at_opt,
        )

        if early_stopping_rounds is not None and valid_features is not None:
            cur = _holdout_loss(root, valid_features, D_v, C_v, loss)
            if cur < es_state["best_loss"] - 1e-12:
                es_state["best_loss"] = cur
                es_state["rounds_since_improve"] = 0
            else:
                es_state["rounds_since_improve"] += 1
                if es_state["rounds_since_improve"] >= early_stopping_rounds:
                    es_state["stop"] = True
                    return

        _recurse(node.left, left_idx)
        _recurse(node.right, right_idx)

    _recurse(root, np.arange(features.shape[0]))
    return root


# ---------------------------------------------------------------------------
# Parent-minus-sibling histogram driver.
#
# When eligible (``hist`` splitter, no categoricals, no max_features
# subsampling), the depthwise recursion threads per-leaf histograms down
# through the tree. After a split:
#   * accumulate the smaller child's histogram from its rows
#   * compute the larger child's histogram by ``parent_hist - smaller_hist``
# This roughly halves the histogram-accumulation work compared to
# rebuilding both children from scratch.

def _recurse_pms_depthwise(
    node: Node,
    idx: np.ndarray,
    *,
    features: np.ndarray,
    D: np.ndarray,
    C: np.ndarray,
    hist_payload: dict,
    leaf_loss,
    alpha_at_opt,
    fast_loss_kind: int,
    bounded_lo: float,
    bounded_hi: float,
    min_samples_split: int,
    eff_min_orig_leaf: int,
    min_impurity_decrease: float,
    max_depth: int,
    es_state: dict,
    valid_features,
    D_v,
    C_v,
    loss,
    early_stopping_rounds: int | None,
    parent_hist,
) -> None:
    if es_state["stop"]:
        return
    if node.depth >= max_depth:
        return
    if node.n_orig < min_samples_split:
        return

    n_features = features.shape[1]
    candidate_features = np.arange(n_features, dtype=np.int32)

    # Build this node's histogram (or use the inherited one from parent-minus-
    # sibling subtraction).
    if parent_hist is None:
        hD, hC, hO, total_D, total_C, total_orig = accumulate_hist(
            hist_payload["X_binned"], D, C, idx,
            candidate_features, hist_payload["max_bins"],
        )
    else:
        hD, hC, hO, total_D, total_C, total_orig = parent_hist

    found = find_best_split_in_hist(
        hD, hC, hO, total_D, total_C, total_orig,
        candidate_features, hist_payload["n_bins_per_feature"],
        loss_kind=fast_loss_kind,
        bounded_lo=bounded_lo, bounded_hi=bounded_hi,
        min_orig_leaf=eff_min_orig_leaf,
    )
    if found is None:
        return
    best_feat, best_bin, gain = found
    if gain <= min_impurity_decrease:
        return

    threshold = float(hist_payload["bin_thresholds"][best_feat][best_bin])
    left_idx, right_idx = partition_idx_by_bin(
        hist_payload["X_binned"], idx, best_feat, best_bin,
    )

    # Mutate node into internal + create child leaves (same as
    # _split_node_into_children but inlined since we have the split shape
    # in (gain, threshold, left_idx, right_idx) form).
    node.is_leaf = False
    node.split_feature = best_feat
    node.split_kind = "continuous"
    node.split_threshold = threshold
    node.split_left_levels = None
    node.split_gain = gain
    node.left = _make_leaf(
        D, C, left_idx, leaf_loss, alpha_at_opt, depth=node.depth + 1
    )
    node.right = _make_leaf(
        D, C, right_idx, leaf_loss, alpha_at_opt, depth=node.depth + 1
    )

    if early_stopping_rounds is not None and valid_features is not None:
        cur = _holdout_loss(
            # Walk via the root, not ``node`` — ``node`` is now internal but
            # we want the whole partial tree. We don't have a handle to the
            # root in this function; use the shared es_state's root reference.
            es_state["root"], valid_features, D_v, C_v, loss,
        )
        if cur < es_state["best_loss"] - 1e-12:
            es_state["best_loss"] = cur
            es_state["rounds_since_improve"] = 0
        else:
            es_state["rounds_since_improve"] += 1
            if es_state["rounds_since_improve"] >= early_stopping_rounds:
                es_state["stop"] = True
                return

    # PMS subtraction. Build the smaller child's histogram explicitly; the
    # larger child's hist is ``parent_hist - smaller_hist``. The parent's
    # histogram is mutated in place (no longer needed after the split) to
    # avoid allocating a fresh per-split array of size (n_features, max_bins).
    #
    # We additionally fall back to plain rebuilding (no subtraction) when
    # the smaller child is so small that the per-bin subtraction cost
    # exceeds the per-row accumulation cost. Threshold: if ``n_smaller <
    # n_features * max_bins / k`` for some small k, plain rebuild wins.
    # Empirically k≈4 is a good balance on macOS arm64 / numpy 1.x.
    n_left = left_idx.size
    n_right = right_idx.size
    n_smaller = n_left if n_left <= n_right else n_right
    n_features_h = hD.shape[0]
    max_bins_h = hD.shape[1]
    pms_worth_it = n_smaller * 4 > n_features_h * max_bins_h

    if not pms_worth_it:
        # Rebuild both children's histograms from rows. Cheaper than the
        # subtraction trick at small leaf sizes.
        left_hist = accumulate_hist(
            hist_payload["X_binned"], D, C, left_idx,
            candidate_features, hist_payload["max_bins"],
        )
        right_hist = accumulate_hist(
            hist_payload["X_binned"], D, C, right_idx,
            candidate_features, hist_payload["max_bins"],
        )
    elif n_left <= n_right:
        # Smaller = left; build its hist from rows; mutate parent → larger.
        sm_hD, sm_hC, sm_hO, sm_tD, sm_tC, sm_tO = accumulate_hist(
            hist_payload["X_binned"], D, C, left_idx,
            candidate_features, hist_payload["max_bins"],
        )
        np.subtract(hD, sm_hD, out=hD)   # in-place: hD now == larger's hist
        np.subtract(hC, sm_hC, out=hC)
        np.subtract(hO, sm_hO, out=hO)
        left_hist = (sm_hD, sm_hC, sm_hO, sm_tD, sm_tC, sm_tO)
        right_hist = (
            hD, hC, hO,
            total_D - sm_tD, total_C - sm_tC, total_orig - sm_tO,
        )
    else:
        # Smaller = right; mirror.
        sm_hD, sm_hC, sm_hO, sm_tD, sm_tC, sm_tO = accumulate_hist(
            hist_payload["X_binned"], D, C, right_idx,
            candidate_features, hist_payload["max_bins"],
        )
        np.subtract(hD, sm_hD, out=hD)
        np.subtract(hC, sm_hC, out=hC)
        np.subtract(hO, sm_hO, out=hO)
        right_hist = (sm_hD, sm_hC, sm_hO, sm_tD, sm_tC, sm_tO)
        left_hist = (
            hD, hC, hO,
            total_D - sm_tD, total_C - sm_tC, total_orig - sm_tO,
        )

    _recurse_pms_depthwise(
        node.left, left_idx,
        features=features, D=D, C=C, hist_payload=hist_payload,
        leaf_loss=leaf_loss, alpha_at_opt=alpha_at_opt,
        fast_loss_kind=fast_loss_kind,
        bounded_lo=bounded_lo, bounded_hi=bounded_hi,
        min_samples_split=min_samples_split,
        eff_min_orig_leaf=eff_min_orig_leaf,
        min_impurity_decrease=min_impurity_decrease,
        max_depth=max_depth,
        es_state=es_state,
        valid_features=valid_features, D_v=D_v, C_v=C_v, loss=loss,
        early_stopping_rounds=early_stopping_rounds,
        parent_hist=left_hist,
    )
    _recurse_pms_depthwise(
        node.right, right_idx,
        features=features, D=D, C=C, hist_payload=hist_payload,
        leaf_loss=leaf_loss, alpha_at_opt=alpha_at_opt,
        fast_loss_kind=fast_loss_kind,
        bounded_lo=bounded_lo, bounded_hi=bounded_hi,
        min_samples_split=min_samples_split,
        eff_min_orig_leaf=eff_min_orig_leaf,
        min_impurity_decrease=min_impurity_decrease,
        max_depth=max_depth,
        es_state=es_state,
        valid_features=valid_features, D_v=D_v, C_v=C_v, loss=loss,
        early_stopping_rounds=early_stopping_rounds,
        parent_hist=right_hist,
    )


# ---------------------------------------------------------------------------
# Leafwise (best-first) growth.

def grow_leafwise(
    features: np.ndarray,
    D: np.ndarray,
    C: np.ndarray,
    loss,
    *,
    max_leaf_nodes: int,
    max_depth: int,
    min_samples_split: int,
    min_orig_leaf: int,
    categorical_features: Sequence[int] = (),
    aug_valid: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    early_stopping_rounds: int | None = None,
    max_features=None,
    min_impurity_decrease: float = 0.0,
    min_weight_fraction_leaf: float = 0.0,
    random_state: int = 0,
    splitter: str = "exact",
    hist_payload: dict | None = None,
) -> Node:
    """Best-first growth: at each step split the leaf whose best candidate
    split has the highest gain, anywhere in the tree.

    Termination: ``max_leaf_nodes`` reached, no above-threshold split remains,
    or early stopping fires.

    sklearn-parity hyperparameters: see :func:`grow_depthwise`.
    """
    leaf_loss, alpha_at_opt = make_leaf_solvers(loss)
    fast_loss_kind, bounded_lo, bounded_hi, user_cfunc_addr = _resolve_fast_loss_args(splitter, loss)
    valid_features = D_v = C_v = None
    if aug_valid is not None:
        valid_features, D_v, C_v = aug_valid

    n_features = features.shape[1]
    random_splitter_rng = (
        np.random.default_rng(random_state + 1) if splitter == "random" else None
    )
    n_features_to_consider = _resolve_max_features(max_features, n_features)
    rng = np.random.default_rng(random_state)
    n_orig_total = int((D > 0).sum())
    eff_min_orig_leaf = _effective_min_orig_leaf(
        min_orig_leaf, min_weight_fraction_leaf, n_orig_total
    )

    root = _make_leaf(
        D, C, np.arange(features.shape[0]), leaf_loss, alpha_at_opt, depth=0
    )

    # Heap entries: (-gain, tiebreaker, node_id, leaf_node, idx, best_feat, best_split, is_cat).
    counter = itertools.count()
    heap: list = []

    def _push_best_split_for_leaf(leaf: Node, idx: np.ndarray) -> None:
        if leaf.depth >= max_depth or leaf.n_orig < min_samples_split:
            return
        if n_features_to_consider < n_features:
            feat_idx = rng.choice(
                n_features, size=n_features_to_consider, replace=False
            )
        else:
            feat_idx = None
        best = best_split_at(
            features, D, C, idx,
            leaf_loss=leaf_loss, alpha_at_opt=alpha_at_opt,
            min_orig_leaf=eff_min_orig_leaf,
            categorical_features=categorical_features,
            feature_indices=feat_idx,
            fast_loss_kind=fast_loss_kind,
            bounded_lo=bounded_lo, bounded_hi=bounded_hi,
            user_cfunc_addr=user_cfunc_addr,
            hist_payload=hist_payload,
            random_splitter_rng=random_splitter_rng,
        )
        if best is None:
            return
        best_feat, best_split = best
        if best_split[0] <= min_impurity_decrease:
            return
        is_cat = best_feat in (set(categorical_features) if categorical_features else set())
        heapq.heappush(
            heap,
            (-float(best_split[0]), next(counter), id(leaf), leaf, idx, best_feat, best_split, is_cat),
        )

    _push_best_split_for_leaf(root, np.arange(features.shape[0]))

    leaves_count = 1
    es_best = _holdout_loss(root, valid_features, D_v, C_v, loss)
    rounds_since_improve = 0

    while heap and leaves_count < max_leaf_nodes:
        neg_gain, _tb, node_id, leaf, idx, best_feat, best_split, is_cat = heapq.heappop(heap)
        if id(leaf) != node_id or not leaf.is_leaf:
            continue  # stale
        left_idx, right_idx = _split_node_into_children(
            leaf, best_feat, best_split,
            is_categorical=is_cat,
            features=features, D=D, C=C,
            leaf_loss=leaf_loss, alpha_at_opt=alpha_at_opt,
        )
        leaves_count += 1

        if early_stopping_rounds is not None and valid_features is not None:
            cur = _holdout_loss(root, valid_features, D_v, C_v, loss)
            if cur < es_best - 1e-12:
                es_best = cur
                rounds_since_improve = 0
            else:
                rounds_since_improve += 1
                if rounds_since_improve >= early_stopping_rounds:
                    break

        _push_best_split_for_leaf(leaf.left, left_idx)
        _push_best_split_for_leaf(leaf.right, right_idx)

    return root
