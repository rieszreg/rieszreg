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
):
    """Best (gain, feature_index, split_payload) across all considered features.

    If ``feature_indices`` is given, only those columns are evaluated
    (sklearn-style ``max_features`` subsampling, drawn by the caller).
    """
    cat_set = set(categorical_features) if categorical_features else set()
    best = None
    best_feat = None
    cols = (
        range(features.shape[1])
        if feature_indices is None
        else [int(j) for j in feature_indices]
    )
    for j in cols:
        if j in cat_set:
            cand = best_split_categorical(
                features[:, j], D, C, idx, leaf_loss, alpha_at_opt,
                min_orig_leaf=min_orig_leaf,
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
    if best is None:
        return None
    return best_feat, best


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
    if aug_valid_features is None or len(aug_valid_features) == 0:
        return float("nan")
    from .tree import predict_array
    alpha_hat = predict_array(root, aug_valid_features)
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

    valid_features = D_v = C_v = None
    if aug_valid is not None:
        valid_features, D_v, C_v = aug_valid

    n_features = features.shape[1]
    n_features_to_consider = _resolve_max_features(max_features, n_features)
    rng = np.random.default_rng(random_state)
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
    }

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
) -> Node:
    """Best-first growth: at each step split the leaf whose best candidate
    split has the highest gain, anywhere in the tree.

    Termination: ``max_leaf_nodes`` reached, no above-threshold split remains,
    or early stopping fires.

    sklearn-parity hyperparameters: see :func:`grow_depthwise`.
    """
    leaf_loss, alpha_at_opt = make_leaf_solvers(loss)
    valid_features = D_v = C_v = None
    if aug_valid is not None:
        valid_features, D_v, C_v = aug_valid

    n_features = features.shape[1]
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
