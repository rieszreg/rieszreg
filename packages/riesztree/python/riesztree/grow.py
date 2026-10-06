"""Tree growth: depthwise (default) and leafwise (best-first) strategies.

Both strategies share the same per-node split search (``_Grower.best_split``);
they differ only in the order in which leaves are considered for splitting.

Validation early stopping: when ``early_stopping_rounds`` is set and a
validation set is given, growth tracks the held-out augmented loss of the
partial tree after each split. Once it has not strictly improved for that
many consecutive splits, growth stops and the tree is rolled back to the
partial tree with the best held-out loss.
"""

from __future__ import annotations

import heapq
import itertools
import math

import numpy as np

from .fast._splitter import (
    LOSS_USER_CFUNC,
    best_split_at_hist,
    best_split_continuous_fast,
    best_split_continuous_random,
    loss_kind_for,
)
from .pruning import _collapse_to_leaf
from .splitter import best_split_categorical, make_leaf_solvers
from .tree import Node


# ---------------------------------------------------------------------------
# Hyperparameter resolution helpers (sklearn parity).

def _resolve_max_features(max_features, n_features: int) -> int:
    """Resolve ``max_features`` to an int in ``[1, n_features]``.

    Mirrors :class:`sklearn.tree.DecisionTreeRegressor`. Accepts:

    - ``None`` (or ``"all"``): all features.
    - ``int``: exact count, clipped to ``[1, n_features]``.
    - ``float`` in ``(0, 1]``: fraction of features (rounded down, ≥ 1).
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
        return max(1, int(max_features * n_features))
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
    return Node(
        is_leaf=True,
        D=D_sum,
        C=C_sum,
        n_orig=int((D[idx] > 0).sum()),
        n_aug=int(idx.size),
        depth=depth,
        alpha=alpha_at_opt(D_sum, C_sum),
        leaf_loss_value=leaf_loss(D_sum, C_sum),
    )


_VALID_SPLITTERS = ("exact", "hist", "random")


def _resolve_fast_loss_args(splitter: str, loss) -> tuple[int, float, float, int]:
    """Decide which split kernel serves this fit.

    Returns ``(loss_kind, bounded_lo, bounded_hi, user_cfunc_addr)``.
    Built-in losses map to their kernel id; user-registered losses map to
    ``LOSS_USER_CFUNC`` with the registered address. Callers run
    ``make_leaf_solvers`` first, which already rejects losses with no solver.
    """
    if splitter not in _VALID_SPLITTERS:
        raise ValueError(
            f"splitter={splitter!r}; expected one of {_VALID_SPLITTERS}."
        )
    kind, lo, hi, user_addr = loss_kind_for(loss)
    if kind == LOSS_USER_CFUNC and splitter != "exact":
        raise NotImplementedError(
            f"splitter={splitter!r} supports the four built-in losses only; "
            f"{type(loss).__name__} is a registered custom loss. Use "
            "splitter='exact'."
        )
    return int(kind), float(lo), float(hi), int(user_addr)


# ---------------------------------------------------------------------------
# Holdout-loss bookkeeping for early stopping.

def _holdout_loss(root: Node, aug_valid, loss) -> float:
    """Held-out mean Riesz loss of the tree rooted at ``root``."""
    from .fast import flat_tree_from_node, predict_alpha as _flat_predict
    return aug_valid.mean_loss(loss, _flat_predict(flat_tree_from_node(root), aug_valid.features))


class _EarlyStopping:
    """Tracks held-out loss after each accepted split and rolls the tree back
    to the best partial tree once ``rounds`` splits in a row fail to improve."""

    def __init__(self, rounds: int | None, valid, loss):
        self.rounds, self.valid, self.loss = rounds, valid, loss
        self.active = rounds is not None and valid is not None
        self.best_loss = float("inf")
        self.since_improve = self.n_best = 0
        self.split_nodes: list[Node] = []

    def start(self, root: Node) -> None:
        if self.active:
            self.best_loss = _holdout_loss(root, self.valid, self.loss)

    def after_split(self, root: Node, node: Node) -> bool:
        """Record the split at ``node``; return True when growth should stop."""
        if not self.active:
            return False
        self.split_nodes.append(node)
        cur = _holdout_loss(root, self.valid, self.loss)
        if cur < self.best_loss - 1e-12:
            self.best_loss = cur
            self.since_improve = 0
            self.n_best = len(self.split_nodes)
            return False
        self.since_improve += 1
        return self.since_improve >= self.rounds

    def rollback(self, leaf_loss) -> None:
        """Undo every split made after the best held-out loss."""
        for node in self.split_nodes[self.n_best:]:
            if not node.is_leaf:
                _collapse_to_leaf(node, leaf_loss)


# ---------------------------------------------------------------------------
# Per-fit grower: split search plus the depthwise and leafwise policies.

class _Grower:
    """Grows one tree. Holds the loss kernels, candidate-feature sampling,
    the column-major feature copy the per-feature kernels read from, and the
    early-stopping tracker. :meth:`grow_depthwise` and :meth:`grow_leafwise`
    differ only in the order in which leaves are considered for splitting.

    sklearn-parity hyperparameters: ``max_features`` sub-samples candidate
    features at each split (see :func:`_resolve_max_features`);
    ``min_impurity_decrease`` rejects splits with gain ≤ that threshold;
    ``min_weight_fraction_leaf`` requires each leaf to hold at least
    ``ceil(min_weight_fraction_leaf * n_original_total)`` original rows
    (combined with ``min_orig_leaf`` via ``max(...)``).

    With ``splitter="hist"``, ``X_binned`` is ``features`` binned by
    ``mapper`` (a :class:`riesztree.fast._binner.BinMapper`).
    """

    def __init__(
        self,
        features, D, C, loss, *,
        max_depth, min_samples_split, min_orig_leaf, categorical_features,
        max_features, min_impurity_decrease, min_weight_fraction_leaf,
        aug_valid, early_stopping_rounds, random_state, splitter,
        X_binned=None, mapper=None,
    ):
        self.features, self.D, self.C, self.loss = features, D, C, loss
        self.max_depth, self.min_samples_split = max_depth, min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.leaf_loss, self.alpha_at_opt = make_leaf_solvers(loss)
        (self.loss_kind, self.bounded_lo, self.bounded_hi,
         self.user_cfunc_addr) = _resolve_fast_loss_args(splitter, loss)
        self.splitter = splitter
        self.cat_set = set(int(j) for j in categorical_features or ())
        self.n_features = features.shape[1]
        self.n_consider = _resolve_max_features(max_features, self.n_features)
        self.rng = np.random.default_rng(random_state)
        # The random splitter draws thresholds from its own stream so it is
        # independent of the max_features subsample stream.
        self.random_rng = (
            np.random.default_rng(None if random_state is None else random_state + 1)
            if splitter == "random" else None
        )
        self.X_binned, self.mapper = X_binned, mapper
        self.min_orig_leaf = _effective_min_orig_leaf(
            min_orig_leaf, min_weight_fraction_leaf, int((D > 0).sum())
        )
        self.es = _EarlyStopping(early_stopping_rounds, aug_valid, loss)
        self._features_T = None

    @property
    def compiled_eligible(self) -> bool:
        """Whether a Cython whole-tree driver can grow this tree: no
        categoricals, no per-split feature subsampling, no early stopping,
        built-in loss, exact or hist splitter."""
        return (
            not self.cat_set
            and self.n_consider == self.n_features
            and self.loss_kind != LOSS_USER_CFUNC
            and not self.es.active
            and self.splitter in ("hist", "exact")
        )

    @property
    def features_T(self) -> np.ndarray:
        # One contiguous copy per fit instead of one per (leaf, feature).
        if self._features_T is None:
            self._features_T = np.ascontiguousarray(self.features.T, dtype=np.float64)
        return self._features_T

    def root(self) -> Node:
        return _make_leaf(
            self.D, self.C, np.arange(self.features.shape[0]),
            self.leaf_loss, self.alpha_at_opt, depth=0,
        )

    def splittable(self, node: Node) -> bool:
        return node.depth < self.max_depth and node.n_orig >= self.min_samples_split

    def best_split(self, idx: np.ndarray):
        """``(feature, split)`` of the best above-threshold split, or None.
        ``split`` is ``(gain, threshold_or_levels, left_idx, right_idx)``.

        Under ``max_features``, a random subset is searched first; as in
        sklearn, the search continues through the remaining features when
        that subset holds no valid split."""
        if self.n_consider >= self.n_features:
            return self._best_among(range(self.n_features), idx)
        perm = self.rng.permutation(self.n_features)
        best = self._best_among(perm[: self.n_consider], idx)
        if best is None:
            best = self._best_among(perm[self.n_consider :], idx)
        return best

    def _best_among(self, cols, idx: np.ndarray):
        cont = [int(j) for j in cols if int(j) not in self.cat_set]
        cats = [int(j) for j in cols if int(j) in self.cat_set]
        D, C = self.D, self.C
        cands = []

        if self.X_binned is not None and cont:
            result = best_split_at_hist(
                self.X_binned, D, C, idx,
                bin_thresholds=self.mapper.bin_thresholds,
                n_bins_per_feature=self.mapper.n_bins,
                candidate_features=np.asarray(cont, dtype=np.int32),
                loss_kind=self.loss_kind,
                bounded_lo=self.bounded_lo, bounded_hi=self.bounded_hi,
                min_orig_leaf=self.min_orig_leaf,
                max_bins=self.mapper.max_bins,
            )
            if result is not None:
                feat, gain, thr, l_idx, r_idx = result
                cands.append((feat, (gain, thr, l_idx, r_idx)))
        else:
            for j in cont:
                col = self.features_T[j]
                if self.random_rng is not None:
                    cand = best_split_continuous_random(
                        col, D, C, idx,
                        loss_kind=self.loss_kind,
                        bounded_lo=self.bounded_lo, bounded_hi=self.bounded_hi,
                        min_orig_leaf=self.min_orig_leaf, rng=self.random_rng,
                    )
                else:
                    cand = best_split_continuous_fast(
                        col, D, C, idx,
                        loss_kind=self.loss_kind,
                        bounded_lo=self.bounded_lo, bounded_hi=self.bounded_hi,
                        min_orig_leaf=self.min_orig_leaf,
                        user_cfunc_addr=self.user_cfunc_addr,
                    )
                if cand is not None:
                    cands.append((j, cand))

        for j in cats:
            cand = best_split_categorical(
                self.features_T[j], D, C, idx, self.leaf_loss, self.alpha_at_opt,
                min_orig_leaf=self.min_orig_leaf,
            )
            if cand is not None:
                cands.append((j, cand))

        if not cands:
            return None
        best = max(cands, key=lambda fc: fc[1][0])
        return best if best[1][0] > self.min_impurity_decrease else None

    def split(self, node: Node, feat: int, split) -> tuple[np.ndarray, np.ndarray]:
        """Turn leaf ``node`` into an internal node; return (left_idx, right_idx)."""
        gain, where, left_idx, right_idx = split
        if feat in self.cat_set:
            node.split_kind, node.split_left_levels, node.split_threshold = "categorical", where, None
        else:
            node.split_kind, node.split_threshold, node.split_left_levels = "continuous", where, None
        node.is_leaf = False
        node.split_feature = feat
        node.split_gain = float(gain)
        for side, rows in (("left", left_idx), ("right", right_idx)):
            setattr(node, side, _make_leaf(
                self.D, self.C, rows, self.leaf_loss, self.alpha_at_opt, depth=node.depth + 1,
            ))
        return left_idx, right_idx

    # ---- growth policies ----

    def grow_depthwise(self) -> Node:
        """Greedy recursive depth-first growth.

        Early stopping counts splits in DFS order, stops expanding once the
        held-out loss fails to strictly improve for that many consecutive
        splits, and rolls back to the DFS prefix with the best held-out
        loss. Use :meth:`grow_leafwise` for textbook best-first early
        stopping.
        """
        if self.compiled_eligible:
            return self._grow_compiled()

        root = self.root()
        self.es.start(root)
        stop = False

        def _recurse(node: Node, idx: np.ndarray) -> None:
            nonlocal stop
            if stop or not self.splittable(node):
                return
            best = self.best_split(idx)
            if best is None:
                return
            left_idx, right_idx = self.split(node, *best)
            if self.es.after_split(root, node):
                stop = True
                return
            _recurse(node.left, left_idx)
            _recurse(node.right, right_idx)

        _recurse(root, np.arange(self.features.shape[0]))
        self.es.rollback(self.leaf_loss)
        return root

    def _grow_compiled(self) -> Node:
        from .fast._tree import node_from_growable_flat_tree
        common = (
            int(self.max_depth), int(self.min_samples_split), int(self.min_orig_leaf),
            float(self.min_impurity_decrease), int(self.loss_kind),
            float(self.bounded_lo), float(self.bounded_hi),
        )
        D_c = np.ascontiguousarray(self.D, dtype=np.float64)
        C_c = np.ascontiguousarray(self.C, dtype=np.float64)
        if self.splitter == "hist":
            from .fast._grow_c import grow_depthwise_hist_c
            growable = grow_depthwise_hist_c(
                self.X_binned, D_c, C_c,
                np.ascontiguousarray(self.mapper.n_bins, dtype=np.int32),
                list(self.mapper.bin_thresholds),
                int(self.mapper.max_bins),
                *common,
            )
        else:
            from .fast._grow_exact_c import grow_depthwise_exact_c
            growable = grow_depthwise_exact_c(
                np.ascontiguousarray(self.features, dtype=np.float64), D_c, C_c, *common,
            )
        return node_from_growable_flat_tree(growable, loss=self.loss)

    def grow_leafwise(self, max_leaf_nodes: int) -> Node:
        """Best-first growth: at each step split the leaf whose best candidate
        split has the highest gain, anywhere in the tree.

        Termination: ``max_leaf_nodes`` reached, no above-threshold split
        remains, or early stopping fires (then the tree rolls back to its
        best held-out loss).
        """
        root = self.root()
        self.es.start(root)

        # Heap entries: (-gain, tiebreaker, leaf, (feature, split)).
        counter = itertools.count()
        heap: list = []

        def _push(leaf: Node, idx: np.ndarray) -> None:
            if not self.splittable(leaf):
                return
            best = self.best_split(idx)
            if best is not None:
                heapq.heappush(heap, (-float(best[1][0]), next(counter), leaf, best))

        _push(root, np.arange(self.features.shape[0]))
        leaves_count = 1
        while heap and leaves_count < max_leaf_nodes:
            _, _, leaf, best = heapq.heappop(heap)
            left_idx, right_idx = self.split(leaf, *best)
            leaves_count += 1
            if self.es.after_split(root, leaf):
                break
            _push(leaf.left, left_idx)
            _push(leaf.right, right_idx)

        self.es.rollback(self.leaf_loss)
        return root
