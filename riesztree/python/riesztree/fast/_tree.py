"""Flat-array tree representation + predict facade.

A ``FlatTree`` mirrors the information in a ``riesztree.tree.Node``
hierarchy as parallel NumPy arrays, sklearn-style. The arrays are
sized to ``n_nodes``; an internal node ``i`` has ``feature[i] >= 0``
and references its children via ``left[i]``, ``right[i]``, while a
leaf has ``feature[i] == -1`` and stores its α* in ``value[i]``.

Categorical splits store the "go-left" level set as a Python
``frozenset[int]`` in ``cat_left_sets[i]``; for non-categorical splits
that slot is ``None``. The Cython predict loop handles continuous
splits at C speed and falls back to a Python-level ``in`` check for
categoricals (Phase 8 will Cythonize that path too).

Building a ``FlatTree`` from a ``Node`` is O(n_nodes) and runs once at
fit time; subsequent prediction is then O(n_rows × depth) at C speed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # avoid circular import at runtime
    from ..tree import Node


@dataclass
class FlatTree:
    """sklearn-style parallel-array tree.

    Attributes
    ----------
    feature : int32[n_nodes]
        Split feature index for internal nodes; ``-1`` at leaves.
    threshold : float64[n_nodes]
        Continuous-split threshold; ``NaN`` at leaves and categorical
        splits.
    left : int32[n_nodes]
        Left child index; ``-1`` at leaves.
    right : int32[n_nodes]
        Right child index; ``-1`` at leaves.
    is_categorical : uint8[n_nodes]
        ``1`` if the node is a categorical split, ``0`` otherwise
        (continuous-split or leaf).
    cat_left_sets : list[frozenset[int] | None]
        Per-node "go-left" level set for categorical splits; ``None``
        otherwise. Length ``n_nodes``.
    value : float64[n_nodes]
        Leaf α*; ``0.0`` at internal nodes (predict never reads it
        for internal nodes).
    """

    feature: np.ndarray
    threshold: np.ndarray
    left: np.ndarray
    right: np.ndarray
    is_categorical: np.ndarray
    cat_left_sets: list
    value: np.ndarray

    @property
    def n_nodes(self) -> int:
        return int(self.feature.shape[0])


def flat_tree_from_node(root: "Node") -> FlatTree:
    """Walk the Node tree once and emit parallel arrays.

    Internal nodes are visited in DFS order; a node's children always
    have indices strictly greater than the node's own index, which the
    Cython predict loop can rely on (no cycles).
    """
    nodes: list = []
    feature: list[int] = []
    threshold: list[float] = []
    left: list[int] = []
    right: list[int] = []
    is_categorical: list[int] = []
    cat_left_sets: list = []
    value: list[float] = []

    def _add(node) -> int:
        idx = len(feature)
        # Reserve the slot first so children get indices > idx.
        feature.append(0)
        threshold.append(np.nan)
        left.append(-1)
        right.append(-1)
        is_categorical.append(0)
        cat_left_sets.append(None)
        value.append(0.0)
        nodes.append(node)
        return idx

    def _walk(node, idx: int) -> None:
        if node.is_leaf:
            feature[idx] = -1
            value[idx] = float(node.alpha)
            return
        feature[idx] = int(node.split_feature)
        if node.split_kind == "categorical":
            is_categorical[idx] = 1
            cat_left_sets[idx] = frozenset(int(v) for v in node.split_left_levels)
        else:
            threshold[idx] = float(node.split_threshold)
        l_idx = _add(node.left)
        r_idx = _add(node.right)
        left[idx] = l_idx
        right[idx] = r_idx
        _walk(node.left, l_idx)
        _walk(node.right, r_idx)

    root_idx = _add(root)
    _walk(root, root_idx)

    return FlatTree(
        feature=np.asarray(feature, dtype=np.int32),
        threshold=np.asarray(threshold, dtype=np.float64),
        left=np.asarray(left, dtype=np.int32),
        right=np.asarray(right, dtype=np.int32),
        is_categorical=np.asarray(is_categorical, dtype=np.uint8),
        cat_left_sets=cat_left_sets,
        value=np.asarray(value, dtype=np.float64),
    )


def node_from_growable_flat_tree(growable, loss=None) -> "Node":
    """Convert a :class:`riesztree.fast._grow_c.GrowableFlatTree`
    (used during fit by the iterative-grow Cython driver) into a
    ``Node`` tree for backward-compat with diagnostics, pruning, and
    serialisation.

    The growable carries per-node ``D_sum``, ``C_sum``, ``n_orig``,
    ``gain``, ``depth``, plus the structural fields
    ``feature``/``threshold``/``left``/``right`` and ``value``
    (per-leaf α*). When ``loss`` is provided, the per-node
    ``leaf_loss_value`` is recomputed via the loss's leaf-loss
    kernel; otherwise it's left at 0.0 (cost-complexity pruning will
    refill it on demand).
    """
    from ..tree import Node

    if loss is not None:
        from ..splitter import make_leaf_solvers
        leaf_loss_fn, _ = make_leaf_solvers(loss)
    else:
        leaf_loss_fn = lambda _D, _C: 0.0

    feature = growable.feature
    threshold = growable.threshold
    left = growable.left
    right = growable.right
    value = growable.value
    D_sum = growable.D_sum
    C_sum = growable.C_sum
    n_orig = growable.n_orig
    gain = growable.gain
    depth = growable.depth
    n_used = int(growable.n_nodes_used)

    nodes = [None] * n_used
    for idx in range(n_used):
        is_leaf = int(feature[idx]) == -1
        node = Node(
            is_leaf=is_leaf,
            D=float(D_sum[idx]),
            C=float(C_sum[idx]),
            n_orig=int(n_orig[idx]),
            n_aug=0,                  # not tracked by the growable; cheap to omit
            depth=int(depth[idx]),
            alpha=float(value[idx]),
            leaf_loss_value=float(leaf_loss_fn(float(D_sum[idx]), float(C_sum[idx])))
                if is_leaf else 0.0,
        )
        if not is_leaf:
            node.split_feature = int(feature[idx])
            node.split_kind = "continuous"
            node.split_threshold = float(threshold[idx])
            node.split_left_levels = None
            node.split_gain = float(gain[idx])
        nodes[idx] = node

    # Wire children. Done after all nodes exist.
    for idx in range(n_used):
        if int(feature[idx]) != -1:
            nodes[idx].left = nodes[int(left[idx])]
            nodes[idx].right = nodes[int(right[idx])]

    return nodes[0]


def node_from_flat_tree(tree: FlatTree) -> "Node":
    """Reverse adapter: build a ``Node`` hierarchy from parallel arrays.

    Loses the per-node aggregated statistics (``D``, ``C``, ``n_orig``,
    ``n_aug``, ``leaf_loss_value``, ``split_gain``, ``depth``) that
    were not stored in the ``FlatTree``; the resulting ``Node`` tree is
    sufficient for prediction but not for diagnostics that consume
    those fields. The serialization path keeps the ``Node`` tree as
    the source of truth; this reverse adapter exists for testing only.
    """
    from ..tree import Node

    def _build(idx: int, depth: int) -> Node:
        if tree.feature[idx] == -1:
            return Node(is_leaf=True, alpha=float(tree.value[idx]), depth=depth)
        node = Node(
            is_leaf=False,
            split_feature=int(tree.feature[idx]),
            depth=depth,
        )
        if tree.is_categorical[idx]:
            node.split_kind = "categorical"
            node.split_left_levels = tuple(sorted(tree.cat_left_sets[idx]))
        else:
            node.split_kind = "continuous"
            node.split_threshold = float(tree.threshold[idx])
        node.left = _build(int(tree.left[idx]), depth + 1)
        node.right = _build(int(tree.right[idx]), depth + 1)
        return node

    return _build(0, 0)


# ---------------------------------------------------------------------------
# Predict facade.

def _predict_alpha_python(tree: FlatTree, X: np.ndarray) -> np.ndarray:
    """Pure-Python fallback used when the Cython extension hasn't been
    compiled. Slow; same correctness contract as the compiled path."""
    feature = tree.feature
    threshold = tree.threshold
    left = tree.left
    right = tree.right
    is_categorical = tree.is_categorical
    cat_left_sets = tree.cat_left_sets
    value = tree.value
    n_rows = X.shape[0]
    out = np.empty(n_rows, dtype=np.float64)
    for i in range(n_rows):
        node = 0
        while feature[node] >= 0:
            feat = feature[node]
            x_val = X[i, feat]
            if is_categorical[node]:
                if int(x_val) in cat_left_sets[node]:
                    node = left[node]
                else:
                    node = right[node]
            else:
                if x_val <= threshold[node]:
                    node = left[node]
                else:
                    node = right[node]
        out[i] = value[node]
    return out


def predict_alpha(tree: FlatTree, X: np.ndarray) -> np.ndarray:
    """Predict α* for each row in ``X``.

    Uses the Cython tight loop ``_tree_c.predict_alpha_c`` when
    available; falls back to ``_predict_alpha_python`` if the extension
    has not been compiled.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    try:
        from . import _tree_c  # type: ignore[attr-defined]
    except ImportError:
        return _predict_alpha_python(tree, X)
    return _tree_c.predict_alpha_c(
        X,
        tree.feature,
        tree.threshold,
        tree.left,
        tree.right,
        tree.is_categorical,
        tree.cat_left_sets,
        tree.value,
    )
