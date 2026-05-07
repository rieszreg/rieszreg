"""Tree datastructure: ``Node`` + traversal + serialization helpers.

Nodes are organised as a Python class hierarchy. For prediction we walk the
tree in pure Python; for n=10⁵ rows × depth ~10 the dispatch overhead is
tens of milliseconds, well below the splitter's cost. A flat-array
representation could replace this once profiling justifies it.

Continuous splits use ``feature_index, threshold`` (left = ``x ≤ thr``).
Categorical splits use ``feature_index, left_levels`` (left = ``x in left_levels``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Node:
    is_leaf: bool = True
    # Statistics aggregated over augmented rows assigned to this node.
    D: float = 0.0
    C: float = 0.0
    n_orig: int = 0          # count of augmented rows with D_r > 0 in this node
    n_aug: int = 0           # total augmented rows in this node
    depth: int = 0
    # Leaf payload (when ``is_leaf``).
    alpha: float = 0.0
    leaf_loss_value: float = 0.0
    # Internal-node payload (when not ``is_leaf``).
    split_feature: int | None = None
    split_kind: str | None = None      # "continuous" | "categorical"
    split_threshold: float | None = None
    split_left_levels: tuple[int, ...] | None = None
    split_gain: float = 0.0
    left: "Node | None" = None
    right: "Node | None" = None

    def predict_one(self, x: np.ndarray) -> float:
        node = self
        while not node.is_leaf:
            if node.split_kind == "continuous":
                node = node.left if x[node.split_feature] <= node.split_threshold else node.right
            else:
                node = node.left if int(x[node.split_feature]) in node.split_left_levels else node.right
        return node.alpha


def predict_array(root: Node, X: np.ndarray) -> np.ndarray:
    out = np.empty(X.shape[0], dtype=float)
    for i in range(X.shape[0]):
        out[i] = root.predict_one(X[i])
    return out


def n_leaves(root: Node) -> int:
    if root.is_leaf:
        return 1
    return n_leaves(root.left) + n_leaves(root.right)


def max_depth(root: Node) -> int:
    if root.is_leaf:
        return root.depth
    return max(max_depth(root.left), max_depth(root.right))


def feature_importance(root: Node, n_features: int) -> np.ndarray:
    """Sum of split-gain across the tree, attributed per feature."""
    out = np.zeros(n_features, dtype=float)
    def _walk(n: Node) -> None:
        if n.is_leaf:
            return
        out[n.split_feature] += float(n.split_gain)
        _walk(n.left)
        _walk(n.right)
    _walk(root)
    total = float(out.sum())
    if total > 0:
        out /= total
    return out


def to_dict(root: Node) -> dict[str, Any]:
    """Serialise a Node tree to a JSON-compatible dict."""
    if root.is_leaf:
        return {
            "is_leaf": True,
            "alpha": float(root.alpha),
            "D": float(root.D),
            "C": float(root.C),
            "n_orig": int(root.n_orig),
            "n_aug": int(root.n_aug),
            "depth": int(root.depth),
            "leaf_loss_value": float(root.leaf_loss_value),
        }
    return {
        "is_leaf": False,
        "split_feature": int(root.split_feature),
        "split_kind": root.split_kind,
        "split_threshold": (
            float(root.split_threshold) if root.split_threshold is not None else None
        ),
        "split_left_levels": (
            list(root.split_left_levels) if root.split_left_levels is not None else None
        ),
        "split_gain": float(root.split_gain),
        "D": float(root.D),
        "C": float(root.C),
        "n_orig": int(root.n_orig),
        "n_aug": int(root.n_aug),
        "depth": int(root.depth),
        "alpha": float(root.alpha),  # the value the node would predict if collapsed
        "left": to_dict(root.left),
        "right": to_dict(root.right),
    }


def from_dict(d: dict[str, Any]) -> Node:
    if d["is_leaf"]:
        return Node(
            is_leaf=True,
            alpha=float(d["alpha"]),
            D=float(d.get("D", 0.0)),
            C=float(d.get("C", 0.0)),
            n_orig=int(d.get("n_orig", 0)),
            n_aug=int(d.get("n_aug", 0)),
            depth=int(d.get("depth", 0)),
            leaf_loss_value=float(d.get("leaf_loss_value", 0.0)),
        )
    return Node(
        is_leaf=False,
        split_feature=int(d["split_feature"]),
        split_kind=d.get("split_kind", "continuous"),
        split_threshold=(
            float(d["split_threshold"]) if d.get("split_threshold") is not None else None
        ),
        split_left_levels=(
            tuple(int(v) for v in d["split_left_levels"])
            if d.get("split_left_levels") is not None
            else None
        ),
        split_gain=float(d.get("split_gain", 0.0)),
        D=float(d.get("D", 0.0)),
        C=float(d.get("C", 0.0)),
        n_orig=int(d.get("n_orig", 0)),
        n_aug=int(d.get("n_aug", 0)),
        depth=int(d.get("depth", 0)),
        alpha=float(d.get("alpha", 0.0)),
        left=from_dict(d["left"]),
        right=from_dict(d["right"]),
    )
