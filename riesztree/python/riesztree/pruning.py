"""Cost-complexity pruning for a riesztree.

Implements the Breiman et al. (1984) weakest-link pruning generalised to
the augmented Bregman loss. The criterion ``R_α(T) = R(T) + α |leaves(T)|``
is minimised; we collapse the subtree rooted at the node with the smallest
``g(t) = (R(t) - R(T_t)) / (|leaves(T_t)| - 1)`` until the pruned-tree
penalty satisfies ``g_min ≥ ccp_alpha``.

For each candidate-collapse subtree this requires per-subtree
``R(T_t)`` (sum of leaf-loss-at-optimum across the subtree's leaves) and
``R(t)`` (the loss the node would have if collapsed to a single leaf, i.e.
``L(α_node*) = leaf_loss(D_node, C_node)``).
"""

from __future__ import annotations

import warnings

from .splitter import make_leaf_solvers
from .tree import Node


def _leaves_loss_sum(node: Node) -> float:
    if node.is_leaf:
        return float(node.leaf_loss_value)
    return _leaves_loss_sum(node.left) + _leaves_loss_sum(node.right)


def _n_leaves(node: Node) -> int:
    if node.is_leaf:
        return 1
    return _n_leaves(node.left) + _n_leaves(node.right)


def _collapse_to_leaf(node: Node, leaf_loss) -> None:
    """Mutate ``node`` from internal to leaf, recomputing leaf payload."""
    node.is_leaf = True
    node.split_feature = None
    node.split_kind = None
    node.split_threshold = None
    node.split_left_levels = None
    node.split_gain = 0.0
    node.left = None
    node.right = None
    node.leaf_loss_value = float(leaf_loss(node.D, node.C))


def _weakest_link_alpha(root: Node, leaf_loss) -> tuple[float, Node | None]:
    """Find the internal node ``t`` with the smallest ``g(t)``.

    Returns ``(g_min, node_to_collapse)``; when no internal node exists
    returns ``(inf, None)``.
    """
    g_min = float("inf")
    best: Node | None = None

    def _walk(n: Node) -> None:
        nonlocal g_min, best
        if n.is_leaf:
            return
        n_leaves_t = _n_leaves(n)
        R_subtree = _leaves_loss_sum(n)
        R_collapsed = float(leaf_loss(n.D, n.C))
        if n_leaves_t > 1:
            g = (R_collapsed - R_subtree) / (n_leaves_t - 1)
            if g < g_min:
                g_min = g
                best = n
        _walk(n.left)
        _walk(n.right)

    _walk(root)
    return g_min, best


def cost_complexity_prune(
    root: Node,
    loss,
    *,
    ccp_alpha: float | None = None,
    pruning_alpha: float | None = None,
) -> Node:
    """Prune the tree in place by repeatedly collapsing the weakest link.

    Stops when ``g_min ≥ ccp_alpha`` or only the root remains. Returns
    the pruned root (same object as input, mutated).

    The keyword is ``ccp_alpha`` to match
    :class:`sklearn.tree.DecisionTreeRegressor`. ``pruning_alpha`` is
    accepted as a deprecated alias and emits ``FutureWarning``.
    """
    if pruning_alpha is not None:
        warnings.warn(
            "`pruning_alpha` is deprecated; use `ccp_alpha` instead "
            "(matches sklearn.tree.DecisionTreeRegressor).",
            FutureWarning,
            stacklevel=2,
        )
        if ccp_alpha is None:
            ccp_alpha = pruning_alpha
    if ccp_alpha is None:
        ccp_alpha = 0.0
    if ccp_alpha <= 0.0:
        return root
    leaf_loss, _ = make_leaf_solvers(loss)
    while not root.is_leaf:
        g_min, weakest = _weakest_link_alpha(root, leaf_loss)
        if weakest is None or g_min >= ccp_alpha:
            break
        _collapse_to_leaf(weakest, leaf_loss)
    return root
