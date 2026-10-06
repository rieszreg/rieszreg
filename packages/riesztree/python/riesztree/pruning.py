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


from .splitter import make_leaf_solvers
from .tree import Node


def _leaves_loss_sum(node: Node) -> float:
    if node.is_leaf:
        return float(node.leaf_loss_value)
    return _leaves_loss_sum(node.left) + _leaves_loss_sum(node.right)


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
    """Find the internal node ``t`` with the smallest ``g(t)`` in one
    bottom-up pass (subtree loss and leaf count are accumulated, not
    recomputed per node). Ties go to the earliest node in pre-order.

    Returns ``(g_min, node_to_collapse)``; when no internal node exists
    returns ``(inf, None)``.
    """
    best = (float("inf"), float("inf"), None)   # (g, preorder index, node)
    counter = 0

    def _walk(n: Node) -> tuple[float, int]:
        nonlocal best, counter
        if n.is_leaf:
            return float(n.leaf_loss_value), 1
        order = counter
        counter += 1
        R_l, k_l = _walk(n.left)
        R_r, k_r = _walk(n.right)
        R_subtree, n_leaves = R_l + R_r, k_l + k_r
        g = (float(leaf_loss(n.D, n.C)) - R_subtree) / (n_leaves - 1)
        if (g, order) < best[:2]:
            best = (g, order, n)
        return R_subtree, n_leaves

    _walk(root)
    return best[0], best[2]


def cost_complexity_prune(
    root: Node,
    loss,
    *,
    ccp_alpha: float = 0.0,
) -> Node:
    """Prune the tree in place by repeatedly collapsing the weakest link.

    Stops when ``g_min ≥ ccp_alpha`` or only the root remains. Returns
    the pruned root (same object as input, mutated).

    The keyword is ``ccp_alpha`` to match
    :class:`sklearn.tree.DecisionTreeRegressor`.
    """
    if ccp_alpha <= 0.0:
        return root
    leaf_loss, _ = make_leaf_solvers(loss)
    while not root.is_leaf:
        g_min, weakest = _weakest_link_alpha(root, leaf_loss)
        if g_min >= ccp_alpha:
            break
        _collapse_to_leaf(weakest, leaf_loss)
    return root


def cost_complexity_pruning_path(root: Node, loss):
    """Compute the cost-complexity pruning path for ``root``.

    Mirrors :meth:`sklearn.tree.DecisionTreeRegressor.cost_complexity_pruning_path`:
    walks the weakest-link sequence from the unpruned tree all the way
    down to the root, recording the effective ``ccp_alpha`` at each
    collapse and the corresponding subtree impurity (sum of leaf-loss
    over the remaining tree).

    Returns
    -------
    ccp_alphas : np.ndarray[float64]
        Sorted ascending. ``ccp_alphas[0] == 0.0`` (no pruning);
        ``ccp_alphas[-1]`` is the smallest α at which the tree
        collapses to the root.
    impurities : np.ndarray[float64]
        ``impurities[i]`` is the sum-of-leaf-loss-at-optimum for the
        subtree that survives at ``ccp_alpha == ccp_alphas[i]``.
        Same length as ``ccp_alphas``.

    The input tree is **not** mutated — internally we deep-copy and
    walk the copy.
    """
    import copy
    work = copy.deepcopy(root)
    leaf_loss, _ = make_leaf_solvers(loss)

    alphas: list[float] = [0.0]
    impurities: list[float] = [_leaves_loss_sum(work)]

    while not work.is_leaf:
        g_min, weakest = _weakest_link_alpha(work, leaf_loss)
        _collapse_to_leaf(weakest, leaf_loss)
        alphas.append(float(max(g_min, alphas[-1])))   # keep monotonic
        impurities.append(_leaves_loss_sum(work))

    import numpy as np
    return np.asarray(alphas, dtype=np.float64), np.asarray(impurities, dtype=np.float64)
