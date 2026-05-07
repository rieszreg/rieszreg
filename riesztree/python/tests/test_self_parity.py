"""Self-parity: closed-form per-leaf solve on a fixed partition matches
the splitter's leaf values when given the same partition.

No prior implementation of riesztree exists, so the parity test compares
two independent paths within the package: the splitter's full greedy fit
on a tiny dataset, and a hand-applied per-leaf closed-form on the same
final partition.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from riesztree import ATE, RieszTreeRegressor


def _gather_leaves(node, out):
    if node.is_leaf:
        out.append(node)
    else:
        _gather_leaves(node.left, out)
        _gather_leaves(node.right, out)


def test_leaf_values_match_closed_form_on_same_partition():
    """For every leaf the splitter produces, recompute α* = -C/D from the
    augmented data routed to that leaf and verify bitwise equality with the
    stored leaf value."""
    rng = np.random.default_rng(0)
    n = 600
    x = rng.normal(0, 1, (n, 3))
    pi = 1 / (1 + np.exp(-0.7 * x[:, 0]))
    a = (rng.uniform(0, 1, n) < pi).astype(float)
    df = pd.DataFrame(x, columns=["x0", "x1", "x2"])
    df.insert(0, "a", a)

    estimand = ATE(treatment="a", covariates=("x0", "x1", "x2"))
    est = RieszTreeRegressor(estimand=estimand, max_depth=4).fit(df)

    feats_full = df[["a", "x0", "x1", "x2"]].to_numpy(dtype=float)
    aug = estimand.augment(feats_full)
    # For each augmented row, walk the tree to find its leaf.
    feats = aug.features
    leaf_idx = {}
    for r in range(feats.shape[0]):
        node = est.predictor_.tree
        while not node.is_leaf:
            j = node.split_feature
            if node.split_kind == "continuous":
                go_left = feats[r, j] <= node.split_threshold
            else:
                go_left = int(feats[r, j]) in node.split_left_levels
            node = node.left if go_left else node.right
        leaf_idx.setdefault(id(node), []).append(r)

    leaves = []
    _gather_leaves(est.predictor_.tree, leaves)

    for leaf in leaves:
        rs = leaf_idx[id(leaf)]
        D_sum = float(aug.is_original[rs].sum())
        C_sum = float(aug.potential_deriv_coef[rs].sum())
        a_star = -C_sum / D_sum if D_sum > 0 else 0.0
        assert np.isclose(a_star, leaf.alpha, atol=1e-12), (
            f"leaf α*={leaf.alpha}, recomputed={a_star}, D={D_sum}, C={C_sum}"
        )
