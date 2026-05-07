"""Compiled fast paths for riesztree.

Phase 3 contributes a flat-array ``FlatTree`` representation with a
Cython ``predict_alpha`` loop. Subsequent phases will add Cython
splitters (``_splitter_exact``, ``_splitter_hist``) and a registry hook
for user-defined Numba ``@cfunc`` leaf-loss kernels.

The Cython extension ``_tree_c`` is imported on demand by
``predict_alpha``; if the extension has not been compiled (e.g. a
fresh source checkout that has not run ``pip install -e .``), the
facade falls back to a pure-Python tree walk. The fallback exists
purely so the package keeps importing — for any real workload, the
compiled extension is required.
"""

from __future__ import annotations

from ._splitter import register_fast_leaf_solver
from ._tree import (
    FlatTree,
    flat_tree_from_node,
    node_from_flat_tree,
    predict_alpha,
)

__all__ = [
    "FlatTree",
    "flat_tree_from_node",
    "node_from_flat_tree",
    "predict_alpha",
    "register_fast_leaf_solver",
]
