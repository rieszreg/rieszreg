"""Compiled fast paths for riesztree.

A flat-array ``FlatTree`` with a Cython ``predict_alpha`` loop, the Cython
split kernels and whole-tree drivers, and the registry hook for
user-defined Numba ``@cfunc`` leaf-loss kernels. The compiled extensions
are required (``pip install -e .`` or ``python setup.py build_ext
--inplace``).
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
