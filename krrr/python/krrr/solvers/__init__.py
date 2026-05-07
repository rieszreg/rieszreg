"""Solvers turn an `AugmentedDataset` and a `Kernel` into dual coefficients γ
satisfying

    (diag(a) · K + λ · I) γ = − b / 2

(or an approximation of the same system). Each solver returns a `SolveResult`
that the predictor uses to evaluate α̂ at new points.

Pick a solver by:
    "direct"      — Cholesky / eigendecomposition. n_aug ≤ ~3000.
    "nystrom_cg"  — Nyström-preconditioned CG. n_aug ≤ ~50k.
    "rff"         — Random Fourier features (primal). n_aug arbitrary,
                    shift-invariant kernel only.
    "falkon"      — `falkon` package (optional dependency).
    "auto"        — pick by n_aug and what's importable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SolveResult:
    """Output of a solver.

    `kind` distinguishes how the predictor uses the payload:
      * "dual"  — γ over support points (the augmented features); predict via
                  `K(x_new, support) @ gamma`.
      * "primal" — explicit feature map weights; predict via `Φ(x_new) @ w`.

    `support` and `gamma` are populated for "dual"; `weights` and `feature_map`
    for "primal". `extra` is solver-specific diagnostics (residual norms,
    iteration counts, eigenvalue spectrum, etc.).
    """

    kind: str
    support: np.ndarray | None = None
    gamma: np.ndarray | None = None
    weights: np.ndarray | None = None
    feature_map: Any | None = None
    extra: dict | None = None


def get_solver(name: str):
    """Return the solver function for a name."""
    if name == "direct":
        from .direct import solve_direct
        return solve_direct
    if name == "nystrom_cg":
        from .nystrom_cg import solve_nystrom_cg
        return solve_nystrom_cg
    if name == "rff":
        from .rff import solve_rff
        return solve_rff
    if name == "falkon":
        from .falkon import solve_falkon
        return solve_falkon
    raise ValueError(f"Unknown solver: {name!r}")


def auto_choose(n_aug: int) -> str:
    """Default solver dispatch by augmented-dataset size."""
    if n_aug <= 3000:
        return "direct"
    if n_aug <= 50_000:
        return "nystrom_cg"
    try:
        import falkon  # noqa: F401
        return "falkon"
    except ImportError:
        return "nystrom_cg"
