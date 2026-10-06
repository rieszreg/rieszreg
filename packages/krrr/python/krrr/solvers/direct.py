"""Direct (eigendecomposition) solver for the augmented kernel ridge system.

The augmented Riesz loss decomposes per-row as

    L_n(α) = (1/n) Σ_r [D_r α(p_r)² + 2 C_r α(p_r)] + λ ‖α‖²_H

with α̂ = Σ_r γ_r k(·, p_r) by the representer theorem. The first-order
condition gives

    (diag(D) K + n λ I) γ = − C

where K[r,s] = k(p_r, p_s). `OBlockSystem` reduces this to a symmetric PSD
system on the rows with D > 0 (see its docstring). A single
eigendecomposition of K̃_oo solves the entire λ path in O(n_o²) per λ after
the O(n_o³) decomposition.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from rieszreg import AugmentedDataset

from ..kernels import Kernel
from . import OBlockSystem, SolveResult


def solve_direct(
    aug: AugmentedDataset,
    kernel: Kernel,
    lambdas: Sequence[float],
    *,
    aug_valid: AugmentedDataset | None = None,
    jitter: float = 1e-10,
) -> tuple[list[SolveResult], np.ndarray | None]:
    """Solve the augmented KRR system at each λ in `lambdas` via a single
    eigendecomposition.

    Returns
    -------
    results : list[SolveResult]
        One per λ. Each `SolveResult.support` is the augmented feature matrix
        and `gamma` is the dual vector over all augmented points (γ_o filled
        in for the D>0 rows; γ_c = -C_c / (n λ) for the D=0 rows).
    val_losses : np.ndarray | None
        Per-λ validation Riesz loss if `aug_valid` is given, else None.
    """
    kernel.fit_data(aug.features)  # resolve e.g. the "median" length scale
    system = OBlockSystem(aug, kernel, aug_valid, jitter)
    eigvals, eigvecs = np.linalg.eigh(system.K_tilde)

    results: list[SolveResult] = []
    val_losses: list[float] = []
    for lam in lambdas:
        n_lam = aug.n_rows * float(lam)
        coeffs = eigvecs.T @ system.rhs_tilde(n_lam)
        res, val = system.result(eigvecs @ (coeffs / (eigvals + n_lam)), lam)
        res.spectrum = eigvals
        results.append(res)
        if val is not None:
            val_losses.append(val)

    return results, (np.asarray(val_losses) if aug_valid is not None else None)
