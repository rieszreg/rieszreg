"""Nyström-preconditioned conjugate gradient on the symmetric o-block.

Uses the o/c reduction in `OBlockSystem`: γ_c is closed-form, γ_o solves the
symmetric PSD system

    (K̃_oo + n λ I) γ̃ = rhs̃

For n_o where the O(n_o³) eigendecomposition is too expensive, run
preconditioned CG with a Nyström preconditioner built from m randomly-sampled
landmark rows: P ≈ (K̃_oo + n λ I)^{-1} via the rank-m approximation
K̃_oo ≈ K̃_nm K̃_mm^{-1} K̃_mn. K̃_oo itself is stored densely, so memory is
O(n_o²) as for the direct solver.

Across the λ path the kernel matrix, the landmark blocks, and K̃_mn K̃_nm are
shared — only the diagonal shift n λ changes.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import scipy.linalg
from scipy.sparse.linalg import LinearOperator, cg

from rieszreg import AugmentedDataset

from ..kernels import Kernel
from . import OBlockSystem, SolveResult


def _make_preconditioner(K_mm, K_nm, K_mn_K_nm, n_lam):
    """Apply P ≈ (K̃_oo + n λ I)^{-1} via the Nyström approximation
    K̃_oo ≈ K̃_nm K̃_mm^{-1} K̃_mn and Sherman-Morrison-Woodbury:

        (n λ I + K̃_nm K̃_mm^{-1} K̃_mn)^{-1}
        = (1/(n λ)) [I − K̃_nm (n λ K̃_mm + K̃_mn K̃_nm)^{-1} K̃_mn]
    """
    inner = n_lam * K_mm + K_mn_K_nm
    inner_chol = scipy.linalg.cho_factor(inner, lower=True)

    def apply(v: np.ndarray) -> np.ndarray:
        u = scipy.linalg.cho_solve(inner_chol, K_nm.T @ v)
        return (v - K_nm @ u) / n_lam

    return apply


def solve_nystrom_cg(
    aug: AugmentedDataset,
    kernel: Kernel,
    lambdas: Sequence[float],
    *,
    aug_valid: AugmentedDataset | None = None,
    n_landmarks: int | None = None,
    cg_tol: float = 1e-6,
    cg_max_iter: int = 200,
    random_state: int = 0,
    jitter: float = 1e-10,
) -> tuple[list[SolveResult], np.ndarray | None]:
    rng = np.random.default_rng(random_state)
    kernel.fit_data(aug.features)
    system = OBlockSystem(aug, kernel, aug_valid, jitter)
    n_o = system.n_o

    if n_landmarks is None:
        n_landmarks = max(50, int(np.sqrt(n_o)) * 4)
    n_landmarks = min(n_landmarks, n_o)

    # Landmark blocks (λ-independent).
    landmark_idx = rng.choice(n_o, size=n_landmarks, replace=False)
    K_nm = system.K_tilde[:, landmark_idx]
    K_mm = K_nm[landmark_idx] + 1e-8 * np.eye(n_landmarks)  # keeps the preconditioner's inner system PD
    K_mn_K_nm = K_nm.T @ K_nm

    results: list[SolveResult] = []
    val_losses: list[float] = []
    for lam in lambdas:
        n_lam = aug.n_rows * float(lam)

        def matvec(v, n_lam=n_lam):
            return system.K_tilde @ v + n_lam * v

        op = LinearOperator(shape=(n_o, n_o), matvec=matvec, dtype=float)
        precond = _make_preconditioner(K_mm, K_nm, K_mn_K_nm, n_lam)
        Mop = LinearOperator(shape=(n_o, n_o), matvec=precond, dtype=float)
        gamma_tilde, info = cg(op, system.rhs_tilde(n_lam), M=Mop, rtol=cg_tol, maxiter=cg_max_iter)

        res, val = system.result(
            gamma_tilde, lam, cg_info=int(info), n_landmarks=int(n_landmarks)
        )
        results.append(res)
        if val is not None:
            val_losses.append(val)

    return results, (np.asarray(val_losses) if aug_valid is not None else None)
