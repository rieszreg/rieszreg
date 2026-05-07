"""Nyström-preconditioned conjugate gradient on the symmetric o-block.

Reuses the o/c partition from `direct.py`: γ_c is closed-form, γ_o solves a
symmetric PSD system

    (K̃_oo + n λ I) γ̃ = D^{-1/2} rhs

with K̃_oo = D^{1/2} K_oo D^{1/2}. For n_o where direct eigendecomposition is
too expensive, run preconditioned CG with a Nyström preconditioner built from
m randomly-sampled landmark rows: P ≈ (K̃_oo + n λ I)^{-1} via the
rank-m approximation K̃_oo ≈ K̃_nm K̃_mm^{-1} K̃_mn.

For multiple λ values we re-use the kernel matrix and landmark factorizations
across the path — only the diagonal shift n λ changes.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import scipy.linalg
from scipy.sparse.linalg import LinearOperator, cg

from rieszreg import AugmentedDataset

from ..kernels import Kernel
from . import SolveResult


def _make_preconditioner(K_mm_chol, K_nm_scaled, n_lam):
    """Build a callable applying P ≈ (K̃_oo + n λ I)^{-1} via the Nyström
    approximation K̃_oo ≈ K̃_nm K̃_mm^{-1} K̃_mn.

    Sherman-Morrison-Woodbury:
        (n λ I + K̃_nm K̃_mm^{-1} K̃_mn)^{-1}
        = (1/(n λ)) [I − K̃_nm (n λ K̃_mm + K̃_mn K̃_nm)^{-1} K̃_mn]
    """
    K_mn = K_nm_scaled.T
    inner = n_lam * (K_mm_chol @ K_mm_chol.T) + K_mn @ K_nm_scaled
    inner_chol = scipy.linalg.cho_factor(
        inner + 1e-10 * np.eye(inner.shape[0]), lower=True
    )

    def apply(v: np.ndarray) -> np.ndarray:
        u = K_mn @ v
        u = scipy.linalg.cho_solve(inner_chol, u)
        return (v - K_nm_scaled @ u) / n_lam

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
    o_mask = aug.is_original > 0
    c_mask = ~o_mask
    p_o = aug.features[o_mask]
    p_c = aug.features[c_mask]
    d_o = aug.is_original[o_mask]
    pdc_o = aug.potential_deriv_coef[o_mask]
    pdc_c = aug.potential_deriv_coef[c_mask]
    n_rows = aug.n_rows
    n_o = p_o.shape[0]
    n_c = p_c.shape[0]

    kernel.fit_data(aug.features)

    if n_landmarks is None:
        n_landmarks = min(n_o, max(50, int(np.sqrt(n_o)) * 4))
    n_landmarks = min(n_landmarks, n_o)

    K_oo = kernel(p_o, p_o)
    sqrt_d = np.sqrt(d_o)
    K_tilde = (sqrt_d[:, None] * sqrt_d[None, :]) * K_oo
    K_tilde = K_tilde + jitter * np.eye(n_o)

    # Landmark indices and gram matrices.
    landmark_idx = rng.choice(n_o, size=n_landmarks, replace=False)
    K_mm = K_tilde[np.ix_(landmark_idx, landmark_idx)]
    K_nm = K_tilde[:, landmark_idx]
    # Cholesky of K_mm (jitter inside K_tilde already).
    K_mm_jit = K_mm + 1e-8 * np.eye(n_landmarks)
    L_mm = np.linalg.cholesky(K_mm_jit)
    K_nm_scaled = K_nm  # alias; the scaling is captured via L_mm in the WMR formula

    # K_oc precomputation.
    if n_c > 0:
        K_oc = kernel(p_o, p_c)
        K_oc_pdc_c = K_oc @ pdc_c
    else:
        K_oc_pdc_c = np.zeros(n_o)

    # Validation slabs.
    if aug_valid is not None:
        K_vo = kernel(aug_valid.features, p_o)
        K_vc = kernel(aug_valid.features, p_c) if n_c > 0 else None
    else:
        K_vo = K_vc = None

    Aop_base = K_tilde  # full materialized; cheap matvec.

    results: list[SolveResult] = []
    val_losses: list[float] = []

    for lam in lambdas:
        n_lam = n_rows * float(lam)
        gamma_c = -pdc_c / n_lam if n_c > 0 else np.zeros(0)
        rhs = -pdc_o + K_oc_pdc_c / n_lam
        rhs_tilde = rhs / sqrt_d

        # Operator: (K̃ + n_lam I) v
        def matvec(v, n_lam=n_lam):
            return Aop_base @ v + n_lam * v

        op = LinearOperator(
            shape=(n_o, n_o), matvec=matvec, rmatvec=matvec, dtype=float
        )
        precond_apply = _make_preconditioner(L_mm, K_nm_scaled, n_lam)
        Mop = LinearOperator(
            shape=(n_o, n_o), matvec=precond_apply, rmatvec=precond_apply, dtype=float
        )

        gamma_tilde, info = cg(
            op, rhs_tilde, M=Mop, rtol=cg_tol, maxiter=cg_max_iter
        )
        gamma_o = sqrt_d * gamma_tilde

        gamma = np.zeros(aug.features.shape[0])
        gamma[o_mask] = gamma_o
        gamma[c_mask] = gamma_c

        results.append(
            SolveResult(
                kind="dual",
                support=aug.features,
                gamma=gamma,
                extra={
                    "lambda": float(lam),
                    "cg_info": int(info),
                    "n_landmarks": int(n_landmarks),
                    "n_o": n_o,
                    "n_c": n_c,
                },
            )
        )

        if K_vo is not None:
            alpha_val = K_vo @ gamma_o
            if K_vc is not None:
                alpha_val = alpha_val + K_vc @ gamma_c
            row_loss = (
                aug_valid.is_original * alpha_val ** 2
                + 2.0 * aug_valid.potential_deriv_coef * alpha_val
            )
            val_losses.append(float(np.sum(row_loss) / aug_valid.n_rows))

    return results, (np.asarray(val_losses) if aug_valid is not None else None)
