"""Direct (eigendecomposition) solver for the augmented kernel ridge system.

The augmented Riesz loss decomposes per-row as

    L_n(α) = (1/n) Σ_r [D_r α(p_r)² + 2 C_r α(p_r)] + λ ‖α‖²_H

with α̂ = Σ_r γ_r k(·, p_r) by the representer theorem. The first-order
condition gives

    (diag(D) K + n λ I) γ = − C

where K[r,s] = k(p_r, p_s). Build_augmented assigns D_r ∈ {0, 1} (D=1 for
the original observation row, D=0 for counterfactual evaluation points
introduced by m). Partition the augmented index set:

    o = {r : D_r > 0}  ("original" points; carry the squared term)
    c = {r : D_r = 0}  ("counterfactual" points; carry only the linear term)

Row r ∈ c reduces to `n λ γ_r = − C_r`, so γ_c is closed-form. Substitute
back: γ_o solves a symmetric PSD system

    (diag(D_o)^{1/2} K_oo diag(D_o)^{1/2} + n λ I) γ̃ = diag(D_o)^{-1/2} rhs

with rhs = − C_o + K_oc C_c / (n λ), and γ_o = diag(D_o)^{1/2} γ̃.

A single eigendecomposition of K̃_oo = diag(D_o)^{1/2} K_oo diag(D_o)^{1/2}
solves the entire λ path in O(n_o²) per λ after the O(n_o³) decomposition.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from rieszreg import AugmentedDataset

from ..kernels import Kernel
from . import SolveResult


def _split_oc(aug: AugmentedDataset):
    o_mask = aug.is_original > 0
    return o_mask, ~o_mask


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
    o_mask, c_mask = _split_oc(aug)
    p_o = aug.features[o_mask]
    p_c = aug.features[c_mask]
    d_o = aug.is_original[o_mask]
    pdc_o = aug.potential_deriv_coef[o_mask]
    pdc_c = aug.potential_deriv_coef[c_mask]
    n_rows = aug.n_rows
    n_o = p_o.shape[0]
    n_c = p_c.shape[0]

    # Pre-fit kernel on the augmented features (for "median" length-scale, etc.)
    kernel.fit_data(aug.features)

    # Symmetric weighted gram matrix K̃_oo = D^{1/2} K_oo D^{1/2}.
    K_oo = kernel(p_o, p_o)
    sqrt_d = np.sqrt(d_o)
    K_tilde = (sqrt_d[:, None] * sqrt_d[None, :]) * K_oo
    # Tiny jitter for numerical PSD-ness (eigh tolerates fp roundoff).
    K_tilde = K_tilde + jitter * np.eye(n_o)

    # K_oc enters the rhs at every λ; precompute once.
    if n_c > 0:
        K_oc = kernel(p_o, p_c)
        K_oc_pdc_c = K_oc @ pdc_c  # shape (n_o,)
    else:
        K_oc_pdc_c = np.zeros(n_o)

    # Eigendecomposition once.
    eigvals, eigvecs = np.linalg.eigh(K_tilde)

    # Validation kernel slabs (lambda-independent).
    if aug_valid is not None:
        kernel_val = kernel  # already fit on training (same kernel obj)
        K_vo = kernel_val(aug_valid.features, p_o)
        K_vc = kernel_val(aug_valid.features, p_c) if n_c > 0 else None
    else:
        K_vo = K_vc = None

    results: list[SolveResult] = []
    val_losses: list[float] = []

    for lam in lambdas:
        n_lam = n_rows * float(lam)

        # γ_c closed form.
        if n_c > 0:
            gamma_c = -pdc_c / n_lam
        else:
            gamma_c = np.zeros(0)

        # rhs for the o-system.
        rhs = -pdc_o + K_oc_pdc_c / n_lam
        rhs_tilde = rhs / sqrt_d  # D^{-1/2} rhs

        # Solve via eigendecomposition: (K̃ + n_lam I)^{-1} rhs_tilde
        coeffs = eigvecs.T @ rhs_tilde
        gamma_tilde = eigvecs @ (coeffs / (eigvals + n_lam))
        gamma_o = sqrt_d * gamma_tilde

        # Re-pack into full-augmented-length γ vector.
        gamma = np.zeros(aug.features.shape[0])
        gamma[o_mask] = gamma_o
        gamma[c_mask] = gamma_c

        results.append(
            SolveResult(
                kind="dual",
                support=aug.features,
                gamma=gamma,
                extra={"lambda": float(lam), "n_o": n_o, "n_c": n_c},
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


def gcv_score(
    aug: AugmentedDataset,
    kernel: Kernel,
    lambdas: Sequence[float],
    *,
    jitter: float = 1e-10,
) -> np.ndarray:
    """Closed-form Generalized Cross-Validation score on the o-block path.

    The augmented squared Riesz loss reformulates as a weighted least-squares
    problem on the o-block: with `t_r = -C_r / D_r` and weights `w_r = D_r`,

        loss_row = D_r (α(p_r) - t_r)² + const

    GCV (Craven-Wahba 1978) for weighted ridge:

        GCV(λ) = (1/n_o) ‖√w (α̂ - t)‖² / (1 - tr(H_λ) / n_o)²

    where H_λ = D^{1/2} K_oo (D K_oo + n λ I)^{-1} D^{1/2}, with eigenvalues
    `μ_i / (μ_i + n λ)` for `μ_i` the eigenvalues of K̃_oo. For the c-block the
    target is irrelevant to GCV (γ_c is closed-form and contributes additively
    to predictions).

    This score is meant for fast bandwidth/λ tuning; it ignores the c-block
    contribution to held-out loss, so for final selection prefer
    `solve_direct(..., aug_valid=...)`.
    """
    o_mask = aug.is_original > 0
    p_o = aug.features[o_mask]
    d_o = aug.is_original[o_mask]
    pdc_o = aug.potential_deriv_coef[o_mask]
    n_rows = aug.n_rows
    n_o = p_o.shape[0]
    if n_o == 0:
        return np.full(len(lambdas), np.inf)

    kernel.fit_data(aug.features)
    K_oo = kernel(p_o, p_o)
    sqrt_d = np.sqrt(d_o)
    K_tilde = (sqrt_d[:, None] * sqrt_d[None, :]) * K_oo + jitter * np.eye(n_o)
    eigvals, eigvecs = np.linalg.eigh(K_tilde)

    target_tilde = -pdc_o / sqrt_d  # √w · t
    coeffs = eigvecs.T @ target_tilde
    out = np.empty(len(lambdas))
    for i, lam in enumerate(lambdas):
        n_lam = n_rows * float(lam)
        # fitted_tilde = K̃ (K̃ + n_lam I)^{-1} target_tilde
        s = eigvals / (eigvals + n_lam)
        fitted_tilde = eigvecs @ (s * coeffs)
        residual = target_tilde - fitted_tilde
        rss = float(np.sum(residual ** 2)) / n_o
        denom = 1.0 - float(np.sum(s)) / n_o
        out[i] = rss / max(denom * denom, 1e-12)
    return out
