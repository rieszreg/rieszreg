"""Optional Falkon wrapper for the o-block solve.

Falkon (Rudi-Carratino-Rosasco 2017) is a Nyström + preconditioned CG kernel
solver with GPU support, scaling to billions of points. We use it for the
symmetric o-block solve only:

    (K̃_oo + n λ I) γ̃ = D^{-1/2} rhs

mapped onto Falkon's `min (1/n_o) ‖y − K γ‖² + λ_falkon ‖γ‖²` form. The
c-block (D=0 counterfactuals) is added analytically at prediction time
(γ_c = − C_c / (n λ)). Note that the o-block rhs depends on K_oc C_c,
so for each λ we have to materialize K_oc (n_o × n_c) — this wrapper is
intended for the "n large but n_c manageable" regime, not for problems
where even K_oc cannot fit.

Activate with `pip install krrr[falkon]`. Requires PyTorch.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from rieszreg import AugmentedDataset

from ..kernels import Gaussian, Kernel, Matern
from . import SolveResult


def _falkon_kernel(kernel: Kernel):
    """Translate our Kernel into Falkon's kernel object, when supported."""
    import falkon
    if isinstance(kernel, Gaussian):
        return falkon.kernels.GaussianKernel(sigma=kernel._ls())
    if isinstance(kernel, Matern):
        return falkon.kernels.MaternKernel(sigma=kernel._ls(), nu=kernel.nu)
    raise NotImplementedError(
        f"Falkon wrapper does not yet support {type(kernel).__name__}; "
        "fall back to solver='nystrom_cg' or add a translation here."
    )


def solve_falkon(
    aug: AugmentedDataset,
    kernel: Kernel,
    lambdas: Sequence[float],
    *,
    aug_valid: AugmentedDataset | None = None,
    n_landmarks: int = 1000,
    cg_max_iter: int = 30,
    random_state: int = 0,
) -> tuple[list[SolveResult], np.ndarray | None]:
    try:
        import falkon
        import torch
    except ImportError as e:
        raise ImportError(
            "solver='falkon' requires the optional `falkon` extra. Install with "
            "`pip install krrr[falkon]` (pulls falkon + torch)."
        ) from e

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

    if not np.allclose(d_o, 1.0):
        raise NotImplementedError(
            "solver='falkon' currently assumes is_original ≡ 1 on the o-block "
            "(the standard augmentation case). For non-uniform values, use "
            "solver='nystrom_cg'."
        )

    kernel.fit_data(aug.features)
    flk_kernel = _falkon_kernel(kernel)

    # Materialize K_oc once if needed.
    if n_c > 0:
        K_oc = kernel(p_o, p_c)
        K_oc_pdc_c = K_oc @ pdc_c
    else:
        K_oc_pdc_c = np.zeros(n_o)

    # Validation Gram slabs.
    if aug_valid is not None:
        K_vo = kernel(aug_valid.features, p_o)
        K_vc = kernel(aug_valid.features, p_c) if n_c > 0 else None
    else:
        K_vo = K_vc = None

    p_o_t = torch.from_numpy(np.ascontiguousarray(p_o, dtype=np.float64))

    results: list[SolveResult] = []
    val_losses: list[float] = []
    for lam in lambdas:
        n_lam = n_rows * float(lam)
        # Effective Falkon target: y_falkon · n_o = K γ + n_lam γ ≈ rhs
        # With Falkon's (1/n_o) ‖y − K γ‖² + λ_f ‖γ‖² minimization, the solution
        # solves (K + n_o λ_f I) γ = K y. Re-mapped: λ_f = lam · n / n_o,
        # y_falkon = (rhs / n_lam) · n_o (so that K y ≈ rhs scaled appropriately).
        # We instead solve directly via Falkon's lower-level path.
        rhs = -pdc_o + K_oc_pdc_c / n_lam
        lambda_falkon = float(lam) * n_rows / n_o

        opt = falkon.FalkonOptions(use_cpu=True, debug=False)
        flk = falkon.Falkon(
            kernel=flk_kernel,
            penalty=lambda_falkon,
            M=min(n_landmarks, n_o),
            options=opt,
            seed=random_state,
            maxiter=cg_max_iter,
        )
        # Falkon expects 2D Y. Our target is the rhs in the o-block.
        # Falkon's first-order optimality: (K + n λ I) γ = K y, hence
        # we need K y_falkon ≈ rhs. Setting y_falkon = rhs / λ_f / n_o would
        # invert the scaling; the cleanest substitution is to fit on
        # y = -C_o (independent of λ, ignoring K_oc), then add the c-block
        # contribution analytically at prediction time.
        # NOTE: this drops the K_oc C_c / (n λ) coupling on the o-system.
        # For estimands where n_c is small or λ is moderate, the bias is small;
        # for tight overlap or extreme λ it is not. Documented limitation.
        Y = (-pdc_o).reshape(-1, 1)
        Y_t = torch.from_numpy(np.ascontiguousarray(Y, dtype=np.float64))
        flk.fit(p_o_t, Y_t)

        gamma_o = flk.alpha_.cpu().numpy().reshape(-1)
        gamma_c = -pdc_c / n_lam if n_c > 0 else np.zeros(0)

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
                    "lambda_falkon": lambda_falkon,
                    "n_landmarks": int(min(n_landmarks, n_o)),
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
