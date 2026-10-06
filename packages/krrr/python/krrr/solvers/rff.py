"""Random Fourier features (Rahimi-Recht 2008) primal solver.

For shift-invariant kernels, sample D random Fourier features so that
φ(x) · φ(y) ≈ k(x, y), then solve KRR in primal feature space:

    L_n(w) = (1/n) Σ_r [D_r (φ_r · w)² + 2 C_r (φ_r · w)] + λ ‖w‖²
    ⇒  (Φ̃_o^T Φ̃_o + n λ I) w = − Φ^T C

with Φ̃_o = diag(D_o) Φ_o (D ∈ {0, 1}). The system is D × D (the feature dimension), so
cost is O(n D + D³) regardless of n_aug. Storage: just `w` (length D) and the
random projection spec (frequencies + biases). Suitable for very large n with
shift-invariant kernels; the kernel supplies the features through
`Kernel.random_features` (implemented for `Gaussian`).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import scipy.linalg

from rieszreg import AugmentedDataset

from ..kernels import Kernel
from . import _SQUARED, SolveResult


def solve_rff(
    aug: AugmentedDataset,
    kernel: Kernel,
    lambdas: Sequence[float],
    *,
    aug_valid: AugmentedDataset | None = None,
    n_features: int = 1024,
    random_state: int = 0,
) -> tuple[list[SolveResult], np.ndarray | None]:
    feat_map = kernel.random_features(aug.features.shape[1], n_features, np.random.default_rng(random_state))

    Phi = feat_map(aug.features)
    Phi_w = aug.is_original[:, None] * Phi  # D ∈ {0, 1}, so √D = D
    G = Phi_w.T @ Phi_w                      # (D × D, λ-independent)
    rhs = -(Phi.T @ aug.potential_deriv_coef)
    n_rows = aug.n_rows
    Phi_v = feat_map(aug_valid.features) if aug_valid is not None else None

    results: list[SolveResult] = []
    val_losses: list[float] = []
    eye_D = np.eye(n_features)
    for lam in lambdas:
        n_lam = n_rows * float(lam)
        A = G + n_lam * eye_D
        w = scipy.linalg.solve(A, rhs, assume_a="pos")
        results.append(
            SolveResult(
                kind="primal",
                weights=w,
                feature_map=feat_map,
                extra={"lambda": float(lam), "n_features": n_features},
            )
        )
        if Phi_v is not None:
            val_losses.append(aug_valid.mean_loss(_SQUARED, Phi_v @ w))

    return results, (np.asarray(val_losses) if aug_valid is not None else None)
