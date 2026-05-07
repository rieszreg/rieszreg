"""Random Fourier features (Rahimi-Recht 2008) primal solver.

For shift-invariant kernels, sample D random Fourier features so that
φ(x) · φ(y) ≈ k(x, y), then solve KRR in primal feature space:

    L_n(w) = (1/n) Σ_r [D_r (φ_r · w)² + 2 C_r (φ_r · w)] + λ ‖w‖²
    ⇒  (Φ̃_o^T Φ̃_o + n λ I) w = − Φ^T C

with Φ̃_o = diag(√D_o) Φ_o. The system is D × D (the feature dimension), so
cost is O(n D + D³) regardless of n_aug. Storage: just `w` (length D) and the
random projection spec (frequencies + biases). Suitable for very large n with
shift-invariant kernels (Gaussian out of the box).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.linalg

from rieszreg import AugmentedDataset

from ..kernels import Gaussian, Kernel
from . import SolveResult


@dataclass
class RFFFeatureMap:
    """Stores the random projection so prediction can recompute Φ(x)."""

    W: np.ndarray   # (d, D)
    b: np.ndarray   # (D,)
    scale: float    # sqrt(2/D) folded in once

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return self.scale * np.cos(X @ self.W + self.b)


def solve_rff(
    aug: AugmentedDataset,
    kernel: Kernel,
    lambdas: Sequence[float],
    *,
    aug_valid: AugmentedDataset | None = None,
    n_features: int = 1024,
    random_state: int = 0,
) -> tuple[list[SolveResult], np.ndarray | None]:
    if not isinstance(kernel, Gaussian):
        raise NotImplementedError(
            "RFF solver currently supports `Gaussian` only (Rahimi-Recht "
            "spectral density). For other shift-invariant kernels, add a "
            "`random_features` method on the kernel and extend this solver."
        )
    rng = np.random.default_rng(random_state)
    kernel.fit_data(aug.features)

    Phi = kernel.random_features(aug.features, n_features, rng)   # (n_aug, D)
    is_original = aug.is_original
    pdc = aug.potential_deriv_coef
    n_rows = aug.n_rows

    sqrt_d = np.sqrt(np.maximum(is_original, 0.0))
    Phi_w = sqrt_d[:, None] * Phi      # weighted rows

    # G = Phi_w^T Phi_w  (D × D, λ-independent)
    G = Phi_w.T @ Phi_w
    rhs = -(Phi.T @ pdc)               # (D,)

    # Validation features (λ-independent).
    if aug_valid is not None:
        # Use the same kernel object (already fit on training).
        Phi_v = kernel.random_features(aug_valid.features, n_features, rng)
        # Replace with the same projection used for training: re-derive from
        # stored frequencies. The above call resamples; instead, sample once
        # and apply manually.
        pass

    # Re-derive features deterministically by storing the projection.
    rng2 = np.random.default_rng(random_state)
    d = aug.features.shape[1]
    ls = kernel._ls()
    W = rng2.normal(0.0, 1.0 / ls, size=(d, n_features))
    bias = rng2.uniform(0.0, 2.0 * np.pi, size=n_features)
    scale = np.sqrt(2.0 / n_features)
    feat_map = RFFFeatureMap(W=W, b=bias, scale=scale)
    Phi = feat_map(aug.features)
    Phi_w = sqrt_d[:, None] * Phi
    G = Phi_w.T @ Phi_w
    rhs = -(Phi.T @ pdc)

    if aug_valid is not None:
        Phi_v = feat_map(aug_valid.features)
    else:
        Phi_v = None

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
            alpha_val = Phi_v @ w
            row_loss = (
                aug_valid.is_original * alpha_val ** 2
                + 2.0 * aug_valid.potential_deriv_coef * alpha_val
            )
            val_losses.append(float(np.sum(row_loss) / aug_valid.n_rows))

    return results, (np.asarray(val_losses) if aug_valid is not None else None)
