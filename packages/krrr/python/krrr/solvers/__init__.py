"""Solvers turn an `AugmentedDataset` and a `Kernel` into dual coefficients γ
satisfying

    (diag(D) · K + n λ · I) γ = − C

(or an approximation of the same system). Each solver returns a `SolveResult`
that the predictor uses to evaluate α̂ at new points.

Pick a solver by:
    "direct"      — eigendecomposition. n_aug ≤ ~3000.
    "nystrom_cg"  — Nyström-preconditioned CG. Stores K_oo densely like
                    "direct" but skips its O(n_o³) eigendecomposition.
    "rff"         — Random Fourier features (primal). n_aug arbitrary,
                    shift-invariant kernel only.
    "auto"        — "direct" for n_aug ≤ 3000, else "nystrom_cg".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rieszreg import AugmentedDataset
from rieszreg.losses import SquaredLoss


@dataclass
class SolveResult:
    """Output of a solver.

    `kind` distinguishes how the predictor uses the payload:
      * "dual"  — γ over support points (the augmented features); predict via
                  `K(x_new, support) @ gamma`.
      * "primal" — explicit feature map weights; predict via `Φ(x_new) @ w`.

    `support` and `gamma` are populated for "dual"; `weights` and `feature_map`
    for "primal". `spectrum` holds the eigenvalues of the o-block Gram matrix
    K̃_oo when the solver computed them (λ-independent; used by diagnostics).
    `extra` is JSON-serializable solver metadata (λ, sizes, CG status, ...).
    """

    kind: str
    support: np.ndarray | None = None
    gamma: np.ndarray | None = None
    weights: np.ndarray | None = None
    feature_map: Any | None = None
    spectrum: np.ndarray | None = None
    extra: dict | None = None


class OBlockSystem:
    """The λ-independent pieces of the dual system shared by the dual solvers.

    Partition augmented rows into o = {D = 1} (rows carrying the squared term)
    and c = {D = 0} (counterfactual points); D ∈ {0, 1} by construction. Row
    r ∈ c gives ``n λ γ_r = −C_r`` in closed form; substituting back, γ_o solves

        (K_oo + n λ I) γ_o = −C_o + K_oc C_c / (n λ).

    ``self.K_tilde`` is K_oo plus ``jitter`` on the diagonal; solvers factor
    or iterate on it.
    """

    def __init__(self, aug: AugmentedDataset, kernel, aug_valid: AugmentedDataset | None, jitter: float):
        self.aug, self.aug_valid = aug, aug_valid
        self.o_mask = aug.is_original > 0
        p_o = aug.features[self.o_mask]
        p_c = aug.features[~self.o_mask]
        self.pdc_o = aug.potential_deriv_coef[self.o_mask]
        self.pdc_c = aug.potential_deriv_coef[~self.o_mask]
        self.n_o, self.n_c = p_o.shape[0], p_c.shape[0]

        self.K_tilde = kernel(p_o, p_o)
        self.K_tilde[np.diag_indices(self.n_o)] += jitter

        # K_oc C_c and K_vc C_c are all γ_c ever multiplies (γ_c ∝ C_c).
        self.K_oc_pdc_c = kernel.matvec(p_o, p_c, self.pdc_c)
        if aug_valid is not None:
            self.K_vo = kernel(aug_valid.features, p_o)
            self.K_vc_pdc_c = kernel.matvec(aug_valid.features, p_c, self.pdc_c)

    def rhs_tilde(self, n_lam: float) -> np.ndarray:
        return -self.pdc_o + self.K_oc_pdc_c / n_lam

    def result(self, gamma_tilde: np.ndarray, lam: float, **extra) -> tuple[SolveResult, float | None]:
        """Pack γ̃ into a full-length dual `SolveResult` and, with a validation
        set, return its mean validation Riesz loss (squared loss)."""
        n_lam = self.aug.n_rows * float(lam)
        gamma_o = gamma_tilde
        gamma = np.empty(self.aug.features.shape[0])
        gamma[self.o_mask] = gamma_o
        gamma[~self.o_mask] = -self.pdc_c / n_lam
        res = SolveResult(
            kind="dual",
            support=self.aug.features,
            gamma=gamma,
            extra={"lambda": float(lam), "n_rows": self.aug.n_rows, "n_o": self.n_o, "n_c": self.n_c, **extra},
        )
        if self.aug_valid is None:
            return res, None
        alpha_val = self.K_vo @ gamma_o - self.K_vc_pdc_c / n_lam
        return res, self.aug_valid.mean_loss(_SQUARED, alpha_val)


_SQUARED = SquaredLoss()


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
    raise ValueError(f"Unknown solver: {name!r}")


def auto_choose(n_aug: int) -> str:
    """Default solver dispatch by augmented-dataset size."""
    return "direct" if n_aug <= 3000 else "nystrom_cg"
