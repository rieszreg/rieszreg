"""KRR-specific diagnostics on top of `rieszboost.diagnose`.

`diagnose_kernel(regressor, X)` returns the same `Diagnostics` fields as the
rieszboost version, plus kernel-specific extras: chosen λ, condition number
of the o-block kernel matrix, and an effective-degrees-of-freedom estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rieszreg.diagnostics import Diagnostics, diagnose

from .estimator import KernelRieszRegressor
from .solvers import SolveResult


@dataclass
class KernelDiagnostics:
    base: Diagnostics
    lambda_selected: float | None
    n_support: int | None
    effective_dof: float | None
    condition_number: float | None
    extra_warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [self.base.summary(), "Kernel diagnostics:"]
        if self.lambda_selected is not None:
            lines.append(f"  λ (selected)       : {self.lambda_selected:.4g}")
        if self.n_support is not None:
            lines.append(f"  support points     : {self.n_support}")
        if self.effective_dof is not None:
            lines.append(f"  effective d.o.f.   : {self.effective_dof:.2f}")
        if self.condition_number is not None:
            lines.append(f"  condition number   : {self.condition_number:.2e}")
        for w in self.extra_warnings:
            lines.append(f"  warning            : {w}")
        return "\n".join(lines)


def _effective_dof(eigvals: np.ndarray, n_lam: float) -> float:
    """tr H_λ where H_λ has eigenvalues μ_i / (μ_i + n λ)."""
    return float(np.sum(eigvals / (eigvals + n_lam)))


def diagnose_kernel(regressor: KernelRieszRegressor, X) -> KernelDiagnostics:
    """Full-fat diagnostics including λ, effective d.o.f. and condition number.

    Effective d.o.f. and condition number are computed from the training-time
    kernel matrix on the support points — this requires a refit-time
    eigendecomposition. For solvers other than "direct" (which already has
    the spectrum), the diagnostic skips these fields.
    """
    base = diagnose(estimator=regressor, X=X)

    result: SolveResult = regressor.predictor_.result
    lambda_selected = result.extra.get("lambda") if result.extra else None
    n_support = result.support.shape[0] if result.support is not None else None

    eff_dof = None
    cond = None
    extra_warnings: list[str] = []

    if result.kind == "dual" and result.support is not None and lambda_selected is not None:
        kernel = regressor.predictor_.kernel
        # Re-derive K̃_oo on the o-block (a > 0 rows). We reproduce the
        # construction in solvers/direct.py — a small redundant cost.
        from rieszreg import build_augmented
        from rieszreg.estimator import _rows_from_X

        rows = _rows_from_X(X, regressor.estimand)
        aug = build_augmented(rows, regressor.estimand)
        n_rows = aug.n_rows
        o_mask = aug.a > 0
        p_o = aug.features[o_mask]
        a_o = aug.a[o_mask]
        if p_o.shape[0] > 0:
            try:
                kernel.fit_data(aug.features)
                K_oo = kernel(p_o, p_o)
                sqrt_a = np.sqrt(a_o)
                K_tilde = (sqrt_a[:, None] * sqrt_a[None, :]) * K_oo
                eigvals = np.linalg.eigvalsh(K_tilde)
                n_lam = n_rows * float(lambda_selected)
                eff_dof = _effective_dof(eigvals, n_lam)
                # Condition number of (K̃_oo + n λ I).
                shifted = eigvals + n_lam
                cond = float(shifted.max() / max(shifted.min(), 1e-30))
                if cond > 1e10:
                    extra_warnings.append(
                        f"effective system is ill-conditioned (κ ≈ {cond:.1e}) — "
                        "consider a larger λ or a different kernel."
                    )
            except Exception as e:
                extra_warnings.append(f"could not compute spectrum: {e}")

    return KernelDiagnostics(
        base=base,
        lambda_selected=lambda_selected,
        n_support=n_support,
        effective_dof=eff_dof,
        condition_number=cond,
        extra_warnings=extra_warnings,
    )
