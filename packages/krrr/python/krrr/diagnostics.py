"""KRR-specific diagnostics extending `rieszreg.diagnose`.

`KernelRieszRegressor.diagnose(Z)` returns the base `Diagnostics` fields
plus kernel-specific extras: chosen λ, condition number of the training-time
o-block system, and an effective-degrees-of-freedom estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rieszreg.diagnostics import Diagnostics, diagnose

from .solvers import SolveResult


@dataclass
class KernelDiagnostics(Diagnostics):
    lambda_selected: float | None = None
    n_support: int | None = None
    effective_dof: float | None = None
    condition_number: float | None = None
    extra_warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [super().summary(), "Kernel diagnostics:"]
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


def diagnose_kernel(regressor, Z, **kwargs) -> KernelDiagnostics:
    """Base diagnostics plus λ, effective d.o.f. and condition number.

    Effective d.o.f. and condition number describe the training-time system
    (K̃_oo + n λ I) on the fitted support. They use the o-block spectrum the
    "direct" solver keeps; for other solvers these fields are ``None``.
    """
    base = diagnose(estimator=regressor, Z=Z, **kwargs)

    result: SolveResult = regressor.predictor_.result
    extra = result.extra or {}
    lambda_selected = extra.get("lambda")
    n_support = result.support.shape[0] if result.support is not None else None

    eff_dof = cond = None
    extra_warnings: list[str] = []
    if result.spectrum is not None:
        n_lam = extra["n_rows"] * float(lambda_selected)
        eff_dof = _effective_dof(result.spectrum, n_lam)
        shifted = result.spectrum + n_lam
        cond = float(shifted.max() / max(shifted.min(), 1e-30))
        if cond > 1e10:
            extra_warnings.append(
                f"effective system is ill-conditioned (κ ≈ {cond:.1e}) — "
                "consider a larger λ or a different kernel."
            )

    if extra.get("cg_info", 0) > 0:
        extra_warnings.append(
            f"conjugate gradient did not converge at the selected λ (stopped "
            f"after {extra['cg_info']} iterations); raise cg_max_iter or "
            "cg_tol, or use a larger λ."
        )

    return KernelDiagnostics(
        **vars(base),
        lambda_selected=lambda_selected,
        n_support=n_support,
        effective_dof=eff_dof,
        condition_number=cond,
        extra_warnings=extra_warnings,
    )
