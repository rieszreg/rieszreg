"""Diagnostics for fitted Riesz representers.

Magnitude (RMS), tail extremes, near-positivity warnings, and held-out
Riesz loss. The extreme-α̂ check is the analogue of "extreme propensity score"
warnings — if the representer takes very large absolute values, the downstream
plug-in estimator's variance will inflate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


@dataclass
class Diagnostics:
    n: int
    rms: float
    mean: float
    min: float
    max: float
    abs_quantiles: dict[float, float]
    n_extreme: int           # rows with |alpha_hat| above `extreme_threshold`
    extreme_fraction: float  # n_extreme / n
    extreme_threshold: float
    riesz_loss: float | None  # held-out per-row Riesz loss, if rows + m given
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Riesz representer diagnostics (n={self.n}):",
            f"  RMS magnitude   : {self.rms:.4f}",
            f"  mean            : {self.mean:.4f}",
            f"  min / max       : {self.min:.4f} / {self.max:.4f}",
            "  |alpha| quantiles:",
        ]
        for q, v in self.abs_quantiles.items():
            lines.append(f"    {q:>5.2f}: {v:.4f}")
        lines.append(
            f"  extreme rows    : {self.n_extreme}/{self.n} "
            f"({100 * self.extreme_fraction:.2f}%) with |alpha| > {self.extreme_threshold}"
        )
        if self.riesz_loss is not None:
            lines.append(f"  held-out Riesz  : {self.riesz_loss:.4f}")
        if self.warnings:
            lines.append("  warnings:")
            for w in self.warnings:
                lines.append(f"    - {w}")
        return "\n".join(lines)


def diagnose(
    alpha_hat: np.ndarray | None = None,
    *,
    booster=None,
    X=None,
    rows: Sequence[dict[str, Any]] | None = None,
    m=None,
    extreme_threshold: float = 30.0,
    extreme_fraction_warn: float = 0.01,
) -> Diagnostics:
    """Compute diagnostics from either pre-computed `alpha_hat` or by
    predicting with `booster.predict(X)`. If `(booster, X)` is given, the
    held-out Riesz loss is computed automatically using the booster's
    estimand and loss spec.

    `rows` / `m` are accepted for backward compatibility (legacy callers) —
    if you have a fitted `RieszBooster`, just pass `booster=…, X=…`.
    """
    if alpha_hat is None:
        if booster is None:
            raise ValueError("diagnose requires either alpha_hat or booster + X")
        if X is None and rows is None:
            raise ValueError("diagnose requires X (or rows for legacy boosters)")
        if X is not None:
            alpha_hat = np.asarray(booster.predict(X))
        else:
            alpha_hat = np.asarray(booster.predict(rows))
    alpha_hat = np.asarray(alpha_hat)

    abs_alpha = np.abs(alpha_hat)
    quantiles = {q: float(np.quantile(abs_alpha, q)) for q in (0.5, 0.9, 0.99, 1.0)}
    n_extreme = int(np.sum(abs_alpha > extreme_threshold))
    extreme_fraction = float(n_extreme / len(alpha_hat))

    riesz_loss = None
    if booster is not None:
        if X is not None and hasattr(booster, "riesz_loss") and callable(booster.riesz_loss):
            try:
                riesz_loss = booster.riesz_loss(X)
            except TypeError:
                # Legacy signature `riesz_loss(rows, m)`
                if rows is not None and m is not None:
                    riesz_loss = booster.riesz_loss(rows, m)
        elif rows is not None and m is not None and hasattr(booster, "riesz_loss"):
            riesz_loss = booster.riesz_loss(rows, m)

    warnings: list[str] = []
    if extreme_fraction > extreme_fraction_warn:
        warnings.append(
            f"{100 * extreme_fraction:.2f}% of rows have |alpha_hat| > "
            f"{extreme_threshold} — possible near-positivity violation; "
            "downstream estimator variance will be inflated."
        )
    if quantiles[1.0] > 10.0 * quantiles[0.99]:
        warnings.append(
            f"max |alpha_hat| ({quantiles[1.0]:.1f}) is >10x the 99th percentile "
            f"({quantiles[0.99]:.1f}) — likely a single extrapolation outlier; "
            "consider tighter tree depth, larger reg_lambda, or earlier stopping."
        )

    return Diagnostics(
        n=len(alpha_hat),
        rms=float(np.sqrt(np.mean(alpha_hat**2))),
        mean=float(np.mean(alpha_hat)),
        min=float(np.min(alpha_hat)),
        max=float(np.max(alpha_hat)),
        abs_quantiles=quantiles,
        n_extreme=n_extreme,
        extreme_fraction=extreme_fraction,
        extreme_threshold=extreme_threshold,
        riesz_loss=riesz_loss,
        warnings=warnings,
    )
