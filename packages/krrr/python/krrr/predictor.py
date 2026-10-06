"""Predictor that satisfies `rieszreg.backends.base.Predictor`.

Wraps a `SolveResult` (from one of the solvers) plus the kernel and loss spec
into something that exposes `predict_eta(X)` / `predict_alpha(X)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from rieszreg.backends.base import register_predictor_loader
from rieszreg.losses import Loss, loss_from_spec

from .kernels import Kernel, RFFFeatureMap, kernel_from_spec
from .solvers import SolveResult


@dataclass
class KernelPredictor:
    """Wraps a `SolveResult` for prediction. Implements the rieszreg
    `Predictor` protocol (`predict_eta`, `predict_alpha`).

    When fit with `keep_path=True`, also stores the full per-λ list of
    `SolveResult`s (`solve_results`) aligned with `lambda_grid`, enabling
    `predict_eta_path` / `predict_alpha_path` to return α̂ at every λ in one
    call by reusing the test-side kernel slab (or feature map) across λ.
    """

    kernel: Kernel
    loss: Loss
    result: SolveResult
    base_score: float = 0.0
    solve_results: list[SolveResult] | None = None
    lambda_grid: tuple[float, ...] | None = None

    kind = "krrr"
    chunk_size = 4096  # test rows per kernel block in predict

    def predict_eta(self, features: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(features, dtype=float))
        if self.result.kind == "dual":
            eta = self.kernel.matvec(X, self.result.support, self.result.gamma, self.chunk_size)
        elif self.result.kind == "primal":
            phi = self.result.feature_map(X)
            eta = phi @ self.result.weights
        else:
            raise ValueError(f"Unknown SolveResult.kind: {self.result.kind!r}")
        return eta + self.base_score

    def predict_alpha(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(self.loss.link_to_alpha(self.predict_eta(features)))

    # ---- Path predict (keep_path=True only) ------------------------------

    def _resolve_lambda_indices(
        self, lambdas: Sequence[float] | None
    ) -> tuple[list[float], list[int]]:
        if self.solve_results is None or self.lambda_grid is None:
            raise RuntimeError(
                "predict_path requires keep_path=True at fit time."
            )
        if lambdas is None:
            return list(self.lambda_grid), list(range(len(self.lambda_grid)))
        out_idx: list[int] = []
        out_lam: list[float] = []
        for lam in lambdas:
            lam_f = float(lam)
            matches = [
                i for i, l in enumerate(self.lambda_grid)
                if np.isclose(l, lam_f, rtol=1e-12, atol=0.0)
            ]
            if not matches:
                raise ValueError(
                    f"lambda={lam_f!r} not in stored lambda_grid "
                    f"{tuple(self.lambda_grid)}."
                )
            out_idx.append(matches[0])
            out_lam.append(self.lambda_grid[matches[0]])
        return out_lam, out_idx

    def predict_eta_path(
        self, features: np.ndarray, lambdas: Sequence[float] | None = None
    ) -> np.ndarray:
        _, indices = self._resolve_lambda_indices(lambdas)
        X = np.atleast_2d(np.asarray(features, dtype=float))
        # Every per-λ SolveResult shares the support (dual) or feature map
        # (primal), so one kernel slab / Φ(X) serves the whole path.
        results = [self.solve_results[i] for i in indices]
        first = results[0]
        if first.kind == "dual":
            eta = self.kernel.matvec(X, first.support, np.column_stack([r.gamma for r in results]), self.chunk_size)
        elif first.kind == "primal":
            eta = first.feature_map(X) @ np.column_stack([r.weights for r in results])
        else:
            raise ValueError(f"Unknown SolveResult.kind: {first.kind!r}")
        return eta + self.base_score

    def predict_alpha_path(
        self, features: np.ndarray, lambdas: Sequence[float] | None = None
    ) -> np.ndarray:
        eta = self.predict_eta_path(features, lambdas)
        return np.asarray(self.loss.link_to_alpha(eta))

    # ---- Serialization ---------------------------------------------------

    def save(self, dir_path) -> None:
        """Write ``predictor.json`` + ``predictor.npz``. The support (dual) or
        feature map (primal) is stored once; with ``keep_path`` the per-λ
        coefficients are stacked into ``path_coef``."""
        import json

        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        r = self.result
        payload = {
            "kernel": self.kernel.to_spec(),
            "loss": self.loss.to_spec(),
            "base_score": float(self.base_score),
            "result_kind": r.kind,
            "result_extra": r.extra or {},
            "lambda_grid": list(self.lambda_grid) if self.lambda_grid is not None else None,
        }
        with open(dir_path / "predictor.json", "w") as f:
            json.dump(payload, f, indent=2)

        coef_name = "gamma" if r.kind == "dual" else "weights"
        arrays = {"coef": getattr(r, coef_name)}
        if r.kind == "dual":
            arrays["support"] = r.support
        else:
            fm = r.feature_map
            arrays.update(rff_W=fm.W, rff_b=fm.b, rff_scale=np.asarray([fm.scale]))
        if r.spectrum is not None:
            arrays["spectrum"] = r.spectrum
        if self.solve_results is not None:
            arrays["path_coef"] = np.stack([getattr(s, coef_name) for s in self.solve_results])
        np.savez(dir_path / "predictor.npz", **arrays)

    @classmethod
    def load(cls, dir_path, base_score=None, loss=None, best_iteration=None):
        import json

        dir_path = Path(dir_path)
        with open(dir_path / "predictor.json") as f:
            payload = json.load(f)
        npz = np.load(dir_path / "predictor.npz")
        kind = payload["result_kind"]
        spectrum = npz["spectrum"] if "spectrum" in npz else None

        def make(coef, extra) -> SolveResult:
            if kind == "dual":
                return SolveResult(kind="dual", support=npz["support"], gamma=coef,
                                   spectrum=spectrum, extra=extra)
            fm = RFFFeatureMap(W=npz["rff_W"], b=npz["rff_b"], scale=float(npz["rff_scale"][0]))
            return SolveResult(kind="primal", weights=coef, feature_map=fm, extra=extra)

        lam_grid = payload.get("lambda_grid")
        solve_results = None
        if "path_coef" in npz and lam_grid is not None:
            solve_results = [
                make(coef, {"lambda": float(lam)})
                for coef, lam in zip(npz["path_coef"], lam_grid)
            ]
        return cls(
            kernel=kernel_from_spec(payload["kernel"]),
            loss=loss if loss is not None else loss_from_spec(payload["loss"]),
            result=make(npz["coef"], payload.get("result_extra") or {}),
            base_score=payload["base_score"] if base_score is None else float(base_score),
            solve_results=solve_results,
            lambda_grid=tuple(lam_grid) if lam_grid is not None else None,
        )


register_predictor_loader("krrr", KernelPredictor.load)
