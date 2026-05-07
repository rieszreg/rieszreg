"""Predictor that satisfies `rieszreg.backends.base.Predictor`.

Wraps a `SolveResult` (from one of the solvers) plus the kernel and loss spec
into something that exposes `predict_eta(X)` / `predict_alpha(X)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from rieszreg.backends.base import register_predictor_loader
from rieszreg.losses import Loss, loss_from_spec

from .kernels import Kernel, kernel_from_spec
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
    feature_keys: tuple[str, ...] = ()
    solve_results: list[SolveResult] | None = None
    lambda_grid: tuple[float, ...] | None = None

    kind = "krrr"

    def predict_eta(self, features: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(features, dtype=float))
        if self.result.kind == "dual":
            K_new = self.kernel(X, self.result.support)
            eta = K_new @ self.result.gamma
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
        n = X.shape[0]
        out = np.empty((n, len(indices)), dtype=float)

        # All current solvers return per-λ SolveResults sharing the same
        # support (dual) or the same feature_map (primal) — exploit that to
        # reuse the test-side kernel slab / feature map across λ.
        first = self.solve_results[indices[0]]
        kind = first.kind
        if kind == "dual":
            K_new = self.kernel(X, first.support)
            for j, idx in enumerate(indices):
                out[:, j] = K_new @ self.solve_results[idx].gamma + self.base_score
        elif kind == "primal":
            phi = first.feature_map(X)
            for j, idx in enumerate(indices):
                out[:, j] = phi @ self.solve_results[idx].weights + self.base_score
        else:
            raise ValueError(f"Unknown SolveResult.kind: {kind!r}")
        return out

    def predict_alpha_path(
        self, features: np.ndarray, lambdas: Sequence[float] | None = None
    ) -> np.ndarray:
        eta = self.predict_eta_path(features, lambdas)
        return np.asarray(self.loss.link_to_alpha(eta))

    # ---- Serialization ---------------------------------------------------

    def save(self, dir_path) -> None:
        import json

        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        payload = {
            "kernel": self.kernel.to_spec(),
            "loss": self.loss.to_spec(),
            "base_score": float(self.base_score),
            "feature_keys": list(self.feature_keys),
            "result_kind": self.result.kind,
            "result_extra": self.result.extra or {},
            "lambda_grid": (
                list(self.lambda_grid) if self.lambda_grid is not None else None
            ),
        }
        with open(dir_path / "predictor.json", "w") as f:
            json.dump(payload, f, indent=2)

        if self.result.kind == "dual":
            np.savez(
                dir_path / "predictor.npz",
                support=self.result.support,
                gamma=self.result.gamma,
            )
        elif self.result.kind == "primal":
            fm = self.result.feature_map
            np.savez(
                dir_path / "predictor.npz",
                weights=self.result.weights,
                rff_W=fm.W,
                rff_b=fm.b,
                rff_scale=np.asarray([fm.scale]),
            )

        if self.solve_results is not None:
            path_dir = dir_path / "solve_results"
            path_dir.mkdir(exist_ok=True)
            for idx, r in enumerate(self.solve_results):
                if r.kind == "dual":
                    np.savez(
                        path_dir / f"lambda_{idx}.npz",
                        support=r.support,
                        gamma=r.gamma,
                    )
                elif r.kind == "primal":
                    fm = r.feature_map
                    np.savez(
                        path_dir / f"lambda_{idx}.npz",
                        weights=r.weights,
                        rff_W=fm.W,
                        rff_b=fm.b,
                        rff_scale=np.asarray([fm.scale]),
                    )

    @classmethod
    def load(cls, dir_path, base_score=None, loss=None, best_iteration=None):
        import json

        from .solvers.rff import RFFFeatureMap

        dir_path = Path(dir_path)
        with open(dir_path / "predictor.json") as f:
            payload = json.load(f)
        npz = np.load(dir_path / "predictor.npz")

        kernel = kernel_from_spec(payload["kernel"])
        loss_loaded = loss if loss is not None else loss_from_spec(payload["loss"])
        result_kind = payload["result_kind"]

        def _load_solve_result(npz_data, kind: str) -> SolveResult:
            if kind == "dual":
                return SolveResult(
                    kind="dual",
                    support=npz_data["support"],
                    gamma=npz_data["gamma"],
                    extra={},
                )
            if kind == "primal":
                fm = RFFFeatureMap(
                    W=npz_data["rff_W"],
                    b=npz_data["rff_b"],
                    scale=float(npz_data["rff_scale"][0]),
                )
                return SolveResult(
                    kind="primal",
                    weights=npz_data["weights"],
                    feature_map=fm,
                    extra={},
                )
            raise ValueError(f"Unknown result_kind: {kind!r}")

        result = _load_solve_result(npz, result_kind)
        result.extra = payload.get("result_extra", {}) or {}

        # Optional path retention
        path_dir = dir_path / "solve_results"
        lam_grid = payload.get("lambda_grid")
        solve_results: list[SolveResult] | None = None
        if path_dir.is_dir() and lam_grid is not None:
            solve_results = []
            for idx, lam in enumerate(lam_grid):
                npz_path = np.load(path_dir / f"lambda_{idx}.npz")
                r = _load_solve_result(npz_path, result_kind)
                r.extra = {"lambda": float(lam)}
                solve_results.append(r)

        bs = payload["base_score"] if base_score is None else float(base_score)
        return cls(
            kernel=kernel,
            loss=loss_loaded,
            result=result,
            base_score=bs,
            feature_keys=tuple(payload.get("feature_keys", ())),
            solve_results=solve_results,
            lambda_grid=tuple(lam_grid) if lam_grid is not None else None,
        )


register_predictor_loader("krrr", KernelPredictor.load)
