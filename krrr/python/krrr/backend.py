"""KernelRidgeBackend — implements `rieszreg.backends.base.Backend`.

Consumes the `AugmentedDataset` produced by `rieszreg.build_augmented` and
returns a `FitResult` whose `predictor` is a `KernelPredictor`. Iterates over
`lambda_grid` and picks the best λ either by validation Riesz loss
(`aug_valid` provided) or the smallest λ (no validation set).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from rieszreg import AugmentedDataset
from rieszreg.backends.base import FitResult
from rieszreg.losses import LossSpec, SquaredLoss

from .kernels import Gaussian, Kernel
from .predictor import KernelPredictor
from .solvers import SolveResult, auto_choose, get_solver


@dataclass
class KernelRidgeBackend:
    """Kernel ridge regression backend for the rieszboost framework.

    Parameters
    ----------
    kernel : Kernel, default=Gaussian(length_scale="median")
    lambda_grid : sequence of float
        Regularization values to sweep. Selection by validation Riesz loss.
    solver : str, default="auto"
        One of "direct", "nystrom_cg", "rff", "falkon", "auto".
    n_landmarks : int or None
        For "nystrom_cg" / "falkon". Defaults to `min(n_o, max(50, 4√n_o))`.
    n_features : int, default=1024
        For "rff" only.
    cg_tol, cg_max_iter : float, int
        For "nystrom_cg" only.
    random_state : int
    """

    kernel: Kernel = field(default_factory=lambda: Gaussian())
    lambda_grid: Sequence[float] = field(default_factory=lambda: tuple(10.0 ** np.linspace(-4, 0, 21)))
    solver: str = "auto"
    n_landmarks: int | None = None
    n_features: int = 1024
    cg_tol: float = 1e-6
    cg_max_iter: int = 200
    validation_fraction: float = 0.2
    random_state: int = 0
    keep_path: bool = True

    def fit_augmented(
        self,
        aug_train: AugmentedDataset,
        aug_valid: AugmentedDataset | None,
        loss: LossSpec,
        *,
        base_score: float,
        random_state: int,
        hyperparams: dict[str, Any],
    ) -> FitResult:
        # KRR is non-iterative; ignore the catch-all hyperparams dict.
        del hyperparams

        if not isinstance(loss, SquaredLoss):
            raise NotImplementedError(
                f"KernelRidgeBackend currently supports SquaredLoss only "
                f"(got {type(loss).__name__}). KLLoss requires Newton iteration "
                "on the kernel system; planned for a future release."
            )

        # base_score (initial α) is folded into the targets: replace
        # potential_deriv_coef with potential_deriv_coef + is_original · base_score
        # so that predicting α̂ = base_score + (kernel part) minimizes the same loss.
        if base_score != 0.0:
            aug_train = AugmentedDataset(
                features=aug_train.features,
                is_original=aug_train.is_original,
                potential_deriv_coef=(
                    aug_train.potential_deriv_coef + aug_train.is_original * base_score
                ),
                origin_index=aug_train.origin_index,
                n_rows=aug_train.n_rows,
            )
            if aug_valid is not None:
                aug_valid = AugmentedDataset(
                    features=aug_valid.features,
                    is_original=aug_valid.is_original,
                    potential_deriv_coef=(
                        aug_valid.potential_deriv_coef + aug_valid.is_original * base_score
                    ),
                    origin_index=aug_valid.origin_index,
                    n_rows=aug_valid.n_rows,
                )

        solver_name = self.solver
        if solver_name == "auto":
            solver_name = auto_choose(aug_train.features.shape[0])
        solver_fn = get_solver(solver_name)

        seed = self.random_state if random_state is None else random_state
        kwargs: dict[str, Any] = {"aug_valid": aug_valid}
        if solver_name == "nystrom_cg":
            kwargs.update(
                n_landmarks=self.n_landmarks,
                cg_tol=self.cg_tol,
                cg_max_iter=self.cg_max_iter,
                random_state=seed,
            )
        elif solver_name == "rff":
            kwargs.update(n_features=self.n_features, random_state=seed)
        elif solver_name == "falkon":
            kwargs.update(
                n_landmarks=self.n_landmarks or 1000,
                cg_max_iter=self.cg_max_iter,
                random_state=seed,
            )

        results, val_losses = solver_fn(aug_train, self.kernel, list(self.lambda_grid), **kwargs)

        if val_losses is not None and len(val_losses) > 0:
            best_idx = int(np.argmin(val_losses))
            best_score = float(val_losses[best_idx])
        else:
            # No validation data: fall back to the largest λ (most regularized,
            # safest default). Users should pass a validation slice for tuning.
            best_idx = int(len(results) - 1)
            best_score = None

        result = results[best_idx]
        predictor = KernelPredictor(
            kernel=self.kernel,
            loss=loss,
            result=result,
            base_score=base_score,
            solve_results=list(results) if self.keep_path else None,
            lambda_grid=tuple(self.lambda_grid) if self.keep_path else None,
        )
        return FitResult(
            predictor=predictor,
            best_iteration=best_idx,
            best_score=best_score,
            history=val_losses.tolist() if val_losses is not None else None,
        )
