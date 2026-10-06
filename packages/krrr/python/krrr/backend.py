"""KernelRidgeBackend — implements `rieszreg.backends.base.Backend`.

Consumes the `AugmentedDataset` produced by `Estimand.augment` and
returns a `FitResult` whose `predictor` is a `KernelPredictor`. Iterates over
`lambda_grid` and picks the best λ either by validation Riesz loss
(`aug_valid` provided) or the largest λ (no validation set).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

import numpy as np

from rieszreg import AugmentedDataset
from rieszreg.backends.base import FitResult
from rieszreg.losses import Loss, SquaredLoss

from .kernels import Gaussian, Kernel
from .predictor import KernelPredictor
from .solvers import auto_choose, get_solver

DEFAULT_LAMBDA_GRID = tuple(10.0 ** np.linspace(-4, 0, 21))


@dataclass
class KernelRidgeBackend:
    """Kernel ridge regression backend for the rieszreg framework.

    Parameters
    ----------
    kernel : Kernel, default=Gaussian(length_scale="median")
        Not modified by fitting: each fit resolves a copy, which the fitted
        predictor keeps.
    lambda_grid : sequence of float
        Regularization values to sweep. Selection by validation Riesz loss;
        without a validation set, the largest λ is used.
    solver : str, default="auto"
        One of "direct", "nystrom_cg", "rff", "auto".
    n_landmarks : int or None
        For "nystrom_cg". Defaults to `min(n_o, max(50, 4√n_o))`.
    n_features : int, default=1024
        For "rff" only.
    cg_tol, cg_max_iter : float, int
        For "nystrom_cg" only.
    """

    kernel: Kernel = field(default_factory=lambda: Gaussian())
    lambda_grid: Sequence[float] = DEFAULT_LAMBDA_GRID
    solver: str = "auto"
    n_landmarks: int | None = None
    n_features: int = 1024
    cg_tol: float = 1e-6
    cg_max_iter: int = 200
    validation_fraction: float = 0.2
    keep_path: bool = True

    def fit_augmented(
        self,
        aug_train: AugmentedDataset,
        aug_valid: AugmentedDataset | None,
        loss: Loss,
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

        # Fold base_score (initial α = b) into the targets: with α = b + f,
        # D(b+f)² + 2C(b+f) = D f² + 2(C + D b) f + (D b² + 2 C b), so the
        # kernel part f solves the same problem with C → C + D b. The dropped
        # constant, the validation loss of α ≡ b, is added back to the
        # reported validation losses.
        val_offset = 0.0
        if base_score != 0.0:
            aug_train = replace(
                aug_train,
                potential_deriv_coef=aug_train.potential_deriv_coef + aug_train.is_original * base_score,
            )
            if aug_valid is not None:
                val_offset = aug_valid.mean_loss(loss, np.full(aug_valid.features.shape[0], base_score))
                aug_valid = replace(
                    aug_valid,
                    potential_deriv_coef=aug_valid.potential_deriv_coef + aug_valid.is_original * base_score,
                )

        solver_name = auto_choose(aug_train.features.shape[0]) if self.solver == "auto" else self.solver
        kwargs: dict[str, Any] = {"aug_valid": aug_valid}
        if solver_name == "nystrom_cg":
            kwargs.update(
                n_landmarks=self.n_landmarks,
                cg_tol=self.cg_tol,
                cg_max_iter=self.cg_max_iter,
                random_state=random_state,
            )
        elif solver_name == "rff":
            kwargs.update(n_features=self.n_features, random_state=random_state)

        kernel = copy.deepcopy(self.kernel)  # solvers resolve data-dependent bandwidths in place
        lambda_grid = tuple(float(lam) for lam in self.lambda_grid)
        results, val_losses = get_solver(solver_name)(aug_train, kernel, list(lambda_grid), **kwargs)

        if val_losses is not None:
            val_losses = val_losses + val_offset
            best_idx = int(np.argmin(val_losses))
            best_score = float(val_losses[best_idx])
        else:
            # No validation data: the most regularized fit is the safest default.
            best_idx = int(np.argmax(lambda_grid))
            best_score = None

        predictor = KernelPredictor(
            kernel=kernel,
            loss=loss,
            result=results[best_idx],
            base_score=base_score,
            solve_results=list(results) if self.keep_path else None,
            lambda_grid=lambda_grid if self.keep_path else None,
        )
        return FitResult(
            predictor=predictor,
            best_iteration=best_idx,
            best_score=best_score,
            history=val_losses.tolist() if val_losses is not None else None,
        )
