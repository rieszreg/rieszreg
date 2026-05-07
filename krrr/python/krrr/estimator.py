"""KernelRieszRegressor — sklearn-compatible kernel ridge Riesz regressor.

Subclass of `rieszreg.RieszEstimator` that defaults the backend to
`KernelRidgeBackend` and surfaces kernel-specific hyperparameters
(`kernel`, `lambda_grid`, `solver`, `n_landmarks`, `n_features`,
`cg_tol`, `cg_max_iter`) as constructor args. Composes with `GridSearchCV`,
`cross_val_predict`, `Pipeline`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from rieszreg.estimands.base import Estimand
from rieszreg.estimator import RieszEstimator, _features_from_rows, _rows_from_Z
from rieszreg.losses import Loss, SquaredLoss

from .backend import KernelRidgeBackend
from .kernels import Gaussian, Kernel


class KernelRieszRegressor(RieszEstimator):
    """Kernel ridge regression for the Riesz representer α₀ of a linear
    functional θ(P) = E[m(Z, g₀)].

    Reuses `rieszreg`'s estimand machinery and Bregman-loss framework;
    swaps in a kernel ridge backend for the actual fit.

    Parameters
    ----------
    estimand : rieszreg.Estimand
        Carries `feature_keys` and the `m(alpha)(z, y)` operator.
    kernel : krrr.Kernel, default=Gaussian(length_scale="median")
        Reproducing kernel. Length-scale "median" resolves to the median
        pairwise Euclidean distance on the augmented training points.
    lambda_grid : sequence of float, default=10**linspace(-4, 0, 21)
        Regularization path. Selection by validation Riesz loss when
        `validation_fraction > 0` or `eval_set` is given.
    solver : {"auto", "direct", "nystrom_cg", "rff", "falkon"}, default="auto"
        "auto" picks "direct" for n_aug ≤ 3000, "nystrom_cg" for ≤ 50k,
        else "falkon" if installed otherwise "nystrom_cg".
    loss : rieszreg.Loss, default=SquaredLoss()
        Currently only SquaredLoss is supported in the kernel backend.
    n_landmarks : int or None
        Nyström landmarks (for "nystrom_cg" / "falkon").
    n_features : int, default=1024
        Random Fourier features (for "rff").
    cg_tol : float, default=1e-6
    cg_max_iter : int, default=200
    init : float, "m1", or None
        α-space initialization. None defers to `loss.default_init_alpha()`.
    validation_fraction : float, default=0.2
        Hold out this fraction of the training data for λ selection.
    keep_path : bool, default=True
        Retain per-λ dual coefficients on the fitted estimator so
        `predict_path(X, lambdas=...)` can return α̂ at each retained λ in
        one call. Storage cost is `n_train × n_lambda × 8` bytes.
    random_state : int, default=0
    """

    def __init__(
        self,
        estimand: Estimand,
        kernel: Kernel | None = None,
        lambda_grid: Sequence[float] | None = None,
        solver: str = "auto",
        loss: Loss | None = None,
        n_landmarks: int | None = None,
        n_features: int = 1024,
        cg_tol: float = 1e-6,
        cg_max_iter: int = 200,
        init: float | str | None = None,
        validation_fraction: float = 0.2,
        keep_path: bool = True,
        random_state: int = 0,
    ):
        super().__init__(
            estimand=estimand,
            backend=None,            # built lazily in _resolved_backend
            loss=loss,
            init=init,
            random_state=random_state,
        )
        self.kernel = kernel
        self.lambda_grid = lambda_grid
        self.solver = solver
        self.n_landmarks = n_landmarks
        self.n_features = n_features
        self.cg_tol = cg_tol
        self.cg_max_iter = cg_max_iter
        self.validation_fraction = validation_fraction
        self.keep_path = keep_path

    # ---- defaults / backend construction ----

    def _resolved_kernel(self) -> Kernel:
        return self.kernel if self.kernel is not None else Gaussian()

    def _resolved_lambda_grid(self) -> tuple[float, ...]:
        if self.lambda_grid is None:
            return tuple(10.0 ** np.linspace(-4, 0, 21))
        return tuple(float(x) for x in self.lambda_grid)

    def _resolved_loss(self) -> Loss:
        return self.loss if self.loss is not None else SquaredLoss()

    def _resolved_backend(self) -> KernelRidgeBackend:
        return KernelRidgeBackend(
            kernel=self._resolved_kernel(),
            lambda_grid=self._resolved_lambda_grid(),
            solver=self.solver,
            n_landmarks=self.n_landmarks,
            n_features=self.n_features,
            cg_tol=self.cg_tol,
            cg_max_iter=self.cg_max_iter,
            validation_fraction=self.validation_fraction,
            keep_path=self.keep_path,
            random_state=self.random_state,
        )

    # ---- fit override exposes lambda_ ----

    def fit(self, Z, y=None, eval_set=None) -> "KernelRieszRegressor":
        super().fit(Z, y=y, eval_set=eval_set)
        # Convenience: surface the chosen λ on the regressor.
        if self.predictor_.result.extra is not None:
            self.lambda_ = self.predictor_.result.extra.get("lambda")
        return self

    def predict_path(
        self, Z, lambdas: Sequence[float] | None = None
    ) -> np.ndarray:
        """Predict α̂ at every λ in the (optionally subset) lambda_grid.

        Returns an array of shape ``(n_rows, n_lambdas)`` whose column ``j``
        is the prediction at ``lambdas[j]`` (or ``self.lambda_grid[j]`` if
        ``lambdas`` is ``None``). Each column is bit-equal to a fresh fit at
        a singleton lambda_grid containing that λ — same Cholesky / RHS / dual.

        Requires ``keep_path=True`` (the default). Raises ``RuntimeError`` if
        the estimator was fit with ``keep_path=False``.
        """
        if not hasattr(self, "predictor_"):
            raise RuntimeError(
                f"{type(self).__name__} is not fitted yet. Call .fit() first."
            )
        rows = _rows_from_Z(Z, self.estimand)
        feats = _features_from_rows(rows, self.estimand)
        return self.predictor_.predict_alpha_path(feats, lambdas)

    # ---- save/load: defer to base class via the registry ----

    def _save_hyperparameters(self) -> dict:
        base = super()._save_hyperparameters()
        base.update(
            validation_fraction=self.validation_fraction,
            keep_path=self.keep_path,
        )
        return base

    @classmethod
    def _construct_for_load(cls, *, estimand, loss, hyperparameters: dict) -> "KernelRieszRegressor":
        return cls(
            estimand=estimand,
            loss=loss,
            init=hyperparameters.get("init"),
            validation_fraction=hyperparameters.get("validation_fraction", 0.2),
            keep_path=hyperparameters.get("keep_path", True),
            random_state=hyperparameters.get("random_state", 0),
        )
