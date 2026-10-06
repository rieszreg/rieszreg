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
from rieszreg.estimator import RieszEstimator
from rieszreg.losses import Loss

from .backend import DEFAULT_LAMBDA_GRID, KernelRidgeBackend
from .kernels import Gaussian, Kernel, kernel_from_spec


class KernelRieszRegressor(RieszEstimator):
    """Kernel ridge regression for the Riesz representer α₀ of a linear
    functional θ(P) = E[m(Z, g₀)].

    Reuses `rieszreg`'s estimand machinery and Bregman-loss framework;
    swaps in a kernel ridge backend for the actual fit.

    Parameters
    ----------
    estimand : rieszreg.Estimand
        What to estimate, e.g. ``ATE(treatment="treated")``. Also names the
        treatment and covariate columns (default: every non-treatment column).
    kernel : krrr.Kernel, default=Gaussian(length_scale="median")
        Reproducing kernel. Length-scale "median" resolves to the median
        pairwise Euclidean distance on the augmented training points.
    lambda_grid : sequence of float, default=10**linspace(-4, 0, 21)
        Regularization path. Selection by validation Riesz loss when
        `validation_fraction > 0` or `eval_set` is given; otherwise the
        largest λ is used.
    solver : {"auto", "direct", "nystrom_cg", "rff"}, default="auto"
        "auto" picks "direct" for n_aug ≤ 3000, otherwise "nystrom_cg".
    loss : rieszreg.Loss, default=SquaredLoss()
        Currently only SquaredLoss is supported in the kernel backend.
    n_landmarks : int or None
        Nyström landmarks (for "nystrom_cg").
    n_features : int, default=1024
        Random Fourier features (for "rff").
    cg_tol : float, default=1e-6
    cg_max_iter : int, default=200
    init : float or None
        α-space starting value. ``None`` (default) starts from the constant
        that minimizes the Riesz loss.
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
        init: float | None = None,
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

    def _resolved_backend(self) -> KernelRidgeBackend:
        return KernelRidgeBackend(
            kernel=self._resolved_kernel(),
            lambda_grid=DEFAULT_LAMBDA_GRID if self.lambda_grid is None else self.lambda_grid,
            solver=self.solver,
            n_landmarks=self.n_landmarks,
            n_features=self.n_features,
            cg_tol=self.cg_tol,
            cg_max_iter=self.cg_max_iter,
            validation_fraction=self.validation_fraction,
            keep_path=self.keep_path,
        )

    @property
    def lambda_(self) -> float:
        """The regularization strength λ selected from lambda_grid."""
        return self.predictor_.result.extra["lambda"]

    def diagnose(self, Z, y=None, **kwargs):
        """Base diagnostics plus kernel extras: selected λ, number of
        support points, effective degrees of freedom, condition number."""
        from .diagnostics import diagnose_kernel
        return diagnose_kernel(self, Z, y=y, **kwargs)

    def predict_path(
        self, Z, lambdas: Sequence[float] | None = None
    ) -> np.ndarray:
        """Predict α̂ at every λ in the (optionally subset) lambda_grid.

        Returns an array of shape ``(n_rows, n_lambdas)`` whose column ``j``
        is the prediction at ``lambdas[j]`` (or ``self.lambda_grid[j]`` if
        ``lambdas`` is ``None``). Each column matches, up to floating-point
        rounding, a fresh fit at a singleton lambda_grid containing that λ.

        Requires ``keep_path=True`` (the default). Raises ``RuntimeError`` if
        the estimator was fit with ``keep_path=False``.
        """
        feats = self._features(Z)
        return self.predictor_.predict_alpha_path(feats, lambdas)

    # ---- save/load: the base class handles every JSON-able param ----

    def _save_hyperparameters(self) -> dict:
        hp = super()._save_hyperparameters()
        if self.kernel is not None:
            hp["kernel"] = self.kernel.to_spec()
        return hp

    @classmethod
    def _construct_for_load(cls, *, estimand, loss, hyperparameters: dict) -> "KernelRieszRegressor":
        hp = dict(hyperparameters)
        if isinstance(hp.get("kernel"), dict):
            hp["kernel"] = kernel_from_spec(hp["kernel"])
        return super()._construct_for_load(estimand=estimand, loss=loss, hyperparameters=hp)
