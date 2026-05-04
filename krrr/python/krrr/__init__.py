"""krrr: kernel ridge Riesz regression.

Sister package to ``rieszboost`` — same scope (Riesz representers for linear
functionals; auto-DML / Lee-Schuler / Singh objective), same user-facing API,
but the backend is kernel ridge regression instead of gradient boosting.

Reuses ``rieszreg``'s ``Estimand``, ``Tracer``, ``AugmentedDataset``,
``LossSpec``, ``Diagnostics``, and sklearn glue. The kernel solve plugs in
through the existing ``Backend`` protocol.

    from rieszreg import ATE
    from krrr import KernelRieszRegressor, Gaussian

    krr = KernelRieszRegressor(estimand=ATE(treatment="a", covariates=("x",)))
    krr.fit(df)
    alpha_hat = krr.predict(df)
    print(krr.diagnose(df).summary())
"""

from .backend import KernelRidgeBackend
from .diagnostics import KernelDiagnostics, diagnose_kernel
from .estimator import KernelRieszRegressor
from .kernels import (
    Gaussian,
    Kernel,
    Linear,
    Matern,
    Polynomial,
    Product,
    Scaled,
    Sum,
    Tensor,
    kernel_from_spec,
)
from .predictor import KernelPredictor
from .solvers import SolveResult, auto_choose, get_solver

# Re-export common rieszreg symbols so users can write `krrr.ATE`, etc.,
# without needing a second import line.
from rieszreg import (
    ATE,
    ATT,
    AdditiveShift,
    Diagnostics,
    Estimand,
    FiniteEvalEstimand,
    LocalShift,
    LossSpec,
    SquaredLoss,
    StochasticIntervention,
    TSM,
)

__all__ = [
    # Estimator
    "KernelRieszRegressor",
    "KernelRidgeBackend",
    "KernelPredictor",
    # Kernels
    "Kernel",
    "Gaussian",
    "Matern",
    "Linear",
    "Polynomial",
    "Tensor",
    "Sum",
    "Product",
    "Scaled",
    "kernel_from_spec",
    # Solvers
    "SolveResult",
    "get_solver",
    "auto_choose",
    # Diagnostics
    "KernelDiagnostics",
    "diagnose_kernel",
    # Re-exports from rieszreg
    "ATE",
    "ATT",
    "AdditiveShift",
    "Diagnostics",
    "Estimand",
    "FiniteEvalEstimand",
    "LocalShift",
    "LossSpec",
    "SquaredLoss",
    "StochasticIntervention",
    "TSM",
]
