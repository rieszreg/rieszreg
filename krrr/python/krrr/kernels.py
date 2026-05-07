"""Kernel functions and a small algebra over them.

Each `Kernel` knows how to produce a Gram matrix `K(X, Y)` and (for
shift-invariant kernels) sample random Fourier features. Kernels compose
via `+` (sum), `*` (product), and `Tensor(k1, k2)` (tensor product over
disjoint feature subsets).

Bandwidth/length-scale resolution is delegated to `bandwidth.py`: a kernel
can be constructed with a numeric value or with the string ``"median"``,
in which case it resolves itself against the training data on first use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.spatial.distance import cdist

from .bandwidth import resolve_length_scale


def _as2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


class Kernel:
    """Base class. Subclasses implement `_gram(X, Y)` returning a (nX, nY)
    matrix and (optionally) `random_features(n_features, rng)` for RFF."""

    def __call__(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        X = _as2d(X)
        Y = X if Y is None else _as2d(Y)
        return self._gram(X, Y)

    def diag(self, X: np.ndarray) -> np.ndarray:
        """Diagonal `K(x_i, x_i)`. Default: extract from full Gram. Override
        for kernels with cheap diagonals (Gaussian: all 1.0)."""
        X = _as2d(X)
        return np.diag(self._gram(X, X))

    def fit_data(self, X: np.ndarray) -> "Kernel":
        """Resolve any data-dependent hyperparameters (e.g. median heuristic)
        against `X`. Returns self for chaining; subclasses mutate in place."""
        return self

    def _gram(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __add__(self, other: "Kernel") -> "Kernel":
        return Sum(self, other)

    def __mul__(self, other) -> "Kernel":
        if isinstance(other, (int, float)):
            return Scaled(float(other), self)
        if isinstance(other, Kernel):
            return Product(self, other)
        return NotImplemented

    def __rmul__(self, other) -> "Kernel":
        return self.__mul__(other)

    def to_spec(self) -> dict:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Shift-invariant kernels (RBF / Matern). All admit RFF.
# ---------------------------------------------------------------------------


@dataclass
class Gaussian(Kernel):
    """Gaussian / RBF kernel: k(x, y) = exp(-||x - y||² / (2 σ²)).

    `length_scale` may be a positive float, the string ``"median"`` (median
    of pairwise Euclidean distances on the training points; resolved by
    `fit_data`), or one of ``"scott"`` / ``"silverman"`` (Scott's / Silverman's
    rule on the training data, requires fit_data).
    """

    length_scale: float | str = "median"
    _resolved: float = field(init=False, default=float("nan"))

    def fit_data(self, X: np.ndarray) -> "Gaussian":
        self._resolved = resolve_length_scale(self.length_scale, X)
        return self

    def _ls(self) -> float:
        if np.isfinite(self._resolved):
            return self._resolved
        if isinstance(self.length_scale, (int, float)):
            return float(self.length_scale)
        raise RuntimeError(
            f"Gaussian.length_scale={self.length_scale!r} is data-dependent; "
            "call .fit_data(X) before evaluating the kernel."
        )

    def _gram(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        ls = self._ls()
        d2 = cdist(X, Y, "sqeuclidean")
        return np.exp(-d2 / (2.0 * ls * ls))

    def diag(self, X: np.ndarray) -> np.ndarray:
        return np.ones(_as2d(X).shape[0])

    def random_features(
        self, X: np.ndarray, n_features: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Rahimi-Recht random Fourier features for the Gaussian kernel.

        Returns a `(n, n_features)` real-valued matrix Φ so that
        Φ @ Φ.T ≈ K(X, X) (in expectation, up to small bias).
        """
        X = _as2d(X)
        d = X.shape[1]
        ls = self._ls()
        # spectral density of RBF is N(0, 1/ls²) per coordinate
        W = rng.normal(0.0, 1.0 / ls, size=(d, n_features))
        b = rng.uniform(0.0, 2.0 * np.pi, size=n_features)
        Z = np.sqrt(2.0 / n_features) * np.cos(X @ W + b)
        return Z

    def to_spec(self) -> dict:
        # Persist the resolved length scale (post-fit) so loaded kernels
        # don't need to re-resolve the median heuristic without their data.
        ls = self._resolved if np.isfinite(self._resolved) else self.length_scale
        return {"type": "Gaussian", "args": {"length_scale": ls}}


@dataclass
class Matern(Kernel):
    """Matern kernel of half-integer smoothness ν ∈ {1/2, 3/2, 5/2}.

    ν=1/2 is the Laplace kernel (least smooth); ν=5/2 is twice differentiable.
    For ν → ∞ this approaches the Gaussian kernel. Same `length_scale` semantics.
    """

    nu: float = 2.5
    length_scale: float | str = "median"
    _resolved: float = field(init=False, default=float("nan"))

    def fit_data(self, X: np.ndarray) -> "Matern":
        if self.nu not in (0.5, 1.5, 2.5):
            raise ValueError(f"Matern.nu must be 0.5, 1.5, or 2.5; got {self.nu!r}.")
        self._resolved = resolve_length_scale(self.length_scale, X)
        return self

    def _ls(self) -> float:
        if np.isfinite(self._resolved):
            return self._resolved
        if isinstance(self.length_scale, (int, float)):
            return float(self.length_scale)
        raise RuntimeError("Matern.length_scale is data-dependent; call fit_data first.")

    def _gram(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        ls = self._ls()
        d = cdist(X, Y, "euclidean") / ls
        if self.nu == 0.5:
            return np.exp(-d)
        if self.nu == 1.5:
            r = np.sqrt(3.0) * d
            return (1.0 + r) * np.exp(-r)
        if self.nu == 2.5:
            r = np.sqrt(5.0) * d
            return (1.0 + r + r * r / 3.0) * np.exp(-r)
        raise ValueError(f"Unsupported nu: {self.nu!r}")

    def diag(self, X: np.ndarray) -> np.ndarray:
        return np.ones(_as2d(X).shape[0])

    def to_spec(self) -> dict:
        ls = self._resolved if np.isfinite(self._resolved) else self.length_scale
        return {"type": "Matern", "args": {"nu": self.nu, "length_scale": ls}}


@dataclass
class Linear(Kernel):
    """Linear kernel: k(x, y) = c + x · y."""

    bias: float = 0.0

    def _gram(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return self.bias + X @ Y.T

    def diag(self, X: np.ndarray) -> np.ndarray:
        X = _as2d(X)
        return self.bias + np.einsum("ij,ij->i", X, X)

    def to_spec(self) -> dict:
        return {"type": "Linear", "args": {"bias": self.bias}}


@dataclass
class Polynomial(Kernel):
    """Polynomial kernel: k(x, y) = (γ · x · y + c)^d."""

    degree: int = 3
    gamma: float = 1.0
    coef0: float = 1.0

    def _gram(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return (self.gamma * (X @ Y.T) + self.coef0) ** self.degree

    def to_spec(self) -> dict:
        return {
            "type": "Polynomial",
            "args": {"degree": self.degree, "gamma": self.gamma, "coef0": self.coef0},
        }


# ---------------------------------------------------------------------------
# Algebra: scaled / sum / product / tensor product
# ---------------------------------------------------------------------------


@dataclass
class Scaled(Kernel):
    """c · k(x, y)."""

    scale: float
    base: Kernel

    def fit_data(self, X: np.ndarray) -> "Scaled":
        self.base.fit_data(X)
        return self

    def _gram(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return self.scale * self.base._gram(X, Y)

    def diag(self, X: np.ndarray) -> np.ndarray:
        return self.scale * self.base.diag(X)

    def to_spec(self) -> dict:
        return {"type": "Scaled", "args": {"scale": self.scale, "base": self.base.to_spec()}}


@dataclass
class Sum(Kernel):
    """k₁(x, y) + k₂(x, y)."""

    a: Kernel
    b: Kernel

    def fit_data(self, X: np.ndarray) -> "Sum":
        self.a.fit_data(X)
        self.b.fit_data(X)
        return self

    def _gram(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return self.a._gram(X, Y) + self.b._gram(X, Y)

    def diag(self, X: np.ndarray) -> np.ndarray:
        return self.a.diag(X) + self.b.diag(X)

    def to_spec(self) -> dict:
        return {"type": "Sum", "args": {"a": self.a.to_spec(), "b": self.b.to_spec()}}


@dataclass
class Product(Kernel):
    """k₁(x, y) · k₂(x, y) (Hadamard product on Gram matrices)."""

    a: Kernel
    b: Kernel

    def fit_data(self, X: np.ndarray) -> "Product":
        self.a.fit_data(X)
        self.b.fit_data(X)
        return self

    def _gram(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return self.a._gram(X, Y) * self.b._gram(X, Y)

    def diag(self, X: np.ndarray) -> np.ndarray:
        return self.a.diag(X) * self.b.diag(X)

    def to_spec(self) -> dict:
        return {"type": "Product", "args": {"a": self.a.to_spec(), "b": self.b.to_spec()}}


@dataclass
class Tensor(Kernel):
    """Tensor-product kernel over disjoint feature subsets:

        k((u, v), (u', v')) = k_a(u, u') · k_b(v, v')

    where `cols_a` / `cols_b` are integer index lists picking columns from
    the joint feature matrix. Useful for `(treatment, covariates)` splits
    where the treatment uses a different kernel from the covariates.
    """

    a: Kernel
    cols_a: Sequence[int]
    b: Kernel
    cols_b: Sequence[int]

    def fit_data(self, X: np.ndarray) -> "Tensor":
        X = _as2d(X)
        self.a.fit_data(X[:, self.cols_a])
        self.b.fit_data(X[:, self.cols_b])
        return self

    def _gram(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        Ka = self.a._gram(X[:, self.cols_a], Y[:, self.cols_a])
        Kb = self.b._gram(X[:, self.cols_b], Y[:, self.cols_b])
        return Ka * Kb

    def diag(self, X: np.ndarray) -> np.ndarray:
        return self.a.diag(X[:, self.cols_a]) * self.b.diag(X[:, self.cols_b])

    def to_spec(self) -> dict:
        return {
            "type": "Tensor",
            "args": {
                "a": self.a.to_spec(),
                "cols_a": list(self.cols_a),
                "b": self.b.to_spec(),
                "cols_b": list(self.cols_b),
            },
        }


# ---------------------------------------------------------------------------
# Spec round-trip
# ---------------------------------------------------------------------------


_REGISTRY = {
    "Gaussian": Gaussian,
    "Matern": Matern,
    "Linear": Linear,
    "Polynomial": Polynomial,
    "Scaled": Scaled,
    "Sum": Sum,
    "Product": Product,
    "Tensor": Tensor,
}


def kernel_from_spec(spec: dict) -> Kernel:
    """Reconstruct a Kernel from its `to_spec()` dict (recursive)."""
    cls = _REGISTRY[spec["type"]]
    args = dict(spec.get("args", {}))
    for child_key in ("base", "a", "b"):
        if child_key in args and isinstance(args[child_key], dict):
            args[child_key] = kernel_from_spec(args[child_key])
    return cls(**args)
