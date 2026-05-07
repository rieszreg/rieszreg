"""Kernel correctness: PSD-ness, diagonals, algebra, spec round-trip."""

from __future__ import annotations

import numpy as np
import pytest

from krrr.kernels import (
    Gaussian,
    Kernel,
    Linear,
    Matern,
    Polynomial,
    Sum,
    Tensor,
    kernel_from_spec,
)


@pytest.fixture
def X():
    rng = np.random.default_rng(0)
    return rng.normal(size=(20, 3))


@pytest.fixture
def Y():
    rng = np.random.default_rng(1)
    return rng.normal(size=(15, 3))


@pytest.mark.parametrize("kernel_factory", [
    lambda: Gaussian(length_scale=1.0),
    lambda: Gaussian(length_scale="median"),
    lambda: Matern(nu=0.5, length_scale=1.0),
    lambda: Matern(nu=1.5, length_scale=1.0),
    lambda: Matern(nu=2.5, length_scale=1.0),
    lambda: Linear(),
    lambda: Polynomial(degree=2, gamma=0.5, coef0=1.0),
])
def test_gram_psd(kernel_factory, X):
    k = kernel_factory().fit_data(X)
    G = k(X, X)
    assert G.shape == (X.shape[0], X.shape[0])
    # Symmetry
    np.testing.assert_allclose(G, G.T, atol=1e-10)
    # PSD up to numerical jitter
    eigs = np.linalg.eigvalsh(G + 1e-10 * np.eye(X.shape[0]))
    assert eigs.min() > -1e-7


def test_diag_consistent_with_full_gram(X):
    k = Gaussian(length_scale=1.0)
    G = k(X)
    d = k.diag(X)
    np.testing.assert_allclose(d, np.diag(G))


def test_kernel_algebra(X):
    k1 = Gaussian(length_scale=1.0)
    k2 = Linear()
    k_sum = (k1 + k2)
    k_prod = (2.0 * k1)
    G_sum = k_sum(X, X)
    G_prod = k_prod(X, X)
    np.testing.assert_allclose(G_sum, k1(X, X) + k2(X, X))
    np.testing.assert_allclose(G_prod, 2.0 * k1(X, X))


def test_tensor_kernel(X):
    # cols (0, 1) | (2,)
    t = Tensor(Gaussian(length_scale=1.0), [0, 1], Linear(), [2])
    G = t(X, X)
    expected = Gaussian(length_scale=1.0)(X[:, [0, 1]], X[:, [0, 1]]) * Linear()(X[:, [2]], X[:, [2]])
    np.testing.assert_allclose(G, expected)


def test_spec_roundtrip(X):
    k = (Gaussian(length_scale="median").fit_data(X) + Linear()) * 0.5
    spec = k.to_spec()
    k2 = kernel_from_spec(spec)
    # k2 needs no further fit_data: median is baked into the spec.
    np.testing.assert_allclose(k(X, X), k2(X, X), atol=1e-10)


def test_random_features_approx(X):
    k = Gaussian(length_scale=1.0)
    rng = np.random.default_rng(0)
    Phi = k.random_features(X, n_features=4096, rng=rng)
    K_approx = Phi @ Phi.T
    K_true = k(X, X)
    err = np.abs(K_approx - K_true).max()
    # 4096 features should give ~0.05 max error on n=20.
    assert err < 0.1
