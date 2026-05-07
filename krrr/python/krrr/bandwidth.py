"""Length-scale resolution for shift-invariant kernels.

The median heuristic (median pairwise Euclidean distance) is a well-known
default that adapts to the scale of the data without requiring CV. Scott's
and Silverman's rules-of-thumb are also exposed for completeness.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist


def median_heuristic(X: np.ndarray, max_pairs: int = 5000, rng: np.random.Generator | None = None) -> float:
    """Median of pairwise Euclidean distances on (a subsample of) `X`.

    For n > sqrt(2·max_pairs), uses a random subsample to avoid the
    O(n²) cost. Falls back to 1.0 if all distances are zero.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n = X.shape[0]
    if n < 2:
        return 1.0
    n_sample_cap = int(np.sqrt(2 * max_pairs)) + 1
    if n > n_sample_cap:
        rng = rng or np.random.default_rng(0)
        idx = rng.choice(n, size=n_sample_cap, replace=False)
        Xs = X[idx]
    else:
        Xs = X
    d = pdist(Xs, "euclidean")
    if d.size == 0:
        return 1.0
    m = float(np.median(d))
    return m if m > 0 else 1.0


def scott_rule(X: np.ndarray) -> float:
    """Scott (1992): h = n^(-1/(d+4)) · σ̄, where σ̄ is the average
    coordinate standard deviation."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n, d = X.shape
    sigma_bar = float(np.mean(np.std(X, axis=0, ddof=1))) if n > 1 else 1.0
    if sigma_bar <= 0:
        sigma_bar = 1.0
    return n ** (-1.0 / (d + 4)) * sigma_bar


def silverman_rule(X: np.ndarray) -> float:
    """Silverman (1986): h = (4 / (d + 2))^(1/(d+4)) · n^(-1/(d+4)) · σ̄."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n, d = X.shape
    sigma_bar = float(np.mean(np.std(X, axis=0, ddof=1))) if n > 1 else 1.0
    if sigma_bar <= 0:
        sigma_bar = 1.0
    return (4.0 / (d + 2)) ** (1.0 / (d + 4)) * n ** (-1.0 / (d + 4)) * sigma_bar


def resolve_length_scale(spec, X: np.ndarray) -> float:
    """Turn a length_scale spec (float or string keyword) into a float
    using `X` as the reference data."""
    if isinstance(spec, (int, float)):
        return float(spec)
    if spec == "median":
        return median_heuristic(X)
    if spec == "scott":
        return scott_rule(X)
    if spec == "silverman":
        return silverman_rule(X)
    raise ValueError(
        f"Unknown length_scale spec: {spec!r}. Pass a number or one of "
        "'median', 'scott', 'silverman'."
    )
