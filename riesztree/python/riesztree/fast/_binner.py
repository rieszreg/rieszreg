"""Quantile-based feature binner for ``splitter='hist'``.

Pre-binning happens once at fit start: each continuous feature is
quantized into ``max_bins`` quantile buckets, producing a
``uint8`` array of bin indices. Subsequent split-finding scans
``max_bins - 1`` candidate thresholds per feature instead of
``n_aug - 1`` distinct values.

Mirrors the spirit of
:class:`sklearn.ensemble._hist_gradient_boosting.binning._BinMapper`,
but reimplemented from scratch (sklearn's bin mapper is private).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BinMapper:
    """Per-feature quantile binner.

    Attributes
    ----------
    bin_thresholds : list[np.ndarray]
        ``bin_thresholds[j]`` is a sorted 1-D array of length
        ``n_bins[j] - 1`` giving the right-inclusive thresholds
        between bins for feature ``j``. A value falls into bin ``b``
        iff ``thresholds[b-1] < value <= thresholds[b]`` (with
        ``thresholds[-1] = -inf`` and ``thresholds[n_bins[j]-1] = +inf``).
    n_bins : np.ndarray[int32]
        Per-feature bin counts; bounded by ``max_bins`` and the number
        of distinct values in that feature.
    max_bins : int
        Configured cap (e.g. 255). Each feature uses ``min(max_bins,
        n_distinct_values)`` actual bins.
    subsample : int
        Number of rows used to compute quantiles.
    """

    bin_thresholds: list[np.ndarray]
    n_bins: np.ndarray
    max_bins: int
    subsample: int


def fit_bin_mapper(
    X: np.ndarray,
    max_bins: int = 255,
    subsample: int = 200_000,
    random_state: int = 0,
) -> BinMapper:
    """Fit per-feature quantile bin thresholds on a (subsample of) ``X``.

    Parameters
    ----------
    X
        Float feature matrix of shape ``(n, p)``. Read-only.
    max_bins
        Maximum bin count per feature. Bin index 0 is reserved for the
        smallest values, ``max_bins - 1`` for the largest. Default 255
        (the sklearn HGB convention; fits in ``uint8``).
    subsample
        Cap on rows sampled for the quantile computation. Default
        200 000. Larger ``X`` is uniformly subsampled.
    random_state
        Seed for the subsample.

    Returns
    -------
    BinMapper
        See :class:`BinMapper`.
    """
    if max_bins < 2:
        raise ValueError(f"max_bins={max_bins!r}; must be >= 2.")
    if max_bins > 255:
        raise ValueError(
            f"max_bins={max_bins!r}; must be <= 255 to fit in uint8."
        )

    n, p = X.shape
    rng = np.random.default_rng(random_state)
    if n > subsample:
        idx = rng.choice(n, size=subsample, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X

    bin_thresholds: list[np.ndarray] = []
    n_bins = np.empty(p, dtype=np.int32)
    quantiles = np.linspace(0.0, 1.0, max_bins + 1, dtype=np.float64)[1:-1]
    # Internal quantiles only; the outer endpoints are implicit (-inf / +inf).

    for j in range(p):
        col = X_sub[:, j]
        # Distinct value count caps the actual bins used.
        uniques = np.unique(col)
        if uniques.size <= max_bins:
            # Use midpoints between consecutive distinct values as
            # thresholds — gives exact bins for low-cardinality features.
            if uniques.size <= 1:
                thresholds = np.empty(0, dtype=np.float64)
                n_bins[j] = 1
            else:
                thresholds = (uniques[:-1] + uniques[1:]) * 0.5
                n_bins[j] = uniques.size
        else:
            thresholds = np.quantile(col, quantiles, method="linear")
            # Drop duplicate thresholds (low-cardinality feature with
            # heavy ties): sklearn's `_BinMapper` uses unique() too.
            thresholds = np.unique(thresholds)
            n_bins[j] = thresholds.size + 1
        bin_thresholds.append(np.asarray(thresholds, dtype=np.float64))

    return BinMapper(
        bin_thresholds=bin_thresholds,
        n_bins=n_bins,
        max_bins=max_bins,
        subsample=subsample,
    )


def transform(X: np.ndarray, mapper: BinMapper) -> np.ndarray:
    """Bin ``X`` using a fitted :class:`BinMapper`.

    Returns a contiguous ``uint8`` array of shape ``(n, p)``. Values
    above the largest threshold land in the rightmost bin; below the
    smallest, in bin 0. ``np.searchsorted`` with side='right' realises
    the "right-inclusive" boundary used at fit time (see
    :func:`fit_bin_mapper`).
    """
    n, p = X.shape
    out = np.empty((n, p), dtype=np.uint8)
    for j in range(p):
        thr = mapper.bin_thresholds[j]
        if thr.size == 0:
            out[:, j] = 0
        else:
            # searchsorted(thr, x, side='right') returns the bin index
            # in {0, ..., len(thr)}.
            out[:, j] = np.searchsorted(thr, X[:, j], side="right").astype(np.uint8)
    return np.ascontiguousarray(out)
