"""Internal BaseGRF subclass that consumes prepacked moment data.

The backend computes the per-row moment vector A = m(W_i; phi) and Jacobian
J = phi(W_i) phi(W_i)' before fit, packs them into the EconML T array, and
delegates split search and leaf solve to BaseGRF's MSE criterion.

EconML's ``LinearMomentGRFCriterion`` requires ``y`` to be scalar, so we use
``T`` to carry both the per-row J flat and the per-row A vector and pass a
dummy zero column for ``y``.

Packing convention used by ``forestriesz.backend.ForestRieszBackend``:

    X       : (n, n_split)      split features
    T       : (n, p² + p)        first p² cols = J flattened (row-major),
                                 next p cols   = A = m(W_i; φ)
    y       : (n, 1)             dummy zeros (unused; required for scalar-y criterion)
"""

from __future__ import annotations

import numpy as np
from econml.grf._base_grf import BaseGRF


class _RieszGRF(BaseGRF):
    """BaseGRF specialized for the Riesz linear-moment equation.

    The basis dimension ``p`` is fixed at construction so the abstract
    methods can recover it without a side-channel. ``l2`` is added to each
    row's Jacobian, so a leaf solves ``(mean J + l2 · I) θ = mean A``, i.e.
    ``(Σ J_i + n_leaf · l2 · I) θ = Σ A_i``.
    """

    def __init__(self, *, n_outputs_riesz: int, l2: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self._n_outputs_riesz = int(n_outputs_riesz)
        self._l2 = float(l2)

    def _get_alpha_and_pointJ(self, X, T, y, **kwargs):
        p = self._n_outputs_riesz
        pointJ = T[:, : p * p] + self._l2 * np.eye(p).ravel()
        alpha = np.ascontiguousarray(T[:, p * p : p * p + p])
        return alpha, np.ascontiguousarray(pointJ)

    def _get_n_outputs_decomposition(self, X, T, y, **kwargs):
        p = self._n_outputs_riesz
        return p, p
