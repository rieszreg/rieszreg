"""The augmented-row packaging used by Bregman-Riesz losses.

Each augmented row r contributes a per-row loss term

    D_r · h_tilde(α(z_r)) + C_r · h'(α(z_r))

(the squared-loss case h_tilde = t², h' = 2t simplifies to D·α² + 2·C·α).
The original observation Z_i seeds row i with (is_original=1,
potential_deriv_coef=0); each (coef, point) pair from m(z_i) contributes
(is_original=0, potential_deriv_coef=-coef) at the point. Duplicate
points within a row are merged by summing the two coefficients.

The row-emission logic lives on :class:`FiniteEvalEstimand` and its
subclasses (``ATE``, ``ATT``, ``TSM``, ``AdditiveShift``, ``LocalShift``);
this module just packages the result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AugmentedDataset:
    features: np.ndarray              # (n_aug, n_features)
    is_original: np.ndarray           # (n_aug,) — 1 if z_r == Z_{i_r}, else 0 (D_r)
    potential_deriv_coef: np.ndarray  # (n_aug,) — coefficient on h'(α) (C_r)
    origin_index: np.ndarray          # (n_aug,) — index into original rows
    n_rows: int                       # number of original rows
