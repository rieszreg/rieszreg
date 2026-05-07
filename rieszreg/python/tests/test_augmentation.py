"""Tests for `Estimand.augment` + `AugmentedDataset`."""

from __future__ import annotations

import numpy as np
import pytest

from rieszreg import (
    ATE,
    AdditiveShift,
    AugmentedDataset,
    FiniteEvalEstimand,
)


def test_ate_augmentation_shape():
    features = np.array([[0.0, 1.0], [1.0, 2.0]])  # (a, x)
    aug = ATE().augment(features)
    assert aug.n_rows == 2
    # Each row gets a treated copy (a=1, x) and a control copy (a=0, x): 2n rows.
    assert aug.features.shape == (4, 2)
    assert aug.is_original.shape == (4,)
    assert aug.potential_deriv_coef.shape == (4,)
    assert aug.origin_index.shape == (4,)
    assert set(aug.origin_index.tolist()) == {0, 1}


def test_original_rows_have_D_one_C_one_under_additive_shift():
    features = np.array([[0.5, 1.5]])
    # AdditiveShift's counterfactual point (a+δ, x) is distinct from (a, x).
    aug = AdditiveShift(delta=0.7).augment(features)
    mask = (aug.features[:, 0] == 0.5) & (aug.features[:, 1] == 1.5)
    assert mask.sum() == 1
    # m(α)(z) = α(a+δ, x) − α(a, x); merging into the original gives C=+1.
    assert aug.is_original[mask].item() == 1.0
    assert aug.potential_deriv_coef[mask].item() == 1.0


def test_counterfactual_rows_have_D_zero():
    features = np.array([[0.5, 1.5]])
    aug = AdditiveShift(delta=0.7).augment(features)
    mask = np.isclose(aug.features[:, 0], 1.2) & (aug.features[:, 1] == 1.5)
    assert mask.sum() == 1
    assert aug.is_original[mask].item() == 0.0
    assert aug.potential_deriv_coef[mask].item() == -1.0


def test_empty_features():
    aug = ATE().augment(np.zeros((0, 2)))
    assert isinstance(aug, AugmentedDataset)
    assert aug.n_rows == 0
    assert aug.features.shape == (0, 2)


def test_origin_index_groups_rows():
    features = np.array([[0.0, 1.0], [1.0, 2.0], [0.0, 3.0]])
    aug = ATE().augment(features)
    counts = np.bincount(aug.origin_index)
    assert (counts > 0).all()
    assert counts.sum() == aug.features.shape[0]


def test_y_dependent_custom_estimand_is_plumbed_through():
    """`FiniteEvalEstimand.augment(features, ys)` passes y into m(α)(z, y)
    via the inherited Tracer-based default."""
    tau = 0.0

    def m(alpha):
        def inner(z, y):
            indicator = 1.0 if y > tau else 0.0
            return indicator * (alpha(a=1, x=z["x"]) - alpha(a=0, x=z["x"]))
        return inner

    estimand = FiniteEvalEstimand(feature_keys=("a", "x"), m=m, name="upper-half-ate")
    features = np.array([[0.0, 0.5], [1.0, 0.5]])

    # y > tau on row 0 → ATE-style augmentation (multiple rows).
    # y ≤ tau on row 1 → trace returns nothing; only the original row remains.
    ys = [1.0, -1.0]
    aug = estimand.augment(features, ys=ys)

    counts_per_row: dict[int, int] = {}
    for i in aug.origin_index.tolist():
        counts_per_row[i] = counts_per_row.get(i, 0) + 1
    assert counts_per_row[0] == 2
    assert counts_per_row[1] == 1


def test_y_length_mismatch_raises():
    features = np.array([[0.0, 0.5], [1.0, 0.5]])
    with pytest.raises(ValueError, match="does not match"):
        ATE().augment(features, ys=[1.0])
