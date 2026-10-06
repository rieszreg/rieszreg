"""xgboost hands the objective and the early-stopping metric float32 margins.
A saturated sigmoid rounds to exactly α = 1 in float32, so both cast first."""

from __future__ import annotations

import numpy as np

from rieszboost.backends.xgboost import _make_metric
from rieszreg import TSM
from rieszreg.losses import BernoulliLoss


def test_metric_reads_float32_margins_in_float64():
    loss = BernoulliLoss()
    aug = TSM(treatment="a", covariates=["x"], level=1).augment(
        np.array([[1.0, 0.3], [0.0, -0.2], [1.0, 1.1]])
    )
    margins = np.full(len(aug.features), 20.0, dtype=np.float32)
    _, value = _make_metric(aug, loss)(margins, None)
    expected = aug.mean_loss(loss, loss.link_to_alpha(margins.astype(np.float64)))
    assert value == expected
