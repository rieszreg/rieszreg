"""save/load round-trip.

Locally constant: full round-trip works without user intervention.
Locally linear with sieve: save works, load needs the user to repass
``riesz_feature_fns=`` because callables don't pickle reliably.
"""

from __future__ import annotations

import numpy as np

from forestriesz import ATE, ForestRieszRegressor, TSM
from forestriesz.feature_fns import default_riesz_features


def test_round_trip_locally_constant(tmp_path, logistic_tsm_df):
    est = ForestRieszRegressor(
        estimand=TSM(level=1),
        n_estimators=30,
        min_samples_leaf=15,
        random_state=0,
    )
    est.fit(logistic_tsm_df)
    pred_before = est.predict(logistic_tsm_df)

    save_dir = tmp_path / "fitted"
    est.save(save_dir)

    loaded = ForestRieszRegressor.load(save_dir)
    pred_after = loaded.predict(logistic_tsm_df)
    np.testing.assert_allclose(pred_before, pred_after)


def test_round_trip_sieve_requires_repassing_callables(
    tmp_path, linear_gaussian_ate_df
):
    fns = default_riesz_features(ATE())
    est = ForestRieszRegressor(
        estimand=ATE(),
        riesz_feature_fns=fns,
        n_estimators=20,
        min_samples_leaf=15,
        random_state=0,
    )
    est.fit(linear_gaussian_ate_df)
    pred_before = est.predict(linear_gaussian_ate_df)

    save_dir = tmp_path / "fitted"
    est.save(save_dir)

    loaded = ForestRieszRegressor.load(
        save_dir, riesz_feature_fns=default_riesz_features(ATE())
    )
    pred_after = loaded.predict(linear_gaussian_ate_df)
    np.testing.assert_allclose(pred_before, pred_after)


def test_metadata_round_trips(tmp_path, logistic_tsm_df):
    import json

    est = ForestRieszRegressor(
        estimand=TSM(level=1),
        n_estimators=15,
        max_depth=4,
        random_state=0,
    )
    est.fit(logistic_tsm_df)

    save_dir = tmp_path / "fitted"
    est.save(save_dir)

    with open(save_dir / "metadata.json") as f:
        meta = json.load(f)
    assert meta["predictor_kind"] == "forestriesz"
    assert meta["estimator_class"] == "ForestRieszRegressor"
    hp = meta["hyperparameters"]
    assert hp["n_estimators"] == 15
    assert hp["max_depth"] == 4
