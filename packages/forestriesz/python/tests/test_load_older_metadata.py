"""Models saved before ``n_features_in`` / ``feature_names_in`` were written
to metadata.json still load."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from forestriesz import TSM, ForestRieszRegressor


def test_load_without_sklearn_feature_metadata(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.binomial(1, 0.5, 300), "x": rng.normal(size=300)})
    est = ForestRieszRegressor(estimand=TSM(level=1), n_estimators=8, random_state=0).fit(df)
    est.save(tmp_path / "m")

    meta_path = tmp_path / "m" / "metadata.json"
    meta = json.loads(meta_path.read_text())
    del meta["n_features_in"], meta["feature_names_in"]
    meta_path.write_text(json.dumps(meta))

    loaded = ForestRieszRegressor.load(tmp_path / "m")
    np.testing.assert_allclose(loaded.predict(df), est.predict(df))
    assert loaded.n_features_in_ == 2
