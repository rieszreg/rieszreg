# Testing plan

Today's suite: 82 Python tests + ~11 R parity tests. The first three layers from the original plan (numerical regression baselines, backend equivalence, edge cases) have shipped; layers 4–6 (sklearn `check_estimator`, hypothesis property tests, performance regression tracking) are next.

## What we have today

| Layer | What's tested | File |
|---|---|---|
| Tracer algebra | `LinearForm` operations, linearity enforcement | `test_tracer.py` |
| ATE end-to-end (DataFrame, ndarray) + early stopping + score = -loss | new estimator API | `test_estimator_ate.py` |
| ATT / LocalShift partial-parameter recovery | per-estimand RMSE on synthetic DGP | `test_att.py`, `test_local_shift.py` |
| StochasticIntervention | trace + augmentation + end-to-end | `test_stochastic.py` |
| Bregman losses | SquaredLoss = default, KL refuses signed coefs, KL produces positive α̂ | `test_bregman.py` |
| `SklearnBackend` | DecisionTree, KernelRidge fit + score | `test_general.py` |
| Diagnostics | array path, booster path, warnings | `test_diagnostics.py` |
| sklearn integration | `clone`, `GridSearchCV`, `cross_val_predict`, `get/set_params` | `test_sklearn_integration.py` |
| Serialization | save/load round-trip across backends, losses, all built-in estimands; cross-language | `test_serialization.py`, `test-parity.R` |
| **Numerical regression baselines** | pinned α-RMSE for ATE / ATT-partial / LASE-partial | `tests/regression/test_baselines.py` + `baselines.json` |
| **Backend equivalence** | XGBoostBackend ≈ SklearnBackend on ATE/ATT/AdditiveShift | `test_backend_equivalence.py` |
| **Edge cases** | extra cols, single row, all-treated, m=0, eval_set override, etc. | `test_edge_cases.py` |
| R parity | predict bitwise-identical across languages, save/load round-trip | `r/rieszboost/tests/testthat/test-parity.R` |
| Reference cross-check | Pearson 0.998 ATE / 0.986 ATT vs Lee-Schuler reference | `examples/lee_schuler/_compare_with_reference.py` (script, not in CI) |

## What's missing

Three layers shipped (✓), three to go.

### 1. ✓ Numerical regression baselines

Pinned α̂-RMSE for three representative cases (ATE, ATT-partial, LASE-partial) on fixed DGPs + seeds + hyperparameters. Tolerances at 10% relative; tighten as the codebase stabilizes.

Lives in `python/tests/regression/baselines.json` + `test_baselines.py`. To regenerate after an intentional algorithmic change, call `compute_baseline(key)` for the affected entry and update the JSON.

### 2. ✓ Backend equivalence

`XGBoostBackend(gradient_only=True)` and `SklearnBackend(DecisionTreeRegressor)` produce strongly correlated predictions on the same data (Pearson > 0.85 on ATE / ATT / AdditiveShift). Lives in `test_backend_equivalence.py`. Skip for KL because exp-link semantics differ between the η-space xgboost step and the slow path's α-space line search.

### 3. ✓ Edge-case coverage

10 edge cases: extra DataFrame columns ignored, single-row data, all-treated / all-control input, LocalShift with no rows below threshold, `validation_fraction=0` with early stopping (auto-split fallback), explicit `eval_set` overriding internal split, StochasticIntervention with empty per-row sample lists, prediction on out-of-distribution X, ndarray with wrong feature count → clear error. Lives in `test_edge_cases.py`.

### 4. Property-based tests (medium-high leverage)

Use `hypothesis` to generate random valid data and assert invariants:

- **Linearity in m**: tracing `c1*m1 + c2*m2` matches `c1*trace(m1) + c2*trace(m2)` term-wise. Catches bugs in the tracer arithmetic.
- **Augmentation invariance**: shuffling row order in input should produce a permuted but otherwise identical augmented dataset.
- **Reproducibility**: same `random_state`, same data, same predictions to floating-point identity.
- **Estimand round-trip**: `Estimand(feature_keys=keys, m=m)` followed by `build_augmented` produces rows whose features match `keys` exactly.
- **Score monotonicity (smoke)**: more boosting rounds → lower training Riesz loss (within reason).

### 5. sklearn conformance (medium leverage)

`sklearn.utils.estimator_checks.check_estimator(RieszBooster(estimand=ATE()))` runs a battery of API conformance tests sklearn ships. We currently pass `clone`, `get_params`, `cross_val_predict`, `GridSearchCV` by hand-rolled tests. `check_estimator` runs ~30 more. Some will fail (we don't accept `y` properly, the augmentation requires float features, etc.); ignore the irrelevant ones via `expected_failed_checks`. **Goal: opt out of the truly-N/A checks explicitly so any new failure is a real bug.**

### 6. Performance regression (low leverage today, useful when the codebase grows)

Track wall time on a reference workload — e.g., ATE fit on n=10000 rows, depth=4, 200 rounds — across commits. Not a unit test; a separate benchmark script that gets run periodically (or on PRs) and writes results to a CSV. We're not going to spend a day on this now, but the entry point should exist so it's cheap to extend.

## Testing infrastructure decisions

- **Framework**: pytest (already in use). Add `hypothesis` for property tests when (3) lands.
- **Reference data**: regression baselines + DGPs live in `tests/regression/`. Baselines are JSON, not pickle, so they're diffable on PRs.
- **CI**: not yet wired up. When we add it, it runs `pytest python/tests` + the R parity test + the cross-vs-reference script.
- **Slow tests**: the bigger n=4000 Lee-Schuler tests live behind `@pytest.mark.slow`; default `pytest` skips them, `pytest -m slow` runs them. We don't have this mark today; introduce when the suite passes 5 seconds.
- **Reproducibility**: every test that randomizes anything sets `numpy.random.default_rng(seed)`; no hidden global state. Already mostly true; audit when adding new tests.

## What's next

Top three layers landed in commit. Remaining priorities:

1. **(4) hypothesis property tests** — start with linearity-in-m and reproducibility-with-seed. Catches bugs in the tracer arithmetic and any future stochastic backend changes.
2. **(5) sklearn `check_estimator`** — opt out of the truly-N/A checks explicitly so any new failure is a real bug. Probably ~half a day to wire up.
3. **(6) performance regression** — only worth doing once the codebase grows; for now the active suite runs in <15 seconds total.
