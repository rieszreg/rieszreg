# Testing plan

Today's suite: 51 Python tests + ~6 R parity tests. Mostly per-feature smoke tests + one Lee-Schuler reference cross-check. That covers "does it work end-to-end" but leaves a lot uncovered. This document catalogs the gaps and proposes layers to add, ordered by leverage.

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
| R parity | predict bitwise-identical across languages | `r/rieszboost/tests/testthat/test-parity.R` |
| Reference cross-check | Pearson 0.998 ATE / 0.986 ATT vs Lee-Schuler reference | `examples/lee_schuler/_compare_with_reference.py` (script, not in CI) |

## What's missing

Six gaps, ranked by impact:

### 1. Numerical regression baselines (high leverage, low cost)

Pin a small handful of α̂-RMSE numbers on a fixed DGP + fixed seed + fixed hyperparameters. Refactors that silently degrade quality become loud test failures instead of hidden regressions.

Concrete: a `tests/regression/baselines.json` file with rows like

```json
{
  "ate_lee_schuler_n4000_seed42": {"alpha_rmse": 0.789, "tol": 0.05},
  "att_lee_schuler_n4000_seed0": {"alpha_rmse": 0.310, "tol": 0.05},
  "lase_lee_schuler_continuous_seed0": {"alpha_rmse": 0.412, "tol": 0.10}
}
```

A single test file (`tests/regression/test_baselines.py`) loops over the baselines, refits, and asserts. Tolerance is generous on first pass; we tighten as we tune. **The point isn't to enforce SOTA — it's to detect drift.**

### 2. Backend equivalence (high leverage)

`XGBoostBackend` and `SklearnBackend(DecisionTreeRegressor(...))` should produce statistically equivalent α̂ when given the same data and matched hyperparameters. We have prediction-correlation checks vs the Lee-Schuler reference; we don't have one *between our two backends*. Add:

- `test_backend_equivalence.py`: fit ATE on the same DataFrame with both backends, assert Pearson correlation > 0.95 and RMSE-vs-each-other < 0.5·RMSE-vs-truth.
- Same for ATT, AdditiveShift, TSM. Skip for KLLoss (too tightly tied to xgboost's eta-space step).

### 3. Property-based tests (medium-high leverage)

Use `hypothesis` to generate random valid data and assert invariants:

- **Linearity in m**: tracing `c1*m1 + c2*m2` matches `c1*trace(m1) + c2*trace(m2)` term-wise. Catches bugs in the tracer arithmetic.
- **Augmentation invariance**: shuffling row order in input should produce a permuted but otherwise identical augmented dataset.
- **Reproducibility**: same `random_state`, same data, same predictions to floating-point identity.
- **Estimand round-trip**: `Estimand(feature_keys=keys, m=m)` followed by `build_augmented` produces rows whose features match `keys` exactly.
- **Score monotonicity (smoke)**: more boosting rounds → lower training Riesz loss (within reason).

### 4. sklearn conformance (medium leverage)

`sklearn.utils.estimator_checks.check_estimator(RieszBooster(estimand=ATE()))` runs a battery of API conformance tests sklearn ships. We currently pass `clone`, `get_params`, `cross_val_predict`, `GridSearchCV` by hand-rolled tests. `check_estimator` runs ~30 more. Some will fail (we don't accept `y` properly, the augmentation requires float features, etc.); ignore the irrelevant ones via `expected_failed_checks`. **Goal: opt out of the truly-N/A checks explicitly so any new failure is a real bug.**

### 5. Edge-case coverage (medium-low leverage, easy to write)

- Empty validation set with `early_stopping_rounds` → clean error, not a crash.
- All-treated / all-control input (degenerate ATE).
- Single-row data.
- `m()` returning `0` for every row → augmentation has no counterfactuals.
- DataFrame with extra unrelated columns → ignored, not failed.
- DataFrame with the wrong dtypes (string treatment column) → clear error message.
- `validation_fraction=0` + `early_stopping_rounds` → auto-split fallback works.
- `init="m1"` for an estimand whose `m(z, 1)` is non-trivially data-dependent.

### 6. Performance regression (low leverage today, useful when the codebase grows)

Track wall time on a reference workload — e.g., ATE fit on n=10000 rows, depth=4, 200 rounds — across commits. Not a unit test; a separate benchmark script that gets run periodically (or on PRs) and writes results to a CSV. We're not going to spend a day on this now, but the entry point should exist so it's cheap to extend.

## Testing infrastructure decisions

- **Framework**: pytest (already in use). Add `hypothesis` for property tests when (3) lands.
- **Reference data**: regression baselines + DGPs live in `tests/regression/`. Baselines are JSON, not pickle, so they're diffable on PRs.
- **CI**: not yet wired up. When we add it, it runs `pytest python/tests` + the R parity test + the cross-vs-reference script.
- **Slow tests**: the bigger n=4000 Lee-Schuler tests live behind `@pytest.mark.slow`; default `pytest` skips them, `pytest -m slow` runs them. We don't have this mark today; introduce when the suite passes 5 seconds.
- **Reproducibility**: every test that randomizes anything sets `numpy.random.default_rng(seed)`; no hidden global state. Already mostly true; audit when adding new tests.

## What I'd write first

If we had one afternoon to spend on this, the order is:

1. (1) Pin three numerical baselines and add the regression test loop.
2. (2) Backend-equivalence test on ATE.
3. (5) The "extra DataFrame columns are ignored" + "all-treated input" + "single-row data" edge cases — five tests, twenty minutes.

Then iterate over (3) hypothesis property tests when we hit a tracer bug. (4) and (6) are good-to-have but not load-bearing for v0.0.x.
