---
name: rieszreg-backend-contract
description: Checklist for adding a new learner package to the rieszreg monorepo. Use when creating a new directory under `packages/<pkg>/`, implementing `Backend.fit_augmented` or `MomentBackend.fit_rows`, registering a predictor loader, writing a convenience subclass of `RieszEstimator`, or any task framed as "wrap my learner" / "add a new backend" / "new package for the family". This skill is the triggered short form of `DESIGN.md` Part B; cite back to it for the full long-form contract.
---

# Adding a learner to the rieszreg monorepo

A new learner means a new workspace member at `packages/<pkg>/`. Six items must be in place for it to fit the family.

## 1. Pick the entry point

Two Protocols, defined in `packages/rieszreg/python/rieszreg/backends/base.py`:

- **`Backend.fit_augmented`** — augmentation-style. The orchestrator pre-computes the augmented `(a, b)` dataset; you receive it and fit. References: `XGBoostBackend` in rieszboost, `KernelRidgeBackend` in krrr, `RieszTreeBackend` in riesztree.
- **`MomentBackend.fit_rows`** — moment-style. You receive raw rows + the estimand and call `rieszreg.trace(estimand, row)` per row to compute moments. References: `ForestRieszBackend` in forestriesz, `TorchBackend` in riesznet.

Pick by the natural decomposition of your learner's loss. Boosted trees, kernel ridge, single-tree → augmented. Forests, neural nets → moments.

## 2. Return shape

Both Protocols return `FitResult(predictor, best_iteration, best_score, history)`. The shape is identical for both entry points; the orchestrator depends on it being uniform.

## 3. Predictor + loader registration

Implement a `<Pkg>Predictor` class (the thing that gets returned in `FitResult.predictor`) with `predict(Z)` and `score(Z, alpha_target)` methods. Then register a loader so `RieszEstimator.load(...)` can deserialize a previously-saved fit:

```python
# in packages/<pkg>/python/<pkg>/__init__.py or similar
from rieszreg.backends import register_predictor_loader
register_predictor_loader("<pkg>-kind", <Pkg>Predictor.load)
```

The loader-kind string must be unique across the family.

## 4. Convenience subclass

Provide a `<Pkg><Estimator>` class subclassing `rieszreg.RieszEstimator` with the learner-specific Tier-3 hyperparameters on its constructor:

```python
class RieszBooster(RieszEstimator):
    def __init__(self, *, n_estimators=100, learning_rate=0.1, max_depth=4, ...):
        backend = XGBoostBackend(n_estimators=n_estimators, ...)
        super().__init__(backend=backend, ...)
```

Tier-3 knobs (learner-specific) live on this constructor only — never on `RieszEstimator` itself or on Protocol methods. See [`rieszreg-coding`](../rieszreg-coding/SKILL.md) §3.

## 5. R wrapper

Subclass `rieszreg::RieszEstimatorR6` for the R API:

```r
# packages/<pkg>/r/<pkg>/R/<pkg>.R
<Pkg>Regressor <- R6::R6Class(
  "<Pkg>Regressor",
  inherit = rieszreg::RieszEstimatorR6,
  public = list(
    initialize = function(...) {
      ...
    }
  )
)
```

`DESCRIPTION` declares `Imports: R6, reticulate (>= 1.30), rieszreg (>= 0.0.1)`. The rieszreg dep resolves at test time via `pkgload::load_all`, not from CRAN. See [`rieszreg-dev-environment`](../rieszreg-dev-environment/SKILL.md) for the R parity-test pattern.

## 6. Tests

Two categories of test are required (DESIGN.md §7):

- **Per-estimand examples** (DESIGN.md §7.1): every built-in estimand (`ATE`, `ATT`, `TSM`, `AdditiveShift`, `LocalShift`) must have a runnable script in `packages/<pkg>/python/examples/`. These double as living documentation and as smoke tests against API drift.
- **Reference-parity test** (DESIGN.md §5.2): if your method has a prior published implementation, ship a parity test against a hand-applied reference computation on a small example. If the method is novel (no prior implementation, e.g. riesztree), ship a self-parity test against a closed-form leaf solution or an independent code path.

Plus the standard sklearn-conformance test (use `rieszreg.testing.sklearn_conformance`) and recovery tests on a known DGP from `rieszreg.testing.dgps`. See [`rieszreg-coding`](../rieszreg-coding/SKILL.md) §5 for which tests are worth writing and which aren't.

## 7. Workspace wire-up

- `packages/<pkg>/python/pyproject.toml` — `dependencies = ["rieszreg"]` (uv resolves to the workspace member; no version pin needed inside the workspace).
- Add `packages/<pkg>/python` to `[tool.uv.workspace] members = [...]` in the root `pyproject.toml`.
- Add `<pkg>` to the matrix in `.github/workflows/test.yml` and `release.yml`.
- Run `uv sync --all-packages --all-extras` to install the new member into the workspace `.venv`.
- Add a backend page to `docs/backends/<pkg>.qmd` showing a worked example with the bilingual Python/R panel-tabset.

## 8. Where the contract is canonical

`DESIGN.md` Part B at the repo root is the long-form authoritative version. This skill is the triggered short form. If they ever conflict, `DESIGN.md` wins; update this skill.
