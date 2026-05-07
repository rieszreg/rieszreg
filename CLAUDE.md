# RieszReg (meta-project)

> **Read [`rieszreg/DESIGN.md`](rieszreg/DESIGN.md) first.** It is the authoritative
> design document for the family — Part A is the meta-package architecture,
> Part B is the contract every learner package (rieszboost, krrr, future)
> must satisfy. Anything in this CLAUDE.md is operational notes; anything in
> DESIGN.md is contract.

> **Notation conventions** (math, prose, code naming) for everything across
> this family — docs, READMEs, docstrings, comments, examples — live in the
> [`rieszreg-notation` skill](.claude/skills/rieszreg-notation/SKILL.md).
> Read it before editing any user-facing prose or docstring that mentions
> $\mu$ / $\alpha$ / $\psi$ / $m$, the data tuple $Z = (A, X)$, or the term
> "learner". Update touched paragraphs to the canonical convention; defer
> bulk sweeps to a dedicated notation-pass PR.

A family of packages for Riesz regression. Top-level coordinator for:

- [`rieszreg/`](rieszreg/) — meta-package: shared abstractions (`Estimand`, `Loss`, `RieszEstimator`, augmentation, diagnostics, `Backend` and `MomentBackend` Protocols, testing utilities, R6 base class).
- [`rieszboost/`](rieszboost/) — gradient-boosting backend (Lee & Schuler 2025; uses `Backend.fit_augmented`).
- [`krrr/`](krrr/) — kernel-ridge backend (Singh 2021; uses `Backend.fit_augmented`).
- [`forestriesz/`](forestriesz/) — random-forest backend (Chernozhukov, Newey, Quintas-Martínez, Syrgkanis ICML 2022; uses `MomentBackend.fit_rows`).
- [`riesznet/`](riesznet/) — neural-network backend (Chernozhukov, Newey, Quintas-Martínez, Syrgkanis 2021, Riesz-rep only; uses `MomentBackend.fit_rows` with PyTorch autograd).

Implementation packages depend on `rieszreg` and provide concrete backends. The design doc lives inside the meta-package itself ([`rieszreg/DESIGN.md`](rieszreg/DESIGN.md)) so every collaborator who clones rieszreg gets it.

## GitHub home

All packages in this family live under the [`rieszreg` GitHub org](https://github.com/rieszreg). When creating a new learner package (or any new repo in the family — examples, downstream wrappers, paper artifacts), create it directly in `rieszreg/`, not under a personal account:

```sh
gh repo create rieszreg/<new-pkg> --public --source=. --remote=origin --description "..."
```

The unified docs CI, the per-package `test.yml` files (which check out `rieszreg` as a sibling via `repository: rieszreg/rieszreg`), and the cross-references in `README.md` / `docs/*.qmd` all assume the org-level path.

## Dependency graph

```
rieszreg (no deps on impl packages)
   ↑
   ├── rieszboost      (XGBoostBackend, SklearnBackend, RieszBooster)        [fit_augmented]
   ├── krrr            (KernelRidgeBackend, kernels, solvers, KernelRieszRegressor)  [fit_augmented]
   ├── forestriesz     (ForestRieszBackend, ForestRieszRegressor)            [fit_rows]
   ├── riesznet        (TorchBackend, TorchPredictor, RieszNet)              [fit_rows]
   └── <future-pkg>    (its backend(s) + thin convenience class)
```

## Where things live

| Concern | Home |
|---|---|
| `Estimand` base class + 5 built-in subclasses (ATE, ATT, TSM, AdditiveShift, LocalShift), `LinearForm`, `Tracer` | `rieszreg/python/rieszreg/estimands/` |
| `Loss` base class + 4 built-in losses | `rieszreg/python/rieszreg/losses/` |
| `AugmentedDataset` packaging | `rieszreg/python/rieszreg/augmentation.py` |
| `Backend` and `MomentBackend` Protocols + predictor-loader registry | `rieszreg/python/rieszreg/backends/base.py` |
| `Diagnostics` base class + `diagnose()` | `rieszreg/python/rieszreg/diagnostics.py` |
| `RieszEstimator` (sklearn orchestrator) | `rieszreg/python/rieszreg/estimator.py` |
| Canonical DGPs, sklearn-conformance, parity helpers | `rieszreg/python/rieszreg/testing/` |
| Shared base R6 class | `rieszreg/r/rieszreg/R/rieszreg.R` |
| User guide (single Quarto site) | `docs/` |
| Reference papers | `reference/` |
| Pre-commit hook template | `.githooks/pre-commit` (each package keeps a copy) |
| CI workflow templates | `.github/workflows/` |
| Concrete backends + convenience subclasses | each implementation package's `python/<pkg>/backends/` and `python/<pkg>/<estimator>.py` |

## Adding a new estimand or loss

Goes in `rieszreg`, not the implementation packages.

1. Add the factory / class in `rieszreg/python/rieszreg/estimands/base.py` (or `losses/<name>.py`).
2. Wire into the relevant `__init__.py` and the `_FACTORY_REGISTRY` / `loss_from_spec` registry.
3. Update the docs page (`docs/estimands.qmd` or `docs/losses.qmd`) and the README.
4. Add a test under `rieszreg/python/tests/`.
5. Implementation packages whose backend supports the new feature should add a smoke test confirming round-trip works.

## Adding a new backend

Pick the entry point that matches your learner's natural loss decomposition:

- **Augmentation-style** (kernel ridge, gradient boosting): implement `Backend.fit_augmented` from `rieszreg/python/rieszreg/backends/base.py`. The orchestrator pre-computes the augmented `(a, b)` dataset for you. References: `KernelRidgeBackend` (krrr), `XGBoostBackend` (rieszboost).
- **Moment-style** (random forests, neural nets): implement `MomentBackend.fit_rows` from the same file. You receive raw rows + the estimand and call `rieszreg.trace(estimand, row)` per row to compute moments. Reference: `ForestRieszBackend` (forestriesz).

Register the predictor loader on import:

```python
from rieszreg.backends import register_predictor_loader
register_predictor_loader("my-kind", MyPredictor.load)
```

Provide a convenience class subclassing `rieszreg.RieszEstimator`. Subclass `rieszreg::RieszEstimatorR6` for the R wrapper. See `rieszreg/DESIGN.md` (Part B) for the full contract.

## Run tests

```sh
.venv/bin/python -m pytest rieszreg/python/tests -q
.venv/bin/python -m pytest rieszboost/python/tests -q
.venv/bin/python -m pytest krrr/python/tests -q
.venv/bin/python -m pytest forestriesz/python/tests -q
.venv/bin/python -m pytest riesznet/python/tests -q

Rscript -e '
  Sys.setenv(RETICULATE_PYTHON = file.path(getwd(), ".venv/bin/python"))
  pkgload::load_all("rieszreg/r/rieszreg")
  pkgload::load_all("rieszboost/r/rieszboost")
  testthat::test_dir("rieszboost/r/rieszboost/tests/testthat")
  pkgload::load_all("krrr/r/krrr")
  testthat::test_dir("krrr/r/krrr/tests/testthat")
  pkgload::load_all("forestriesz/r/forestriesz")
  testthat::test_dir("forestriesz/r/forestriesz/tests/testthat")
  pkgload::load_all("riesznet/r/riesznet")
  testthat::test_dir("riesznet/r/riesznet/tests/testthat")
'
```

## Doc-tone rules (enforced by .githooks/pre-commit)

User-facing docs describe what's currently in the package, in plain instructive prose matching the [ngboost user guide](https://stanfordmlgroup.github.io/ngboost/intro.html). Two failure modes the hook checks for:

1. **No design-decision metacommentary.** Don't explain the API's negative space — what we removed, intentionally didn't build, or chose between. Just describe what the function does and how to use it.
2. **No AI-flavored hedge or editorial framing.** Avoid phrases like "the workhorse", "the right choice for almost every", "almost never needs tuning", "the natural way/API", "rather than reinvent". Avoid em-dashes peppered through prose. Sentences should be short (8–15 words on average), active voice.

## sklearn-first rule

Before writing any procedural code with loops, splits, grids, or folds, ask *"is there an sklearn way?"*. If yes, use it. Bespoke is reserved for things sklearn genuinely doesn't cover (the `LinearForm` tracer, the custom xgboost objective, the Bregman `Loss`).

## Status

- rieszreg: 71 Python tests passing (unit tests for tracer, losses, estimands, augmentation, diagnostics, orchestrator with stub backends — both `Backend` and `MomentBackend` dispatch paths covered, testing utilities).
- rieszboost: 112 Python tests + 11 R parity tests passing.
- krrr: 36 Python tests + 1 R parity test passing.
- forestriesz: 55 Python tests + 1 R parity test passing.
- riesznet: 41 Python tests + 1 R parity test passing.
- Unified Quarto docs site renders all 17 pages (neural backend page added).
- Pre-commit hook + CI workflow templates wired but not yet activated by `git config core.hooksPath` in any clone.
