---
name: rieszreg-coding
description: How to write code in the rieszreg monorepo — research-code mindset, OOP-first organization, Tier-1/2/3 placement across packages, sklearn-first patterns, and test philosophy (test what the library does statistically, not that warnings fire). Use whenever you are about to add or modify code under `packages/<pkg>/python/<pkg>/`, write tests, design new abstractions, or make planning decisions about where new functionality should live.
---

# Coding in the rieszreg monorepo

This skill encodes the recurring style and architecture decisions for this codebase. Apply on every code edit. The five concerns below interact — a single new method usually touches three of them.

## 1. Research code, not production code

This is research code that supports active development of statistical methods. The maintenance posture is different from a stable production library:

- **Breaking changes are fine.** A rename, a signature change, a removal — any of those land as a single commit. Don't write deprecation shims. Don't keep old kwargs as aliases. Don't preserve old API names "in case someone is using them" — the only people using them will rebase or update.
- **Don't paranoidly validate user inputs.** Internal calls trust their callers. Validation happens at the genuine boundary (the public `RieszEstimator.fit` entry point, file I/O, deserialization). Inside the codebase, a function that takes an `Estimand` trusts that it received an `Estimand`.
- **Don't add error handling, fallbacks, or retries for scenarios that can't happen.** If a code path can only be reached via an internal call site that always passes valid data, don't write defensive checks. Trust the framework guarantees (sklearn, numpy, torch).
- **Don't gate features behind backwards-compat flags.** If a new fit signature is better, change it. Update every call site in the same commit.

Production-style hygiene that *is* worth keeping: type hints on public functions, docstrings on public classes, a clear sklearn-conformant API surface. Production-style hygiene that is *not* worth keeping: deprecation warnings, FutureWarnings about future renames, version-gated behavior, exhaustive input validation.

## 2. OOP-first; extend existing classes; fewer lines is better

Default to extending an existing class when a new piece of functionality conceptually belongs there. Reach for a new module or a free function only when the new functionality is genuinely orthogonal to existing classes.

Concrete heuristics:

- **New per-loss behavior** → method on `Loss` (`packages/rieszreg/python/rieszreg/losses/base.py`). Not a `compute_X(loss, ...)` free function elsewhere.
- **New per-estimand behavior** → method on `Estimand` (`packages/rieszreg/python/rieszreg/estimands/base.py`).
- **New per-backend behavior** → method on `Backend` / `MomentBackend` Protocol or on the concrete `<Pkg>Backend` class.
- **New diagnostic** → field on the per-package `Diagnostics` dataclass.
- **New shared helper** that several backends would use → in `rieszreg`, not in any one impl package.

When tempted to add a new file with a new free function, pause and ask: *which existing class would naturally own this method?* If the answer is "this class, but it doesn't have that method yet," add the method.

**Prefer fewer lines to more.** A 6-line method that does the thing is better than a 22-line method that does the thing with three intermediate-named locals and a docstring repeating the function name. Don't expand `return self._loss.tilde_potential(alpha) - m_alpha` into a 12-line block to "explain" it.

When writing a *plan* (in conversation or in a plan file), explicitly think through: *which existing abstractions do I extend?* not just *which new file do I create?* If the plan introduces a new top-level concept (a new Protocol, a new dataclass), that's a real architectural move and should be justified. If it doesn't, the plan should fit into existing files.

## 3. Tier-1 / Tier-2 / Tier-3 placement

The architectural rule from `DESIGN.md` Part A §2. Every parameter, method, or abstraction in this codebase belongs to exactly one tier:

- **Tier 1 — universal across all backends.** Lives in `packages/rieszreg/`. Examples: `Estimand`, `Loss`, `RieszEstimator`, `Backend` and `MomentBackend` Protocols, `Diagnostics` base, `AugmentedDataset`. A would-be Tier-1 thing must work for *every* plausible backend the family might add. If a moment-style backend (forest, neural net) would just *ignore* the new kwarg, it isn't Tier 1.
- **Tier 2 — orchestrator dispatch flag.** Lives on `RieszEstimator` only when it picks between Tier-1 dispatch paths (e.g. `fit_augmented` vs `fit_rows`). Rare; don't invent new ones casually.
- **Tier 3 — learner-specific knob.** Lives on the concrete backend's constructor (`XGBoostBackend.n_estimators`, `KernelRidgeBackend.kernel`, `RieszTreeRegressor.max_depth`). Not on `RieszEstimator`, not on Protocol methods.

The would-be-ignored lint test: before adding a kwarg to `RieszEstimator.__init__` or to a `Backend` Protocol method, ask *would a moment-style backend (forest, neural net) ignore this kwarg?* If yes, it's Tier 3 — move it to the concrete backend constructor.

Concrete failure modes the codebase has had: `n_estimators` on `RieszEstimator` (was Tier 3), `feature_keys` on `fit()` (no, plumb through `Z` directly), custom `crossfit()` orchestration function (no, use `cross_val_predict`).

## 4. sklearn-first

Before writing procedural code with loops, splits, grids, or folds, ask *"is there an sklearn way?"*. If yes, use it. The estimator's user-facing API must compose with `clone`, `GridSearchCV`, `cross_val_predict`, `Pipeline` — verify those four work on every change to the orchestrator, the predictor, or any concrete `<Pkg>Backend`.

Bespoke is reserved for things sklearn genuinely doesn't cover: the `LinearForm` tracer, the custom xgboost objective, the Bregman `Loss` Protocol, the augmentation engine. If you find yourself writing `for fold in folds:` with a manual `train_test_split`, that's a flag — almost certainly `cross_val_predict` does what you want in one line.

## 5. Test philosophy

Tests should verify the library is doing what it's *meant* to do, statistically and conceptually. Not that the right warning fires on the wrong input.

**The interesting tests** (write more of these):

- *Recovery*: on a known DGP where the true Riesz representer is closed-form, the estimator recovers it within Monte-Carlo error. (`testing.dgps` provides ATE, TSM, AdditiveShift DGPs with analytic ground truth.)
- *Parity*: when there's a published method this implements, the result matches a hand-applied reference computation on a small example. (DESIGN.md §5.2.) For methods with no prior implementation, ship a self-parity test against a closed-form leaf solution or an independent code path.
- *sklearn conformance*: `clone`, `cross_val_predict`, `Pipeline`, `GridSearchCV` compose with the new estimator. (`testing.sklearn_conformance` provides this.)
- *Cross-loss correctness*: behavior matches across the four built-in losses, especially around link-function boundaries.

**Tests not worth writing** (delete these on touch):

- Defensive input validation tests: "raises if `n_estimators=0`", "raises if `Z` is None", "raises if `loss` is unrecognized".
- Warning-firing tests: "FutureWarning when `old_param` is set". (We don't emit FutureWarnings.)
- Backwards-compat shim tests: "old API name still works".
- Trivial smoke tests that mirror existing higher-level tests: "import works", "constructor returns an instance".

If a test you're writing only fails when someone removes a guard rail (an `if x is None: raise` block), the test is testing the guard rail, not the statistical method. Skip it.

## 6. When in doubt

Refer to `DESIGN.md` Part A (the meta-package architecture) and Part B (the contract every learner package satisfies). When the rules above conflict with a specific situation, favor: (a) what makes the public API clean and sklearn-conformant, (b) what makes the test suite genuinely informative about statistical correctness, (c) fewer files, fewer lines, fewer abstractions.
