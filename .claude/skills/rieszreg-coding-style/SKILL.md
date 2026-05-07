---
name: rieszreg-coding-style
description: How to write code in the rieszreg monorepo — research-code mindset (breaking changes are fine, no deprecation shims, no paranoid input validation), OOP-first organization (extend existing classes when conceptually right, prefer fewer lines, work within existing frameworks), and test philosophy (test what the library does statistically, not that warnings fire). Use whenever you are about to add or modify code in any `packages/<pkg>/`, write tests, design new abstractions, or make planning decisions about where new functionality should live. Complementary to `rieszreg-architecture` (which covers tier classification and the load-bearing structural rules).
---

# Coding style for the rieszreg monorepo

Three style rules apply on every code edit. Each one routinely gets violated in the absence of an explicit reminder; this skill is the reminder.

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

## 3. Test philosophy

Tests should verify the library is doing what it's *meant* to do, statistically and conceptually. Not that the right warning fires on the wrong input.

**The interesting tests** (write more of these):

- *Recovery*: on a known DGP where the true Riesz representer is closed-form, the estimator recovers it within Monte-Carlo error. (`packages/rieszreg/python/rieszreg/testing/dgps.py` provides ATE, TSM, AdditiveShift DGPs with analytic ground truth.)
- *Parity*: when there's a published method this implements, the result matches a hand-applied reference computation on a small example. For methods with no prior implementation, ship a self-parity test against a closed-form leaf solution or an independent code path.
- *sklearn conformance*: `clone`, `cross_val_predict`, `Pipeline`, `GridSearchCV` compose with the new estimator. (`rieszreg.testing.sklearn_conformance` provides this.)
- *Cross-loss correctness*: behavior matches across the four built-in losses, especially around link-function boundaries.

**Tests not worth writing** (delete these on touch):

- Defensive input validation tests: "raises if `n_estimators=0`", "raises if `Z` is None", "raises if `loss` is unrecognized".
- Warning-firing tests: "FutureWarning when `old_param` is set". (We don't emit FutureWarnings.)
- Backwards-compat shim tests: "old API name still works".
- Trivial smoke tests that mirror existing higher-level tests: "import works", "constructor returns an instance".

If a test you're writing only fails when someone removes a guard rail (an `if x is None: raise` block), the test is testing the guard rail, not the statistical method. Skip it.

## When in doubt

Favor: (a) what makes the public API clean and sklearn-conformant, (b) what makes the test suite genuinely informative about statistical correctness, (c) fewer files, fewer lines, fewer abstractions. The architectural rules (tier classification, the agnostic-orchestrator principle, lazy imports, module separation) live in `rieszreg-architecture`; this skill is the style layer that sits on top of them.
