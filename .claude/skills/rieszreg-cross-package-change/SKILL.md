---
name: rieszreg-cross-package-change
description: Sweep checklist for a public-API change in `packages/rieszreg/` — renames, signature changes, removals, deprecations of anything re-exported from `rieszreg/__init__.py`, the Estimand factories, the Loss Protocol, Backend / MomentBackend signatures. Use whenever a commit message starts with "rename", "refactor", "drop", "deprecate", or when modifying a public surface in `packages/rieszreg/python/rieszreg/`. The goal is to catch stale prose, examples, and docstrings in the same commit as the API change.
---

# Cross-package sweep after a rieszreg API change

In the monorepo, the rename itself is a single commit (no coordinated PRs across repos). The remaining work is the *prose-and-examples sweep* — every README, every docstring, every `docs/*.qmd`, every `packages/<pkg>/python/examples/*` that mentions the old symbol. Catching this in the same commit avoids the "Tidy stale X-as-predictor-matrix prose remnants"-style follow-up PRs that used to land a week later.

## The sweep

```sh
grep -rn '<old>' --include='*.py' --include='*.qmd' --include='*.md' --include='*.R' \
  packages/ docs/ README.md DESIGN.md CLAUDE.md
```

Hit list to mentally walk through:

1. **Source** of every package — `packages/<pkg>/python/<pkg>/**/*.py`, `packages/<pkg>/r/<pkg>/R/*.R`. Update call sites and docstrings.
2. **Tests** — `packages/<pkg>/python/tests/**/*.py`, `packages/<pkg>/r/<pkg>/tests/testthat/*.R`. Update imports, fixtures, assertions.
3. **Examples** — `packages/<pkg>/python/examples/*.py` (one per built-in estimand, per package — `ATE`, `ATT`, `TSM`, `AdditiveShift`, `LocalShift`; this is the per-estimand-examples requirement enforced by `rieszreg-add-learner-package`). Update example code and any inline comments.
4. **READMEs** — `README.md` at root + `packages/<pkg>/README.md` for each package. Update the API description and any code blocks.
5. **CLAUDE.md files** — root and any per-package `packages/<pkg>/CLAUDE.md`.
6. **DESIGN.md** — at root. Cross-reference notation tables and API descriptions.
7. **User-guide docs** — `docs/*.qmd`, especially the per-backend pages and the estimands/losses pages. The `docs/_freeze/` cache will regenerate on the next render.
8. **The notation skill** — `.claude/skills/rieszreg-notation/SKILL.md`. If the rename has notation implications (a symbol changed, a term changed), update the canonical convention there. The rest of the codebase follows from that.

## When the rename has notation consequences

If you're renaming a math symbol or a domain term — e.g. `g → mu`, `theta → psi`, "target parameter" → "estimand" — the rename is bigger than a code change. Update the [`rieszreg-notation`](../rieszreg-notation/SKILL.md) skill **first**, then sweep prose to match. Doc-tone rules (also in the notation skill) apply to any prose you touch in the sweep — replace any "the workhorse" / "the natural way" / em-dash-laden phrasing while you're there.

## When the rename has API consequences for downstream users

This is research code (see [`rieszreg-coding-style`](../rieszreg-coding-style/SKILL.md) §1) — breaking changes are fine, no deprecation shims. The sweep is for *internal* consistency across the monorepo, not for backwards compatibility.

If a downstream user (paper, downstream wrapper) is pinned to a specific tagged release, that pin keeps working — the rename only affects users on `main` or on tags after the rename. Don't add aliases or shims.

## Verification

After the sweep, before committing:

```sh
grep -rn '<old>' --include='*.py' --include='*.qmd' --include='*.md' --include='*.R' \
  packages/ docs/ README.md DESIGN.md CLAUDE.md
```

Should return nothing (or only intentional historical references — e.g. a CHANGELOG entry). Then run the full test matrix locally before pushing:

```sh
for p in rieszreg rieszboost krrr forestriesz riesznet riesztree; do
  uv run pytest packages/$p/python/tests -q
done
```
