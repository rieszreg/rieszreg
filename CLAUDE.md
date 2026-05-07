# RieszReg

Family of packages for Riesz regression in a uv-workspace monorepo.

Domain-specific rules live in skills under `.claude/skills/` (architecture, notation/doc-tone, dev environment, learner-package contract). They auto-trigger on relevant tasks; you don't need to invoke them by hand.

## Research code, not production

This is a research codebase, not a released library. There are no users to keep on old APIs, no deprecation cycles, no backwards-compatibility shims. Breaking changes are fine — when refactoring, just change the call sites. Don't add `_legacy` paths, deprecation warnings, or "still supported for now" branches. If something is wrong, fix it; if a name is bad, rename it.

For tests, the priority is statistical and conceptual correctness — does the estimator recover the truth on a known DGP, do the math identities hold, do the parity tests against reference implementations agree. Tests that check input-validation messages, that a specific warning fires, or that boilerplate plumbing didn't break are low value here. Spend test budget on what the math says should be true.

## Code style

Prefer fewer lines to more. When adding functionality, tie it into the existing OOP scaffolding as a method on the right class — don't reach for a new free function or a new file unless the existing structure genuinely doesn't fit. The classes here (`Estimand`, `Loss`, `Backend`, `RieszEstimator`, the R6 wrappers) are the abstractions; new behavior usually wants to live on one of them.

When you find yourself writing a procedural loop or a helper script, ask whether it's really a method that belongs on an existing class.

## Keeping this file lightweight

CLAUDE.md is loaded into every agent context in this repo. Only add to it information that every agent doing any task here needs. Anything narrower belongs in a skill under `.claude/skills/` — skills load on trigger, not always, which is the whole point.

Don't restate things visible from the source tree: package list, directory layout, dependency graph, file-by-file responsibilities, status snapshots, test counts. Agents read those from `ls`, `cat`, and `git log`, and any prose copy here drifts out of sync. The repo should be self-documenting; this file is for things that aren't.

When you find yourself writing a rule that applies only to agents editing a particular kind of file (docs, R code, CI workflows) or doing a particular kind of task (adding a backend, running tests), write it as a skill keyed to those files / tasks instead of adding a section here.
