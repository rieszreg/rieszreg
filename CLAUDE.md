# RieszReg

Family of packages for Riesz regression in a uv-workspace monorepo.

## Read first

[`DESIGN.md`](DESIGN.md) is the contract for this family. Part A is the meta-package architecture; Part B is what every learner package must satisfy. Read it before any non-trivial change.

## Keeping this file lightweight

CLAUDE.md is loaded into every agent context in this repo. Only add to it information that every agent doing any task here needs. Anything narrower belongs in a skill under `.claude/skills/` — skills load on trigger, not always, which is the whole point.

Don't restate things visible from the source tree: package list, directory layout, dependency graph, file-by-file responsibilities, status snapshots, test counts. Agents read those from `ls`, `cat`, and `git log`, and any prose copy here drifts out of sync. The repo should be self-documenting; this file is for things that aren't.

When you find yourself writing a rule that applies only to agents editing a particular kind of file (docs, R code, CI workflows), write it as a skill keyed to those files instead of adding a section here.

## Doc-tone rules (enforced by .githooks/pre-commit)

User-facing docs describe what's currently in the package, in plain instructive prose matching the [ngboost user guide](https://stanfordmlgroup.github.io/ngboost/intro.html). Two failure modes the hook checks for:

1. **No design-decision metacommentary.** Don't explain the API's negative space — what we removed, intentionally didn't build, or chose between. Just describe what the function does and how to use it.
2. **No AI-flavored hedge or editorial framing.** Avoid phrases like "the workhorse", "the right choice for almost every", "almost never needs tuning", "the natural way/API", "rather than reinvent". Avoid em-dashes peppered through prose. Sentences should be short (8–15 words on average), active voice.

## sklearn-first rule

Before writing any procedural code with loops, splits, grids, or folds, ask *"is there an sklearn way?"*. If yes, use it. Bespoke is reserved for things sklearn genuinely doesn't cover (the `LinearForm` tracer, the custom xgboost objective, the Bregman `Loss`).
