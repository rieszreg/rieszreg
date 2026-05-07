# RieszReg

Family of packages for Riesz regression in a uv-workspace monorepo.

## Keeping this file lightweight

CLAUDE.md is loaded into every agent context in this repo. Only add to it information that every agent doing any task here needs. Anything narrower belongs in a skill under `.claude/skills/` — skills load on trigger, not always, which is the whole point.

Don't restate things visible from the source tree: package list, directory layout, dependency graph, file-by-file responsibilities, status snapshots, test counts. Agents read those from `ls`, `cat`, and `git log`, and any prose copy here drifts out of sync. The repo should be self-documenting; this file is for things that aren't.

When you find yourself writing a rule that applies only to agents editing a particular kind of file (docs, R code, CI workflows) or doing a particular kind of task (adding a backend, running tests), write it as a skill keyed to those files / tasks instead of adding a section here.
