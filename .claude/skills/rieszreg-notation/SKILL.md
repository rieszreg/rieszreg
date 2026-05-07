---
name: rieszreg-notation
description: Notation, prose style, and doc-tone for the rieszreg family — math symbols ($\mu$, $\alpha$, $\psi$, $m$), code identifiers ($Z = (A, X)$), terminology ("estimand" not "target", "learner" not "model"), audience expectations for `docs/*.qmd`, and the doc-tone rules enforced by `.githooks/pre-commit` (no design-decision metacommentary, no AI-flavored hedging). Use whenever editing user-facing prose — Quarto pages (`docs/**/*.qmd`), READMEs, docstrings, code comments, and example scripts — that mentions an estimand, a Riesz representer, an outcome regression, a learner, or treatment / outcome variables.
---

# Notation conventions for the rieszreg family

These rules apply uniformly across the meta-package, all implementation packages, all docs (Quarto pages, READMEs), all docstrings, all code comments, and all example scripts. Apply on every edit; do not invent inconsistent local notation. If existing prose uses old notation, update it as part of any edit that touches the surrounding paragraph.

## 1. Outcome regression: $\mu$, never $g$

The conditional expectation of the outcome is $\mu(z) = \mathbb{E}[Y \mid Z = z]$. Always use $\mu$ (or `mu` in code). Reserve $f, g$ for generic functions (mathematical filler). Rationale: $\mu$ matches the standard statistical convention for a mean and pairs visually with $\alpha$ for the Riesz representer (both Greek letters for unknown functional parameters).

## 2. Operator notation: $m(\mu)(Z)$, never $m(Z, \mu)$

$m$ is an **operator**: it takes a function $\mu$ as input and produces another function $m(\mu)$ as output. Always write $m(\mu)$ for the result of applying $m$ to $\mu$, and $m(\mu)(Z)$ for that result evaluated at $Z$. Code mirrors this: prefer `m(mu)(z)` over `m(z, mu)`.

Note the difference between an *operator* (function → function, e.g. $m$) and a *functional* (function → number, e.g. $\mu \mapsto \mathbb{E}[m(\mu)(Z)]$). Use the right word — $m$ is an operator; the full target $\psi = \mathbb{E}[m(\mu)(Z)]$ as a function of $\mu$ is a functional. Don't call $m$ a functional.

Do **not** use empirical-process notation like $\mathbb{P}m(\mu)$ or $\mathbb{P}_n m(\mu)$. Many users will not have seen it. Always spell expectations as $\mathbb{E}[\cdot]$ and distributions as $\mathbb{P}$.

## 3. Estimands: $\psi$, never $\theta$

The scalar quantity is $\psi$. When an explicit reference distribution $\mathbb{P}$ matters, write $\psi = \Psi(\mathbb{P})$ (i.e. $\Psi$ is the functional, $\psi$ its value at $\mathbb{P}$). Capital and lowercase $\psi$ are visually distinct, unlike $\theta$ / $\Theta$. Avoid $\theta$ entirely — it's overloaded in the wider literature (angles, generic parameters).

The English word for $\psi$ is **estimand** — uniformly across docs, READMEs, docstrings, comments, examples, and code identifiers. Do **not** use "target", "target parameter", "parameter of interest", "estimation target", or bare "parameter" (in the statistical sense) as synonyms. The pedagogical aside `estimand (a.k.a. target parameter)` is fine the first time the word is introduced on a page, but every subsequent reference uses "estimand" only. "Target" / "parameter" remain valid in their non-statistical senses (target variable $Y$ in ML, function parameters, hyperparameters, model weights / parameters of a fitted estimator) — discriminate by context.

## 4. Data convention: $Z = (A, X)$, $Y$ outcome

The observed predictors are $Z = (A, X)$, where $A$ is the treatment (or any intervention, e.g. exposure level for a continuous shift) and $X$ are covariates. $Y$ is the outcome. $A$ at the front of the alphabet distinguishes it from the back-of-alphabet covariates $X$. This matches ML / TML conventions.

This applies to **code identifiers** as well. The predictor frame in `fit / predict / score / riesz_loss / diagnose` signatures, in tests, in docs, and in user-supplied examples is `Z`, not `X`. Sklearn's idiomatic `X` parameter name does **not** override this — `Z` wins because in our world `X` is reserved for the covariate sub-block. A function whose entire input is just covariates (no treatment column) may keep `X`, but estimator entry points that take the full predictor matrix do not.

## 5. Reference distribution: $\alpha_0$ vs $\alpha$

When a reference (true / population) distribution $\mathbb{P}_0$ is explicitly in scope, use $\alpha_0$, $\mu_0$ to denote the values of these functions at $\mathbb{P}_0$. When $\mathbb{P}_0$ is implicit, just use $\alpha$, $\mu$ — but **do not mix**. If you write $\alpha_0$ anywhere in a derivation, also write $\mu_0$. In user guide prose, prefer the unsubscripted form for visual cleanliness.

## 6. Algorithm terminology: "learner", not "backend" or "model"

The user-facing word for an algorithm that ingests data and produces an estimate of a Riesz representer is **learner**. Use "learner family" when emphasizing that the same structural learner with different hyperparameters yields different concrete fits. Reserve "backend" for the internal Protocol-level abstraction (`Backend`, `MomentBackend`); user-facing prose should rarely need that word.

## 7. Riesz loss derivation

When introducing the Riesz loss, prefer the four-line derivation that demystifies it:

$$
\begin{aligned}
\text{ideal MSE: } & \mathbb{E}\!\left[\,(\alpha(Z) - \alpha_0(Z))^2\,\right] \\
&= \mathbb{E}[\alpha(Z)^2] - 2\,\mathbb{E}[\alpha(Z)\,\alpha_0(Z)] + \mathbb{E}[\alpha_0(Z)^2] \\
&= \mathbb{E}[\alpha(Z)^2] - 2\,\mathbb{E}[m(\alpha)(Z)] + \text{const} \quad \text{(Riesz identity)} \\
\Rightarrow\, & L(\alpha) = \mathbb{E}\!\left[\,\alpha(Z)^2 - 2\,m(\alpha)(Z)\,\right]
\end{aligned}
$$

The point is that $\alpha_0$ is not observed, but the loss can be computed by evaluating $m(\alpha)$ — which only requires knowing $m$ (the user-supplied estimand) and the candidate $\alpha$.

## 8. Voice and audience for user-guide prose

**The audience is a reader who knows calculus and calculus-based statistics, plus basic causal-inference vocabulary (treatment, propensity, IPW). Nothing more.** Write everything in `docs/*.qmd` to that bar. This is the single most common notation-skill failure — prose drifts toward measure-theoretic / functional-analytic phrasing because it's natural for the writer. Catch it.

Concretely:

- **No** "Gateaux derivative", "square-integrable", "weakly converges", "Riesz identity" without an immediate plain-language gloss, "Hilbert space" / "Banach space" anywhere, empirical-process notation, "consistent" without saying what converges to what, "tangent space".
- **Don't use "map" as a noun in prose.** Say "function", "rule", "formula", or "estimand" depending on context. `\mapsto` is fine in math notation; "map" as an English noun reads as jargon to this audience.
- **Pace.** When introducing a math concept, lead with a concrete example (often the ATE), then state the general rule. A new technical idea gets its own paragraph; do not stack two new ideas in one sentence.
- **Plain-language equivalents** are required wherever a technical term shows up. Technical terms in parentheticals are fine when the prose works without them.

## 9. Doc-tone rules (enforced by .githooks/pre-commit)

User-facing prose describes what's currently in the package, in plain instructive prose matching the [ngboost user guide](https://stanfordmlgroup.github.io/ngboost/intro.html). Two failure modes the pre-commit hook checks for:

1. **No design-decision metacommentary.** Don't explain the API's negative space — what we removed, intentionally didn't build, or chose between. The user only cares what they can call. Just describe what the function does and how to use it. No "intentionally no X", "by design", "we chose Y over Z", "the rationale is...".
2. **No AI-flavored hedge or editorial framing.** Avoid phrases like "the workhorse", "the right choice for almost every", "almost never needs tuning", "the natural way / API", "rather than reinvent". Avoid em-dashes peppered through prose. Sentences should be short (8–15 words on average), active voice.

Also: don't add backend-specific framing to tier-1 docs ("the booster does X", "the kernel matrix is Y"). Use neutral language ("the backend produces η").

## 10. Docstrings

- Module-level docstrings on every `.py` file explaining semantics.
- Class/function docstrings on all public types.

## 11. Where this lives

This skill is canonical. Update here first if the conventions ever change.

## 12. Drift handling

A non-trivial fraction of existing files still uses the old notation ($g$, $\theta$, $m(z, g)$). When you edit a paragraph that uses old notation, update that paragraph fully to the new convention. Do not bulk-rewrite untouched files — drift gets fixed on touch. The exceptions are user guide pages explicitly being polished in a notation-pass PR.

## 11. Riesz representable Estimand

The estimand is $\mathbb E[m(\mu)(Z,Y)]$. Note the $Y$. That is, $m$ maps from whatever space $\mu$ lives in (functions of $Z$) to a space of functions that *can* depend on $Y$ as well. This is important for supporting estimands that depend on the joint distribution of $(Z, Y)$.

## 12. Finite-evaluation linear forms

When prose or code refers to estimands whose $m$ is a finite linear combination of point evaluations of $\mu$, use the canonical $(a, c)$ pair:

- $a: \mathcal Z \times \mathcal Y \to \mathcal Z^k$ — the **point generator**. Per row, $a(z, y)$ returns the $k$ points at which $\mu$ is evaluated. For ATE: $a(z, y) = ((1, z_x), (0, z_x))$ with $k = 2$.
- $c: \mathcal Z \times \mathcal Y \to \mathbb R^k$ — the **weight function**. Per row, $c(z, y)$ returns the linear-combination coefficients. For ATE: $c(z, y) = (1, -1)$.

The canonical decomposition is $m(\mu)(z, y) = c(z, y)^\top \vec\mu(a(z, y))$, where $\vec\mu$ denotes elementwise application.

Empirical pseudo-data convention (when writing the empirical loss):

- $Z_{ij} = a_j(Z_i, Y_i)$ — the $j$-th pseudo-datapoint generated from row $i$.
- $C_{ij} = c_j(Z_i, Y_i)$ — its scalar coefficient.

These objects belong to a `FiniteEvalEstimand` (the concrete subclass of `Estimand` that the tracer and augmentation engine support). Use "finite-evaluation estimand" in prose; reserve plain "estimand" for the abstract concept.

## 13. Bregman-Riesz potential: math vs code names

In math and prose use $h$, $h'$, and $\tilde h$:

- $h: \mathbb R \to \mathbb R$ — the Bregman potential (matches Hines & Miles, Kato; called $F$ in their papers).
- $h'(t)$ — its derivative.
- $\tilde h(t) = t h'(t) - h(t)$. Equivalently $\tilde h = h^* \circ h'$ where $h^*$ is the Legendre transform of $h$ — i.e. $\tilde h(t)$ is the value of $h^*$ at the slope $h'(t)$, *not* the Legendre transform itself. The population loss is $\mathbb E[\tilde h(\alpha(Z)) - m(h' \circ \alpha)(Z, Y)]$.

In code, the same three concepts are spelled out as descriptive method names on the `Loss` class:

- `potential(alpha)` ↔ $h(\alpha)$
- `potential_deriv(alpha)` ↔ $h'(\alpha)$
- `tilde_potential(alpha)` ↔ $\tilde h(\alpha)$

Don't use Greek letters (e.g. $\phi$, $\psi$) for the potential or its derivatives anywhere — Greek letters in this codebase denote functionals of distributions ($\mu$ for the conditional mean, $\alpha$ for the Riesz representer, $\psi$ for the estimand), not arbitrary scalar functions.