# Predictability of the failure-crossing event on `instance_10p`

> **Note (2026-06-19):** This report analyzes the **original noise-dominated** instance, now saved as
> `instances/instance_10p_noisy_alpha0p05.json`. Based on these findings, the default
> `instances/instance_10p.json` was **regenerated to be anticipatable** (`alpha0_mean` 0.05→0.8 →
> time-to-failure CV ≈ 0.6→0.22, same E[lifetime]=60 yr). All numbers below describe the *noisy*
> instance; the new default has the predictable dynamics. Exp 0 is being re-run on the new default.

**Question.** Is the single most consequential event in this MDP — an asset crossing the
functional→failed threshold (`d_i ≥ d_fail`), after which escalating risk cost `C_risk` dominates —
*predictable* far enough ahead to be **anticipated**, or is its timing so noisy that the best any
policy can do is **react**? If the latter, that explains why MC rollout beats a tuned reactive
heuristic by only ~6–9% on mean cost on this instance, and it is a reportable finding.

**Method.** Analysis only — no model/agent code changed, no new long runs. Analytical
characterization from the instance parameters, plus Monte-Carlo using the env's own
`gamma_step` (exact, same inverse-CDF transform as the simulator), cross-checked against the 50
existing eval episodes (× 239 epochs) on disk. Scripts: `_diag_predictability.py`,
`_diag_empirical.py`.

---

## Headline numbers

| Metric | Value on `instance_10p` | Reading |
|---|---|---|
| **Per-epoch increment CV** | **5.5 – 9.2** (median **6.9**) | Each step's degradation is ~7× noisier than its mean. Pure noise at the step level. |
| **Time-to-failure CV** (from d=0.5) | **0.56 – 0.71** (median **0.63**) | Failure timing has a ±63% relative spread — intrinsically unpredictable. From d=0.9 it is still 0.76–0.86. |
| **k-step spread** `std(d_{t+k}\|d_t=0.5)` | k=8 → **0.12**, k=16 → **0.11–0.15** | By the renovation lead time the conditional spread is ~0.15 of the whole [0,1] scale. |
| **Failure-prob sharpness** (10→90% width in `d_t` at the renovation lead) | width **0.20 – 0.63**, median **0.35** | The "safe→doomed" transition is a *wide ramp*, not a cliff. You cannot cleanly tell at lead time which assets will fail. |
| **Foresight ceiling** R²(future cost \| current state) | full horizon **0.58**, next-16-epoch **0.32** | Current state explains only ~58% of cost-to-go variance (and only ~32% over the actionable lead horizon). **~42–68% is irreducible future noise.** |
| **Action signal vs noise** over the lead horizon | restriction shifts `d` by **~9%** of one noise SD | The controllable effect is an order of magnitude smaller than the process noise it must fight. |

**Verdict: failure timing on `instance_10p` is fundamentally unpredictable at the horizon
needed to act on it. Anticipatory RL is structurally limited here; the achievable policy is
essentially reactive.** This is consistent with the thin RL margin. (Important caveat below: MC
rollout still cuts *outcome variance* sharply, even though it barely moves the *mean*.)

---

## 1. Per-epoch increment — the step is dominated by noise

`ΔG_i ~ Gamma(shape = α_i·Δt, rate = β_i)`. The increment CV is `1/√(α_i·Δt)`, independent of
`β`. On this instance `α_i·Δt ≈ 0.02`, so:

- mean increment 0.0073–0.0092, **CV 5.5–9.2 (median 6.9)**.

A single epoch's degradation is essentially noise: the standard deviation is ~7× the mean. Signal
only emerges by *averaging over many steps* — which is exactly what removes the ability to predict
any individual crossing. Mean epochs pristine→fail = `β/(α·Δt)` ≈ 109–137, i.e. the planning
horizon T=120 is roughly one expected asset lifetime.

## 2. Time-to-failure (first passage `d→1.0`)

Monte-Carlo first-passage via `gamma_step` (ℓ=0), 4000 paths/asset:

| start `d` | mean TTF (epochs) | **CV of TTF** |
|---|---|---|
| 0.0 | 132–161 | 0.43–0.62 (med 0.52) |
| 0.5 | 75–95 | 0.56–0.71 (med **0.63**) |
| 0.7 | 52–72 | 0.63–0.77 (med 0.71) |
| 0.9 | 26–42 | 0.76–0.86 (med **0.81**) |

A CV around 0.6–0.8 means that even knowing an asset's exact current condition, the *timing* of its
failure is uncertain to ±60–80%. Strikingly, **the CV grows as the asset gets closer to failure**:
the closer you are, the more the (now-small) remaining headroom is dominated by a few noisy
increments. So "watch it closely as it nears failure" does not buy sharper timing — the opposite.

## 3. k-step-ahead conditional spread of `d_{t+k} | d_t`

MC (ℓ=0), and **cross-checked empirically** against the rollout-policy trajectories (windows with no
renovation in them, so it is pure degradation):

| | k=1 | k=4 | k=8 | k=16 |
|---|---|---|---|---|
| analytic std (d_t=0.5) | 0.04 | 0.085 | 0.115 | 0.151 |
| **empirical std (d_t~0.5)** | 0.048 | 0.073 | 0.090 | **0.109** |
| empirical std (d_t~0.3) | 0.069 | 0.108 | 0.137 | 0.172 |

Analytic and empirical agree closely (the empirical is slightly tighter because of capping at 1 and
mild policy selection). By k≈8–16 epochs — the time it takes to *complete a renovation* — the
conditional spread of `d` is ~0.11–0.17, i.e. **10–17% of the entire [0,1] condition scale**. At the
lead time you actually need, the forecast of an asset's condition is broad.

## 4. Failure-probability sharpness at the renovation lead time

Failure is only *avoidable* if foreseeable at least one renovation duration ahead (you must finish
renovating before `d` hits 1). Renovation lead on this instance is **8–26 epochs (median 15)**.
Sweeping `d_t` and measuring `P(failed within k=lead | d_t)`:

- 10%→90% transition **width in `d_t`: 0.20–0.63, median 0.35**.

A *predictable* problem would have a sharp cliff (narrow width): below some `d` you're safe, above it
you're doomed. Here the transition spans a third of the condition scale on average — a **gradual
ramp**. Over the band of conditions the policy actually operates in (`d≈0.4–0.8`), an asset's
probability of failing within the actionable horizon is genuinely uncertain. The slow-degrading
asset 2 (lead 19 epochs) has width 0.51; even the fastest asset 9 (lead 8) has width 0.19.

## 5. Empirical cross-check

- **Conditional spread** matches the analytic numbers (table in §3).
- **Realized first-crossing epoch** (reactive heuristic; censored by renovations): mean 84,
  std 55, **CV 0.65** — consistent with the analytic TTF CV of ~0.6–0.7 despite censoring.

## 6. Value of foresight / the reactivity gap

**(a) Irreducible outcome noise.** All 50 eval episodes start from the *same fixed* `d_init` and run
the *same policy* (shared CRN across agents); the only thing that differs is process noise. The
spread of total discounted episode cost is therefore pure irreducible outcome noise:

| agent | mean total disc cost | CV |
|---|---|---|
| MC rollout (policy warmstart) | 6.96e8 | **0.235** |
| MC rollout (empty warmstart) | 7.05e8 | **0.208** |
| reactive (tuned) | 7.16e8 | **0.741** |
| paced | 8.66e8 | 0.894 |
| per-asset | 1.01e9 | 0.659 |

**(b) Foresight ceiling.** Regressing realized discounted cost-to-go on the current full 5N state
(rollout policy, 6050 decision points):

- full-state **R² = 0.58** → ~42% of cost-to-go variance is unexplained by the current state
  (= future noise the agent cannot see).
- `d`-only R² = 0.44; `n_fail`-only R² = 0.04.
- Predicting only the **next k epochs** of cost from the current state: R² = 0.26 (k=4), 0.28 (k=8),
  **0.32 (k=16)**, 0.35 (k=32). Over the actionable lead horizon, **~68% of near-term cost is
  irreducible noise.**

So even a perfect state-only predictor would explain barely a third of what happens over the window
in which an intervention could change the outcome. There is little foresight to be had.

## 7. Why MC rollout barely separates from reactive — and the important caveat

The argument and the caveat both follow from the numbers above.

**Why the means are close.** MC rollout ranks candidate actions by simulating futures under CRN.
But over the relevant lead horizon the *action effect is dwarfed by process noise*: a sustained load
restriction (the only "slow it down" lever, and `restrict_degrad_multiplier = 0.90` here — just a
10% rate cut) shifts `d` by ~0.013 over 16 epochs, **~9% of one noise SD (~0.15)**. Renovation is
binary and takes 8–26 epochs to bite. So when the rollout simulates "restrict" vs "do nothing", the
two clouds of sampled futures overlap almost completely — the noise, shared by CRN or not, swamps the
controllable difference. The rollout cannot reliably *anticipate* a crossing it cannot predict, so it
converges to reacting once `n_fail`/`d` make the failed state unambiguous — which is what the tuned
reactive heuristic already does. Hence ~6–9% on the mean, not significant at n=50.

**The caveat (report this honestly).** MC rollout is *not* worthless here: it **cuts the outcome CV
from 0.74 (reactive) to 0.23** — a ~3× reduction in the spread/tail of total cost, at a similar mean.
The value of rollout on this instance is **risk reduction (avoiding bad realizations), not mean-cost
anticipation.** A mean-only comparison understates it; a risk-aware or tail metric (CVaR, P90 cost)
would show a clearer, likely significant separation. This is itself a paper-worthy framing: in a
noise-dominated maintenance MDP, the gain from planning shows up in *variance*, not *expectation*.

---

## Does this generalize? Instance-design knobs for Exp 1

The unpredictability here is driven by three compounding properties of `instance_10p`, each a tunable
knob:

1. **Gamma increment CV = `1/√(α·Δt)` ≈ 7.** This is the root cause. It is high because `α·Δt ≈ 0.02`
   is tiny (slow mean degradation, so each step is mostly noise). To make failure *more* anticipatable,
   raise the per-step shape `α·Δt` — e.g. larger `α0` with `β` rescaled to hold `e_fail` fixed, or a
   larger `Δt`. The generator's `e_fail_cv` (currently 0.1) controls *cross-asset* heterogeneity but
   **not** the within-asset step CV; the step CV is set by `α·Δt`. A dedicated "degradation
   smoothness" knob (effectively `α0` magnitude, or a Gamma→more-deterministic shape) is what changes
   predictability.
2. **`d_fail` headroom.** With `d_fail = 1.0` and `d_init ≈ 0.4–0.8`, assets sit close to failure and
   the remaining headroom is a small, noise-dominated gap (this is why TTF CV *rises* near failure,
   §2). Larger headroom (lower `d_init` band, or treating `d_fail` as the design target with more
   room) gives more steps to average over → lower TTF CV → more anticipatable.
3. **Renovation lead vs failure-ramp width.** Anticipation needs the failure to be foreseeable at
   least one renovation duration ahead. Here lead is 8–26 epochs and the 10→90 failure ramp is ~0.35
   wide — the forecast is broad over exactly that horizon. **Shorter renovation lead** (larger `mu_h`)
   or a **sharper failure ramp** (higher step shape, knob 1) both widen the window in which a
   confident, actionable forecast exists.

**Recommendation for Exp 1:** include at least one "anticipatable" instance — higher `α·Δt` (lower
increment CV, say CV≈2–3 instead of 7) and/or more `d_fail` headroom and/or shorter renovation lead —
to demonstrate the *contrast*: that anticipatory RL pulls clearly ahead of reactive precisely when
the process is predictable, and collapses toward reactive when it is not. That contrast is a stronger
paper result than either instance alone.

---

## Proposed logging / metrics to capture predictability (NOT implemented)

Each is opt-in diagnostic logging, off the hot path, hooking into the eval loop and the rollout
action generator. None changes the model.

1. **Predicted vs realized time-to-failure (per asset, per decision).** At each decision epoch, log
   the model/rollout-implied expected epochs-to-failure for each not-yet-failed asset, and later
   join to the *realized* crossing epoch in the same episode. *Reveals:* calibration of the agent's
   failure-timing forecast and the realized prediction error (the empirical TTF CV, per condition
   band). Hook: eval loop, after each `env.step`, plus a post-hoc join pass.
2. **Conditional failure probability at the renovation lead time.** Per decision, per asset, log
   `P(d_{t+lead} ≥ d_fail | s_t)` (rollout already samples futures — read it off the candidate=none
   rollouts) alongside the eventual outcome. *Reveals:* whether the agent has any sharp signal at the
   actionable horizon (Brier score / reliability curve); directly operationalizes §4. Hook: rollout
   generator (reuse the do-nothing rollout cloud).
3. **Value-function calibration: implied future cost vs realized.** Log `V'(s_post)` (already
   computed in the action generators) vs the realized discounted cost-to-go from that state.
   *Reveals:* the §6b R²/calibration as a training-time signal — how much of cost-to-go the VFA can
   actually predict, and whether it over/under-confident. Hook: action generator (it already has
   `immediate_cost_components`) + post-hoc cost-to-go pass over the episode.
4. **Anticipation-benefit metric ("oracle minus reactive").** Per episode, alongside the policy cost,
   log the cost of a fixed reactive baseline on the *same CRN noise*, and (offline) a CRN-aware
   hindsight/oracle policy. *Reveals:* the gap the agent could in principle close (oracle−reactive)
   and the gap it actually closes (policy−reactive) — a direct "how much foresight is on the table"
   number per instance. Hook: eval loop runs the paced/reactive baseline on the same
   `begin_episode("evaluation", ep)` CRN stream; oracle computed offline.
5. **Action-cloud separation in rollouts.** Per decision, log the overlap (e.g. mean/std, or a
   distributional distance) between the sampled-return clouds of the top-2 candidate actions.
   *Reveals:* directly measures §7 — how often the rollout has a statistically resolvable preference
   vs noise-dominated ties (and would justify the adaptive-budget Wilcoxon stopping already in
   `agents/rollout.py`). Hook: rollout generator, where candidate Q-samples are compared.
6. **Outcome-variance / tail metrics in eval aggregation.** Already-loggable: report total-cost CV,
   P90, and CVaR per agent (not just the mean). *Reveals:* the §6a risk-reduction effect that the
   mean comparison hides — likely the cleanest place MC rollout shows a significant win. Hook: eval
   summary/`comparison_dashboard.py`.

---

*Files:* analysis scripts `_diag_predictability.py`, `_diag_empirical.py` (throwaway, repo root).
*Data:* `results/exp0/{i10p_rollout_policy,i10p_optuna_reactive,...}/eval_episodes.csv` (50 ep × 239
epochs). *No model or agent code was modified.*
