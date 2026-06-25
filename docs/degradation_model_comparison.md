# Degradation model comparison: continuous Gamma process vs. discrete 5-state chain

**Question.** This codebase models asset condition with a *continuous* Gamma-process degradation
(`d ∈ [0,1]`, `d_{t+1} = min(1, d_t + ΔG)`). A common alternative in the maintenance literature is a
*discrete* deterioration chain with a transition matrix. How do the two compare on the quantity that
actually drives the MDP — the **lifespan** (time from pristine to failure) — in terms of **mean** and
**coefficient of variation (CV)**, and how does the condition trajectory look over time? And: **what
`alpha0` (and `e_fail`) would make the current Gamma model reproduce the discrete model's
failure-timing uncertainty**, versus what `generate_instance.py` uses by default?

## The two models

**Continuous (this codebase, `env/degradation.py::gamma_step`).**
`ΔG ~ Gamma(shape = α₀·Δt, rate = β)`, `β = α₀·e_fail`, `d_{t+1} = min(1, d_t + ΔG)`; failure at
`d ≥ 1`. Lifespan = first epoch `d ≥ 1` from a pristine asset (`d=0`). `α₀` is the *smoothness knob*:
it sets within-asset increment variance **without changing the mean degradation rate** `Δt/e_fail`.

**Discrete 5-state "doubling-hazard" chain.**
States `1=good, 2, 3, 4=imminent collapse, 5=collapsed` (state 5 is absorbing; the analogue of
`d ≥ 1`). Each epoch the advance probability **doubles per state**: `p = [X, 2X, 4X, 8X]` for
transitions `1→2, 2→3, 3→4, 4→5`. `X` is calibrated to a target mean lifespan. For the plot the
discrete state is mapped to the continuous scale by **`state / 5`** (so collapse = `5/5 = 1.0`).

**Common calibration.** Target mean lifespan `E[life] = 60 yr` (= `e_fail_mean` default), `Δt = 0.5 yr`.

## Method

Analytic where closed forms exist, Monte-Carlo (50 000 paths) otherwise. The Gamma side uses the
**actual simulator increment** (`env.degradation.gamma_step`, same inverse-CDF transform), so the MC
includes the real `min(1,·)` cap and discrete-epoch observation. Script:
[`scratch/degradation_model_comparison.py`](../scratch/degradation_model_comparison.py); figure
`scratch/degradation_model_comparison.png`.

## Headline numbers (E[life] target = 60 yr, Δt = 0.5 yr)

| Model / calibration | Mean lifespan | **Lifespan CV** | Notes |
|---|---|---|---|
| **Discrete 5-state doubling** | 60.0 yr | **0.608** | `X=0.0156`; CV essentially fixed by the doubling structure (→ 0.6146 as `X→0`) |
| **Gamma @ α₀ = 0.8** (current default) | 60.9 yr | **0.142** | ~4.3× *less* lifespan variability than the discrete chain |
| Gamma @ α₀ = 0.05 (old "noisy" default) | 70.0 yr | 0.489 | mean already inflated above 60 (see below) |
| Gamma, **CV-matched** (α₀=0.025, e_fail kept 60) | **79.9 yr** | 0.608 | matches CV but **mean balloons to 80 yr** |
| Gamma, **mean+CV matched** (α₀=0.033, e_fail=45.0) | 60.0 yr | 0.608 | reproduces the discrete lifespan distribution |

**Bottom line.** The discrete doubling chain is intrinsically *much noisier* (CV ≈ 0.61) than the
codebase's current default Gamma calibration (CV ≈ 0.14). Reproducing the discrete model's
failure-timing uncertainty drives `α₀` back down by ~30× into the old noise-dominated regime.

## 1. Discrete model — closed form

Holding time in state `k` is Geometric(`p_k`) with mean `1/p_k` epochs and variance `(1−p_k)/p_k²`.
Lifespan = sum of the four independent geometrics, so (in epochs):

```
mean = Σ 1/p_k = (1/X)(1 + ½ + ¼ + ⅛) = (15/8)/X
var  = Σ (1−p_k)/p_k²
CV²  = var / mean² = (85 − 120·X) / 225
```

Calibrating the mean to `E[life]` years gives `X = Δt·(15/8)/E[life]` (= 0.015625 for 60 yr, Δt=0.5),
and then **`CV = 0.6078`**. Crucially, `CV → √(85/225) = 0.6146` as `X → 0`, so **the discrete CV is
basically a structural constant ≈ 0.61 regardless of the calibrated lifespan**. The doubling means
the asset spends most of its (highly variable, geometric) life waiting in the early "good" states —
that early-state dwell-time dominates and fixes the CV.

## 2. Gamma model — first passage, and why the mean inflates

For the continuous Gamma process, `G(t) ~ Gamma(α₀·t, rate β)`, so the survival function is exact:
`S(t) = P(T>t) = P(G(t)<1) = gammainc(α₀·t, β)` (regularized lower incomplete gamma). Integrating,
`E[T] = ∫S dt`, `E[T²] = ∫2t·S dt`. Two facts fall out:

- **CV depends on `β = α₀·e_fail` alone** (time-scaling: scaling `(α₀, e_fail) → (α₀/s, s·e_fail)`
  scales the lifespan by `s`). A renewal approximation gives `CV ≈ 1/√(α₀·e_fail)`, but it
  *overestimates* in the noisy regime; the exact integral is the reference.
- **The mean first-passage time exceeds the nominal `e_fail`.** `β = α₀·e_fail` targets the *mean
  accumulated damage* reaching 1 at `t = e_fail`, but the *crossing-time* mean is larger
  (`E[T] = e_fail·g(β)`, with the inflation factor `g→1` only as `β→∞`). At α₀=0.8, `g≈1.01`
  (60.6 yr); at α₀=0.05, `g≈1.17` (70 yr); at the CV-matched α₀≈0.025, `g≈1.33` (80 yr).

This is why you **cannot** match the discrete CV by lowering `α₀` alone without also distorting the
mean.

## 3. How to tune the Gamma model to the discrete CV

**If you only need the CV to match** (and accept a longer mean life): hold `e_fail = 60` and set
**`α₀ ≈ 0.025`** (exact continuous) — the renewal rule of thumb `α₀ ≈ 1/(CV²·e_fail) = 0.045` is in
the right ballpark but optimistic. This gives CV = 0.61 but mean ≈ 80 yr.

**To match mean AND CV** (reproduce the discrete lifespan distribution): solve for `β` from the CV,
then pull the mean back with `e_fail`:

```
β = α₀·e_fail ≈ 1.48   (sets CV = 0.61)
e_fail = 60 / g(β) ≈ 45.0 yr   (g ≈ 1.33 here)
α₀ = β / e_fail ≈ 0.033
```

i.e. **`α₀ ≈ 0.033`, `e_fail ≈ 45.0 yr`** makes the Gamma model's lifespan mean (60.0 yr) and CV (0.608)
both coincide with the discrete chain — panels A–C of the figure show the distributions and
mean±(10–90%) condition bands essentially overlay.

## 4. Line-up with `generate_instance.py` defaults

| `generate_instance.py` default | value | implied lifespan CV | vs. discrete (0.61) |
|---|---|---|---|
| `alpha0_mean` | **0.8** | **≈ 0.14** (from pristine) | ~4.3× *smoother / more predictable* |
| `e_fail_mean` | 60 yr | — | mean matches (Gamma mean ≈ 60.9 here) |
| `alpha0_mean` (old) | 0.05 | ≈ 0.49 | closer, but still below 0.61, and mean inflated to 70 yr |

So the **current default instance is deliberately far more anticipatable than a discrete 5-state
doubling chain would be.** This is consistent with — and quantifies — the earlier predictability work
([`docs/i10p_predictability_analysis.md`](i10p_predictability_analysis.md)): `alpha0_mean` was raised
0.05→0.8 precisely to move from a noise-dominated (CV≈0.6, ≈ discrete-chain regime) to an
anticipatable instance (CV≈0.2 at mid-life, ≈0.14 from pristine). A discrete-state model with this
doubling structure would land back near the *old* noisy regime, not the current default.

### Practical note for the paper
- The CV mismatch is the substantive difference, not the mean — both models can be calibrated to any
  target mean lifespan. If the discrete chain is intended as a comparison/ablation, either (a) match
  the Gamma model to it via `α₀≈0.033, e_fail≈45.0` (joint match above), or (b) keep `α₀=0.8` and
  report the deliberate contrast: continuous model = anticipatable, discrete doubling chain =
  reactive-only — which is itself a clean experimental axis.

## 5. End-of-life: how predictable is the *remaining* lifespan?

Second figure: `scratch/degradation_eol_analysis.png` (panels E–G). Conditional remaining-life MC
(20k paths) from each observed condition. **Crucial mechanic first:** renovation **freezes**
degradation the epoch it starts and risk cost needs `h==0` (`env/mdp.py`: `d_new = where(h>0, d, …)`,
`still_failed = (d>=d_fail) & (h==0)`), so **an asset under renovation cannot fail.** Therefore
*failure avoidance* is a **~1-step** problem — renovate before `d` crosses 1 *next* epoch — **not** a
"start one renovation-duration ahead" problem. The 8–26-epoch renovation duration matters for *cost*
(traffic disruption, staggering concurrent jobs), i.e. **scheduling**, not for failure-avoidance lead.

| condition | E[rem] | CV(rem) | **P(collapse next epoch)** | P(collapse ≤16 ep) |
|---|---|---|---|---|
| **Discrete** state 1 (c=0.2) | 59.7 yr | 0.60 | **0.000** | 0.004 |
| Discrete state 2 (c=0.4) | 27.7 yr | 0.64 | **0.000** | 0.069 |
| Discrete state 3 (c=0.6) | 11.9 yr | 0.73 | **0.000** | 0.408 |
| Discrete state 4 *imminent* (c=0.8) | 4.0 yr | 0.94 | **0.123** | 0.881 |
| **Gamma α₀=0.8** d=0.8 | 12.9 yr | 0.30 | **7e-6** | 0.108 |
| Gamma α₀=0.8 d=0.9 | 6.9 yr | 0.39 | **1e-3** | 0.711 |
| Gamma α₀=0.8 d=0.95 | 3.9 yr | 0.49 | **0.020** | 0.976 |
| Gamma α₀=0.8 d=0.99 | 1.4 yr | 0.62 | **0.262** | 1.000 |
| **Gamma matched** (α₀=0.033) d=0.8 | 21.8 yr | 0.81 | **0.015** | 0.252 |
| Gamma matched d=0.95 | 12.4 yr | 0.89 | **0.033** | 0.455 |

Three findings:

1. **Remaining-life timing uncertainty rises toward EOL in *every* model** (panel E): the closer to
   failure, the *fewer* noisy increments are left to average, so CV(remaining) grows
   (discrete 0.60→0.94; Gamma α₀=0.8 0.20→0.62; matched 0.70→0.90). "Watch it closely as it nears
   failure" does **not** buy sharper timing — the opposite.
2. **But the discrete chain has a guaranteed, observable warning** (panel G): `P(collapse next
   epoch)=0` *exactly* from every state below the "imminent-collapse" state — you **cannot** skip a
   state. So a trivial policy "renovate iff in state 4" (or state 3 for buffer) **never** eats a
   surprise collapse, regardless of the high CV. The high timing-CV is harmless because you act on a
   *categorical flag*, not on a timing forecast.
3. **The Gamma model has no such flag** — `P(collapse next epoch)` is a smooth ramp with no `d` below
   which it is exactly zero. At **α₀=0.8** the ramp is steep and a threshold around `d≈0.9` is very
   safe (P(next)≈0.001), so a good just-in-time policy *exists* and the model is in fact **more**
   predictable than the discrete chain for most of life (CV 0.2–0.4). At the **noisy/matched α₀≈0.03**
   the single-step crossing is already 1–3% from a *moderate* `d=0.7–0.9` → assets fail abruptly from
   mid-condition and **no safe threshold exists**.

## 6. Why might RL performance have dropped after the discrete→Gamma switch?

The data above narrows it to **one decisive check, then two regimes.** (None of this required running
the agents; all are cheap diagnostics on the instance + existing eval logs.)

**Check first — which noise regime is your Gamma instance in?** Open the instance JSON and look at
`alpha0` (and compute the lifespan CV / single-step crossing with this script).

- **If α₀ is small (≈0.03–0.1, e.g. the old 0.05 default):** the Gamma model genuinely allows
  **abrupt failures from a moderate condition** (P(collapse next epoch) ≈ 1–3% already at d≈0.7–0.9)
  and there is **no `d`-threshold that is both safe and not wasteful** — anticipation is impossible and
  the escalating risk cost `C_risk` punishes every surprise. This is a *real* increase in problem
  difficulty and the most likely culprit if you regenerated instances at low α₀. **Fix:** raise
  `alpha0_mean` toward 0.8 (the smoothness knob — same E[life]) and re-evaluate.

- **If α₀ ≈ 0.8 (current default):** the Gamma model is *more* anticipatable than the discrete chain,
  so the regression is almost certainly **representational / optimization**, not predictability:
  1. **Lost the categorical trigger.** Discrete gave a 5-value sufficient statistic where the optimal
     policy is a lookup ("renovate in state 4"). The Gamma agent must learn an *implicit threshold on a
     continuous `d`*, and small value-function error near the boundary turns into either wasted life or
     catastrophic (escalating-`C_risk`) failures. Tree/MLP approximation of a sharp threshold on a
     continuous axis is genuinely harder than a 5-state table.
  2. **Continuous state × escalating risk = a cost cliff.** `C_risk` grows with `n_fail`; a slightly
     mistimed threshold on continuous `d` falls off that cliff. Check value-function **calibration**
     (predicted vs realized cost-to-go R²) and whether the agent **under-** vs **over-maintains** vs the
     reactive baseline (failure count / total `C_risk` vs renovation spend).
  3. **Renovation representation.** Because `d` is *frozen* (not reset) during renovation, the value
     function sees high `d` with `h>0` (safe) and must learn it differs from high `d` with `h==0`
     (dangerous). Worth testing: feed an "effective condition" `d·(h==0)`, or reset `d→0` at renovation
     *start* (your intuition) — it cannot change cost/dynamics (the frozen `d` feeds nothing but the VFA
     input) but it **declutters the value-function's job**.
  4. **Scale / normalization.** Euro-scale cost-to-go (~1e11) collapses an un-normalized MLP value
     function (see CLAUDE.md / `NeuralValueFn`). Confirm input+target normalization is active for the
     Gamma instance.
  5. **Stale tuning.** Warmstart heuristic, `init_action`, thresholds, learning rates may have been
     tuned for the discrete dynamics — re-tune on the Gamma instance.

**Fastest triage:** on the Gamma instance, log per-episode failure count, total `C_risk`, and total
cost CV/P90 for your agent vs a tuned reactive heuristic. If `C_risk`/failures dominate → surprise-
failure problem (regime A, raise α₀, or the threshold-learning problem 6.1–6.3). If renovation spend
dominates → the agent over-maintains (mis-set threshold / cost weighting). Also histogram the realized
single-epoch `d` jumps that crossed into failure: many crossings from `d<0.85` ⇒ you are in the noisy
regime.

## Caveats

- **`state/5` floor.** Mapping the discrete "good" state to `1/5 = 0.2` means the discrete trajectory
  starts at 0.2 while the Gamma asset starts at `d=0`. This is an inherent discretization offset; an
  alternative `(state−1)/4` would start both at 0 but put collapse at the wrong place relative to the
  4 transitions. The plot follows the requested `÷5` convention and the curves are compared on shape.
- **Lifespan CV is ~Δt-independent.** It is governed by `β = α₀·e_fail` (Gamma) / `X` (discrete);
  `Δt` only changes discretization granularity, not the lifespan CV.
- All Gamma numbers are from a **pristine** start (`d=0`, full-life CV). Starting mid-life (`d=0.5`)
  raises the CV (less headroom to average over) — e.g. α₀=0.8 gives CV≈0.20 from `d=0.5`, matching the
  `generate_instance.py` docstring.

*Reproduce:* `python scratch/degradation_model_comparison.py` → prints both reports and writes
`scratch/degradation_model_comparison.png` (lifespan, §1–4) and
`scratch/degradation_eol_analysis.png` (end-of-life, §5).
