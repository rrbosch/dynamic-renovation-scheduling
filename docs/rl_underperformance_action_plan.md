# RL underperformance — diagnostics & action plan (sf15)

Working checklist. Each item = one diagnosed cause or agreed fix. We tackle them one at a time and
tick the box when done. Companion to [EXPERIMENTS.md](../EXPERIMENTS.md); numbers below are recomputed
from `results/exp0/Code_v2/results/exp0/*/eval_episodes.csv` (50 shared-CRN eval eps, γ_pe = 0.97^0.5).

## Evidence baseline (discounted cost, M€)

| Run | mean | p50 | max | travel | maint | **risk** | failure signature |
|---|---|---|---|---|---|---|---|
| **perasset heuristic** | **3110** | 3037 | 5345 | 895 | 2206 | **10** | — (baseline) |
| adp2_ctrl_emptyinit | 3424 | **2908** | 17357 | 988 | 1973 | 463 | median-wins, one 17B cascade tail |
| seq_rollout_empty | 3489 | **2949** | 8021 | 807 | 1487 | **1195** | defers maint, pays risk cascades |
| rollout_empty / policy | 3504 / 3547 | 3099 / 3307 | ~8018 | ~800 | ~1485 | **~1240** | defers maint, pays risk cascades |
| dcl_seq_xgb_vfa | 3768 | 3770 | **4613** | 1380 | 2296 | 92 | uniform +21%, tighter tail than heuristic |
| dcl_independent_nn | 4455 | 3367 | 12432 | 2343 | 2027 | 85 | under-trained |
| adp2_ctrl_seq | 19928 | 14224 | 78343 | **13345** | 6335 | 248 | collapsed → over-acts, travel blow-up |
| adp2_xgb_abon_ns4 | 44050 | 44165 | 98713 | **36716** | 7334 | 0 | collapsed → over-acts, travel blow-up |
| ppo_curriculum | 196111 | 199137 | 220752 | **189301** | 6809 | 0 | never trained on travel cost |

**One-line story:** the learners win the *median* episode but lose the *mean* through rare
risk-cascade (rollout/ADP-empty) or congestion-explosion (collapsed ADP/PPO) episodes. The heuristic's
edge is tail control, not average decision quality.

## Status board

| # | Item | Type | Status |
|---|---|---|---|
| 1 | Cap renovation duration (decouple from avg-sim target) | instance redesign | ✅ tested (`instance_sf20`) |
| 2 | Rollout truncation bias (V=0 past horizon) | fixed by #1? | ☐ |
| 3 | ADP collapse (flat V′, no exploration) | needs fix | ☐ |
| 4 | PPO curriculum: 0-travel → naive-additive travel | code | ✅ done |
| 5 | Early-stopping criterion | code | ✅ done (base Trainer) |
| 6 | Decouple evaluation from training budget (post-mortem eval) | infra | ✅ done (Trainer + DCL) |
| 7 | Report CVaR / P90 (not just mean) | reporting | ✅ done |
| 8 | Non-anticipative (SDP) lower bound | new workstream | ☐ |

---

## 1. Renovation duration is 22.7 yr — cap it to a few years ☐

**Diagnosis.** The instance builder makes **renovation duration the primary calibrated quantity**:
`calibrate_e_ren` binary-searches `e_ren_years` so a renovate-at-failure workload hits
`--target-avgsim 3.5`, then *derives* physical length from it
([build_synergy_instance.py:257-269](../experiments/build_synergy_instance.py)):
`lengths_mean = 5·(e_ren·52 − 10)`. The governing identity in steady state is

```
avg_simultaneous_renovations  ≈  N · e_ren / e_fail
```

With N=15 and e_fail=100 yr, hitting avg-sim 3.5 *forces* e_ren ≈ 22.7 yr (≈45 epochs) and, via the
derived length, an absurd **mean asset length of 5.4 km** and c_ren ≈ 271 M€. Consequences:

- A renovation spans **~45 of the 190 eval epochs** → only ~1 cycle fits in the horizon → the "end of
  the tunnel" is never in view, so *every* method (rollouts, MC targets, PPO curriculum) is reasoning
  about an effectively-infinite tail from truncated information.
- Renovation at η_ren=0.05 (95 % capacity loss) held for 22 yr makes any *cluster* of simultaneous
  renovations catastrophic → the fat travel/risk tails that sink the mean.

**Decision (agreed).** Cap renovation duration to a realistic **~2–4 yr** (not 40+). This also fixes the
5.4 km lengths (they drop to ~700 m — realistic) since length is derived from e_ren. Accept the
necessary evil: to preserve avg-sim ≈ 3.5 we must **lower e_fail** (shorter lifespans → several
renovation cycles per episode) or **raise N**. Per the identity above, for e_ren=3 yr and avg-sim 3.5:
e_fail ≈ N·e_ren/3.5 → e_fail≈13 yr at N=15, or N≈70 at e_fail=60 yr.

Multiple cycles per episode is a *feature* for learning: shorter credit horizon (γ^6 vs γ^45), the full
renovate→reset→re-degrade loop is observed in-horizon, MC-target variance shrinks relative to the
per-action signal, and rollout windows span many cycles. This is expected to help learning **immensely**
and is the single highest-leverage change.

**Open questions to resolve when we do #1.**
- Which knob absorbs avg-sim: shorter e_fail (keep N=15, small portfolio) vs larger N (keep realistic
  60-yr lifespan)? Larger N also makes the joint-action problem harder — could be good or bad for the
  paper's narrative.
- Shrinking length shrinks c_ren, c_rep and c_risk (all ∝ L) but **not** travel (∝ capacity, not L) →
  tilts the balance toward travel-dominated. Per [[gap_equals_travel_ceiling]] that *raises* the gap;
  re-tune risk_base / vot / tcf so the gap and cost balance stay sane.
- Requires regenerating the instance → **re-run 0A (heuristic tuning) and all of 0B**. Budget for it.
- Decouple option: we could add a `--max-e-ren-years` cap flag and let length be set independently,
  rather than the rigid `L ↔ e_ren` coupling. Cleaner and more honest.

**Result — `instance_sf20` built & validated (2026-07-10).** Chose N=20, e_ren≈3 yr, e_fail≈14 yr
(kept avg-sim≈3.3 via the identity `avg-sim ≈ N·e_ren/(e_fail+e_ren)`; the builder's own
`--target-avgsim` proxy is single-failure and invalid for short lifespans, so length/e_fail were set
directly and avg-sim verified empirically). Builder cmd + full details in [[instance_sf20_redesign]].

| Property | sf15 | **sf20** |
|---|---|---|
| N | 15 | 20 |
| renovation duration | 22.7 yr (45 ep) | **3 yr (6 ep)** |
| mean asset length / c_ren | 5421 m / 271 M | **683 m / 34 M** |
| cycles per asset (horizon) | ~1 | **~5** |
| realized avg-sim | ~2.7 | 2.93 |
| **travel fraction (gap ceiling)** | 81 % | **79 %** |

Gap (20 CRN eps, seed 42, disc γ_pe=0.97^0.5):

| agent | mean disc | gap vs clairvoyant |
|---|---|---|
| clairvoyant (floor) | 1653 M (CV 0.03) | — |
| **reactive (tuned)** | **4806 M** | **61 %** |
| per-asset (tuned, laptop) | 5130 M | 62 % |
| renovate@0.95 (untuned) | 6484 M | 71 % |

**Verdict: the redesign works.** The anticipation gap survived (**~61 % vs the reliable reactive
baseline**) while renovations became realistic and there are now ~5 learning cycles per asset in-horizon
— exactly the regime that should help #2 and #3. Slightly below sf15's reactive gap (76 %), expected
because a 3-yr mis-scheduled cut locks in less congestion than a 22-yr one; still comfortably large.
**Caveat:** the 60-dim per-asset heuristic does **not** converge at laptop tuning budget (445 trials,
only 4 completed — WilcoxonPruner over-prunes; its `best_value` overfits the T-tuning episodes so it
generalizes *worse* than the 3-param reactive). On sf15 per-asset needed a 24 h Snellius tune to become
the decisive baseline; expect the same here → the *true* gap vs a fully-tuned per-asset will land ~45–55 %
(cf. sf15's 41 %). **Next:** decide the operating point (N=20/e_fail=14 vs a bigger portfolio), then
re-run 0A + 0B on `instance_sf20`.

**Decision (2026-07-10): LOCKED IN — N=20 / e_fail=14 is the sf20 subject.** The full per-asset tune +
0A/0B re-run are deferred; next work is items 6→5→7 (cheap, instance-independent infra/reporting).

---

## 2. Rollout truncation bias — likely fixed by #1 ☐

**Diagnosis.** Confirmed real: with `rollout_horizon: 100`, the return is **cut at V=0** — no terminal
value ([rollout.py:249-255](../agents/rollout.py); the γ^(K+1)·V bootstrap at
[rollout.py:522-527](../agents/rollout.py) exists only in the DCL VFA path). At the cut only γ^100 =
0.218 of present value remains, and the 100-epoch window holds barely **one** 45-epoch renovation cycle.
So "wait" pushes renovation cost deeper into the discount and shoves the *second* cycle + late-escalating
risk past the horizon → each deferral looks marginally free → rolling-horizon procrastination. This is
exactly the data: rollouts spend 33 % less on maintenance than the heuristic and pay **120× the risk**
(1.2 B vs 10 M). It is **bias, not noise** (Wilcoxon budgeting itself is sound: p=0.02, 20/100, paired CRN).

**Answer to "fixed by #1?": mostly yes.** Once a renovation is ~6 epochs, a 100-epoch window spans
~15 cycles → the truncation weight γ^100 covers well past the decision-relevant future, the "end of
tunnel" is visible, and deferral no longer hides cost. Recommend: **do #1 first, then re-measure the
rollout risk fraction.** If a residual bias remains, add the cheap terminal value at the cut (the DCL
VFA-bootstrap code path shows how) or set `rollout_horizon` close to T+tail with adaptive budgeting
absorbing the cost. Keep this box open until re-measured.

---

## 3. ADP collapse — proposal ☐

**Diagnosis (two regimes).**
- `ctrl_emptyinit` is **not** collapsed — it converged (training log 27 B → 3.4 B over 7250 eps) and
  *beats the heuristic's median* (2908 vs 3037). It fails only on the #1 cascade tail (one 17 B episode).
- The `seq` / `policy-init` / `abon_ns4` cells **are** collapsed (20 B–44 B, travel-dominated, risk≈0 →
  over-acting with terrible clustering). Root causes (from the audit, consistent with the i10p backlog):
  (a) warmstart buffer ~98 % do-nothing → V′ never sees acting post-states; (b) MC target
  `mc_return − cost` over 190 epochs has residual noise ≈ the per-action signal (~271 M);
  (c) **no online exploration** — collection is pure greedy argmin, so a bad V′ self-reinforces
  (garbage policy → garbage data → garbage V′); (d) `b(t)` advantage baseline is frozen at warmstart and
  goes stale the moment the policy shifts.

**Proposal (fixes only — no variant dropped).**
1. **Re-run after #1.** Shorter cycles → lower-variance MC targets and natural acting-post-state
   coverage. Much of (a)/(b) is downstream of the 45-epoch renovation. Cheapest high-value step.
2. **Add online exploration** to the collection loop (ε-greedy or softmax over the ranked Q), with an
   explicit rng for reproducibility. This is the missing structural piece that lets a cell *escape* a
   bad V′. Currently absent entirely.
3. **Unfreeze `b(t)`**: refresh the per-epoch advantage baseline every K updates (or use a rolling
   estimate) so it tracks the policy distribution instead of the warmstart distribution.
4. **Diagnose the collapsing cells by action-ranking, not r²** (per [[project_adp_valuefn_backlog]]):
   log Spearman / top-1 of V′ on held-out acting states each eval so we can see *when* seq/policy-init
   diverge, rather than only the final cost.
5. Keep all grid cells; report the collapsed ones honestly (they are a finding: "MC-ADP targets can't
   rank act-vs-wait on the long-cycle instance").

**Open question.** Does `init_action='policy'` recover once V′ is informative (post-#1 + exploration),
or is seeding local search from the heuristic action structurally worse here? Ablate after #1.

---

## 4. PPO curriculum: replace 0-travel with naive-additive travel ✅

**Diagnosis.** Phases 0/1 run on a curriculum env built with **`traffic_cost_factor=0`**
([configs.py:439](../experiments/configs.py)) + NullTAP, so PPO **never receives a gradient involving
travel cost** — which is **96 % (189 B / 196 B) of its eval cost**. Compounded by a graduation gate that
needs *both* a plateau *and* beating the phase-0 baseline, with no episode cap
([ppo_trainer.py:129-138](../training/ppo_trainer.py)): 435 000 episodes, all stuck in phase 1. Its 63×
number measures a curriculum design flaw, not PPO-on-this-problem.

**Decision (agreed): make the curriculum travel cost "naive-additive" instead of zero.** Give each
renovation/restriction its *independent* marginal travel impact, summed over active assets, ignoring the
super-additive congestion synergy (which is exactly what phase 2 then adds).

**Proposed implementation.**
- Precompute once, per asset i: `ct_ren[i] = travel_cost(only asset i under renovation, η_ren)` and
  `ct_res[i] = travel_cost(only asset i restricted, η_load)` — 2N TAP solves total. (The clairvoyant's
  travel-free decomposition `LB0` already builds per-asset marginals — reuse it if possible.)
- Curriculum env overrides `travel_cost(s_post) = Σ_i ct_ren[i]·1{h_i>0} + ct_res[i]·1{ell_i=1}`.
  Purely additive, no TAP in the loop, no synergy.
- Effect: PPO learns in phase 1 that "renovations aren't free" and roughly *how* costly each is, so
  phase 2 is a moderate step up (adding synergy) instead of a 0→full cliff.

**Also fix the gate regardless:** add an episode/time cap that force-graduates phase 1 → phase 2, and
reconsider whether "beat the phase-0 baseline on the *simplified* env" is even the right criterion. Until
the gate is fixed, PPO's number should not be interpreted.

**Open question.** The naive-additive cost *under*-estimates true travel (no synergy) — do we scale it
(e.g. ×1.2) to soften the phase-2 jump, or leave it honest?

**Implemented (2026-07-10).** Both fixes landed and are verified on sf20.
- *Naive-additive travel:* `env/mdp.compute_additive_travel_lut(env)` precomputes per-asset marginal
  single-closure costs (`ct_ren`/`ct_res`, ~2N TAP solves on the real env). `InfraEnv._additive_travel`
  (opt-in attr) makes `travel_cost`/`_compute_cost` return the **sum of active assets' marginals** (no
  synergy, no TAP-in-loop). `configs.py` computes the LUT on the real env and attaches it to the
  curriculum env (`training.curriculum_additive_travel`, default **on**); the real env is untouched.
  Verified on sf20: additive = Σ marginals (5.0M), real-with-synergy = 6.8M → **1.34× super-additive**
  (curriculum captures ~74% of the true cost; Phase 2 adds the synergy premium). Pristine ⇒ 0.
- *Gate fix:* `PPOConfig.curriculum_phase_budget_frac` (default 0.5) force-graduates Phases 0+1 once
  they have used that fraction of `time_budget`, guaranteeing Phase 2 (real env) always runs; plus an
  optional `curriculum_phase1_max_episodes` cap. Verified: on a 90s run with frac=0.4, Phase 1
  force-graduated at ~36s (never beating the 2168M baseline — the exact old infinite-loop condition)
  and Phase 2 ran to budget. Full suite (111) green.
- *Phase-0 warm-start rework (companion):* `curriculum_phase0_mode` (default **`"bc"`**). The old Phase 0
  ran `update_ppo` on heuristic trajectories — advantage-weighted imitation through a cold critic, so it
  never actually started near the heuristic (the sf15 "50× worse than its own warm-start" symptom). The
  new `"bc"` mode (`PPOAgent.update_bc`) trains the **actor** with a direct per-asset cross-entropy to the
  (deterministic) heuristic action and the **critic** with the normal standardized-return regression —
  warming *both* heads. The actor target is env-agnostic; the critic term is what makes the additive-travel
  curriculum (above) pay off (it warm-starts a value fn on a realistic cost, ~1.34× low on congested
  states vs the real env — corrected fast in Phase 2). A small entropy bonus prevents deterministic
  collapse. Verified on sf20: BC loss 16.2 → 6.6 over 20 episodes (~72% mass on the heuristic action/asset),
  critic MSE stable. `"ppo"` mode retained for A/B. sf20 PPO config uses `"bc"`.

---

## 5. Early-stopping criterion ✅

**Decision (agreed).** Longer training is acceptable; add early stopping to avoid wasting the 24 h budget
once converged.

**Proposal.**
- **Value-based / rollout / PPO:** patience on a *cheap* periodic eval signal (small #eps, possibly
  shortened horizon) — stop when best-so-far mean cost hasn't improved by > tol over K consecutive evals,
  subject to a **minimum training-budget floor** (don't stop during warmstart / early churn).
- **DCL:** stop when round-over-round improvement < tol (it already logs per-round mean cost; the 2 we
  have went 6.98 B → 3.77 B and were still descending steeply → this is the family most starved by
  walltime, see #6). Also raise rounds-per-walltime once #6 frees the eval cost.
- Ties into #6: use a *cheap* in-loop eval purely as the stopping signal; the *authoritative* 50-ep CRN
  eval runs post-mortem.

**Implemented (2026-07-10, base Trainer).** `TrainingConfig.early_stop_{patience,tol,min_seconds,episodes,interval_seconds}`
(all default off). `Trainer._maybe_early_stop` runs a *cheap* wall-clock-gated eval (`early_stop_episodes`,
falling back to `n_eval_episodes`; cadence = `early_stop_interval_seconds` or `eval_interval_seconds`),
tracks the best mean cost, and breaks the training loop once it has not improved by > `early_stop_tol`
(relative) for `early_stop_patience` consecutive checks — never before `early_stop_min_seconds`. It runs
**even under `defer_eval`** (it is the stopping signal; the authoritative eval is post-mortem, #6). Unit-tested
(plateau → stops after N stale checks; `patience=0` → never). DCL is naturally bounded by `n_rounds`
(round-based early-stop is a trivial future add); PPO deferred to #4.

---

## 6. Decouple evaluation from training budget ✅

**Decision (agreed).** Don't let expensive-to-evaluate methods (rollout, DCL) burn their own training
time on eval. **Checkpoint (pickle) the agent periodically during training; run the authoritative eval
post-mortem** from the checkpoints after training finishes.

**Proposal.**
- During training, save `agent` + minimal env-reconstruction metadata at fixed **wall-clock** intervals
  (e.g. hourly) → `checkpoints/<run>/ckpt_<elapsed>.pkl`.
- A separate `evaluate_checkpoints.py` loads each checkpoint and runs the 50-ep shared-CRN eval offline
  (eval is a pure function of seed → fully reproducible, parallelizable, and identical to in-loop eval).
- Benefits: (a) training budget spent on training; (b) uniform cost-vs-wallclock curves across *all*
  methods for free; (c) the in-loop eval can shrink to just the #5 stopping signal.

**Open question.** Pickle-ability: rollout/DCL agents hold TAP/numba refs and (for rollout) persistent
pools. Confirm they pickle cleanly, or checkpoint only the learned artifacts (V′ / classifier / policy
net weights) + rebuild the agent shell at eval time (mirrors how workers already reconstruct env+TAP).

**Implemented (2026-07-10, Trainer + DCL).** Resolved the pickle question by checkpointing **only the learned
artifacts** via the existing `agent.save()` (XGBoost booster / torch state_dict / DCL classifier) — no buffer,
no TAP/pools. `TrainingConfig.snapshot_interval_seconds` (>0 = on) writes a **history-kept** agent-only
snapshot to `results/<run>/snapshots/t_<elapsed>s/` every N wall-clock seconds **and** once at train end;
`defer_eval` skips the in-loop eval so the budget goes to training (run.py then skips its hardcoded 50-ep
final eval and prints the post-mortem command). New **`experiments/evaluate_checkpoints.py`** reconstructs
env+agent from the config, `agent.load()`s each snapshot, runs the shared-CRN eval offline, and writes
`eval_curve.csv` (cost-vs-wallclock: mean/P50/P90/CVaR + component split) plus per-snapshot
`eval_summary.csv`. Same hooks added to `DCLTrainer` (per-round, wall-clock gated). **PPO excluded** — its
eval is *cheap* (policy forward pass, not rollout/DCL search) and its curriculum needs in-loop eval for
phase advancement; revisit under #4. Tested end-to-end on an sf20 ADP run (snapshots written under
`defer_eval`; `evaluate_checkpoints.py` produced the curve); full suite (111) green.

---

## 7. Report CVaR / P90, not just mean ✅

**Decision (agreed).** The whole story is in the tail — report it explicitly.

**Proposal.** Add to `utils/metrics.py` and surface in `comparison_dashboard.py` (per
[[feedback_visualization]]): mean, P50, **P90**, **CVaR@10 %** (mean of worst 10 % episodes), max, CV.
For sf15 this immediately reframes the result: several learners *win P50* and lose on CVaR — that is the
scientific finding, and it matches the earlier noisy-i10p arm.

**Implemented (2026-07-10).** `utils/metrics.py` gained `cvar(costs, alpha)` and `cost_summary(costs)`
(→ n/mean/p50/p90/cvar/std/cv/max). `comparison_dashboard.py`'s summary table now shows **P50, P90,
CVaR 10%, Max Ep** columns alongside mean over the pooled per-episode costs. The same `cost_summary`
powers `evaluate_checkpoints.py`'s post-mortem curve (#6), so in-loop and post-mortem reporting share
one definition. (The box-plot distribution chart already existed.)

---

## 8. Non-anticipative (SDP) lower bound ☐

**Decision (agreed).** Compute the information-respecting lower bound (the SDP / non-anticipative
counterpart to the clairvoyant, deferred per EXPERIMENTS.md §0c).

**Why it matters.** The clairvoyant floor (1824 M, gap 41 %) is a *wait-and-see* bound — it replays exact
noise, so it includes value-of-perfect-information. Nobody knows how much of the 1286 M gap is
*achievable* without clairvoyance. If the true non-anticipative optimum sits near ~3000 M, the tuned
heuristic is already near-optimal and "RL doesn't meaningfully improve" is the **correct result**, not a
failure — and the paper should say so. If it sits near ~2000 M, there is real room and the agent fixes
matter.

**Proposal (sketch).** Exact SDP is why it was deferred. Candidates: a per-asset SDP relaxation that
ignores congestion coupling (valid lower bound), coordinated via the same block-coordinate-descent /
travel lookup the clairvoyant already uses, but with **non-anticipative** per-asset value functions
(expectation over future noise, not the realized path). Scope this as its own workstream.

---

## Suggested order

**#1 first** (it's the anchor — likely also resolves #2 and much of #3), in parallel with the
cheap/independent wins **#6 → #5 → #7** (infra/reporting, no instance dependency), then **#4** (PPO
curriculum) and the **#3** ADP fixes on the regenerated instance, with **#8** as a parallel bounding
workstream to calibrate what "success" even means.
