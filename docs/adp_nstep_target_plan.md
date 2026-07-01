# Plan (not yet built): n-step ADP target + do-nothing warmstart mix

Status: **design only** — approved to plan, not to implement/run (2026-06-22). Feeds a *future*
ADP iteration; cannot be retrofitted to the completed Exp-0B ADP run.

## Context / why

Exp-0B ADP underperforms: best cell `normal_policy_fifo_xgb` = **1991M** vs the 795M heuristic /
746M rollout (~2.5× off). The value-fn/degradation deep-dive (`docs/adp_value_fn_improvements.md`,
`docs/degradation_model_comparison.md`) found this is **not** a hard ceiling:

- Measured by the right yardstick (**action-ranking, not r²**), V' ranks actions decently
  (Spearman 0.62–0.68, top-1 63–77%); r²_within ≈ 0.05 badly understates decision usefulness.
- **The full-horizon MC target is the bottleneck.** Within-epoch R² ≈ **0.52 at an n-step target
  (n ≤ 4)** vs ~0 at the full horizon. A short/bootstrapped target is the single biggest untried
  lever, and is neutral-to-faster to train.
- On *noisy* instances a separate −74% under-prediction bias is a **warmstart-coverage** problem,
  fixed by mixing **~25% pure do-nothing episodes** (expensive-state R² −4.6 → +0.69; n_fail
  coverage ↑~1000×). Reactive warmstart policies never let `n_fail` accumulate, so V' never sees
  failed states. (Current `instance_10p` is predictable, α₀=0.8, where calibration is already
  fine — so this knob mainly matters if/when noisy instances are revisited.)

**Caution that must shape validation:** twice now an *offline proxy* improved while the *deployed
full-run policy* got worse (advantage_baseline lifted offline ranking but pushed NN 1444M→2971M).
So action-ranking is necessary-but-not-sufficient; **the full-run eval is the arbiter.**

## Change 1 — n-step / bootstrapped ADP target (the main lever)

**Current target** (`agents/dqn.py` `ADPAgent.update`, `training/trainer.py` backward MC pass):
`mc_return_t = cost_t + γ·mc_return_{t+1}` (full horizon), V' trained on post-decision features with
`y = mc_return − cost` (future-only, optionally minus the per-epoch advantage baseline b(t)).

**Proposed:** replace the full-horizon return with an **n-step truncated return that bootstraps off
V'** at the truncation point:
`y_t^(n) = Σ_{k=1..n} γ^{k-1}·cost_{t+k} + γ^n·V'(post_state_{t+n})` (post-decision indexing — exact
γ exponents to be pinned in code), then minus b(t) as today.

Design decisions:
- **Where:** compute in the trainer **backward pass at collection time**, using the agent's current
  V' snapshot to bootstrap (stable, no moving-target refit loop). Near the episode end (< n steps
  left) fall back to the available horizon (≡ full return for the tail).
- **Knob:** `agent.extra.n_step` (int; default `0`/`None` ⇒ full horizon = **current behavior**).
  Suggested screen value **n = 3–4** (from the within-epoch-R² curve).
- **Warmstart:** V' is untrained during warmstart → bootstrap term = 0 (n-step degenerates to the
  available-horizon return), same as today.
- **Invariant / test:** `n_step` large (≥ T+tail) must reproduce the current full-horizon target
  **bit-identically** (guard with a unit test); `b(t)` advantage baseline still applies on top.
- **Touch points:** `training/trainer.py` (`_run_train_episode` + `_train_sequential` backward
  passes, and `_run_warmstart` MC pass), `agents/dqn.py` (`ADPAgent.update` already consumes
  `tr.mc_return`; keep that interface — populate it with the n-step value), `experiments/configs.py`
  (read `n_step`), `training/buffer.py` (no change — `Transition` already stores cost/post_state).

## Change 2 — do-nothing warmstart mixture (coverage; noisy-instance fix)

**Where:** superseded — implemented as the `MixtureAgent` warmstart (`agents/mixture.py`,
`agent_type: 'mixture'`); acting-flip coverage is the `explore_flip` heuristic. (Original note: add a
per-episode coin flip: with prob `p_donothing` (~0.25) run the **entire** warmstart
episode as all-zeros (do-nothing), so V' sees high-`n_fail` failed states the reactive policy never
reaches.

- **Knob:** `training.warmstart_donothing_frac` (float, default `0.0` ⇒ current behavior). Screen
  value **0.25** for noisy instances.
- **Reproducible:** keyed on `self.seed` like the existing exploration flips.
- **Recipe (deep-dive):** per-episode sampled-threshold reactive + acting flips + ~25% do-nothing.
  (Sampled-threshold reactive is optional extra diversity; the do-nothing fraction is the part that
  actually moved the bias.)

## Validation (before any Snellius run)

1. **Offline action-ranking** (`scratch/test_action_ranking.py`): Spearman, top-1, and **decision
   regret** + bias — NOT r²_within. Compare n-step vs full-horizon target.
2. **Within-epoch R² curve** vs n (`scratch/test_horizon_targets.py`) to confirm the n choice.
3. **Short full-run smoke** (one cell, reduced `time_budget`): the policy-eval curve must actually
   descend toward the heuristic — the full-run eval is the arbiter, given the offline/full-run
   disconnect above. Only schedule the Snellius grid if the smoke trends right.

## Open questions for whoever implements

- n-step is validated mostly on the **noisy** regime (within-epoch R² 0.52 @ n≤4). Confirm it also
  helps on the **predictable** `instance_10p` (α₀=0.8) where full-horizon R² is already 0.83 — the
  lever may be smaller there. Run `test_horizon_targets.py` on the predictable instance first.
- Does the advantage baseline interact with the n-step bootstrap (both reshape the target)? Consider
  an ablation grid cell with `advantage_baseline:false` + n-step, since advantage_baseline hurt the
  NN full-run.
