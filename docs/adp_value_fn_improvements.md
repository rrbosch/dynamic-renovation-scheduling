# Improving the ADP agent via its value function — what we tried

A record of the experiments we ran to make the ADP agent perform better (lower cost) by
improving its value function `V'(s_post)`, what worked, what didn't, and the expected
training-speed cost of each. All evidence is offline analysis of the trained ADP runs
(`results/exp0*/i10p_adp_*_xgb`) plus a CRN-rollout action-ranking harness; no full
retrains yet. Scripts in §Reproduce.

Two instances recur: **predictable** (`exp0`, α₀=0.8) and **noisy** (`exp0_noisy_alpha0p05`,
α₀=0.05) — the same MDP at two degradation-noise levels (see
[degradation_model_comparison.md](degradation_model_comparison.md)).

---

## Key takeaways (do these, in order)

1. **Measure the value function by action-ranking, not by `r²`.** The trained V' ranks
   candidate actions well (Spearman **0.62–0.68**, picks the rollout-optimal action **63–77%**
   of the time) even though its within-epoch eval `r²` is ~0.05. `r²_within` badly
   *understates* decision usefulness. Use `scratch/test_action_ranking.py` as the yardstick for
   any change. *(Measurement change; no training-speed cost.)*

2. **Biggest performance lever: train V' on a short / n-step (bootstrapped) target, not the
   full-horizon return.** Near-term cost is highly state-predictable (within-epoch R² ≈ **0.52**
   at n≤4 on the noisy instance) while the full horizon is ~unpredictable (≈0). A short target is
   far more learnable and decision-relevant, *especially on noisy instances where the
   full-horizon V' is broken*. *Training speed: neutral-to-faster* (shorter rollouts + one
   bootstrap lookup vs a full Monte-Carlo return).

3. **`advantage_baseline` is now ON for all ADP configs** (was NN-only). The target is ~94%
   epoch-trend; subtracting `b(t)` focuses the fit on the controllable advantage and should cut
   the calibration bias. Mechanism is sound; **not yet validated on XGBoost** — confirm with the
   yardstick after the next runs. *Training speed: free* (a per-epoch add/subtract).

4. **The noisy-regime failure is a coverage problem, and the fix is warmstart NEGLECT episodes —
   not exploration tweaks.** V' there under-predicts future cost by **−74%** because the warmstart
   never visits *sustained-failure* (high-`n_fail`) states: the tuned heuristic renovates failures
   within ~1 epoch, so `n_fail` never accumulates. Tested (`test_warmstart_diversity.py`): exploration
   flips and broad initial conditions **don't help** (the corrective base policy pulls trajectories
   back; cost tail stuck at ~1.6e9, expensive-state bias −99%). **Mixing in ~25% do-nothing
   (sustained-neglect) episodes** covers the escalating-risk region (`n_fail` ↑ ~1000×, cost tail ↑
   ~100× to 2e11) and a value fn trained on it predicts expensive states well (R² **+0.71**, tail bias
   **−10%** vs −99%). The recommended recipe is a **3-way behavior-policy mixture**: per-episode
   sampled-threshold reactive (action/`d` diversity) + acting flips + ~25% do-nothing (failure tail).
   Action-*type* coverage is already adequate in the baseline; sampling thresholds only helps modestly,
   and the do-nothing fraction is the decisive ingredient. *Training speed: ~neutral* (one-time
   warmstart; do-nothing episodes are if anything cheaper to roll out).

5. **Things that did NOT help on these instances — skip them:** physics-engineered features
   (no gain, and slightly slower predict on the hot path), heavy-tail loss transforms
   (`log1p`/Huber/quantile didn't fix the bias), and a ranking loss (marginal). Details below.

6. **Know the ceiling: α₀ decides whether ADP can help at all.** On the noisy instance *no*
   value function can predict the full-horizon future (a strong RandomForest gets R²≈0 too), so
   anticipatory value is capped by the instance. ADP's edge over a reactive heuristic only exists
   when degradation is predictable enough (high α₀). This is an instance-design lever, not an
   agent one.

---

## How we diagnosed it

**Calibration — predicted V'(s_post) vs realized future cost** (`adp_vfa_calibration.py`):

| | R² (incl. trend) | within-epoch r² | signed bias |
|---|---|---|---|
| predictable (α₀=0.8) | 0.83 | 0.05 | −5% |
| noisy (α₀=0.05) | **−1.48** | 0.10 | **−74%** |

Predictable: well-calibrated in *level* but state-blind within an epoch. Noisy: **broken** —
worse than predicting a constant, and a massive under-prediction.

**The right metric — action ranking** (`test_action_ranking.py`, fast TAP): at sampled states,
rank {do-nothing, renovate-worst-asset-i} by `Q_vfa = C+V'(s_post)` vs a CRN-rollout ground
truth `Q_true`:

| | Spearman | top-1 match | regret |
|---|---|---|---|
| predictable | **0.68** | 77% | 3.9% |
| noisy | 0.62 | 63% | 7.7% |

V' ranks actions well despite the ~0 `r²_within` — and the noisy regime's bias shows up
concretely as **~2× the decision regret** and lower top-1 accuracy.

**The ceiling** (RF probe + horizon sweep): within-epoch full-horizon future cost is
~unpredictable from state (RF held-out R² ≈ −0.3 to +0.14); the predictable signal lives at
short horizons (next few epochs), which is what motivates takeaway #2.

---

## What we tried — results

| # | Lever | What we did | Verdict | Expected effect on agent cost | Training-speed cost |
|---|---|---|---|---|---|
| 4D | **n-step / bootstrapped target** | measured state→future predictability vs horizon | **WORKS (strong)** — within-R² 0.52 (noisy, n≤4) vs ~0 full | **Largest** — learnable, decision-relevant target, esp. noisy | neutral / slightly **faster** |
| 1 | **`advantage_baseline`** | enabled for all ADP configs; decomposed target variance (94% trend) | **PLAUSIBLE** — mechanism sound, not yet retrained | improves ranking + cuts bias | **free** |
| 4 | **warmstart NEGLECT episodes** (coverage) | mixed ~25–50% do-nothing episodes into collection vs exploration/broad-init tweaks | **WORKS (key for noisy)** — expensive-state bias −99% → −10%, R² −4.6 → +0.71; exploration/broad-init don't help | fixes under-maintenance in noisy regime | ~neutral (one-time warmstart) |
| 2 | **ranking loss** (`value_fn: ranking`) | rank:pairwise vs MSE on n=8 target | **MARGINAL** — within-t Spearman 0.38 vs 0.34 (pred), tie (noisy) | small | neutral |
| 5 | **heavy-tail loss / transform** | MSE vs log1p / Huber / quantile | **DID NOT FIX** — log1p over-shrinks; Huber degenerate at €-scale; q50 ≈ MSE | none (as tried) | neutral |
| 6/4C | **physics-engineered features** | E[TTF], P(fail≤k), aggregates vs raw state | **NO GAIN** — raw already sufficient (pred); nothing predicts (noisy) | none | slightly **slower** (more features on the predict hot path) |
| 3 | **K-rollout target averaging** | designed, not run | UNTESTED | would reduce target variance | **slower** (K× rollouts per target) |

Notes:

- **Warmstart neglect (4) — the diversification that works.** On the noisy instance the tuned
  heuristic renovates failures within ~1 epoch, so `n_fail` never accumulates (`sumNf_mean ≈ 0.01`,
  cost tail ~1.6e9) and a value fn trained on it under-predicts expensive states by −99% (R² −4.6).
  Exploration flips (`wait_explore`) and broad initial conditions (`broad_init`) **do not** change
  this — a corrective base policy over a 120-epoch episode washes out the init and renovates away the
  flips. Mixing in **sustained do-nothing episodes** (25–50%) builds the escalating-risk region
  (`sumNf_mean` 48–94, `max` ~820, cost tail ~2e11) → expensive-state bias −10%/−6%, R² +0.71/+0.79.
  Use a *modest* fraction (~25%, or tune lower): too much neglect over-represents failure and
  starves the normal operating region. This needs a `warmstart` knob for "fraction of collection
  episodes run under an under-maintaining policy" (currently the exploration flips are the only
  diversification lever, and they're acting-biased — the wrong direction for *failure* coverage).
- **Action-mix coverage — already present; sampled thresholds help only modestly.** Tried sampling a
  fresh Reactive policy per episode with thresholds `restrict<repair<renovate` drawn from ranges
  (`mixed_thr`). Findings via the action-taken distribution (per-asset decisions are ~97% do-nothing
  for *every* strategy — intrinsic to the problem): the baseline tuned heuristic already exercises all
  three action types (renov 1.5% / repair 0.4% / restrict 1.4% of decisions). Sampled thresholds raise
  repair ~50% (0.44→0.66%) and slightly widen the `d`-band each action is taken at — a nice-to-have,
  but it does **not** fix the failure bias (still −99%; any *reactive* policy responds before `n_fail`
  builds). Note a tension: neglect episodes *dilute* acting coverage (renovate rate 1.49→1.09%), so
  don't over-do them. **Recommended recipe `mixed+negl25`:** per-episode sampled-threshold reactive +
  acting flips + ~25% do-nothing — best of both (expensive-state R² +0.68 *and* the widest action/`d`
  coverage). Reproduce: `scratch/test_warmstart_diversity.py`.

Notes on the negatives:

- **Physics features (4C):** a linear model on the *raw* state already hits aggregate R²=0.92 on
  the predictable instance (the `P(fail≤k)` hazards even *saturate* into step functions at high
  α₀ and lose signal); on the noisy instance nothing predicts the full horizon. Feature
  representation is not the bottleneck — and extra features slow the per-candidate `predict` call
  the action generator runs in its inner loop.
- **Loss transforms (5):** the −74% bias reproduces offline as a *tail* effect (MSE under-predicts
  the costliest 10% of states by −71%), but `log1p` makes the bias worse, `reg:pseudohubererror`
  is degenerate at euro scale (predicts ≈0 unless `huber_slope`/target is rescaled), and the
  median (`q50`) is ≈ MSE. If anything, an **asymmetric high-quantile** loss (α≈0.7–0.9) to
  *counter* under-prediction is the only variant worth a follow-up — but the cleaner fix is
  coverage (#4).
- **Ranking loss (2):** MSE already orders near-term states fine; the rank loss adds little.

---

## Recommended changes (for ADP performance)

1. **Switch the ADP target to n-step + bootstrap** (the trainer change implied by 4D). Highest
   expected payoff, neutral/positive on speed.
2. **Validate `advantage_baseline`** (already enabled) on the next runs using the action-ranking
   yardstick + bias, not `r²_within`.
3. **Warmstart from a *mixture* of behavior policies**, not the single tuned heuristic. Per the §4
   experiment the recipe is: per-episode Reactive with *sampled* thresholds (`restrict<repair<renovate`)
   for action/`d` diversity + the existing acting-biased flips + **~25% do-nothing episodes** for the
   escalating-risk tail. The do-nothing fraction is the critical ingredient (it's what kills the −74%
   bias; exploration flips, broad init, and threshold variation alone don't). Needs a `warmstart` knob
   for the neglect fraction (and optionally the threshold ranges); tune neglect down if it starves the
   normal region. This sits squarely in the offline-RL coverage / mixed-quality-dataset literature.
4. **Pick instances where ADP can win** (high enough α₀); report the reactive-vs-ADP contrast as
   a function of α₀ rather than fighting an unpredictable instance.
5. Skip physics features and loss transforms; treat ranking loss as a low-priority extra.

Caveat threaded through all offline tests: they use *eval* states, a narrow on-policy slice that
under-samples the hard/expensive decision states. Dumping the **replay-buffer states** (state +
`mc_return` + t) to CSV would let every offline test run on the actual training distribution and
would sharpen the bias/coverage conclusions (#1, #4) in particular.

---

## Reproduce
- Calibration table + figure: `python scratch/adp_vfa_calibration.py`
- Action-ranking yardstick: `python scratch/test_action_ranking.py --tap fast`
- Horizon / n-step target (4D): `python scratch/test_horizon_targets.py`
- Physics features (4C): `python scratch/test_physics_features.py`
- Loss / transform (5): `python scratch/test_loss_transform.py`
- Ranking loss (2): `python scratch/test_ranking_loss.py`
- Warmstart diversity / neglect (4): `python scratch/test_warmstart_diversity.py`
