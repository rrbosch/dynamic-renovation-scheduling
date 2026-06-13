# Registry Conventions — `n_workers` and Replications

Conventions governing how Phase-1 configs are parallelised and replicated when building
`hpc/registry.json` (via `hpc/generate_registry.py`) for Snellius dispatch. Referenced from the
docstring of `generate_registry.py`. See also [experiment_setup.md](experiment_setup.md) (compute
budget) and [../EXPERIMENTS.md](../EXPERIMENTS.md) (experiment plan).

---

## `n_workers` — intra-run parallelism

- **Learner configs set `"n_workers": 8`** on Snellius (in the `training` block). HyperQueue reserves a
  matching 8 cores per task: `hq submit ... --cpus=8`. The two numbers **must agree** — `n_workers`
  controls Python-level parallelism, `--cpus` controls the cores HQ pins to the task.
- **Optuna heuristic search is single-threaded — `"n_workers": 1`, submit with `--cpus=1`.**
  `OptunaHeuristicTrainer` (`experiments/optuna_heuristic_search.py`) runs `study.optimize` with no
  `n_jobs` (one trial at a time), evaluates each trial's tuning episodes in a plain loop, and runs the
  final evaluation sequentially. It does **not** accept or use `n_workers`. Giving an Optuna task 8
  cores leaves 7 idle for the whole run — pure waste. Keep these on a separate 1-core HQ array.
- **On the laptop, use `n_workers: 1`** everywhere to avoid multiprocessing/spawn overhead.
- Only **one layer** of parallelism is ever active (no nesting) — see CLAUDE.md §8. Per agent class,
  the 8 workers are spent on whichever layer applies:
  - value-based (ADP): parallel **episode collection** between VFA updates;
  - evaluation (all agents): parallel **eval episodes**;
  - `MonteCarloRolloutAgent`: parallel **rollouts** per Q-estimate.
- `n_workers: 1` is guaranteed to reproduce the parallel result bit-for-bit (env randomness is
  phase-keyed, not worker-dependent) — parallelism only changes speed, never outcomes.

## Replications (seeds)

Replication policy is **per experiment** (see [../EXPERIMENTS.md](../EXPERIMENTS.md)):

- **Exp 0 (exploration, `instance_10p`): 1 replication per config.** No seed expansion — just enough to
  see which algorithm variants are viable before scaling up. No aggregate statistics are claimed here.
- **Exp 1 / Exp 2 (larger instances): 5 seeds (`0 1 2 3 4`)** — the defensible minimum for the
  `rliable` aggregate statistics we report (stratified bootstrap CIs); see
  [experiment_setup.md](experiment_setup.md).
- **Optuna heuristic tuning is NEVER replicated — 1 run per config**, in every experiment. Rationale:
  an Optuna run is a *tuning procedure* whose deliverable is a single tuned parameter set, not a
  stochastic learning curve. Its objective already averages over `n_eval_episodes` per trial, and the
  search is driven by its own internal sampler seed, so a second "replication" adds cost without a
  meaningful independent sample for the cross-agent comparison.
- **Curbing the winner's-curse bias in heuristic tuning.** The reported Optuna `best_value` is the
  minimum over many noisy trials and is optimistically biased (~1.8x) vs. held-out evaluation — mostly
  optimization overfitting, not horizon. To mitigate, `OptunaHeuristicTrainer` now (a) raises tuning
  episodes from 30 → **100** (`n_tuning_episodes`), and (b) uses Optuna's **`WilcoxonPruner`**
  (`p_threshold=0.1`, `n_startup_steps=5`) to stop disappointing trials early and reallocate budget.
  Pruning requires **common random numbers (CRN)**: tuning episode `k` is keyed
  `("optuna_trial", k, base_seed)` with a *fixed* base seed (no `trial.number`), so episode `k` is the
  same scenario in every trial — the paired-comparison setup the signed-rank pruner needs. The separate
  held-out `"evaluation"` phase (seed 42) remains the **unbiased** final metric reported for comparison,
  so the fixed tuning CRN set does not reintroduce the bias.
- Seed expansion is applied by `generate_registry.py --seeds`; each seeded run gets
  `run_name = "<config_stem>/s<seed>"` and a distinct `seed`, so results nest under
  `results/<config_stem>/s<seed>/` and the dashboard aggregates seeds per experiment.

## Regenerating the registry

**Exp 0** — 1 replication per config; learners first (so they occupy the low indices for an 8-core
array), Optuna appended (high indices, 1-core array):

```bash
# Learners: 1 run each  (24 ADP + 4 rollout + 1 PPO = 29 configs)
python hpc/generate_registry.py \
    --configs configs/i10p_adp_*.json configs/i10p_rollout_*.json \
              configs/i10p_seq_rollout_*.json configs/i10p_ppo_curriculum.json

# Optuna heuristics: append  (3 configs)
python hpc/generate_registry.py --configs configs/i10p_optuna_*.json --append

# Total: 32 runs. Submit as two arrays (different --cpus per group):
#   hq submit --array 0-28  --pin taskset --cpus=8 hpc/hq_task.sh   # learners
#   hq submit --array 29-31 --pin taskset --cpus=1 hpc/hq_task.sh   # optuna
```

**Exp 1 / Exp 2** — same pattern but add `--seeds 0 1 2 3 4` to the learner call (Optuna stays
un-seeded). `generate_registry.py` skips already-finished runs (by `run_name`), so re-running after a
partial batch only enqueues what's left.
