# Reinforcement Learning for Dynamic Infrastructure Renovation Scheduling

Research code accompanying the paper *"Reinforcement Learning for Dynamic Infrastructure Renovation Scheduling"* by Bosch, Rogetzer, van Heeswijk, and Mes — University of Twente.

## Problem

Schedule maintenance actions (renovate, repair, restrict, or do nothing) across a portfolio of *N* infrastructure assets (road edges) over a finite planning horizon of *T* half-year epochs, to minimize expected discounted cost. The formulation accounts for:

- **Stochastic degradation** via a usage-dependent Gamma process
- **Renovation duration uncertainty** via a Wiener process
- **Network traffic-flow impacts** solved via the Traffic Assignment Problem (TAP) at each epoch
- **Reduced capacity** during active renovations and under load restrictions
- **Escalating risk cost** that grows with how long a failed asset is left unaddressed

This is the dynamic, sequential-decision counterpart to the static genetic-algorithm formulation in [`road-maintenance-scheduling`](https://github.com/rrbosch/road-maintenance-scheduling).

## MDP Formulation

**State** `S_t = (d, h, ell, r, n_fail)`, with one value per asset (flat shape `(5N,)`):

| Variable | Description |
|---|---|
| `d` | Condition in [0, 1] (0 = pristine, 1 = critical failure); monotone increasing |
| `h` | Remaining renovation work (> 0 = under renovation; reset to 0 on completion) |
| `ell` | Load-restriction indicator |
| `r` | Repair-used indicator (at most one repair per renovation cycle) |
| `n_fail` | Consecutive failed epochs (drives the escalating risk cost) |

**Actions** per asset: `0` do nothing, `1` repair, `2` renovate, `3` restrict.

**Cost** `C(S_t, A_t) = C_maint + C_travel + C_risk`:
- `C_maint` — length-proportional renovation / repair costs
- `C_travel` — extra vehicle-hours from TAP flows, valued at the value-of-time (VOT) and annualized
- `C_risk` — escalating risk `risk_base · dt · Σ n_fail_i · L_i` for failed assets not yet under renovation

**Objective**: `π* = argmin_π E[ Σ_t γ^t · C(S_t, π(S_t)) ]`.

See [`CLAUDE.md`](CLAUDE.md) for the full design-decisions and notation reference.

## Project Structure

```
.
├── agents/           # Heuristics, ADP/DQN, actor-critic, PPO, rollout, DCL, value/policy fns
├── env/              # MDP environment, degradation, network, TAP solvers, phase-keyed noise
├── training/         # Replay buffer and training loops (value-based + PPO)
├── experiments/      # Config wiring, instance generation, CLI entry point, sweeps
├── configs/          # Experiment JSON configs
├── instances/        # Generated instance JSON files (asset parameters)
├── utils/            # Logging and metrics
├── vis/              # Visualization (comparison dashboard, episode viewer)
└── tests/            # Pytest suite
```

## Installation

```bash
pip install -r requirements.txt
```

Core dependencies: `numpy`, `scipy`, `xgboost`, `numba`, `torch`, `pandas`, `optuna`, `tqdm`. The default TAP backend is a pure-NumPy/Numba Frank-Wolfe solver — no external traffic-modelling library is required.

## Usage

The workflow is: **generate an instance → write a config that points at it → run**.

### 1. Generate an instance

An *instance* is a JSON file of per-asset parameters (lengths, degradation rates, costs, horizon). Defaults reproduce the paper's setup; everything is overridable:

```bash
python experiments/generate_instance.py --n-assets 10 --seed 0 --output instances/instance_10p.json
```

### 2. Run an experiment

```bash
python experiments/run.py --config configs/i10p_adp_normal_empty_fifo_xgb.json
```

Run with no `--config` to pick interactively from `configs/` (each entry shows its status: *not started / in progress / finished*). Override the run name with `--run-name`.

Runs are **interruptible and resumable**: stop at any time and rerun the same config — an incomplete checkpoint with a matching config hash is auto-detected and resumed (or pass `--resume <checkpoint_dir>` explicitly). Every run is fully seeded and deterministic given `(seed, instance, config)`.

### 3. Sweeps

```bash
python experiments/sweep.py configs/*.json --workers 4
```

### Results

Each run writes to `results/{run_name}/`:
- `config.json` — full resolved experiment configuration
- `training_log.csv` — per-evaluation mean / std discounted cost
- `eval_episodes.pkl` — pickled evaluation trajectories for post-hoc analysis
- `checkpoints/` — resumable checkpoints

(`results/` is git-ignored — outputs are regenerable from instances + configs + seeds.)

## Configuration

A config is a thin JSON wrapper: it selects the network, TAP backend, seed, the **instance file** (which supplies all environment/physics parameters), and the training + agent settings.

```json
{
  "network": "sioux_falls",
  "tap_backend": "fast",
  "seed": 42,
  "run_name": "i10p_adp_normal_empty_fifo_xgb",
  "instance": "instances/instance_10p.json",
  "training": {
    "time_budget": 86400,
    "eval_interval": 1000000,
    "update_interval": 50,
    "truncation_mode": "bootstrap",
    "buffer_capacity": 200000,
    "buffer_strategy": "fifo",
    "n_eval_episodes": 10,
    "n_workers": 8
  },
  "agent": {
    "agent_type": "adp",
    "value_fn": "xgboost",
    "action_gen": "local_search"
  }
}
```

| Key | Options |
|---|---|
| `tap_backend` | `fast` (Numba Frank-Wolfe, default), `null` (no traffic coupling), `surrogate` (learned TAP) |
| `training.truncation_mode` | `none`, `horizon_rollout`, `bootstrap` (terminal handling for training targets) |
| `training.buffer_strategy` | `fifo`, `lowest_error`, `stochastic_knockout` |
| `training.n_workers` | Multi-core parallelism (episode collection / evaluation / rollouts) |

See `configs/` for complete examples.

## Agents

| `agent_type` | Description |
|---|---|
| `reactive` | Renovate/repair/restrict when condition crosses per-action thresholds |
| `paced` | Schedule renovations to track expected asset lifespans |
| `adp` | Approximate dynamic programming; value fn trained on post-decision state, target = future-only return |
| `dqn` | TD(0) bootstrap; value fn trained on pre-decision state, target = `cost + γ·V(s_next)` |
| `actor_critic` | ADP critic + policy network trained by imitating local search |
| `ppo` | Proximal Policy Optimization with optional curriculum |
| `rollout` / `sequential_rollout` | Monte Carlo rollout over a base policy (with common random numbers; optional adaptive budgeting) |
| `optuna_heuristic` | Optuna-tuned reactive/paced heuristics |

**Value functions** (`value_fn`): `xgboost`, `neural`, `ranking` (LambdaRank).

**Action generators** (`action_gen`, search over the combinatorial joint action): `local_search` (greedy single-asset deviations), `sequential` (commit assets one at a time).

## Network

The default network is **Sioux Falls** (24 nodes, 76 edges). An Amsterdam network interface is defined but not yet implemented.

## Reproducibility

All environment randomness is **stateless and phase-keyed** (blake2b → Philox): episodes are a pure function of `(seed, phase, episode_idx)`, making runs reproducible, resume-safe, and invariant to the number of parallel workers. Evaluation uses shared common random numbers across agents so comparisons are paired. No mutable RNG state is checkpointed.

## Citation

Citation metadata is in [`CITATION.cff`](CITATION.cff). A Zenodo DOI will be minted on first release; please cite the accompanying paper and the archived software version once available.

## License

Released under the [MIT License](LICENSE).
