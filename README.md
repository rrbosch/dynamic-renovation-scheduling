# Reinforcement Learning for Dynamic Infrastructure Renovation Scheduling

Research code accompanying the paper *"Reinforcement Learning for Dynamic Infrastructure Renovation Scheduling"* by Bosch, Rogetzer, van Heeswijk, and Mes — University of Twente.

## Problem

Schedule maintenance actions (renovate, repair, restrict, or do nothing) across a portfolio of infrastructure assets (road edges) over a multi-period planning horizon to minimize expected discounted cost. The problem accounts for:

- **Stochastic degradation** via a usage-dependent Gamma process
- **Renovation duration uncertainty** via a Wiener process
- **Traffic flow impacts** solved via the Traffic Assignment Problem (TAP) at each epoch
- **Dynamic capacity constraints** during active renovations

## Project Structure

```
Code v2/
├── agents/           # Agent implementations (heuristics, DQN, actor-critic)
├── env/              # MDP environment, degradation, network, TAP solvers
├── training/         # Replay buffer and training loop
├── experiments/      # Config dataclasses, CLI entry point, parallel sweep
├── configs/          # Example experiment JSON configs
├── utils/            # Logging and metrics
└── results/          # Auto-generated experiment outputs
```

## MDP Formulation

**State** `S_t = (d, h, ell, r)` per asset:
- `d` — condition level in [0, 1] (0 = pristine, 1 = failure)
- `h` — remaining renovation work (> 0 means under renovation)
- `ell` — load restriction indicator
- `r` — repair-used indicator (once per renovation cycle)

**Actions** per asset: `0` do nothing, `1` repair, `2` renovate, `3` restrict

**Cost**: travel time cost (from TAP flows) + maintenance cost + failure penalties

## Installation

```bash
pip install numpy scipy xgboost torch scikit-learn pandas
```

Optional (recommended for faster TAP solving):
```bash
pip install aequilibrae
```

On Windows, aequilibrae requires SpatiaLite. Download the DLL and add it to your `PATH` before running. The code auto-falls back to a pure Python/NumPy Frank-Wolfe solver if aequilibrae is unavailable.

## Usage

### Run a single experiment

```bash
python experiments/run.py --config configs/exp1_dqn_localsearch.json
```

Optionally override the run name:
```bash
python experiments/run.py --config configs/exp1_dqn_localsearch.json --run-name my_run
```

### Run a parallel sweep

```bash
python experiments/sweep.py configs/*.json --workers 4
```

### Results

Each run saves to `results/{run_name}/`:
- `config.json` — full experiment configuration
- `training_log.csv` — per-evaluation mean and std discounted cost
- `eval_episodes.pkl` — pickled episode trajectories for post-hoc analysis

## Agents

| Agent | Description |
|---|---|
| `ReactiveAgent` | Renovate when condition exceeds a threshold |
| `PacedAgent` | Schedule renovations to match expected asset lifespans |
| `DQNAgent` | Approximate dynamic programming with XGBoost or neural value function |
| `ActorCriticAgent` | DQN critic + policy network trained via imitation of local search |

**Action generators** (used by learning agents to search over the combinatorial action space):
- `LocalSearchGenerator` — greedy local search over single-asset deviations
- `SequentialGenerator` — commit assets one by one in random order

**Value functions**: `XGBoostValueFn`, `NeuralValueFn`, `RankingValueFn` (LambdaRank)

## Configuration

Experiments are defined as JSON files. Example:

```json
{
  "run_name": "exp1_dqn_localsearch",
  "network": "sioux_falls",
  "tap_backend": "aequilibrae",
  "seed": 42,
  "env": {
    "n_epochs": 20,
    "gamma": 0.95,
    "cost_renovation": 50.0,
    "cost_repair": 10.0,
    "cost_failure": 200.0
  },
  "training": {
    "n_episodes": 500,
    "eval_interval": 50,
    "update_interval": 10,
    "buffer_capacity": 200000,
    "buffer_strategy": "fifo",
    "batch_size": 512,
    "bootstrap_truncation": true
  },
  "agent": {
    "agent_type": "dqn",
    "value_fn": "xgboost",
    "action_gen": "local_search"
  }
}
```

See `configs/` for complete examples.

## Key Design Notes

- **Post-decision state** is central: value functions are trained on `s_post` (state after action, before stochastic transitions), making learning targets less noisy.
- **TAP is swappable**: switch between exact (`aequilibrae`, `fallback`) and surrogate (XGBoost-based) solvers via config.
- **Local search avoids TAP**: during action search, approximate flows (30% utilization) are used for speed; exact TAP is only called once per environment step.
- **No global random state**: all randomness flows through a `np.random.Generator` instance for full reproducibility.
- **Frozen dataclasses** for all configs ensure immutability and clean JSON serialization.

## Network

The default network is **Sioux Falls** (24 nodes, 76 edges). An Amsterdam network interface is defined but not yet implemented.

## Citation

If you use this code, please cite the accompanying paper (details TBD upon publication).
