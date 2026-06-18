# Experiment Config Reference

All options that can appear in an experiment config JSON file.
The JSON is loaded by `ExperimentConfig.from_json()` / `from_file()` in `experiments/configs.py`,
and wired into `(InfraEnv, Agent, Trainer)` by `build_experiment()`. Environment/physics
parameters do **not** live here — they come from the **instance JSON** named by `instance`.

---

## Top-level fields

| Field | Type | Required | Description |
|---|---|---|---|
| `network` | string | yes | `"sioux_falls"` or `"amsterdam"` (`load_amsterdam` is a stub) |
| `tap_backend` | string | yes | `"fast"` (Numba Frank-Wolfe), `"null"` (no traffic coupling), or `"surrogate"` (learned TAP) |
| `seed` | int | yes | Master seed (environment phase-keying + training) |
| `run_name` | string | yes | Results written to `results/<run_name>/` |
| `instance` | string | yes | Path to instance JSON (relative to project root) — supplies all env parameters |
| `training` | object | yes | See **Training options** below |
| `agent` | object | yes | See **Agent options** below |

---

## Training options (`training` object)

| Key | Type | Default | Description |
|---|---|---|---|
| `time_budget` | float | `3600.0` | Wall-clock seconds before stopping |
| `n_episodes` | int | `10_000_000` | Max episodes (effectively unbounded; `time_budget` usually stops first) |
| `eval_interval` | int | `50` | Evaluate every N training episodes |
| `update_interval` | int | `10` | Call `agent.update()` every N episodes |
| `truncation_mode` | string | `"bootstrap"` | `"none"`, `"horizon_rollout"`, or `"bootstrap"` (terminal handling for training targets). Legacy `bootstrap_truncation: true/false` is still accepted and mapped to `bootstrap`/`none`. |
| `buffer_capacity` | int | `200_000` | Max transitions in replay buffer |
| `buffer_strategy` | string | `"fifo"` | `"fifo"`, `"lowest_error"`, or `"stochastic_knockout"` |
| `n_eval_episodes` | int | `10` | Episodes per evaluation pass |
| `n_warmstart_episodes` | int | `0` | Heuristic warm-start episodes before RL training |
| `warmstart` | object/null | `null` | Agent config dict for the warm-start agent (same schema as `agent`) |
| `checkpoint_interval` | int | `0` | Checkpoint every N episodes (`0` = disabled, use time-based) |
| `checkpoint_interval_seconds` | float | `1800.0` | Checkpoint every N seconds |
| `n_workers` | int | `1` | Multi-core parallelism (episode collection / evaluation / rollouts). `1` reproduces parallel results bit-for-bit |

> `T_tail` (evaluation tail length) is **not** a training key — it is sourced from the instance JSON.

PPO and Optuna-heuristic agents read additional keys from `training` (e.g.
`curriculum_phase0_episodes`, `early_stopping_seconds`, `n_tuning_episodes`); see
`build_experiment()` for the full list.

---

## Agent options (`agent` object)

| Key | Type | Default | Description |
|---|---|---|---|
| `agent_type` | string | — | **Required.** `"reactive"`, `"paced"`, `"adp"`, `"dqn"`, `"actor_critic"`, `"ppo"`, `"rollout"`, `"sequential_rollout"`, `"dcl"`, `"optuna_heuristic"`, `"marl"` (stub) |
| `value_fn` | string | `"xgboost"` | `"xgboost"`, `"neural"`, `"ranking"` — used by `adp`/`dqn`/`actor_critic` |
| `action_gen` | string | `"local_search"` | `"local_search"`, `"sequential"`, `"bdq"` (stub) — used by `adp`/`dqn`/`actor_critic` |
| `extra` | object | `{}` | Agent-type-specific options (see below) |

> **`adp` vs `dqn` are separate classes, not a `bootstrap_mode` flag.** `adp` trains the value
> function on post-decision-state features with a future-only return target; `dqn` trains on
> pre-decision-state features with a `cost + γ·V(s_next)` TD(0) target.

### `extra` options by agent type

#### `reactive`
| Key | Default | Description |
|---|---|---|
| `threshold` | `0.7` | Condition threshold triggering renovation |
| `repair_threshold` | `null` | Threshold for the repair action (optional) |
| `restrict_threshold` | `null` | Threshold for the restrict action (optional) |

#### `paced`
| Key | Default | Description |
|---|---|---|
| `threshold` | `0.7` | Condition threshold triggering renovation |
| `pace_threshold` | `null` | Secondary threshold for pacing logic |

#### `adp` / `dqn`
| Key | Default | Description |
|---|---|---|
| `finite_horizon` | `true` | Append epoch index `t` to features → shape `(5N+1,)`; `false` → `(5N,)` |

#### `actor_critic`
| Key | Default | Description |
|---|---|---|
| `finite_horizon` | `true` | Same as ADP/DQN |
| `patience` | `20` | Early-stopping patience for policy training |
| `hidden_dims` | `[256, 256]` | Policy-network hidden layer sizes |

#### `ppo`
| Key | Default | Description |
|---|---|---|
| `finite_horizon` | `true` | As above |
| `hidden_dims` | `[256, 256]` | Policy/value network hidden sizes |
| `ppo_kwargs` | `{}` | Extra kwargs forwarded to `PPOAgent` |

#### `rollout` / `sequential_rollout`
| Key | Default | Description |
|---|---|---|
| `rollout_policy` | reactive (thr 0.7) | Base policy config rolled out (nested agent dict) |
| `n_rollouts` | `30` | Rollouts per Q estimate |
| `rollout_horizon` | `null` → `T` | **Fixed** lookahead window (epochs) per decision. `null` defaults to the planning-horizon length `T`. Do **not** expect it to shrink with `t`: evaluation runs `T + tail_epochs`, so a `T - t` window would collapse to `0` in the tail and silently turn the agent into a do-nothing policy. (There is no `max_steps` key.) |
| `initial_action` | `"policy"` | `"policy"` or `"empty"` initial action |
| `action_threshold` | `0.5` | Local-search acceptance threshold |
| `rollout_selection` | `"adaptive"` | `"fixed"` or `"adaptive"` (sequential Wilcoxon budgeting; see `docs/adaptive_rollout_literature.md`) |
| `p_threshold`, `min_rollouts`, `max_rollouts`, `rollout_batch` | `0.02`, `20`, `100`, `5` | Adaptive-budget controls |

> Unknown keys under `agent.extra` for `reactive`, `paced`, `rollout`, and `sequential_rollout` are **rejected** with a "did you mean…?" error (`_check_extra_keys` in `configs.py`). This prevents a misspelled/renamed key from being silently ignored and falling back to a default.

#### `dcl`
| Key | Default | Description |
|---|---|---|
| `policy_type` | `"xgboost"` | `"xgboost"` or `"nn"` policy |
| `value_fn` | `"xgboost"` | Value function for the DCL critic |
| `heuristic_policy` | reactive (thr 0.8) | Base heuristic config (nested agent dict) |
| `action_gen` | `"local_search"` | Action generator |
| `rollout_horizon` | `10` | Rollout depth |
| `n_rollouts` | `5` | Rollouts per estimate |
| `finite_horizon` | `true` | As above |

#### `optuna_heuristic`
| Key | Default | Description |
|---|---|---|
| `heuristic_type` | `"reactive"` | `"reactive"`, `"paced"`, or `"reactiveperasset"` |
| `param_space` | type-specific default | Optuna search space (overrides built-in defaults) |
| `n_tuning_episodes` | `30` | Tuning episodes per trial |

---

## Instance JSON fields (read by `build_experiment`)

Instance files live in `instances/` and are generated by `experiments/generate_instance.py`
(schema_version 5). Per-asset arrays have length N.

### Generated per-asset arrays (length N)

| Field | Description |
|---|---|
| `d_init` | Initial condition values in [0,1] (or `null` to sample) |
| `alpha0` | Baseline Gamma shape rates |
| `beta` | Gamma rate parameters (`= alpha0 · e_fail`) |
| `mu_h` | Renovation Wiener drift (years⁻¹) |
| `sigma_h` | Renovation Wiener volatility |
| `asset_lengths_m` | Physical length per asset (metres) |
| `c_ren` | Renovation cost per asset (€, length-proportional) |
| `c_rep` | Repair cost per asset (€, length-proportional) |

### Environment scalars

| Field | Default | Description |
|---|---|---|
| `n_assets` | — | Number of assets N |
| `network` | — | Network name used during generation |
| `years` | — | Planning horizon in years (`T = round(years / dt)`) |
| `dt` | — | Epoch length (years) |
| `T_tail` | `1× e_fail_mean` | Evaluation tail length in **years** (converted to epochs) |
| `gamma` | — | Annual discount factor (per-epoch = `gamma ** dt`) |
| `d_fail` | `1.0` | Condition threshold for failure |
| `eta_ren` | `0.05` | Capacity factor during renovation |
| `eta_load` | `0.50` | Capacity factor under load restriction |
| `restrict_degrad_multiplier` | `0.5` | Multiplier on degradation rate α under load restriction |
| `delta_repair` | — | Condition improvement from repair action |
| `vot` | `10.76` | Value of time (€/vehicle-hour) |
| `traffic_cost_factor` | `1.0` | Scales raw traffic cost |
| `risk_base` | `10000.0` | €/m/year per consecutive failed epoch |
| `allow_repair` | `true` | Set `false` to permanently disable the repair action |
| `allow_restrict` | `true` | Set `false` to permanently disable the load-restriction action |

---

## Minimal example config

```json
{
  "network": "sioux_falls",
  "tap_backend": "fast",
  "seed": 42,
  "run_name": "exp_001",
  "instance": "instances/instance_10p.json",
  "training": {
    "time_budget": 3600,
    "eval_interval": 50,
    "update_interval": 10,
    "truncation_mode": "bootstrap",
    "buffer_strategy": "fifo",
    "n_workers": 1
  },
  "agent": {
    "agent_type": "adp",
    "value_fn": "xgboost",
    "action_gen": "local_search",
    "extra": {
      "finite_horizon": true
    }
  }
}
```
