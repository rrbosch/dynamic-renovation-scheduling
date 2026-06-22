# CLAUDE.md — RL Infrastructure Maintenance Scheduling

Authoritative design-decisions and context file for Claude sessions.

## 0. Usage Intent

This is scientific code written for a research paper. The following principles govern how it is used:

- **Laptop-first, Snellius-ready.** Development and preliminary experiments run on a personal laptop. Once code is stable, production runs move to Snellius (Dutch national supercomputer). Entry points such as `experiments/run.py` must remain fully CLI-driven with argument support (e.g. `python run.py configs/foo.json`) so they can be submitted as batch jobs.
- **Reproducibility first.** Every experiment is seeded and fully deterministic given the seed. Instance files, config JSONs, and checkpoints together must be sufficient to reproduce any result.
- **Interruptible / resumable.** Experiments can be stopped at any time and resumed from the latest checkpoint. Checkpoint save/load logic must remain robust.
- **Speed matters in hot paths.** Training and evaluation loops run for many episodes; unnecessary overhead compounds quickly. Prioritise vectorised numpy operations, avoid Python-level per-step loops, and prefer compiled/numba paths where they already exist. Do not add slowdowns to code that runs inside the training or evaluation loop.
- **Scientific insight.** The primary output is not software — it is results for a paper. Changes should support clean comparisons, stable logging, and interpretable outputs rather than engineering convenience for its own sake.

---

## 1. Project Overview

Schedule maintenance (renovate / repair / restrict / do nothing) for a portfolio of N infrastructure assets (road edges) over a finite planning horizon of T half-year epochs. Decisions affect traffic flows across the whole network. Goal: minimise expected discounted cost.

Paper: **"Reinforcement Learning for Dynamic Infrastructure Renovation Scheduling"**
(Bosch, Rogetzer, van Heeswijk, Mes — University of Twente).

---

## 2. MDP Formulation

### State (5 variables, shape `(5N,)`)

```
S_t = (d, h, ell, r, n_fail)
```

| Variable | Shape | Description |
|----------|-------|-------------|
| `d` | `(N,)` | Condition in [0,1]. 0=pristine, 1=critical failure. Monotone increasing. |
| `h` | `(N,)` | Remaining renovation work. >0 = under renovation; reset to 0 on completion. |
| `ell` | `(N,)` | Load restriction indicator (bool-like float). |
| `r` | `(N,)` | Repair-used indicator (bool-like float). At most one repair per renovation cycle. |
| `n_fail` | `(N,)` | Consecutive failed steps (n_fail[i] increments each epoch d[i] >= d_fail). |

`state.features()` returns flat `np.concatenate([d, h, ell, r, n_fail])`, shape `(5N,)`.

### Actions

`{0=none, 1=repair, 2=renovate, 3=restrict}` — joint action shape `(N,)`.

### Degradation

```
ΔG_i ~ Gamma(α_i(ℓ_i) · Δt,  β_i)
α_i(ℓ_i) = (1 - (1 - restrict_degrad_multiplier) · ℓ_i) · α_i^0
d_i^{t+1} = min(1, d_i + ΔG_i)
```

Shape rate depends only on the load restriction indicator. `restrict_degrad_multiplier` (default 0.5) controls how much restriction reduces degradation, independently of the capacity effect `eta_load`. No traffic-flow coupling. `EnvConfig` has no `kappa` field.

### Renovation duration

```
h^{t+1} = h - μ_h·Δt + σ_h·√Δt·ε,   ε ~ N(0,1)
```
Completes at first epoch where h ≤ 0. On completion: `d←0, ell←0, r←0, h←0`.

### Cost function

```
C(S_t, A_t) = C_maint + C_travel + C_risk
```

**C_maint**: `Σ c_ren_i · 1{a_i=renovate} + c_rep_i · 1{a_i=repair}`

**C_travel**: Extra vehicle-hours × VOT, annualized:
```
C_travel = traffic_cost_factor · vot · (Σ f_e·τ_e − B_travel) / 60 · dt · 365
```
where `B_travel` = baseline total travel time at nominal capacities (pre-computed once).

**C_risk**: Escalating risk for failed assets not yet under renovation:
```
C_risk = risk_base · dt · Σ n_fail_i · L_i · 1{d_i >= d_fail} · 1{h_i = 0}
```
Risk per epoch increases linearly with duration of failure (`n_fail_i` counter). Assets under active renovation (`h_i > 0`) are excluded.

### Objective

```
π* = argmin_π  E[ Σ_{t=0}^{T-1} γ^t · C(S_t, π(S_t)) ]
```

### Truncation modes (`TrainingConfig.truncation_mode`)

| Mode | Simulates | Trains on | Terminal G |
|------|-----------|-----------|------------|
| `"none"` | T epochs only | all T transitions | `G = 0` at t=T |
| `"horizon_rollout"` | T + tail_epochs | all T + tail_epochs transitions | `G = 0` at end |
| `"bootstrap"` | T epochs only | all T transitions | `G = V(s_T)` where s_T is treated as t=0 (stationary VFA) |

- **T_tail** is an **instance-level** parameter (years), set at instance generation and stored in
  the instance JSON. Default = **1× `e_fail_mean`** (the expected asset lifespan), overridable via
  `generate_instance.py --t-tail-years`. It is converted to epochs via `int(T_tail / dt)`.
  `configs.py` sources it from the instance (`inst['T_tail']`); old instances lacking the field fall
  back to `mean(beta/alpha0)`. The `training.T_tail` config key is no longer consulted.
- **Evaluation always uses `horizon_rollout`** (T + tail_epochs) regardless of training mode. The
  eval loops run the full `eval_length = T + tail_epochs` and **do not break on `env.step`'s
  `done`** (which fires at the planning horizon `t = T`); the env keeps stepping into the tail. This
  holds for `Trainer` (sequential + parallel worker), `OptunaHeuristicTrainer`, and `PPOTrainer`.
  Note: `horizon_rollout` *training* now also simulates the tail — both training paths (sequential +
  parallel worker) skip the `done`-break when `truncation_mode == 'horizon_rollout'`, collecting all
  T + tail_epochs transitions (verified). `bootstrap`/`none` still stop at `done` (= T).

### Terminology

- **Warmstart** = filling the replay buffer with heuristic-generated transitions before training begins.
  Sized in **states/transitions** via `training.n_warmstart_states` (not episodes); the loop collects
  whole episodes until that many transitions are buffered. Heuristic = `training.warmstart`
  (`{agent_type, extra}`).
- **init_action** (ADP only, `agent.extra.init_action`) = the seed of the local action search:
  `'empty'` starts from do-nothing (all zeros); `'policy'` starts from the warmstart heuristic's action
  at each decision (same heuristic as the buffer warmstart; wired in `build_experiment`). Independent of
  whether/how the buffer is warmstarted. The action generators (`LocalSearchGenerator`,
  `SequentialGenerator`) take an optional `init_action` arg (None ⇒ zeros).
- **Truncation** = how the end of an episode is handled for training targets (none / horizon_rollout / bootstrap)

---

## 3. Design Decisions

**Why escalating risk (C_risk) instead of flat penalty (C_fail):**
A flat per-epoch penalty `c_fail · 1{d >= d_fail}` provides no incentive to address failure *soon*. The escalating formulation `n_fail · L_i · risk_base · dt` penalizes the *duration* of failure, creating a gradient that increases over time spent in the failed state and naturally reflects accumulating risk (structural weakening, liability).

**Why length-proportional costs:**
`c_ren = 50_000 · L_i`, `c_rep = 25_000 · L_i`. Renovation and repair cost should scale with the physical size of the intervention. This also couples the risk cost to asset size via `L_i`.

**Why 5-variable state (include n_fail):**
Without `n_fail` in the state, the agent cannot distinguish an asset that just entered failure from one that has been failed for 10 epochs. Since cost is escalating, these states have different cost-to-go and the value function must see the counter.

**Two agent classes implement different bootstrapping strategies:**
- `ADPAgent` (agent_type `'adp'`): target = `mc_return - cost` (future-only discounted return). V' approximates future costs only; current-step costs (c_maint, c_travel, c_risk) are added explicitly in action generators as `Q(s,a) = C(s,a) + V'(s_post)`. Trains on post-decision state features. Lower-variance targets; natural for ADP/XGBoost.
- `DQNAgent` (agent_type `'dqn'`): target = `cost + γ·V(s_next; θ_prev)`. TD(0) bootstrap on `s_next` (post-stochastic-transition). Trains on pre-decision state features. More natural for standard DQN, actor-critic, MARL CTDE.
Both `post_state` and `next_state` are stored in every `Transition`.

**Feature space difference (ValueFn vs PolicyNetwork):**
- `ValueFn`: flat 5N state features `[d, h, ell, r, n_fail]` (+ optional `t` column for finite_horizon=True → 5N+1)
- `PolicyNetwork`: per-asset features with global context `[d_i, h_i, ell_i, r_i, n_fail_i, idx, mean_d, mean_nf, t_norm]` (9 features per asset)

This is intentional — the value function operates on the full state vector while the policy network processes per-asset observations with shared context for parameter efficiency.

---

## 4. Project Structure

```
project/
├── env/
│   ├── network.py          NetworkData, load_sioux_falls, load_amsterdam (stub)
│   ├── degradation.py      gamma_step, wiener_step
│   ├── tap.py              TAPSolver protocol, FastTAP (numba), NullTAP, make_tap
│   ├── surrogate.py        SurrogateTAP (load_pretrained is stub)
│   └── mdp.py              State, EnvConfig, InfraEnv
├── agents/
│   ├── base.py             Agent ABC
│   ├── heuristics.py       ReactiveAgent, PacedAgent, PerAssetReactiveAgent, LeadTimeAgent,
│   │                       NetConcurrencyAgent, HoldingAgent, ValueDensityAgent, WorstFirstAgent
│   ├── dqn.py              DQNAgent
│   ├── value_fn.py         ValueFn ABC, XGBoostValueFn, NeuralValueFn
│   ├── action_gen.py       LocalSearchGenerator, SequentialGenerator, BDQGenerator (stub)
│   ├── ranking.py          RankingValueFn
│   ├── actor_critic.py     PolicyNetwork, ActorCriticAgent
│   ├── rollout.py          MonteCarloRolloutAgent
│   └── marl.py             CTDEAgent (stub)
├── training/
│   ├── buffer.py           Transition, ReplayBuffer
│   └── trainer.py          TrainingConfig, Trainer
├── experiments/
│   ├── configs.py          AgentConfig, ExperimentConfig, build_experiment
│   ├── generate_instance.py  Generate per-asset parameters JSON
│   ├── run.py              CLI entry point
│   └── sweep.py            Parallel sweep over configs
├── instances/              Generated instance JSON files
├── utils/
│   ├── logging.py          RunLogger
│   └── metrics.py          discounted_cost, per_asset_stats
└── vis/                    Visualization tools
```

---

## 5. Key File Reference

### `env/degradation.py`

```python
def gamma_step(d, alpha0, beta, ell, dt, u, restrict_degrad_multiplier=0.5) -> np.ndarray:
    """Inverse-CDF Gamma increment from externally supplied uniforms u, shape (N,).
    alpha = (1 - (1-f)*ell) * alpha0;  delta = gammaincinv(alpha*dt, u) / beta."""

def wiener_step(h, mu_h, sigma_h, dt, eps) -> np.ndarray:
    """Wiener step from externally supplied standard normals eps, shape (N,).
    Only advances assets where h > 0. Returns updated h, shape (N,)."""
```

**Noise is supplied by the caller, not drawn inside.** `stochastic_transition(s_post, t, noise)`
takes `noise = (u, eps)`, each `(N,)`. **All randomness is stateless and phase-keyed** via
`env.noise.keyed_philox(*key_parts)` (blake2b → Philox); there is **no mutable `env.rng`**.
Before each episode the caller invokes `env.begin_episode(phase, episode_idx, base_seed=None)`,
which arms two episode-scoped generators keyed on `("transition"/"reset", phase, seed, episode_idx)`.
`env.step` then pulls `(u, eps)` from the transition generator and `reset()` samples `d_init` from
the reset generator. The **phase tag** (`"training"`, `"evaluation"`, `"warmstart"`,
`"optuna_trial"`, `"rollout"`, ...) makes streams structurally independent: an agent's internal
`"rollout"` sims can never coincide with the real `"evaluation"` transitions (no clairvoyance).
Episodes are a pure function of `(phase, seed, episode_idx)` → reproducible, resume-safe, and
parallelism-invariant; no env RNG is checkpointed. **Evaluation uses shared CRN across agents**:
the eval key omits agent identity, so every agent sees identical episodes/initial conditions per
seed (paired comparison). `reset()` auto-begins a `"default"` episode if `begin_episode` was not
called (keeps ad-hoc/test use deterministic).

Rollout agents (`MonteCarloRolloutAgent`, `SequentialMCRolloutAgent`, `DCLAgent`) derive `noise`
via `agents.rollout.rollout_noise`, now keyed `("rollout", seed, root-state, decision-epoch,
rollout-index)` on the same `keyed_philox` primitive. Because the key **excludes the candidate
action**, every candidate at a decision step replays identical noise → **common random numbers
(CRN)**. The Gamma uses the inverse-CDF transform (`scipy.special.gammaincinv`) so a shared
uniform maps to the same quantile regardless of the action-dependent shape — exact CRN with
fixed per-asset randomness consumption (no rejection-sampling desync). Rollout agents hold no
mutable rng state (their `save`/`load` are no-ops).

### `env/mdp.py — State`

```python
@dataclass
class State:
    d:      np.ndarray  # (N,) condition
    h:      np.ndarray  # (N,) renovation progress
    ell:    np.ndarray  # (N,) load restriction
    r:      np.ndarray  # (N,) repair-used
    n_fail: np.ndarray  # (N,) consecutive failed steps

    def features(self) -> np.ndarray:
        return np.concatenate([self.d, self.h, self.ell, self.r, self.n_fail])
        # shape (5N,) — NOT (4N,)
```

### `env/mdp.py — EnvConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `n_assets` | int | — | N |
| `T` | int | 120 | Planning horizon |
| `dt` | float | 0.5 | Epoch length (years) |
| `gamma` | float | — | Annual discount factor (per-epoch = gamma**dt) |
| `d_fail` | float | 1.0 | Failure threshold |
| `eta_ren` | float | 0.05 | Capacity factor during renovation |
| `eta_load` | float | 0.50 | Capacity factor under load restriction |
| `restrict_degrad_multiplier` | float | 0.5 | Multiplier on α under load restriction: 0.5=rate halved, 1.0=no effect, 0.0=fully stopped |
| `mu_h` | float\|array | — | Wiener drift (per asset or scalar) |
| `sigma_h` | float\|array | — | Wiener volatility |
| `delta_repair` | float | — | Condition improvement from repair |
| `alpha0` | `(N,)` | — | Baseline shape rates |
| `beta` | `(N,)` | — | Gamma rate parameters |
| `c_ren` | `(N,)` | — | Renovation costs (length-proportional) |
| `c_rep` | `(N,)` | — | Repair costs (length-proportional) |
| `asset_lengths_m` | `(N,)` | — | Physical length per asset (metres) |
| `vot` | float | 10.76 | Value of time (€/vehicle-hour) |
| `traffic_cost_factor` | float | 1.0 | Scales raw traffic cost |
| `risk_base` | float | 10_000 | €/m/year per failure step |
| `d_init` | `(N,)\|None` | None | Fixed initial conditions (or sample Uniform) |

**No `c_fail` field. No `kappa` field. No traffic-flow coupling in degradation.**

### `experiments/generate_instance.py`

- `asset_lengths_m ~ LogNormal(mean=200m, CV=0.5)`
- `alpha0 ~ LogNormal(mean=alpha0_mean, sigma_log=0.3)`, **default `alpha0_mean=0.8`** (CLI
  `--alpha0-mean`). Higher `alpha0` ⇒ tighter Gamma increments per step ⇒ failure timing is more
  *predictable* — **without changing E[lifetime]** (since `beta = alpha0·e_fail`, the mean per-epoch
  increment `alpha0·dt/beta = dt/e_fail` is independent of `alpha0`). The old default 0.05 gave a
  noise-dominated, near-unpredictable instance (TTF-CV ≈ 0.6); 0.8 gives TTF-CV ≈ 0.22. See
  `docs/i10p_predictability_analysis.md`.
- `e_fail ~ LogNormal(mean=60yr, CV=0.1)`, then `beta = alpha0 * e_fail`
- `e_ren_weeks = 10 + L_i / 5` (base 10 weeks + 1 week per 5m length)
- `mu_h = 1 / e_ren_years`, `sigma_h = 0.2 * mu_h`
- `c_ren = 50_000 * L_i`, `c_rep = 25_000 * L_i`

**No geometric sampling. No c_fail.**

### Instance JSON schema (schema_version: 5)

Required fields: `n_assets`, `network`, `generation_seed`, `d_init`, `alpha0`, `beta`,
`mu_h`, `sigma_h`, `asset_lengths_m`, `c_ren`, `c_rep`, `years`, `dt`, `T_tail`, `gamma`, `d_fail`,
`eta_ren`, `eta_load`, `restrict_degrad_multiplier`, `delta_repair`, `vot`, `traffic_cost_factor`, `risk_base`.

`T_tail` is the evaluation tail length in **years** (default = 1× `e_fail_mean`); `configs.py`
converts it to epochs and adds it past the planning horizon `T = round(years / dt)`.

### `experiments/configs.py`

`input_dim = 5 * env.config.n_assets` (line ~213) — correct, 5N not 4N.

### `agents/value_fn.py — NeuralValueFn`

Input dim = **5N** (not 4N). **Implemented** — torch MLP (`hidden_dims` default `(256, 256)`),
selected via `value_fn: "neural"`. File lives at `agents/fn/value_fn.py`. **Normalizes inputs and
targets** (z-scores features; standardizes the target, frozen on first `fit`, un-normalized on
`predict`). This is essential: ADP value targets are euro cost-to-go ~1e11, and an un-normalized MLP
collapses to a constant (→ do-nothing policy). XGBoost needs no such scaling (tree, scale-invariant).
The same return/input normalization was applied to the PPO critic and the policy networks
(`agents/ppo.py`, `agents/fn/policy.py`, `build_policy_input`).

---

## 6. Current Status

### Implemented and working
- `env/network.py`: Sioux Falls network, `load_sioux_falls(n_assets=N)`
- `env/degradation.py`: `gamma_step`, `wiener_step`
- `env/tap.py`: `FastTAP` (numba FW), `NullTAP`, `make_tap`
- `env/mdp.py`: Full MDP with 5-variable state, escalating risk cost, VOT travel cost
- `agents/heuristics.py`: `ReactiveAgent`, `PacedAgent`, `PerAssetReactiveAgent`, and (network-aware
  / lifetime-aware baselines) `LeadTimeAgent`, `NetConcurrencyAgent`, `HoldingAgent`,
  `ValueDensityAgent`, `WorstFirstAgent`. All tunable via `agent_type: optuna_heuristic`
  (`heuristic_type`); any can be an ADP warmstart or MC-rollout base (see `_build_agent`).
- `agents/value_fn.py`: `XGBoostValueFn`, `NeuralValueFn` (input/target normalized)
- `agents/action_gen.py`: `LocalSearchGenerator` (steepest-descent, **batched** —
  one `value_fn.predict` per sweep, not per candidate; ~5.5x faster for XGBoost),
  `SequentialGenerator`
- `agents/dqn.py`: `DQNAgent`
- `agents/actor_critic.py`: `PolicyNetwork`, `ActorCriticAgent`; `agents/ppo.py`: PPO (normalized)
- `agents/ranking.py`: `RankingValueFn`
- `agents/rollout.py`: `MonteCarloRolloutAgent` (fixed/adaptive Wilcoxon budgeting)
- `training/buffer.py`: FIFO, lowest-error, stochastic knockout
- `training/trainer.py`: Full training loop; `none`/`bootstrap`/`horizon_rollout` truncation (tail
  simulated in training for `horizon_rollout`); time-based periodic eval (`eval_interval_seconds`)
- `experiments/configs.py`: Full wiring via `build_experiment`
- `experiments/generate_instance.py`: Length-based parameter generation

### Stubs (raise NotImplementedError)
- `CTDEAgent` (`agents/marl.py`)
- `BDQGenerator` (`agents/action_gen.py`)
- `SurrogateTAP.load_pretrained` (`env/surrogate.py`)
- `load_amsterdam` (`env/network.py`)

---

## 7. Notation Mapping (paper → code)

| Paper | Code |
|-------|------|
| `d_i^t` | `state.d[i]` |
| `h_i^t` | `state.h[i]` |
| `ℓ_i^t` | `state.ell[i]` |
| `r_i^t` | `state.r[i]` |
| `n_i^t` | `state.n_fail[i]` |
| `α_i^0` | `config.alpha0[i]` |
| `β_i` | `config.beta[i]` |
| `μ^h` | `config.mu_h` (scalar or per-asset array) |
| `σ^h` | `config.sigma_h` (scalar or per-asset array) |
| `η^ren` | `config.eta_ren` |
| `η^load` | `config.eta_load` |
| `f_degrad` | `config.restrict_degrad_multiplier` |
| `δ` | `config.delta_repair` |
| `L_i` | `config.asset_lengths_m[i]` |
| `r_base` | `config.risk_base` |
| `α_tcf` | `config.traffic_cost_factor` |
| `ν_vot` | `config.vot` |
| `B_travel` | `env._c_travel_baseline` |
| `T` | `config.T` |
| `Δt` | `config.dt` |
| `γ` | `config.gamma` (annual; per-epoch = `gamma**dt`, derived in `configs.py`) |

---

## 8. Key Design Principles

- **No global random state; environment randomness is stateless and phase-keyed.** There is no mutable `env.rng`. All env noise derives from `env.noise.keyed_philox("phase", seed, episode_idx, ...)` (blake2b→Philox); callers arm each episode with `env.begin_episode(phase, episode_idx, base_seed=None)`. Distinct `phase` tags ("training"/"evaluation"/"warmstart"/"optuna_trial"/"rollout") give structurally independent streams; evaluation omits agent identity from the key → **shared CRN across agents** (paired comparison). Runs are a pure function of `(config.seed, phase, episode_idx)` — reproducible, resume-safe, parallelism-invariant; no env RNG is checkpointed. Other RNGs (warmstart action-flip, action_gen permutation, model fitting) still pass an explicit `np.random.Generator`.
- **`ADPAgent` vs `DQNAgent` (separate classes, no `bootstrap_mode` param).** `ADPAgent`: target = `mc_return - cost`; V' trained on post-decision features. `DQNAgent`: target = `cost + γ·V(s_next; θ_prev)` — natural for standard DQN/MARL. Both `post_state` and `next_state` are stored in every `Transition`.
- **TAP as swappable callable.** `InfraEnv` takes `tap_fn: TAPSolver`. Config selects `'fast'` (default) or `'null'`. `make_tap` raises `ValueError` for any other backend string.
- **Action generators rank by `Q(s,a) = c_maint + c_risk + c_travel + V'(s_post)`.** All three cost components that are deterministic given (s,a) are computed explicitly. `env.travel_cost(s_post)` calls TAP during search — this is intentional. `env.step()` also calls TAP once for the committed action.
- **Feasibility is checked strictly.** `env.assert_feasible()` raises `ValueError` on infeasible actions (unlike the spec which said silent masking — code validates).
- **Action generators must pre-check feasibility before calling `post_decision_state()`.** Compute `feas = env.feasible_actions(state)` once and check `feas[i, a]` before evaluating each candidate. Pass `check=False` to `post_decision_state()` to skip the redundant internal `assert_feasible()` call. External / novel agents that skip pre-checking should use the default `check=True`.
- **Vectorize over assets.** Degradation and renovation updates are one numpy call each.
- **AequilibraE graph reuse.** Built once on `AequilibraeTAP.__init__`, capacity arrays updated in place per `solve()`.
- **TAP caches are bounded.** Both `FastTAP` and `AequilibraeTAP` use `_BoundedCache(maxsize=10_000)` (LRU eviction) to prevent unbounded memory growth over long training runs.
- **Multi-core parallelization via `n_workers`.** Single config parameter in `TrainingConfig` (JSON: `training.n_workers`, default 1). Rules: (1) evaluation always parallelizes episodes across workers; (2) value-based training parallelizes episode collection between VFA updates; (3) `MonteCarloRolloutAgent` parallelizes the N rollouts per Q estimation call using a persistent pool; (4) only ONE layer of parallelism active at a time (no nesting); (5) uses `multiprocessing.get_context('spawn')` on Windows; (6) each worker reconstructs its own `InfraEnv` + TAP from `(network, tap_backend, env_config, seed)` to avoid pickling numba/TAP objects; (7) `n_workers=1` produces identical results to pre-parallelization code.
