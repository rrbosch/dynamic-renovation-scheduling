"""Deep Controlled Learning training loop (approximate policy iteration).

Runs `n_rounds` of: (1) collect a fresh ON-POLICY dataset under the current
policy (warm-up L steps, then label visited states with the rollout-improvement
oracle and step forward under the label); (2) train the classifier from scratch;
(3) the trained classifier becomes the base/rollout policy for the next round.

Mirrors PPOTrainer: dispatched from `build_experiment`, owns the loop, exposes
`train` / `evaluate` / checkpoint hooks with the same signatures `run.py` expects.
The optional truncated-rollout VFA bootstrap (DCL's compute shortcut, active only
when `rollout_horizon` is set) is owned here and passed to the oracle.
"""
from __future__ import annotations

import math
import os
import json
import shutil
import time
from dataclasses import dataclass

import numpy as np

try:
    from tqdm import tqdm
except ImportError:                       # progress bar is cosmetic; degrade gracefully
    def tqdm(iterable=None, **kwargs):
        if iterable is not None:
            return iterable

        class _Null:
            def update(self, *a, **k): pass
            def __enter__(self): return self
            def __exit__(self, *a): return False
        return _Null()

from env.mdp import InfraEnv


# ---------------------------------------------------------------------------
# Module-level collection helpers (top-level so they pickle for `spawn` workers)
# ---------------------------------------------------------------------------

def _init_worker():
    """Pin BLAS/OpenMP/torch to a single thread inside each collection worker, so
    the parallelism stays at the PROCESS layer (one layer at a time, CLAUDE.md §8
    rule 4). Without this, 16 worker processes each running multi-threaded xgboost
    `predict` (the round>=1 base policy) would oversubscribe the 16 cores."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


def _collect_one_episode(agent, env, oracle, value_fn, round_idx, global_ep,
                         n_collect, eval_length, gamma, base_seed):
    """Collect one on-policy episode: label `n_collect` states (starting at t=0)
    with the rollout oracle and step forward under the label. Returns
    (states, labels, vf_states, vf_targets). Keyed on (base_seed, GLOBAL episode
    index) so the result is independent of worker assignment AND of how each
    worker's env was constructed.

    No burn-in: this is a finite-horizon MDP, so labelling starts at t=0 to cover
    the full horizon the deployed classifier sees (see
    docs/dcl_faithful_vs_hybrid.md)."""
    env.begin_episode("dcl_collect", round_idx * 1_000_000 + global_ep, base_seed)
    s = env.reset()
    t = 0

    states, labels, ep_states, ep_costs = [], [], [], []
    for _ in range(n_collect):
        if t >= eval_length:
            break
        label = oracle.act(s)
        states.append(s.copy())
        labels.append(label.copy())
        ep_states.append(s.copy())
        s, cost, _ = env.step(s, label)
        ep_costs.append(cost)
        t += 1

    vf_states, vf_targets = [], []
    if value_fn is not None and ep_states:
        # n-step return-to-go under the improved policy; bootstrap with the VFA
        # when we stopped before the horizon end, else 0 at the end.
        G = 0.0 if t >= eval_length else float(value_fn.predict([s])[0])
        for st, c in zip(reversed(ep_states), reversed(ep_costs)):
            G = c + gamma * G
            vf_states.append(st)
            vf_targets.append(G)
    return states, labels, vf_states, vf_targets


def _run_dcl_collect_episode(args):
    """Worker entry: reconstruct env + oracle from the pickled agent, then collect
    one episode. Module-level for `spawn` pickling (mirrors `_run_train_episode`)."""
    (agent, env_config, network, tap_backend, base_seed, round_idx, global_ep,
     dcl_cfg, value_fn, n_collect, eval_length, gamma) = args
    from env.tap import make_tap
    env = InfraEnv(network, make_tap(network, backend=tap_backend), env_config,
                   rng_seed=base_seed)
    # Point the policy at this worker's fresh env so the oracle's rollouts and the
    # warm-up share one (config-identical) env instance.
    agent.policy.env = env
    agent.policy.cfg = env.config
    agent.policy.set_predict_threads(1)        # avoid xgboost thread oversubscription
    oracle = agent.policy.make_oracle(agent, dcl_cfg, base_seed + round_idx, value_fn)
    return _collect_one_episode(agent, env, oracle, value_fn, round_idx, global_ep,
                                n_collect, eval_length, gamma, base_seed)


@dataclass
class DCLConfig:
    # Approximate-policy-iteration loop
    n_rounds: int = 3
    samples_per_round: int = 2000          # labelled (state, action) pairs per round
    collect_steps: int = 0                 # per-episode labelled cap; 0 = to episode end
    # Rollout-improvement oracle
    rollout_horizon: int | None = None     # None = full rollout (value-function-free)
    n_rollouts: int = 30
    rollout_selection: str = 'fixed'       # 'fixed' | 'wilcoxon' | 'sequential_halving'
    sh_budget_per_arm: int | None = None
    p_threshold: float = 0.02
    min_rollouts: int = 20
    max_rollouts: int = 100
    rollout_batch: int = 5
    action_threshold: float = 0.0
    initial_action: str = 'policy'
    # Optional VFA tail bootstrap (only built/used when rollout_horizon is set)
    value_fn_kind: str = 'xgboost'
    finite_horizon: bool = True
    # Evaluation / infra
    eval_interval: int = 1                 # evaluate every N rounds
    n_eval_episodes: int = 10
    time_budget: float = 86400.0
    T_tail: float = 10.0
    seed: int = 0
    config_hash: str = ''
    n_workers: int = 1                     # process pool for collection + eval
    tap_backend: str = 'fast'              # for env reconstruction in workers


class DCLTrainer:
    def __init__(self, agent, env: InfraEnv, config: DCLConfig, logger):
        self.agent = agent                  # DCLAgent (policy = decomposition)
        self.env = env
        self.config = config
        self.logger = logger
        self.seed = config.seed
        self.value_fn = self._build_value_fn() if config.rollout_horizon is not None else None
        self._last_checkpoint_dir: str | None = None
        # TAP backend for env reconstruction in spawn workers. Prefer the config
        # value; fall back to the tag build_experiment stamps on the env.
        self._tap_backend = config.tap_backend or getattr(env, '_tap_backend_name', 'fast')

    def _build_value_fn(self):
        if self.config.value_fn_kind == 'neural':
            from agents.fn.value_fn import NeuralValueFn
            return NeuralValueFn(finite_horizon=self.config.finite_horizon)
        from agents.fn.value_fn import XGBoostValueFn
        return XGBoostValueFn(finite_horizon=self.config.finite_horizon)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def train(self, start_ep: int = 0, already_elapsed: float = 0.0) -> None:
        cfg = self.config
        t_start = time.monotonic()
        decomposition = self.agent.policy

        for rnd in range(start_ep, cfg.n_rounds):
            elapsed = already_elapsed + (time.monotonic() - t_start)
            if cfg.time_budget and elapsed >= cfg.time_budget:
                print(f"[DCL] Time budget reached after {rnd} rounds.")
                break

            # Base/rollout policy for this round = current deployed agent (round 0
            # acts via the heuristic; round >= 1 via the trained classifier).
            oracle = decomposition.make_oracle(
                self.agent, cfg, seed=self.seed + rnd, value_fn=self.value_fn)

            states, labels, vf_states, vf_targets = self._collect_dataset(oracle, rnd)
            print(f"[DCL] Round {rnd}: collected {len(states)} labelled states "
                  f"(selection={cfg.rollout_selection}, "
                  f"horizon={cfg.rollout_horizon}).")

            # Train the classifier from scratch on this round's fresh dataset.
            decomposition.fit(states, np.stack(labels))

            # Fit the optional VFA (pre-decision cost-to-go) for the next round's
            # truncation bootstrap.
            if self.value_fn is not None and vf_states:
                self.value_fn.fit_targets(vf_states, np.asarray(vf_targets, dtype=float))

            if (rnd + 1) % cfg.eval_interval == 0 or rnd == cfg.n_rounds - 1:
                results = self.evaluate(cfg.n_eval_episodes)
                elapsed = already_elapsed + (time.monotonic() - t_start)
                self.logger.log_step(rnd + 1, {
                    'round': rnd,
                    'mean_cost': results['mean_cost'],
                    'std_cost': results['std_cost'],
                    'n_labelled': len(states),
                })
                print(f"[DCL] Round {rnd:3d} | {elapsed:6.0f}s | "
                      f"mean_cost={results['mean_cost']:.2f} ± {results['std_cost']:.2f}")

            self._save_checkpoint(rnd, already_elapsed + (time.monotonic() - t_start))

        self._mark_complete()

    # ------------------------------------------------------------------
    # Dataset collection
    # ------------------------------------------------------------------
    def _collect_dataset(self, oracle, round_idx: int):
        """Collect this round's on-policy dataset over a FIXED number of episodes
        (so the result is a pure function of (seed, round) — identical for any
        `n_workers`, including the serial path). Returns (states, labels,
        vf_states, vf_targets); the vf_* lists are empty when no VFA is used.

        Each episode warms up under the current policy then labels `n_collect`
        states with the oracle. Episodes are independent → embarrassingly parallel
        across `n_workers` (DCL's "W threads")."""
        cfg = self.config
        gamma = self.env.config.gamma
        eval_length = self.env.config.T + int(cfg.T_tail / self.env.config.dt)
        cap = cfg.collect_steps if cfg.collect_steps > 0 else eval_length
        # No burn-in (finite horizon): every episode labels from t=0. Per-episode yield is fixed.
        n_collect = max(1, min(cap, eval_length))
        n_episodes = max(1, math.ceil(cfg.samples_per_round / n_collect))

        if cfg.n_workers > 1:
            from multiprocessing import get_context
            args_list = [
                (self.agent, self.env.config, self.env.network, self._tap_backend,
                 self.seed, round_idx, e, cfg, self.value_fn,
                 n_collect, eval_length, gamma)
                for e in range(n_episodes)
            ]
            ctx = get_context('spawn')
            with ctx.Pool(processes=cfg.n_workers, initializer=_init_worker) as pool:
                # imap (not imap_unordered) preserves order → deterministic concat.
                results = list(tqdm(pool.imap(_run_dcl_collect_episode, args_list),
                                    total=n_episodes,
                                    desc=f"DCL round {round_idx} collect", unit="ep"))
        else:
            results = [
                _collect_one_episode(self.agent, self.env, oracle, self.value_fn,
                                     round_idx, e, n_collect, eval_length,
                                     gamma, self.seed)
                for e in tqdm(range(n_episodes),
                              desc=f"DCL round {round_idx} collect", unit="ep")
            ]

        states, labels, vf_states, vf_targets = [], [], [], []
        for st, lb, vs, vt in results:
            states += st
            labels += lb
            vf_states += vs
            vf_targets += vt
        return states, labels, vf_states, vf_targets

    # ------------------------------------------------------------------
    # Evaluation (classifier-only; mirrors Trainer/PPOTrainer.evaluate)
    # ------------------------------------------------------------------
    def evaluate(self, n_episodes: int | None = None, resume: bool = False,
                 save_episodes: bool = False, env: InfraEnv | None = None) -> dict:
        if n_episodes is None:
            n_episodes = self.config.n_eval_episodes
        env = env or self.env
        gamma = env.config.gamma
        tail_epochs = int(self.config.T_tail / env.config.dt)
        eval_length = env.config.T + tail_epochs
        agent = self.agent
        persist = bool(save_episodes)

        completed = 0
        prior_costs: list[float] = []
        if resume and persist:
            completed = self.logger.count_completed_eval_episodes()
            if completed > 0:
                prior_costs = self.logger.load_eval_episode_costs(gamma)
        remaining = n_episodes - completed
        if remaining <= 0:
            episode_costs = prior_costs[:n_episodes]
            return {'mean_cost': float(np.mean(episode_costs)),
                    'std_cost': float(np.std(episode_costs)), 'episodes': []}

        if persist:
            self.logger.start_eval(append=(completed > 0))

        episode_costs, episodes = [], []
        if self.config.n_workers > 1 and env is self.env:
            # Parallel eval — reuse the standard episode worker (shared-CRN keying,
            # full T+tail horizon). Each worker reconstructs its own env.
            from multiprocessing import get_context
            from training.trainer import _run_eval_episode
            args_list = [
                (agent, env.config, env.network, self._tap_backend,
                 self.seed, completed + i, eval_length)
                for i in range(remaining)
            ]
            ctx = get_context('spawn')
            with ctx.Pool(processes=self.config.n_workers, initializer=_init_worker) as pool:
                for ep_idx, cost, ep_data in tqdm(
                    pool.imap_unordered(_run_eval_episode, args_list),
                    total=remaining, desc="Evaluating", unit="ep",
                ):
                    if persist:
                        self.logger.append_episode(ep_idx, ep_data)
                        self.logger.append_agent_metrics(ep_idx, ep_data)
                    episode_costs.append(cost)
                    episodes.append(ep_data)
        else:
            for i in tqdm(range(remaining), desc="Evaluating", unit="ep"):
                ep_idx = completed + i
                env.begin_episode("evaluation", ep_idx, self.seed)   # shared CRN across agents
                state = env.reset()
                total_cost = 0.0
                ep_data = []
                for t in range(eval_length):
                    action = agent.act(state)
                    _m = dict(getattr(agent, 'step_metrics', None) or {})
                    next_state, cost, done = env.step(state, action)
                    c_travel, c_maint, c_risk = env.last_cost_breakdown
                    total_cost += (gamma ** t) * cost
                    ep_data.append({'t': t, 'state': state.copy(), 'action': action,
                                    'cost': cost, 'c_travel': c_travel, 'c_maint': c_maint,
                                    'c_risk': c_risk, 'agent_metrics': _m})
                    state = next_state
                    # No break on done: simulate the full T + tail horizon.
                if persist:
                    self.logger.append_episode(ep_idx, ep_data)
                    self.logger.append_agent_metrics(ep_idx, ep_data)
                episode_costs.append(total_cost)
                episodes.append(ep_data)

        all_costs = prior_costs + episode_costs
        return {'mean_cost': float(np.mean(all_costs)),
                'std_cost': float(np.std(all_costs)), 'episodes': episodes}

    # ------------------------------------------------------------------
    # Checkpoint save / load (round-based; reuses Trainer's ep_<N> layout so
    # run.py auto-resume finds it)
    # ------------------------------------------------------------------
    def _save_checkpoint(self, rnd: int, elapsed: float) -> None:
        ckpt_dir = os.path.join(str(self.logger.run_dir), 'checkpoints', f'ep_{rnd}')
        agent_dir = os.path.join(ckpt_dir, 'agent')
        os.makedirs(agent_dir, exist_ok=True)

        metadata = {'episode': rnd, 'elapsed_seconds': elapsed,
                    'config_hash': self.config.config_hash}
        meta_path = os.path.join(ckpt_dir, 'metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f)
        shutil.copy2(meta_path, os.path.join(ckpt_dir, 'metadata_backup.json'))

        self.agent.save(agent_dir)
        if self.value_fn is not None:
            self.value_fn.save(os.path.join(ckpt_dir, 'value_fn.pkl'))

        if self._last_checkpoint_dir and os.path.exists(self._last_checkpoint_dir) \
                and self._last_checkpoint_dir != ckpt_dir:
            shutil.rmtree(self._last_checkpoint_dir, ignore_errors=True)
        self._last_checkpoint_dir = ckpt_dir
        print(f"[Checkpoint] DCL round {rnd} -> {ckpt_dir}")

    def _mark_complete(self) -> None:
        if self._last_checkpoint_dir is None:
            return
        meta_path = os.path.join(self._last_checkpoint_dir, 'metadata.json')
        try:
            with open(meta_path) as f:
                metadata = json.load(f)
            metadata['complete'] = True
            with open(meta_path, 'w') as f:
                json.dump(metadata, f)
            shutil.copy2(meta_path, meta_path.replace('metadata.json', 'metadata_backup.json'))
        except OSError:
            pass

    def load_checkpoint(self, checkpoint_dir: str) -> int:
        with open(os.path.join(checkpoint_dir, 'metadata.json')) as f:
            metadata = json.load(f)
        rnd = metadata['episode']
        self.agent.load(os.path.join(checkpoint_dir, 'agent'))
        vf_path = os.path.join(checkpoint_dir, 'value_fn.pkl')
        if self.value_fn is not None and os.path.exists(vf_path):
            self.value_fn.load(vf_path)
        self._last_checkpoint_dir = checkpoint_dir
        print(f"[Checkpoint] DCL resumed from round {rnd}")
        return rnd + 1
