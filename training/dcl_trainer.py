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


@dataclass
class DCLConfig:
    # Approximate-policy-iteration loop
    n_rounds: int = 3
    samples_per_round: int = 2000          # labelled (state, action) pairs per round
    warmup_steps: int = 100                # L: on-policy warm-up before labelling
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


class DCLTrainer:
    def __init__(self, agent, env: InfraEnv, config: DCLConfig, logger):
        self.agent = agent                  # DCLAgent (policy = decomposition)
        self.env = env
        self.config = config
        self.logger = logger
        self.seed = config.seed
        self.value_fn = self._build_value_fn() if config.rollout_horizon is not None else None
        self._last_checkpoint_dir: str | None = None

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
        """Warm-up under the current policy, then label states with the oracle and
        step forward under the label. Returns (states, labels, vf_states,
        vf_targets); the vf_* lists are empty when no VFA is used."""
        cfg = self.config
        gamma = self.env.config.gamma
        eval_length = self.env.config.T + int(cfg.T_tail / self.env.config.dt)
        cap = cfg.collect_steps if cfg.collect_steps > 0 else eval_length
        # Always leave room for >= 1 labelled step after warm-up, else an episode
        # contributes nothing and the collection loop would spin forever.
        warmup = max(0, min(cfg.warmup_steps, eval_length - 1))
        max_episodes = 100 + 10 * cfg.samples_per_round   # safety against no progress

        states, labels = [], []
        vf_states, vf_targets = [], []
        ep_idx = 0
        with tqdm(total=cfg.samples_per_round,
                  desc=f"DCL round {round_idx} collect", unit="state") as pbar:
            while len(states) < cfg.samples_per_round and ep_idx < max_episodes:
                # A dedicated phase tag keeps this stream independent from the
                # real "training"/"evaluation" streams (no clairvoyance).
                self.env.begin_episode("dcl_collect", round_idx * 1_000_000 + ep_idx)
                s = self.env.reset()
                ep_idx += 1

                # Warm-up L steps under the base policy (no labelling).
                t = 0
                for _ in range(warmup):
                    a = self.agent.act(s)
                    s, _, _ = self.env.step(s, a)
                    t += 1

                # Label and step forward under the improved action.
                ep_states, ep_costs = [], []
                steps = 0
                stopped_at_done = False
                while steps < cap and t < eval_length and len(states) < cfg.samples_per_round:
                    label = oracle.act(s)
                    states.append(s.copy())
                    labels.append(label.copy())
                    ep_states.append(s.copy())
                    s_next, cost, done = self.env.step(s, label)
                    ep_costs.append(cost)
                    s = s_next
                    t += 1
                    steps += 1
                    pbar.update(1)
                    if t >= eval_length:
                        stopped_at_done = True
                        break

                # Value targets: n-step return-to-go under the improved policy.
                # Bootstrap with the (previous round's) VFA when we stopped before
                # the horizon end; 0 at the horizon end.
                if self.value_fn is not None and ep_states:
                    if stopped_at_done:
                        G = 0.0
                    else:
                        G = float(self.value_fn.predict([s])[0])
                    for st, c in zip(reversed(ep_states), reversed(ep_costs)):
                        G = c + gamma * G
                        vf_states.append(st)
                        vf_targets.append(G)
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
        for i in tqdm(range(remaining), desc="Evaluating", unit="ep"):
            ep_idx = completed + i
            env.begin_episode("evaluation", ep_idx)     # shared CRN across agents
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
