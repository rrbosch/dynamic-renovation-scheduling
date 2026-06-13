"""Training loop and configuration."""
from __future__ import annotations

import gc
import os
import sys
import time
import warnings
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING

from env.mdp import InfraEnv
from training.buffer import ReplayBuffer, Transition

if TYPE_CHECKING:
    from agents.base import Agent
    from utils.logging import RunLogger

try:
    import psutil
    _proc = psutil.Process(os.getpid())
    def _mem_mb(): return _proc.memory_info().rss / (1024**2)
except ImportError:
    def _mem_mb(): return 0.0  # fallback: never triggers


def _run_eval_episode(args):
    """Run a single evaluation episode. Module-level for pickling (multiprocessing)."""
    agent, env_config, network, tap_backend, base_seed, episode_idx, eval_length = args
    from env.mdp import InfraEnv
    from env.tap import make_tap
    env = InfraEnv(network, make_tap(network, backend=tap_backend), env_config, rng_seed=base_seed)
    # Shared-CRN evaluation: keyed only on (base_seed, episode_idx), not the agent.
    env.begin_episode("evaluation", episode_idx, base_seed)
    state = env.reset()
    total_cost = 0.0
    gamma = env.config.gamma
    ep_data = []
    for t in range(eval_length):
        action = agent.act(state)
        _step_metrics = dict(getattr(agent, 'step_metrics', None) or {})
        next_state, cost, done = env.step(state, action)
        c_travel, c_maint, c_risk = env.last_cost_breakdown
        discounted = (gamma ** t) * cost
        total_cost += discounted
        ep_data.append({'t': t, 'state': state.copy(), 'action': action,
                        'cost': cost, 'c_travel': c_travel,
                        'c_maint': c_maint, 'c_risk': c_risk,
                        'agent_metrics': _step_metrics})
        state = next_state
        # Intentionally do NOT break on `done`: evaluation uses the
        # horizon_rollout horizon (T + tail_epochs), simulating the tail past the
        # planning horizon T. `done` fires at t=T but the env keeps stepping.
    # Echo episode_idx back: imap_unordered returns out of order, but the saved
    # index must match the seed used (shared-CRN evaluation).
    return episode_idx, total_cost, ep_data


def _run_train_episode(args):
    """Collect one training episode. Module-level for pickling (multiprocessing)."""
    (agent, env_config, network, tap_backend, base_seed, episode_idx,
     ep_length, truncation_mode, gamma) = args
    from env.mdp import InfraEnv
    from env.tap import make_tap
    env = InfraEnv(network, make_tap(network, backend=tap_backend), env_config, rng_seed=base_seed)
    env.begin_episode("training", episode_idx, base_seed)
    state = env.reset()
    ep_transitions = []
    for t in range(ep_length):
        action = agent.act(state)
        next_state, cost, done = env.step(state, action)
        post_state = env.post_decision_state(state, action)
        tr = Transition(
            state=state.copy(), action=action.copy(), cost=cost,
            next_state=next_state.copy(), post_state=post_state.copy(), done=done,
        )
        ep_transitions.append(tr)
        state = next_state
        if done:
            break
    # MC backward pass
    if truncation_mode == 'bootstrap' and hasattr(agent, 'value_fn'):
        G = agent.value_fn.predict([ep_transitions[-1].post_state])[0]
    elif truncation_mode == 'none':
        G = 0.0
    else:
        G = 0.0
    for tr in reversed(ep_transitions):
        G = tr.cost + gamma * G
        tr.mc_return = G
    return ep_transitions


def _rmtree_with_retry(path: str, retries: int = 10, delay: float = 3.0) -> None:
    """Delete a directory tree, retrying on PermissionError (e.g. OneDrive sync locks)."""
    import shutil
    import stat

    def _onerror(func, fpath, _exc_info):
        # Make read-only files writable (common on Windows / OneDrive) then retry.
        try:
            os.chmod(fpath, stat.S_IWRITE)
            func(fpath)
        except Exception:
            pass

    for attempt in range(retries):
        try:
            shutil.rmtree(path, onerror=_onerror)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(delay)


@dataclass(frozen=True)
class TrainingConfig:
    time_budget: float = 3600.0   # wall-clock seconds; 0 or None disables
    n_episodes: int | None = None  # None = unlimited when time_budget set, else 10_000_000
    eval_interval: int = 50
    update_interval: int = 10
    truncation_mode: str = 'bootstrap'  # 'none', 'horizon_rollout', or 'bootstrap'
    T_tail: float = 10.0                # tail horizon in years (converted to epochs via T_tail/dt)
    buffer_capacity: int = 200_000
    buffer_strategy: str = 'fifo'
    n_eval_episodes: int = 10
    batch_size: int = 2048
    n_warmstart_episodes: int = 0           # 0 = disabled
    warmstart_agent_config: dict | None = None  # parsed from JSON 'warmstart' key
    checkpoint_interval: int = 0                 # 0 = disabled; save checkpoint every N episodes (legacy)
    checkpoint_interval_seconds: float = 1800.0  # 0 = disabled; save checkpoint every N wall-clock seconds
    config_hash: str = ''                        # stable hash of ExperimentConfig (excl. seed)
    n_workers: int = 1                           # parallel workers for eval/training episodes (1=sequential)


class Trainer:
    def __init__(
        self,
        agent: 'Agent',
        env: InfraEnv,
        config: TrainingConfig,
        logger: 'RunLogger',
        seed: int = 0,
        tap_backend: str = 'fast',
    ):
        from agents.base import Agent
        self.agent = agent
        self.env = env
        self.config = config
        self.logger = logger
        self.seed = seed
        self._tap_backend = tap_backend
        self.buffer = ReplayBuffer(
            capacity=config.buffer_capacity,
            strategy=config.buffer_strategy,
        )
        # Detect non-learning agents (those that use the base no-op update)
        self._is_learner = type(agent).update is not Agent.update
        if not self._is_learner:
            warnings.warn(
                f"{type(agent).__name__} does not implement update(); "
                "value function updates will be skipped.",
                stacklevel=2,
            )
        # Backstop for stochastic_knockout: if eviction is triggered before any
        # model has populated prediction errors, train one on the spot so the
        # knockout has a real error signal to rank by.
        if self._is_learner:
            self.buffer.refresh_errors_fn = self._refresh_buffer_errors
        self._last_checkpoint_dir: str | None = None
        self._last_checkpoint_wall: float = 0.0  # tracks elapsed time of last time-based checkpoint

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def _save_checkpoint(self, ep: int, elapsed: float) -> None:
        """Save complete training state to results/<run>/checkpoints/ep_<N>/."""
        import os, json, pickle, shutil

        ckpt_dir = os.path.join(str(self.logger.run_dir), 'checkpoints', f'ep_{ep}')
        agent_dir = os.path.join(ckpt_dir, 'agent')
        os.makedirs(agent_dir, exist_ok=True)

        def _write_primary_and_backup(primary: str, write_fn) -> None:
            write_fn(primary)
            base, ext = os.path.splitext(primary)
            shutil.copy2(primary, base + '_backup' + ext)

        # metadata.json
        metadata = {
            'episode': ep,
            'elapsed_seconds': elapsed,
            'buffer_size': len(self.buffer),
            'config_hash': self.config.config_hash,
        }
        meta_path = os.path.join(ckpt_dir, 'metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f)
        shutil.copy2(meta_path, os.path.join(ckpt_dir, 'metadata_backup.json'))

        # NOTE: no env RNG is saved — environment randomness is stateless and
        # fully determined by (phase, base_seed, episode_idx), so resume is
        # deterministic from the episode counter alone (see env.begin_episode).

        # buffer
        _write_primary_and_backup(
            os.path.join(ckpt_dir, 'buffer.pkl'),
            lambda p: self.buffer.save(p),
        )

        # agent (save() creates its own files inside agent_dir)
        self.agent.save(agent_dir)
        for fname in sorted(os.listdir(agent_dir)):
            fpath = os.path.join(agent_dir, fname)
            if not os.path.isfile(fpath):
                continue
            base, ext = os.path.splitext(fname)
            if base.endswith('_backup'):
                continue
            shutil.copy2(fpath, os.path.join(agent_dir, base + '_backup' + ext))

        # Delete the previous checkpoint to keep only the latest
        if self._last_checkpoint_dir is not None and os.path.exists(self._last_checkpoint_dir):
            _rmtree_with_retry(self._last_checkpoint_dir)
        self._last_checkpoint_dir = ckpt_dir
        agent_name = type(self.agent).__name__
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        print(f"[Checkpoint] saving {agent_name} after {h:02d}:{m:02d}:{s:02d} of training -> {ckpt_dir}")

    def _mark_complete(self) -> None:
        """Write 'complete': true into the last checkpoint's metadata.json."""
        import json, shutil
        if self._last_checkpoint_dir is None:
            return
        meta_path = __import__('os').path.join(self._last_checkpoint_dir, 'metadata.json')
        try:
            with open(meta_path) as f:
                metadata = json.load(f)
            metadata['complete'] = True
            with open(meta_path, 'w') as f:
                json.dump(metadata, f)
            shutil.copy2(meta_path, meta_path.replace('metadata.json', 'metadata_backup.json'))
        except OSError:
            pass

    @staticmethod
    def _safe_load(primary: str, backup: str, load_fn):
        """Try loading `primary`; fall back to `backup` on failure."""
        for path, label in ((primary, 'primary'), (backup, 'backup')):
            if not __import__('os').path.exists(path):
                continue
            try:
                return load_fn(path)
            except Exception as exc:
                warnings.warn(f"[Checkpoint] {label} file {path} failed to load: {exc}. "
                              "Trying backup.")
        raise RuntimeError(f"Both primary ({primary}) and backup ({backup}) failed to load.")

    def load_checkpoint(self, checkpoint_dir: str) -> int:
        """
        Restore full training state from `checkpoint_dir`.
        Returns the episode number to resume from (i.e. ep + 1).
        """
        import os, json, pickle

        ckpt_dir = checkpoint_dir
        agent_dir = os.path.join(ckpt_dir, 'agent')

        # metadata
        def _load_json(p):
            with open(p) as f:
                return json.load(f)
        metadata = self._safe_load(
            os.path.join(ckpt_dir, 'metadata.json'),
            os.path.join(ckpt_dir, 'metadata_backup.json'),
            _load_json,
        )
        ep = metadata['episode']

        # No env RNG to restore — environment randomness is stateless and keyed
        # on (phase, base_seed, episode_idx). Older checkpoints may still carry
        # an env_rng.pkl; it is intentionally ignored.

        # buffer
        def _load_buffer(p):
            self.buffer.load(p)
        self._safe_load(
            os.path.join(ckpt_dir, 'buffer.pkl'),
            os.path.join(ckpt_dir, 'buffer_backup.pkl'),
            _load_buffer,
        )

        # agent — delegate to agent.load() if it exists (non-learning agents are no-ops)
        if hasattr(self.agent, 'load') and callable(self.agent.load):
            self.agent.load(agent_dir)

        self._last_checkpoint_dir = ckpt_dir
        print(f"[Checkpoint] resumed from ep={ep}, buffer_size={len(self.buffer)}, "
              f"elapsed={metadata.get('elapsed_seconds', 0):.0f}s")
        return ep + 1  # start_ep for train()

    # ------------------------------------------------------------------
    # Buffer sampling
    # ------------------------------------------------------------------

    def _sample_batch(self) -> list[Transition]:
        """Return full buffer for value functions that retrain from scratch (e.g. XGBoost),
        or a batch_size sample for incremental learners (e.g. neural networks)."""
        vf = getattr(self.agent, 'value_fn', None)
        if getattr(vf, 'prefers_full_dataset', False):
            return self.buffer.sample(len(self.buffer))
        return self.buffer.sample(min(self.config.batch_size, len(self.buffer)))

    def _refresh_buffer_errors(self) -> None:
        """Knockout backstop: train the value function on the current buffer so
        every transition gets a finite pred_error. `agent.update` both fits the
        model and writes pred_error onto the transition objects (which the
        buffer holds by reference), so this populates the whole buffer's errors.
        Invoked by ReplayBuffer when knockout eviction fires with no error
        signal yet (all pred_error == inf)."""
        if not self._is_learner or len(self.buffer) == 0:
            return
        print(f"[Knockout backstop] no prediction-error signal yet; "
              f"training value function on {len(self.buffer)} transitions...")
        batch = self._sample_batch()
        self.agent.update(batch)

    # ------------------------------------------------------------------
    # Warmstart helpers
    # ------------------------------------------------------------------

    def _run_warmstart(self, warmstart_agent: 'Agent') -> None:
        env, cfg = self.env, self.config
        print(f"Warmstarting buffer with {type(warmstart_agent).__name__} "
              f"for {cfg.n_warmstart_episodes} episodes...")
        rng = np.random.default_rng(self.seed)
        n = env.config.n_assets
        p_flip = 1 / env.config.T
        _ws_t0 = time.monotonic()
        _ws_report_every = max(1, cfg.n_warmstart_episodes // 20)  # ~20 progress lines
        for _ws_ep in range(cfg.n_warmstart_episodes):
            env.begin_episode("warmstart", _ws_ep)
            state = env.reset()
            ep_transitions: list[Transition] = []
            for _ in range(env.config.T):
                action = warmstart_agent.act(state)
                feas = env.feasible_actions(state)        # (N, 4) bool
                for i in range(n):
                    if rng.random() < p_flip:
                        feasible_i = np.where(feas[i])[0]
                        action[i] = int(rng.choice(feasible_i))
                next_state, cost, done = env.step(state, action)
                post_state = env.post_decision_state(state, action)
                tr = Transition(
                    state=state.copy(), action=action.copy(), cost=cost,
                    next_state=next_state.copy(), post_state=post_state.copy(), done=done,
                )
                ep_transitions.append(tr)
                self.buffer.add(tr)
                state = next_state
                if done:
                    break
            # Backward MC pass — VF is untrained here so always start G=0
            G = 0.0
            for tr in reversed(ep_transitions):
                G = tr.cost + env.config.gamma * G
                tr.mc_return = G

            # Periodic progress so a slow eviction strategy is visible in the log
            if (_ws_ep + 1) % _ws_report_every == 0:
                rate = (_ws_ep + 1) / max(1e-9, time.monotonic() - _ws_t0)
                print(f"Warmstart progress: {_ws_ep + 1}/{cfg.n_warmstart_episodes} episodes, "
                      f"buffer={len(self.buffer)}, {rate:.2f} ep/s")
        print(f"Warmstart complete. Buffer size: {len(self.buffer)}")

        # Pre-train VFA on the collected heuristic data
        if self._is_learner and len(self.buffer) > 0:
            batch = self._sample_batch()
            self.agent.update(batch)
            print("VFA pre-trained on warmstart data.")

    def _resolve_warmstart_agent(self, d: dict) -> 'Agent':
        from experiments.configs import AgentConfig, _build_agent
        return _build_agent(AgentConfig.from_dict(d), self.env, self.seed)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, start_ep: int = 0, already_elapsed: float = 0.0) -> None:
        if not self._is_learner:
            return

        cfg = self.config
        env = self.env
        agent = self.agent

        if start_ep == 0 and cfg.n_warmstart_episodes > 0:
            if cfg.warmstart_agent_config is not None:
                self._run_warmstart(self._resolve_warmstart_agent(cfg.warmstart_agent_config))
            else:
                warnings.warn("n_warmstart_episodes > 0 but no 'warmstart' config provided. Skipping.")

        t_start = time.monotonic()
        self._last_checkpoint_wall = already_elapsed  # reset clock relative to resume point
        use_budget = bool(cfg.time_budget)
        # Remaining budget accounts for time already spent before this call
        remaining_budget = max(0.0, cfg.time_budget - already_elapsed) if use_budget else 0.0

        if cfg.n_episodes is not None:
            n_episodes = cfg.n_episodes
        elif use_budget:
            n_episodes = sys.maxsize       # time_budget is the sole termination criterion
        else:
            n_episodes = 10_000_000        # preserve original default

        tail_epochs = int(cfg.T_tail / env.config.dt)
        if cfg.truncation_mode == 'horizon_rollout':
            ep_length = env.config.T + tail_epochs
        else:
            ep_length = env.config.T

        _baseline_mb = _mem_mb()
        _gc_check_interval = 100

        if cfg.n_workers > 1:
            self._train_parallel(start_ep, n_episodes, ep_length, cfg, env, agent,
                                 use_budget, remaining_budget, t_start, already_elapsed,
                                 tail_epochs, _baseline_mb, _gc_check_interval)
        else:
            self._train_sequential(start_ep, n_episodes, ep_length, cfg, env, agent,
                                   use_budget, remaining_budget, t_start, already_elapsed,
                                   _baseline_mb, _gc_check_interval)

        self._mark_complete()

    def _train_sequential(self, start_ep, n_episodes, ep_length, cfg, env, agent,
                          use_budget, remaining_budget, t_start, already_elapsed,
                          _baseline_mb, _gc_check_interval):
        for ep in range(start_ep, n_episodes):
            wall_elapsed = time.monotonic() - t_start
            if use_budget and wall_elapsed >= remaining_budget:
                print(f"Time budget of {cfg.time_budget:.0f}s reached after {ep} episodes.")
                break
            env.begin_episode("training", ep)
            state = env.reset()
            ep_transitions: list[Transition] = []

            # Episode rollout
            for t in range(ep_length):
                action = agent.act(state)
                next_state, cost, done = env.step(state, action)
                post_state = env.post_decision_state(state, action)

                tr = Transition(
                    state=state.copy(),
                    action=action.copy(),
                    cost=cost,
                    next_state=next_state.copy(),
                    post_state=post_state.copy(),
                    done=done,
                )
                ep_transitions.append(tr)
                self.buffer.add(tr)
                state = next_state

                if done:
                    break

            # Compute MC returns (backward pass)
            if (cfg.truncation_mode == 'bootstrap'
                    and hasattr(self.agent, 'value_fn')
                    and len(ep_transitions) > 0):
                G = self.agent.value_fn.predict([ep_transitions[-1].post_state])[0]
            elif cfg.truncation_mode == 'none':
                G = 0.0
            else:  # horizon_rollout
                G = 0.0

            for tr in reversed(ep_transitions):
                G = tr.cost + env.config.gamma * G
                tr.mc_return = G

            # Value function update
            if self._is_learner and (ep + 1) % cfg.update_interval == 0 and len(self.buffer) > 0:
                batch = self._sample_batch()
                agent.update(batch)
                del batch

            # Memory-pressure-based GC
            if (ep + 1) % _gc_check_interval == 0:
                if _mem_mb() > _baseline_mb * 1.5:
                    gc.collect()
                    _baseline_mb = _mem_mb()

            # Evaluation
            if (ep + 1) % cfg.eval_interval == 0:
                results = self.evaluate(cfg.n_eval_episodes)
                self.logger.log_step(ep + 1, results)
                elapsed = already_elapsed + (time.monotonic() - t_start)
                print(f"Episode {ep+1:4d} | {elapsed:6.0f}s | "
                      f"mean_cost={results['mean_cost']:.2f} "
                      f"± {results['std_cost']:.2f}")

            # Checkpoint
            if cfg.checkpoint_interval_seconds > 0:
                elapsed = already_elapsed + (time.monotonic() - t_start)
                if elapsed - self._last_checkpoint_wall >= cfg.checkpoint_interval_seconds:
                    self._save_checkpoint(ep + 1, elapsed)
                    self._last_checkpoint_wall = elapsed
            elif cfg.checkpoint_interval > 0 and (ep + 1) % cfg.checkpoint_interval == 0:
                elapsed = already_elapsed + (time.monotonic() - t_start)
                self._save_checkpoint(ep + 1, elapsed)

    def _train_parallel(self, start_ep, n_episodes, ep_length, cfg, env, agent,
                        use_budget, remaining_budget, t_start, already_elapsed,
                        tail_epochs, _baseline_mb, _gc_check_interval):
        """Training loop that collects batches of episodes in parallel between updates."""
        from multiprocessing import get_context

        batch_size = cfg.update_interval  # collect this many episodes per parallel batch
        ep = start_ep

        while ep < n_episodes:
            wall_elapsed = time.monotonic() - t_start
            if use_budget and wall_elapsed >= remaining_budget:
                print(f"Time budget of {cfg.time_budget:.0f}s reached after {ep} episodes.")
                break

            # How many episodes in this batch
            n_batch = min(batch_size, n_episodes - ep)

            # Episode env seeds are stateless: each worker keys its env on
            # ("training", self.seed, global_episode_idx) via begin_episode, so
            # results are parallelism- and resume-invariant.
            args_list = [
                (agent, env.config, env.network, self._tap_backend,
                 self.seed, ep + i, ep_length, cfg.truncation_mode, env.config.gamma)
                for i in range(n_batch)
            ]

            ctx = get_context('spawn')
            with ctx.Pool(processes=cfg.n_workers) as pool:
                all_ep_transitions = pool.map(_run_train_episode, args_list)

            # Add transitions to buffer
            for ep_transitions in all_ep_transitions:
                for tr in ep_transitions:
                    self.buffer.add(tr)

            ep += n_batch

            # Value function update
            if self._is_learner and len(self.buffer) > 0:
                batch = self._sample_batch()
                agent.update(batch)
                del batch

            # Memory-pressure-based GC
            if ep % _gc_check_interval == 0:
                if _mem_mb() > _baseline_mb * 1.5:
                    gc.collect()
                    _baseline_mb = _mem_mb()

            # Evaluation
            if ep % cfg.eval_interval == 0:
                results = self.evaluate(cfg.n_eval_episodes)
                self.logger.log_step(ep, results)
                elapsed = already_elapsed + (time.monotonic() - t_start)
                print(f"Episode {ep:4d} | {elapsed:6.0f}s | "
                      f"mean_cost={results['mean_cost']:.2f} "
                      f"± {results['std_cost']:.2f}")

            # Checkpoint
            if cfg.checkpoint_interval_seconds > 0:
                elapsed = already_elapsed + (time.monotonic() - t_start)
                if elapsed - self._last_checkpoint_wall >= cfg.checkpoint_interval_seconds:
                    self._save_checkpoint(ep, elapsed)
                    self._last_checkpoint_wall = elapsed
            elif cfg.checkpoint_interval > 0 and ep % cfg.checkpoint_interval == 0:
                elapsed = already_elapsed + (time.monotonic() - t_start)
                self._save_checkpoint(ep, elapsed)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, n_episodes: int | None = None, resume: bool = False,
                 save_episodes: bool = False) -> dict:
        """
        Run agent greedily for n_episodes.
        Always uses T + tail_epochs (horizon_rollout) regardless of training truncation mode.
        Returns {'mean_cost': float, 'std_cost': float, 'episodes': list}.

        If save_episodes=True, each completed episode is written to eval_episodes.csv
        incrementally. If resume=True, already-completed episodes are skipped.
        """
        if n_episodes is None:
            n_episodes = self.config.n_eval_episodes

        env = self.env
        agent = self.agent
        tail_epochs = int(self.config.T_tail / env.config.dt)
        eval_length = env.config.T + tail_epochs

        # Resume: detect already-completed episodes
        completed = 0
        prior_costs: list[float] = []
        if resume and save_episodes:
            completed = self.logger.count_completed_eval_episodes()
            if completed > 0:
                print(f"Resuming evaluation: {completed}/{n_episodes} episodes already done")
                prior_costs = self.logger.load_eval_episode_costs(env.config.gamma)
        remaining = n_episodes - completed

        if remaining <= 0:
            # All episodes already done — return stats from disk
            episode_costs = prior_costs[:n_episodes]
            return {
                'mean_cost': float(np.mean(episode_costs)),
                'std_cost': float(np.std(episode_costs)),
                'episodes': [],
            }

        # Set up incremental saving
        if save_episodes:
            self.logger.start_eval(append=(completed > 0))

        episode_costs = []
        episodes = []

        if self.config.n_workers > 1:
            from multiprocessing import get_context
            # Shared-CRN evaluation: each episode env is keyed on
            # ("evaluation", self.seed, episode_idx) inside the worker.
            args_list = [
                (agent, env.config, env.network, self._tap_backend,
                 self.seed, completed + i, eval_length)
                for i in range(remaining)
            ]
            ctx = get_context('spawn')
            with ctx.Pool(processes=self.config.n_workers) as pool:
                for ep_idx, cost, ep_data in tqdm(
                    pool.imap_unordered(_run_eval_episode, args_list),
                    total=remaining, desc="Evaluating", unit="ep",
                ):
                    if save_episodes:
                        self.logger.append_episode(ep_idx, ep_data)
                        self.logger.append_agent_metrics(ep_idx, ep_data)
                    episode_costs.append(cost)
                    episodes.append(ep_data)
        else:
            for i in tqdm(range(remaining), desc="Evaluating", unit="ep"):
                ep_idx = completed + i
                env.begin_episode("evaluation", ep_idx)
                state = env.reset()
                total_cost = 0.0
                gamma = env.config.gamma
                ep_data = []

                for t in range(eval_length):
                    action = agent.act(state)
                    _step_metrics = dict(getattr(agent, 'step_metrics', None) or {})
                    next_state, cost, done = env.step(state, action)
                    c_travel, c_maint, c_risk = env.last_cost_breakdown
                    discounted = (gamma ** t) * cost
                    total_cost += discounted
                    ep_data.append({'t': t, 'state': state.copy(), 'action': action,
                                     'cost': cost, 'c_travel': c_travel,
                                     'c_maint': c_maint, 'c_risk': c_risk,
                                     'agent_metrics': _step_metrics})
                    state = next_state
                    # No break on `done`: eval simulates the full T + tail_epochs
                    # horizon (the tail past the planning horizon T).

                if save_episodes:
                    self.logger.append_episode(ep_idx, ep_data)
                    self.logger.append_agent_metrics(ep_idx, ep_data)
                episode_costs.append(total_cost)
                episodes.append(ep_data)

        # Combine with prior costs for overall stats
        all_costs = prior_costs + episode_costs
        return {
            'mean_cost': float(np.mean(all_costs)),
            'std_cost': float(np.std(all_costs)),
            'episodes': episodes,
        }
