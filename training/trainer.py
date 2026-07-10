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
    # Anticipative baselines pre-solve the episode here from its exact replayed
    # noise (no-op for ordinary agents). Keyed identically to the env above.
    agent.on_episode_start("evaluation", episode_idx, base_seed, env, eval_length)
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


def _assign_mc_returns(ep_transitions, gamma, truncation_mode, agent):
    """Assign `tr.mc_return` for one episode's transitions (the ADP target source).

    Default: full-horizon discounted return `mc_return_i = cost_i + γ·mc_return_{i+1}`
    with the bootstrap/none terminal anchor. If the agent sets `n_step > 0`, use an
    **n-step return that bootstraps off V'** at the post-decision state n steps ahead:
        mc_return_i = Σ_{k=0..n} γ^k·cost_{i+k} + γ^n·V'(post_{i+n})      (when i+n < L)
    and the exact full-horizon return for the tail (i+n ≥ L). `n_step` of 0 or ≥ L is
    **bit-identical** to the full-horizon target. The n-step bootstrap uses the
    un-baselined `value_fn.predict` (the advantage baseline b(t) is added back inside
    predict), so it remains a valid cost estimate. Falls back to full-horizon if V' is
    not yet fitted. See docs/adp_nstep_target_plan.md / docs/adp_value_fn_improvements.md.
    """
    L = len(ep_transitions)
    if L == 0:
        return
    if truncation_mode == 'bootstrap' and hasattr(agent, 'value_fn'):
        boot_state = ep_transitions[-1].post_state.copy()
        boot_state.t = 0
        G = agent.value_fn.predict([boot_state])[0]
    else:
        G = 0.0
    full = [0.0] * L
    for i in range(L - 1, -1, -1):
        G = ep_transitions[i].cost + gamma * G
        full[i] = G

    n = int(getattr(agent, 'n_step', 0) or 0)
    vpost = None
    if 0 < n < L:
        try:
            vpost = agent.value_fn.predict([tr.post_state for tr in ep_transitions])
        except Exception:
            vpost = None  # V' not fitted yet → full-horizon fallback
    if vpost is None:
        for i in range(L):
            ep_transitions[i].mc_return = full[i]
        return

    costs = [tr.cost for tr in ep_transitions]
    gpow = [gamma ** k for k in range(n + 1)]
    for i in range(L):
        if i + n < L:
            acc = gpow[n] * vpost[i + n]
            for k in range(n + 1):
                acc += gpow[k] * costs[i + k]
            ep_transitions[i].mc_return = acc
        else:
            ep_transitions[i].mc_return = full[i]


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
        # horizon_rollout simulates the tail in training too (to T + tail_epochs), so do
        # NOT stop at the planning-horizon `done` (= t=T) in that mode; other modes stop at T.
        if done and truncation_mode != 'horizon_rollout':
            break
    # MC backward pass (full-horizon, or n-step bootstrap if agent.n_step > 0)
    _assign_mc_returns(ep_transitions, gamma, truncation_mode, agent)
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
    eval_interval_seconds: float = 0.0   # 0 = disabled; run a periodic eval every N wall-clock
                                         # seconds and log the policy-improvement curve to
                                         # training_log.csv. Episode-based eval_interval is
                                         # unreliable under a wall-clock time_budget (episode
                                         # count is data-dependent), so prefer this for long runs.
    update_interval: int = 10
    truncation_mode: str = 'bootstrap'  # 'none', 'horizon_rollout', or 'bootstrap'
    T_tail: float = 10.0                # tail horizon in years (converted to epochs via T_tail/dt)
    buffer_capacity: int = 200_000
    buffer_strategy: str = 'fifo'
    n_eval_episodes: int = 10
    batch_size: int = 2048
    n_warmstart_states: int = 0             # 0 = disabled; fill buffer with this many
                                            # heuristic transitions (states) before training
    warmstart_agent_config: dict | None = None  # parsed from JSON 'warmstart' key
    # Warmstart exploration is no longer a trainer step. To add acting-biased flips
    # (the old (b) mechanism) wrap the warmstart agent in the 'explore_flip' heuristic
    # (agents.heuristics.FlipWrapperAgent) via the config — it composes per-policy
    # (e.g. flip-wrap a reactive mode but not the do-nothing mode of a mixture).
    checkpoint_interval: int = 0                 # 0 = disabled; save checkpoint every N episodes (legacy)
    checkpoint_interval_seconds: float = 1800.0  # 0 = disabled; save checkpoint every N wall-clock seconds
    config_hash: str = ''                        # stable hash of ExperimentConfig (excl. seed)
    n_workers: int = 1                           # parallel workers for eval/training episodes (1=sequential)
    # Post-mortem evaluation (decouple eval cost from the training budget). When
    # snapshot_interval_seconds > 0, a lightweight agent-only snapshot (no buffer) is
    # written to results/<run>/snapshots/t_<elapsed>s/ every N wall-clock seconds AND
    # once at the end of training — history is kept (never deleted). experiments/
    # evaluate_checkpoints.py then reconstructs env+agent and runs the shared-CRN eval
    # offline, producing a cost-vs-wallclock curve (mean + CVaR/P90). When defer_eval
    # is True, the in-loop evaluate() calls (episode-based + wall-clock periodic) are
    # skipped so the training budget goes entirely to training.
    snapshot_interval_seconds: float = 0.0       # 0 = disabled
    defer_eval: bool = False                     # True = skip in-loop eval; rely on post-mortem snapshots
    # Early stopping (item 5). A *cheap* wall-clock-gated eval (early_stop_episodes, small)
    # is the stopping signal — it coexists with defer_eval (whose authoritative 50-ep eval runs
    # post-mortem). Stop when the best mean cost has not improved by > early_stop_tol (relative)
    # for early_stop_patience consecutive checks, but never before early_stop_min_seconds.
    early_stop_patience: int = 0                 # 0 = disabled
    early_stop_tol: float = 0.005                # relative improvement threshold
    early_stop_min_seconds: float = 0.0          # floor: no stop before this much wall-clock
    early_stop_episodes: int = 0                 # 0 = use n_eval_episodes; small = cheap signal
    early_stop_interval_seconds: float = 0.0     # 0 = use eval_interval_seconds as the cadence


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
        self._last_snapshot_wall: float = 0.0    # tracks elapsed time of last eval snapshot
        self._last_es_wall: float = 0.0          # tracks elapsed time of last early-stop check
        self._es_best: float = float('inf')      # best (lowest) mean cost seen by early stopping
        self._es_stale: int = 0                  # consecutive early-stop checks without improvement

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

    def _save_snapshot(self, ep: int, elapsed: float) -> None:
        """Lightweight, history-kept agent-only snapshot for post-mortem evaluation.

        Unlike _save_checkpoint (buffer + latest-only, for resume), this writes just the
        learned artifacts to results/<run>/snapshots/t_<elapsed>s/agent/ plus a small
        meta.json, and never deletes prior snapshots — so evaluate_checkpoints.py can
        build a cost-vs-wallclock curve. No buffer, no backup copies (snapshots are
        frequent and disposable; the resume checkpoint remains the source of truth)."""
        import os, json
        snap_dir = os.path.join(str(self.logger.run_dir), 'snapshots', f't_{int(elapsed)}s')
        agent_dir = os.path.join(snap_dir, 'agent')
        os.makedirs(agent_dir, exist_ok=True)
        self.agent.save(agent_dir)
        with open(os.path.join(snap_dir, 'meta.json'), 'w') as f:
            json.dump({'episode': int(ep), 'elapsed_seconds': round(float(elapsed), 1),
                       'config_hash': self.config.config_hash}, f)

    def _maybe_snapshot(self, ep: int, t_start: float, already_elapsed: float, cfg,
                        final: bool = False) -> None:
        """Save an eval snapshot if the wall-clock interval elapsed (or final=True)."""
        if cfg.snapshot_interval_seconds <= 0:
            return
        elapsed = already_elapsed + (time.monotonic() - t_start)
        if final or (elapsed - self._last_snapshot_wall >= cfg.snapshot_interval_seconds):
            self._save_snapshot(ep, elapsed)
            self._last_snapshot_wall = elapsed

    def _maybe_early_stop(self, ep: int, t_start: float, already_elapsed: float, cfg) -> bool:
        """Wall-clock-gated patience early stop on a *cheap* eval signal (item 5).

        Runs a small eval every `early_stop_interval_seconds` (falling back to
        `eval_interval_seconds`); stops when the best mean cost has not improved by
        > early_stop_tol for early_stop_patience consecutive checks — but never before
        early_stop_min_seconds. Runs even under defer_eval (it is the stopping signal;
        the authoritative eval still happens post-mortem). Returns True to end training."""
        if cfg.early_stop_patience <= 0:
            return False
        cadence = cfg.early_stop_interval_seconds or cfg.eval_interval_seconds
        if cadence <= 0:
            return False
        elapsed = already_elapsed + (time.monotonic() - t_start)
        if elapsed - self._last_es_wall < cadence:
            return False
        self._last_es_wall = elapsed
        n = cfg.early_stop_episodes or cfg.n_eval_episodes
        cur = self.evaluate(n)['mean_cost']
        if cur < self._es_best * (1.0 - cfg.early_stop_tol):
            self._es_best = cur
            self._es_stale = 0
        else:
            self._es_stale += 1
        if elapsed >= cfg.early_stop_min_seconds and self._es_stale >= cfg.early_stop_patience:
            print(f"[early stop] no improvement (> {cfg.early_stop_tol:.1%}) for "
                  f"{self._es_stale} checks; best mean_cost={self._es_best:.3e}. "
                  f"Stopping at ep {ep}, {elapsed:.0f}s.")
            return True
        return False

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
        target = cfg.n_warmstart_states
        print(f"Warmstarting buffer with {type(warmstart_agent).__name__} "
              f"for {target} states...")
        _ws_t0 = time.monotonic()
        _ws_report_every = max(1, target // 20)  # ~20 progress lines
        _next_report = _ws_report_every
        n_states = 0          # transitions generated (may exceed buffer len under eviction)
        _ws_ep = 0
        # Collect whole episodes until the requested number of states is reached.
        while n_states < target:
            env.begin_episode("warmstart", _ws_ep)
            state = env.reset()
            ep_transitions: list[Transition] = []
            for _ in range(env.config.T):
                action = warmstart_agent.act(state)
                next_state, cost, done = env.step(state, action)
                post_state = env.post_decision_state(state, action)
                tr = Transition(
                    state=state.copy(), action=action.copy(), cost=cost,
                    next_state=next_state.copy(), post_state=post_state.copy(), done=done,
                )
                ep_transitions.append(tr)
                self.buffer.add(tr)
                n_states += 1
                state = next_state
                if done or n_states >= target:
                    break
            # Backward MC pass — VF is untrained here so always start G=0
            G = 0.0
            for tr in reversed(ep_transitions):
                G = tr.cost + env.config.gamma * G
                tr.mc_return = G
            _ws_ep += 1

            # Periodic progress so a slow eviction strategy is visible in the log
            if n_states >= _next_report:
                rate = n_states / max(1e-9, time.monotonic() - _ws_t0)
                print(f"Warmstart progress: {n_states}/{target} states "
                      f"({_ws_ep} episodes), buffer={len(self.buffer)}, {rate:.0f} states/s")
                _next_report += _ws_report_every
        print(f"Warmstart complete. {n_states} states from {_ws_ep} episodes. "
              f"Buffer size: {len(self.buffer)}")

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

        if start_ep == 0 and cfg.n_warmstart_states > 0:
            if cfg.warmstart_agent_config is not None:
                self._run_warmstart(self._resolve_warmstart_agent(cfg.warmstart_agent_config))
            else:
                warnings.warn("n_warmstart_states > 0 but no 'warmstart' config provided. Skipping.")

        t_start = time.monotonic()
        self._last_checkpoint_wall = already_elapsed  # reset clock relative to resume point
        self._last_eval_wall = already_elapsed        # wall-clock anchor for periodic eval
        self._last_snapshot_wall = already_elapsed    # wall-clock anchor for eval snapshots
        self._last_es_wall = already_elapsed          # wall-clock anchor for early-stop checks
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
        ep = start_ep - 1  # defensive: keep `ep` bound if the loop body never runs
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

                # horizon_rollout simulates the tail in training too (to T + tail_epochs),
                # so don't stop at the planning-horizon `done` (= t=T) in that mode.
                if done and cfg.truncation_mode != 'horizon_rollout':
                    break

            # Compute MC returns (full-horizon, or n-step bootstrap if agent.n_step > 0)
            _assign_mc_returns(ep_transitions, env.config.gamma, cfg.truncation_mode, self.agent)

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

            # Evaluation (skipped when eval is deferred to post-mortem snapshots)
            if not cfg.defer_eval and (ep + 1) % cfg.eval_interval == 0:
                results = self.evaluate(cfg.n_eval_episodes)
                self.logger.log_step(ep + 1, results)
                elapsed = already_elapsed + (time.monotonic() - t_start)
                print(f"Episode {ep+1:4d} | {elapsed:6.0f}s | "
                      f"mean_cost={results['mean_cost']:.2f} "
                      f"± {results['std_cost']:.2f}")

            # Wall-clock-gated periodic eval (policy-improvement curve)
            if not cfg.defer_eval:
                self._maybe_periodic_eval(ep + 1, t_start, already_elapsed, cfg)

            # Checkpoint
            if cfg.checkpoint_interval_seconds > 0:
                elapsed = already_elapsed + (time.monotonic() - t_start)
                if elapsed - self._last_checkpoint_wall >= cfg.checkpoint_interval_seconds:
                    self._save_checkpoint(ep + 1, elapsed)
                    self._last_checkpoint_wall = elapsed
            elif cfg.checkpoint_interval > 0 and (ep + 1) % cfg.checkpoint_interval == 0:
                elapsed = already_elapsed + (time.monotonic() - t_start)
                self._save_checkpoint(ep + 1, elapsed)

            # Lightweight eval snapshot for post-mortem evaluation (history kept)
            self._maybe_snapshot(ep + 1, t_start, already_elapsed, cfg)

            # Early stopping (cheap patience eval; coexists with defer_eval)
            if self._maybe_early_stop(ep + 1, t_start, already_elapsed, cfg):
                break

        # Final snapshot capturing the end-of-training policy
        self._maybe_snapshot(ep + 1, t_start, already_elapsed, cfg, final=True)

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

            # Evaluation (skipped when eval is deferred to post-mortem snapshots)
            if not cfg.defer_eval and ep % cfg.eval_interval == 0:
                results = self.evaluate(cfg.n_eval_episodes)
                self.logger.log_step(ep, results)
                elapsed = already_elapsed + (time.monotonic() - t_start)
                print(f"Episode {ep:4d} | {elapsed:6.0f}s | "
                      f"mean_cost={results['mean_cost']:.2f} "
                      f"± {results['std_cost']:.2f}")

            # Wall-clock-gated periodic eval (policy-improvement curve)
            if not cfg.defer_eval:
                self._maybe_periodic_eval(ep, t_start, already_elapsed, cfg)

            # Checkpoint
            if cfg.checkpoint_interval_seconds > 0:
                elapsed = already_elapsed + (time.monotonic() - t_start)
                if elapsed - self._last_checkpoint_wall >= cfg.checkpoint_interval_seconds:
                    self._save_checkpoint(ep, elapsed)
                    self._last_checkpoint_wall = elapsed
            elif cfg.checkpoint_interval > 0 and ep % cfg.checkpoint_interval == 0:
                elapsed = already_elapsed + (time.monotonic() - t_start)
                self._save_checkpoint(ep, elapsed)

            # Lightweight eval snapshot for post-mortem evaluation (history kept)
            self._maybe_snapshot(ep, t_start, already_elapsed, cfg)

            # Early stopping (cheap patience eval; coexists with defer_eval)
            if self._maybe_early_stop(ep, t_start, already_elapsed, cfg):
                break

        # Final snapshot capturing the end-of-training policy
        self._maybe_snapshot(ep, t_start, already_elapsed, cfg, final=True)

    def _maybe_periodic_eval(self, ep, t_start, already_elapsed, cfg) -> None:
        """Wall-clock-gated policy eval logged to training_log.csv.

        Gives a policy-improvement curve over a long time-budget run, where the
        episode-based eval_interval is unreliable (episode count is data-dependent).
        No-op unless cfg.eval_interval_seconds > 0.
        """
        if cfg.eval_interval_seconds <= 0:
            return
        elapsed = already_elapsed + (time.monotonic() - t_start)
        if elapsed - self._last_eval_wall < cfg.eval_interval_seconds:
            return
        self._last_eval_wall = elapsed
        results = self.evaluate(cfg.n_eval_episodes)
        self.logger.log_step(ep, {
            'elapsed_seconds': round(elapsed, 1),
            'mean_cost': results['mean_cost'],
            'std_cost': results['std_cost'],
        })
        print(f"[periodic eval] ep {ep} | {elapsed:6.0f}s | "
              f"mean_cost={results['mean_cost']:.3e} ± {results['std_cost']:.3e}")

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
                # Anticipative baselines pre-solve the episode here (no-op
                # otherwise). self.seed == env._base_seed used by begin_episode.
                agent.on_episode_start("evaluation", ep_idx, self.seed, env, eval_length)
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
