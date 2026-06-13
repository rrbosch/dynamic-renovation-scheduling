"""Optuna-based hyperparameter search for HeuristicAgent subclasses."""
from __future__ import annotations

import csv
import json
import os
import pickle
import shutil
import time
from typing import Callable

import numpy as np
from tqdm import tqdm

from env.mdp import InfraEnv
from utils.logging import RunLogger


class OptunaHeuristicTrainer:
    """
    Tunes any HeuristicAgent subclass using Optuna's TPE sampler with a
    wall-clock time budget and early stopping.

    Parameters
    ----------
    param_space : dict[str, dict]
        Maps parameter name → spec dict with key ``"type"`` one of
        ``"float"``, ``"log_float"``, ``"int"``, ``"categorical"``.
        Float/int specs require ``"low"`` and ``"high"``; categorical
        requires ``"choices"``.
    agent_factory : Callable[[dict], HeuristicAgent]
        Receives a ``{name: value}`` params dict and returns a fresh agent.
    checkpoint_interval_seconds : float
        Save a checkpoint every this many wall-clock seconds (default 1800 = 30 min).
        Set to 0 to disable periodic checkpoints.
    config_hash : str
        Stable hash of the experiment config (excluding seed).  Used to match
        checkpoints to the correct experiment on resume.

    Seed separation:
    - Tuning episodes share a FIXED CRN set keyed ("optuna_trial", k, base_seed)
      — episode k is identical across all trials (required by WilcoxonPruner,
      a paired test). No trial.number in the key.
    - Final evaluation uses the ("evaluation", base_seed, ep_idx) key.
    The "optuna_trial" and "evaluation" phase tags keep the streams structurally
    separate (no clairvoyance).
    """

    def __init__(
        self,
        env: InfraEnv,
        agent,                              # prototype — updated in-place after tuning
        logger: RunLogger,
        base_seed: int,
        time_budget: int,
        n_tuning_episodes: int,
        early_stopping_seconds: int,
        param_space: dict,                  # {name: spec_dict}
        agent_factory: Callable,            # (params: dict) -> HeuristicAgent
        checkpoint_interval_seconds: float = 1800.0,
        config_hash: str = '',
        T_tail: float = 10.0,
    ):
        self.env = env
        self.agent = agent
        self.logger = logger
        self.base_seed = base_seed
        self.time_budget = time_budget
        self.n_tuning_episodes = n_tuning_episodes
        self.early_stopping_seconds = early_stopping_seconds
        self.param_space = param_space
        self.agent_factory = agent_factory
        self.checkpoint_interval_seconds = checkpoint_interval_seconds
        self.config_hash = config_hash
        self.T_tail = T_tail

    # ------------------------------------------------------------------
    # Checkpoint directory
    # ------------------------------------------------------------------

    @property
    def _checkpoint_dir(self) -> str:
        return os.path.join(str(self.logger.run_dir), 'checkpoints', 'optuna')

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def _save_optuna_checkpoint(self, study, elapsed: float) -> None:
        """Persist study + metadata to checkpoints/optuna/."""
        ckpt_dir = self._checkpoint_dir
        os.makedirs(ckpt_dir, exist_ok=True)

        # Study pickle (contains sampler state + all completed trials)
        study_path = os.path.join(ckpt_dir, 'optuna_study.pkl')
        with open(study_path, 'wb') as f:
            pickle.dump(study, f)

        meta = {
            'elapsed_seconds': elapsed,
            'n_trials': len(study.trials),
            'config_hash': self.config_hash,
        }
        meta_path = os.path.join(ckpt_dir, 'metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f)
        shutil.copy2(meta_path, os.path.join(ckpt_dir, 'metadata_backup.json'))

        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        print(
            f"[Optuna checkpoint] {len(study.trials)} trials, "
            f"elapsed {h:02d}:{m:02d}:{s:02d} -> {ckpt_dir}"
        )

    def _load_optuna_checkpoint(self):
        """
        Try to load an existing incomplete checkpoint.

        Returns (study, elapsed_seconds) if a valid checkpoint is found,
        or None if no checkpoint exists / it belongs to a different config /
        it is already marked complete.
        """
        ckpt_dir = self._checkpoint_dir
        meta_path = os.path.join(ckpt_dir, 'metadata.json')
        study_path = os.path.join(ckpt_dir, 'optuna_study.pkl')
        if not (os.path.exists(meta_path) and os.path.exists(study_path)):
            return None
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception:
            return None
        # Guard: different experiment
        if self.config_hash and meta.get('config_hash') != self.config_hash:
            return None
        # Guard: already finished
        if meta.get('complete'):
            return None
        try:
            with open(study_path, 'rb') as f:
                study = pickle.load(f)
        except Exception:
            return None
        return study, float(meta.get('elapsed_seconds', 0.0))

    def _mark_optuna_complete(self, study, elapsed: float) -> None:
        """Save final checkpoint and mark it complete."""
        self._save_optuna_checkpoint(study, elapsed)
        meta_path = os.path.join(self._checkpoint_dir, 'metadata.json')
        with open(meta_path) as f:
            meta = json.load(f)
        meta['complete'] = True
        with open(meta_path, 'w') as f:
            json.dump(meta, f)

    # ------------------------------------------------------------------
    # Param suggestion dispatcher
    # ------------------------------------------------------------------

    def _suggest(self, trial, name: str, spec: dict):
        t = spec.get('type', 'float')
        if t == 'float':
            return trial.suggest_float(name, spec['low'], spec['high'])
        elif t == 'log_float':
            return trial.suggest_float(name, spec['low'], spec['high'], log=True)
        elif t == 'int':
            return trial.suggest_int(name, int(spec['low']), int(spec['high']))
        elif t == 'categorical':
            return trial.suggest_categorical(name, spec['choices'])
        else:
            raise ValueError(f"Unknown param type: {t!r}")

    # ------------------------------------------------------------------
    # Optuna objective
    # ------------------------------------------------------------------

    def _objective(self, trial) -> float:
        import optuna
        params = {name: self._suggest(trial, name, spec)
                  for name, spec in self.param_space.items()}
        agent = self.agent_factory(params)

        # CRN across trials (required by WilcoxonPruner — it is a paired
        # signed-rank test, so episode k must be the SAME scenario in every
        # trial). We key env noise on a FIXED base_seed (no trial.number), so
        # episode k is identical across trials. Tuning therefore uses a fixed
        # n_tuning_episodes-element CRN set rather than fresh episodes per trial
        # — the correct setup for paired pruning. The separate held-out
        # "evaluation" phase (seed 42) remains the unbiased final metric, so
        # this fixed CRN set does not reintroduce optimization-overfitting bias.
        # The "optuna_trial" phase tag keeps these streams structurally separate
        # from the real "evaluation" phase (no clairvoyance).
        costs = []
        for k in range(self.n_tuning_episodes):
            costs.append(self._run_episode(agent, k))
            trial.report(float(np.mean(costs)), step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(costs))

    def _run_episode(self, agent, episode_idx: int) -> float:
        env = self.env
        env.begin_episode("optuna_trial", episode_idx, base_seed=self.base_seed)
        state = env.reset()
        total_cost = 0.0
        gamma = env.config.gamma
        for t in range(env.config.T):
            action = agent.act(state)
            next_state, cost, done = env.step(state, action)
            total_cost += (gamma ** t) * cost
            state = next_state
            if done:
                break
        return total_cost

    # ------------------------------------------------------------------
    # Early stopping (inner closure, called from combined callback)
    # ------------------------------------------------------------------

    def _make_early_stop_fn(self):
        best_val = [float('inf')]
        best_time = [time.time()]

        def early_stop(study, trial):
            if study.best_value < best_val[0]:
                best_val[0] = study.best_value
                best_time[0] = time.time()
            if time.time() - best_time[0] > self.early_stopping_seconds:
                print(
                    f"[Optuna] Early stopping: no improvement for "
                    f"{self.early_stopping_seconds}s."
                )
                study.stop()

        return early_stop

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, start_ep: int = 0, already_elapsed: float = 0.0) -> None:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # WilcoxonPruner: stops disappointing trials early (paired signed-rank
        # test vs. the best trial over the shared CRN episodes — see _objective).
        # Frees budget to curb the winner's-curse / optimization-overfitting bias.
        def _make_pruner():
            return optuna.pruners.WilcoxonPruner(p_threshold=0.1, n_startup_steps=5)

        # --- Resume detection ---
        loaded = self._load_optuna_checkpoint()
        if loaded is not None:
            study, prev_elapsed = loaded
            already_elapsed = max(already_elapsed, prev_elapsed)
            # A reloaded (pickled) study may not retain the pruner; re-attach it
            # so pruning stays active across resume.
            study.pruner = _make_pruner()
            print(
                f"[Optuna] Resuming: {len(study.trials)} trials done, "
                f"{already_elapsed:.0f}s elapsed."
            )
        else:
            study = optuna.create_study(
                direction='minimize',
                sampler=optuna.samplers.TPESampler(seed=self.base_seed),
                pruner=_make_pruner(),
            )

        remaining = (self.time_budget - already_elapsed) if self.time_budget else None

        # --- Combined callback: early stopping + periodic checkpoints ---
        t0 = time.time()
        last_ckpt_session = [0.0]   # session seconds at last checkpoint
        early_stop_fn = self._make_early_stop_fn()

        def combined_callback(study, trial):
            early_stop_fn(study, trial)

            if self.checkpoint_interval_seconds > 0:
                session_elapsed = time.time() - t0
                if session_elapsed - last_ckpt_session[0] >= self.checkpoint_interval_seconds:
                    total_elapsed = already_elapsed + session_elapsed
                    self._save_optuna_checkpoint(study, total_elapsed)
                    last_ckpt_session[0] = session_elapsed

        study.optimize(
            self._objective,
            timeout=remaining,
            callbacks=[combined_callback],
        )

        # Mark complete (saves final study + sets complete=True)
        self._mark_optuna_complete(study, already_elapsed + (time.time() - t0))

        # Update shared agent in-place so run.py's save_agent(agent) works
        bp = study.best_params
        best_agent = self.agent_factory(bp)
        self.agent.__dict__.update(best_agent.__dict__)

        results_dir = self.logger.run_dir

        # Save best_params.json
        with open(os.path.join(results_dir, 'best_params.json'), 'w') as f:
            json.dump({**bp, 'best_value': study.best_value}, f, indent=2)

        # Save full Optuna study (also in results root for backwards compat)
        with open(os.path.join(results_dir, 'optuna_study.pkl'), 'wb') as f:
            pickle.dump(study, f)

        # Write per-trial CSV
        self._write_trial_csv(study)

        # Write one row to training_log.csv for the best result
        self.logger.log_step(
            len(study.trials),
            {'mean_cost': study.best_value, 'std_cost': 0.0},
        )

        print(f"[Optuna] Best cost: {study.best_value:.4f}  params: {bp}")

    # ------------------------------------------------------------------
    # Evaluation (mirrors Trainer.evaluate signature)
    # ------------------------------------------------------------------

    def evaluate(self, n_episodes: int | None = None, resume: bool = False,
                 save_episodes: bool = False) -> dict:
        """
        Run the tuned heuristic greedily for n_episodes.

        Mirrors ``Trainer.evaluate``: always uses the T + tail_epochs horizon
        (so the heuristic baseline is comparable to the learning agents), saves
        each completed episode incrementally when ``save_episodes`` is set, and
        skips already-completed episodes when ``resume`` is set.

        Per-episode seeds are derived deterministically from
        (base_seed, 0xFFFF0000, ep_idx) so a resumed run reproduces the same
        scenarios. This seed space never overlaps the tuning seeds.
        """
        if n_episodes is None:
            n_episodes = self.n_tuning_episodes

        env = self.env
        agent = self.agent
        gamma = env.config.gamma
        tail_epochs = int(self.T_tail / env.config.dt)
        eval_length = env.config.T + tail_epochs

        # Resume: detect already-completed episodes on disk
        completed = 0
        prior_costs: list[float] = []
        if resume and save_episodes:
            completed = self.logger.count_completed_eval_episodes()
            if completed > 0:
                print(f"Resuming evaluation: {completed}/{n_episodes} episodes already done")
                prior_costs = self.logger.load_eval_episode_costs(gamma)
        remaining = n_episodes - completed

        if remaining <= 0:
            episode_costs = prior_costs[:n_episodes]
            return {
                'mean_cost': float(np.mean(episode_costs)),
                'std_cost': float(np.std(episode_costs)),
                'episodes': [],
            }

        if save_episodes:
            self.logger.start_eval(append=(completed > 0))

        episode_costs = []
        episodes = []

        for i in tqdm(range(remaining), desc="Evaluating", unit="ep"):
            ep_idx = completed + i
            # Shared-CRN evaluation: same ("evaluation", base_seed, ep_idx) key
            # as the standard Trainer, so the tuned heuristic is evaluated on the
            # exact same episodes as the ADP / rollout agents.
            env.begin_episode("evaluation", ep_idx, base_seed=self.base_seed)
            state = env.reset()
            total_cost = 0.0
            ep_data = []

            for t in range(eval_length):
                action = agent.act(state)
                next_state, cost, done = env.step(state, action)
                c_travel, c_maint, c_risk = env.last_cost_breakdown
                discounted = (gamma ** t) * cost
                total_cost += discounted
                ep_data.append({
                    't': t,
                    'state': state.copy(),
                    'action': action,
                    'cost': cost,
                    'c_travel': c_travel,
                    'c_maint': c_maint,
                    'c_risk': c_risk,
                    'agent_metrics': {},
                })
                state = next_state
                # No break on `done`: eval simulates the full T + tail_epochs
                # horizon (matches the standard Trainer's evaluation).

            if save_episodes:
                self.logger.append_episode(ep_idx, ep_data)
                self.logger.append_agent_metrics(ep_idx, ep_data)
            episode_costs.append(total_cost)
            episodes.append(ep_data)

        all_costs = prior_costs + episode_costs
        return {
            'mean_cost': float(np.mean(all_costs)),
            'std_cost': float(np.std(all_costs)),
            'episodes': episodes,
        }

    # ------------------------------------------------------------------
    # CSV output
    # ------------------------------------------------------------------

    def _write_trial_csv(self, study) -> None:
        path = os.path.join(self.logger.run_dir, 'optuna_trials.csv')
        param_keys = list(self.param_space.keys())
        fieldnames = ['trial_number'] + param_keys + ['mean_cost', 'state']
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for trial in study.trials:
                p = trial.params
                row = {'trial_number': trial.number}
                for k in param_keys:
                    row[k] = p.get(k, '')
                row['mean_cost'] = trial.value if trial.value is not None else ''
                row['state'] = trial.state.name
                writer.writerow(row)
