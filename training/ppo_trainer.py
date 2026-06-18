"""PPO training loop."""
from __future__ import annotations

import time
import sys
import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING
from tqdm import tqdm

from env.mdp import InfraEnv
from agents.ppo import PPOAgent, RolloutBuffer

if TYPE_CHECKING:
    from utils.logging import RunLogger
    from agents.base import Agent


@dataclass(frozen=True)
class PPOConfig:
    n_episodes: int = 1000
    eval_interval: int = 50
    n_eval_episodes: int = 10
    time_budget: float = 3600.0
    T_tail: float = 10.0          # eval tail horizon in years (epochs = T_tail/dt); matches Trainer
    # Phase 0
    curriculum_phase0_episodes: int = 0      # 0 = skip Phase 0
    # Phase 1 (dynamic exit)
    curriculum_phase1_plateau_window: int = 5    # # eval checkpoints to look back
    curriculum_phase1_plateau_tol: float = 0.01  # relative improvement < tol → plateau
    # Phase 2
    curriculum_reset_critic: bool = False


class PPOTrainer:
    def __init__(
        self,
        agent: PPOAgent,
        env: InfraEnv,
        config: PPOConfig,
        logger: 'RunLogger',
        curriculum_env: InfraEnv | None = None,
        heuristic_agent: 'Agent | None' = None,
    ):
        self.agent = agent
        self.env = env
        self.curriculum_env = curriculum_env
        self.heuristic_agent = heuristic_agent
        self.config = config
        self.logger = logger

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, start_ep: int = 0, already_elapsed: float = 0.0) -> None:
        cfg = self.config
        t_start = time.monotonic()
        use_budget = bool(cfg.time_budget)
        global_ep = 0   # global episode counter for logger

        def _budget_exceeded():
            return use_budget and (already_elapsed + time.monotonic() - t_start) >= cfg.time_budget

        # ── Phase 0: heuristic behavioral cloning ───────────────────────────
        phase0_baseline_cost = None
        if cfg.curriculum_phase0_episodes > 0 and self.heuristic_agent is not None \
                and self.curriculum_env is not None:
            print(f"[Curriculum] Phase 0: {cfg.curriculum_phase0_episodes} episodes "
                  f"(heuristic imitation, NullTAP)")
            phase0_baseline_cost = self._evaluate_heuristic(
                self.curriculum_env, cfg.n_eval_episodes)
            print(f"[Curriculum] Phase 0 heuristic baseline cost: "
                  f"{phase0_baseline_cost:.2f}")

            for ep in range(cfg.curriculum_phase0_episodes):
                if _budget_exceeded():
                    break
                rollout = self._collect_phase0_rollout(self.curriculum_env, episode_idx=global_ep)
                stats = self.agent.update_ppo(rollout)
                global_ep += 1

                if global_ep % cfg.eval_interval == 0:
                    results = self.evaluate(cfg.n_eval_episodes,
                                            env=self.curriculum_env)
                    elapsed = time.monotonic() - t_start
                    self.logger.log_step(global_ep, {
                        **stats,
                        'mean_cost': results['mean_cost'],
                        'std_cost': results['std_cost'],
                        'phase': 0,
                    })
                    print(f"[Phase 0] ep {global_ep:5d} | {elapsed:6.0f}s | "
                          f"mean_cost={results['mean_cost']:.2f} "
                          f"(baseline={phase0_baseline_cost:.2f})")

            print("[Curriculum] Phase 0 done.")

        # ── Phase 1: PPO acts on NullTAP env; dynamic exit ───────────────────
        if self.curriculum_env is not None:
            print("[Curriculum] Phase 1: PPO acting on simplified env (NullTAP)")
            phase1_costs = []
            plateau_window = cfg.curriculum_phase1_plateau_window
            plateau_tol = cfg.curriculum_phase1_plateau_tol

            while not _budget_exceeded():
                rollout = self._collect_rollout(self.curriculum_env,
                                                phase="curriculum_train", episode_idx=global_ep)
                stats = self.agent.update_ppo(rollout)
                global_ep += 1

                if global_ep % cfg.eval_interval == 0:
                    results = self.evaluate(cfg.n_eval_episodes,
                                            env=self.curriculum_env)
                    mean_cost = results['mean_cost']
                    phase1_costs.append(mean_cost)
                    elapsed = time.monotonic() - t_start
                    self.logger.log_step(global_ep, {
                        **stats,
                        'mean_cost': mean_cost,
                        'std_cost': results['std_cost'],
                        'phase': 1,
                    })
                    print(f"[Phase 1] ep {global_ep:5d} | {elapsed:6.0f}s | "
                          f"mean_cost={mean_cost:.2f}")

                    # Dynamic exit: plateau AND beats Phase 0 baseline
                    if len(phase1_costs) >= plateau_window:
                        window = phase1_costs[-plateau_window:]
                        rel_improvement = (window[0] - window[-1]) / (abs(window[0]) + 1e-9)
                        plateau = rel_improvement < plateau_tol
                        beats_baseline = (phase0_baseline_cost is None
                                          or mean_cost < phase0_baseline_cost)
                        if plateau and beats_baseline:
                            print(f"[Curriculum] Phase 1 graduation: "
                                  f"improvement={rel_improvement:.4f} < {plateau_tol}, "
                                  f"cost={mean_cost:.2f} < baseline={phase0_baseline_cost:.2f}")
                            break

            print("[Curriculum] Phase 1 done. Transferring to full environment.")

            if cfg.curriculum_reset_critic:
                self.agent.critic._build()
                import torch
                self.agent._critic_opt = torch.optim.Adam(
                    self.agent.critic._model.parameters(),
                    lr=self.agent._critic_lr,
                )
                print("[Curriculum] Critic reset for Phase 2.")

        # ── Phase 2: full environment ─────────────────────────────────────────
        print("[Curriculum] Phase 2: full environment training.")
        n_episodes = cfg.n_episodes if cfg.n_episodes else sys.maxsize
        for ep in range(n_episodes):
            if _budget_exceeded():
                print(f"Time budget reached after {global_ep} total episodes.")
                break
            rollout = self._collect_rollout(self.env, phase="training", episode_idx=global_ep)
            stats = self.agent.update_ppo(rollout)
            global_ep += 1

            if global_ep % cfg.eval_interval == 0:
                results = self.evaluate(cfg.n_eval_episodes)
                elapsed = time.monotonic() - t_start
                self.logger.log_step(global_ep, {
                    **stats,
                    'mean_cost': results['mean_cost'],
                    'std_cost': results['std_cost'],
                    'phase': 2,
                })
                print(f"[Phase 2] ep {global_ep:5d} | {elapsed:6.0f}s | "
                      f"mean_cost={results['mean_cost']:.2f} "
                      f"± {results['std_cost']:.2f} | "
                      f"l_clip={stats['l_clip']:.4f} entropy={stats['entropy']:.4f}")

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def _collect_rollout(self, env: InfraEnv | None = None,
                         phase: str = "training", episode_idx: int = 0) -> RolloutBuffer:
        import torch
        env = env or self.env
        agent = self.agent

        env.begin_episode(phase, episode_idx)
        state = env.reset()
        buf = RolloutBuffer()
        done = False

        while not done:
            action, log_prob, value = agent.act_with_info(state)
            mask = env.feasible_actions(state)            # (N, 4) bool
            next_state, cost, done = env.step(state, action)
            buf.append(
                state_feat=agent._state_to_tensor(state).squeeze(0).numpy(),
                action=action,
                mask=mask,
                reward=-cost,                             # PPO maximises reward
                log_prob=log_prob,
                value=value,
            )
            state = next_state

        # Bootstrap last value
        if not done:
            with torch.no_grad():
                x = agent._state_to_tensor(state)
                last_value = agent.critic.forward(x).item()
        else:
            last_value = 0.0

        buf.compute_gae(last_value, env.config.gamma, agent.gae_lambda)
        return buf

    def _collect_phase0_rollout(self, env: InfraEnv, episode_idx: int = 0) -> RolloutBuffer:
        """Heuristic acts; PPO evaluates its own log_prob/value for those actions."""
        agent = self.agent
        heuristic = self.heuristic_agent

        env.begin_episode("curriculum_phase0", episode_idx)
        state = env.reset()
        buf = RolloutBuffer()
        done = False

        while not done:
            action = heuristic.act(state)                   # heuristic decides
            mask = env.feasible_actions(state)              # (N, 4) bool
            log_prob, value = agent.evaluate_action(state, action, mask)
            next_state, cost, done = env.step(state, action)
            buf.append(
                state_feat=agent._state_to_tensor(state).squeeze(0).numpy(),
                action=action,
                mask=mask,
                reward=-cost,
                log_prob=log_prob,
                value=value,
            )
            state = next_state

        buf.compute_gae(0.0, env.config.gamma, agent.gae_lambda)
        return buf

    def _evaluate_heuristic(self, env: InfraEnv, n_episodes: int) -> float:
        """Mean discounted cost of the heuristic over n_episodes greedy rollouts.
        Uses the same T + tail_epochs horizon as evaluate() so the Phase-0
        baseline is comparable to the Phase-1 plateau evaluations."""
        heuristic = self.heuristic_agent
        gamma = env.config.gamma
        eval_length = env.config.T + int(self.config.T_tail / env.config.dt)
        costs = []
        for i in range(n_episodes):
            env.begin_episode("curriculum_eval", i)
            state = env.reset()
            total = 0.0
            for t in range(eval_length):
                action = heuristic.act(state)
                state, cost, done = env.step(state, action)
                total += (gamma ** t) * cost
                # No break on `done`: simulate the full T + tail_epochs horizon.
            costs.append(total)
        return float(np.mean(costs))

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, n_episodes: int | None = None, resume: bool = False,
                 save_episodes: bool = False, env: InfraEnv | None = None) -> dict:
        """Greedy rollout using agent.act() (argmax).

        Signature and persistence protocol mirror ``Trainer.evaluate`` so PPO
        runs are directly comparable (paired CRN) to the value-based agents:

        * Full env  → shared-CRN ``"evaluation"`` phase (omits agent identity →
          identical episodes per seed across all agents) and the full
          ``T + tail_epochs`` horizon (no break on ``done`` at ``t = T``).
        * Curriculum (NullTAP simplified) env → internal ``"curriculum_eval"``
          diagnostic, never persisted.

        When ``save_episodes=True`` each completed episode is written to
        ``eval_episodes.csv`` incrementally via the same ``RunLogger`` calls
        (``start_eval`` / ``append_episode`` / ``append_agent_metrics``) used by
        ``Trainer.evaluate``. ``resume=True`` skips already-completed episodes.
        Returns ``{'mean_cost', 'std_cost', 'episodes'}`` (discounted-per-episode
        costs, per-epoch gamma = gamma_annual ** dt).
        """
        if n_episodes is None:
            n_episodes = self.config.n_eval_episodes
        env = env or self.env

        # Full env → shared-CRN "evaluation" namespace (paired with ADP/rollout/
        # optuna). Curriculum (NullTAP simplified) env → internal diagnostic.
        is_full_env = env is self.env
        eval_phase = "evaluation" if is_full_env else "curriculum_eval"
        # Only persist for the real (full-env) evaluation; the curriculum
        # diagnostic eval is never written to eval_episodes.csv.
        persist = bool(save_episodes) and is_full_env

        # Evaluation horizon = T + tail_epochs (horizon_rollout), matching the
        # standard Trainer / optuna evaluators so PPO is directly comparable.
        tail_epochs = int(self.config.T_tail / env.config.dt)
        eval_length = env.config.T + tail_epochs

        agent = self.agent
        gamma = env.config.gamma

        # Resume: detect already-completed episodes (same protocol as Trainer).
        completed = 0
        prior_costs: list[float] = []
        if resume and persist:
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

        if persist:
            self.logger.start_eval(append=(completed > 0))

        episode_costs = []
        episodes = []

        for i in tqdm(range(remaining), desc="Evaluating", unit="ep"):
            ep_idx = completed + i
            env.begin_episode(eval_phase, ep_idx)
            state = env.reset()
            total_cost = 0.0
            ep_data = []

            for t in range(eval_length):
                action = agent.act(state)
                _step_metrics = dict(getattr(agent, 'step_metrics', None) or {})
                next_state, cost, done = env.step(state, action)
                c_travel, c_maint, c_risk = env.last_cost_breakdown
                total_cost += (gamma ** t) * cost
                ep_data.append({
                    't': t, 'state': state.copy(), 'action': action,
                    'cost': cost, 'c_travel': c_travel,
                    'c_maint': c_maint, 'c_risk': c_risk,
                    'agent_metrics': _step_metrics,
                })
                state = next_state
                # No break on `done`: simulate the full T + tail_epochs horizon.

            if persist:
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
