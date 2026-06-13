"""Deep Controlled Learning (DCL) Agent.

DCL extends MonteCarloRolloutAgent by also training a policy and VFA from
rollout data, progressively replacing the heuristic rollout policy with the
learned policy.

Key loop:
  better rollout policy → better Q estimates → better actions
  → better training labels → better policy

The actions stored in the replay buffer are already rollout-optimal (chosen by
rollout Q search during act()), so no re-rollout is needed during update().
"""
from __future__ import annotations

import numpy as np

from agents.base import Agent
from agents.fn.policy import XGBoostPolicy, NNPolicy, _asset_features, _build_dataset
from agents.rollout import rollout_noise
from env.mdp import State, InfraEnv, EnvConfig


# ---------------------------------------------------------------------------
# DCLAgent
# ---------------------------------------------------------------------------

class DCLAgent(Agent):
    """
    Deep Controlled Learning agent.

    Acts via rollout Q local search (like MonteCarloRolloutAgent).
    Also trains a policy and VFA from rollout data.
    Progressively replaces the heuristic rollout policy with the learned policy.
    """

    def __init__(
        self,
        policy,               # NNPolicy | XGBoostPolicy
        value_fn,             # XGBoostValueFn | NeuralValueFn
        env: InfraEnv,
        heuristic_policy: Agent,
        action_gen,           # ActionGenerator (used optionally; we do inline search)
        rollout_horizon: int,
        n_rollouts: int,
        min_samples_train: int,
        finite_horizon: bool,
        rng: np.random.Generator,
    ):
        self.policy = policy
        self.value_fn = value_fn
        self.env = env
        self.heuristic_policy = heuristic_policy
        self.action_gen = action_gen
        self.rollout_horizon = rollout_horizon
        self.n_rollouts = n_rollouts
        self.min_samples_train = min_samples_train
        self.finite_horizon = finite_horizon
        # Rollout noise is key-derived (CRN); derive a fixed integer seed from the
        # supplied rng so reproducibility still flows from the config seed.
        self.seed = int(rng.integers(0, 2**63 - 1))

        self._rollout_policy: Agent = heuristic_policy
        self._n_updates: int = 0
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def act(self, state: State) -> np.ndarray:
        """
        Greedy local search over single-asset deviations using MC rollout Q.
        Identical structure to MonteCarloRolloutAgent.act().
        """
        cfg = self.env.config
        n = cfg.n_assets
        feas = self.env.feasible_actions(state)

        current_action = np.zeros(n, dtype=int)
        # current_action = self._rollout_policy.act(state)
        current_q = self._rollout_q(state, current_action)

        improved = True
        while improved:
            improved = False
            for i in range(n):
                for a in range(0, 4):
                    if a == current_action[i]:
                        continue
                    if not feas[i, a]:
                        continue
                    candidate = current_action.copy()
                    candidate[i] = a
                    q = self._rollout_q(state, candidate)
                    if q < current_q:
                        current_q = q
                        current_action = candidate
                        improved = True

        return current_action

    def update(self, transitions: list) -> None:
        """
        1. Fit VFA on post-decision future targets.
        2. Fit policy on stored (already rollout-optimal) actions.
        3. Increment counter; switch rollout policy at threshold.
        """

        # 1. VFA update: target = mc_return - cost (future-only discounted return)
        X = self.value_fn._feats([t.post_state for t in transitions])
        y = np.array([t.mc_return - t.cost for t in transitions])
        self.value_fn.fit(X, y)
        self._fitted = True

        # 2. Policy update: stored actions are already rollout-optimal
        states = [t.state for t in transitions]
        actions = np.stack([t.action for t in transitions])  # (B, N)
        self.policy.fit(states, actions)

        # 3. Increment; switch rollout policy if threshold reached
        if len(transitions) > self.min_samples_train:
            self._rollout_policy = self.policy
            print(
                f"[DCLAgent] Switched rollout policy from heuristic to learned policy "
                f"(after collecting {self.min_samples_train} samples)."
            )

    # ------------------------------------------------------------------
    # Q estimation via truncated rollout
    # ------------------------------------------------------------------

    def _rollout_q(self, state: State, action: np.ndarray) -> float:
        """
        Q(s, a) = c_imm(s, a) + mean over n_rollouts of truncated rollout.
        """
        cfg = self.env.config
        s_post = self.env.post_decision_state(state, action)
        c_imm = self.env.immediate_cost(state, action, s_post)

        t_next = state.t + 1
        K = min(self.rollout_horizon, cfg.T - t_next)
        if K <= 0:
            return c_imm

        # Noise keyed on the root `state` (shared across candidate actions) and
        # the rollout index, giving common random numbers across candidates.
        futures = [
            self._single_truncated_rollout(s_post, t_next, state, r)
            for r in range(self.n_rollouts)
        ]
        return c_imm + float(np.mean(futures))

    def _single_truncated_rollout(
        self, s_post: State, t_start: int, root_state: State, rollout_idx: int
    ) -> float:
        """
        Simulate K steps under _rollout_policy; bootstrap with VFA at truncation.

        Noise is drawn once for this rollout from a (seed, root_state, decision
        epoch, rollout_idx) key, so candidate actions share common random numbers.
        Discount starts at gamma^1 relative to the current epoch.
        """
        cfg = self.env.config
        current = s_post.copy()
        current.t = t_start - 1  # s_post epoch (action taken at t_start-1)

        K = min(self.rollout_horizon, cfg.T - t_start)
        u_bank, eps_bank = rollout_noise(
            self.seed, root_state, root_state.t, rollout_idx, K, cfg.n_assets
        )
        discount = cfg.gamma
        total = 0.0
        t = t_start

        for k in range(K):
            # Action from rollout policy; mask infeasible to ACTION_NONE
            raw_action = self._rollout_policy.act(current)
            feas = self.env.feasible_actions(current)
            action = np.where(
                feas[np.arange(cfg.n_assets), raw_action],
                raw_action,
                InfraEnv.ACTION_NONE,
            )

            s_post_step = self.env.post_decision_state(current, action)
            cost = self.env.immediate_cost(current, action, s_post_step)

            total += discount * cost
            discount *= cfg.gamma

            # Stochastic transition using this rollout's CRN noise
            current, t = self.env.stochastic_transition(
                s_post_step, t, (u_bank[k], eps_bank[k])
            )

        # Bootstrap with VFA at truncation (returns 0 if not yet fitted)
        total += discount * float(self.value_fn.predict([current])[0])

        return total

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        import os
        os.makedirs(path, exist_ok=True)
        self.value_fn.save(os.path.join(path, 'value_fn.pkl'))
        self.policy.save(os.path.join(path, 'policy.pkl'))

    def load(self, path: str) -> None:
        import os
        self.value_fn.load(os.path.join(path, 'value_fn.pkl'))
        self.policy.load(os.path.join(path, 'policy.pkl'))
