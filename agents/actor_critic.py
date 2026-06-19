"""Actor-Critic agent combining DQN value function with a policy network."""
from __future__ import annotations

import numpy as np

from agents.base import Agent
from agents.dqn import ValueBasedAgent
from agents.fn.policy import PolicyNetwork, build_policy_input
from env.mdp import State
from training.buffer import Transition


class ActorCriticAgent(Agent):
    """
    Combines a DQNAgent (critic/value fn) with a PolicyNetwork (actor).
    Action selection: best-of-patience candidates from policy, evaluated by Q.
    Policy update: supervised imitation of local-search-optimal actions.
    """

    def __init__(
        self,
        dqn_agent: ValueBasedAgent,
        policy_network: PolicyNetwork,
        patience: int = 20,
    ):
        self.dqn_agent = dqn_agent
        self.policy_network = policy_network
        self.patience = patience

    def act(self, state: State) -> np.ndarray:
        """
        1. Init best action as all-none, evaluate with value fn.
        2. Sample candidate from policy network.
        3. Evaluate candidate Q. If better, update best; reset patience counter.
        4. Repeat until patience exhausted.
        5. Return best action.
        """
        env = self.dqn_agent.env
        cfg = env.config
        vf = self.dqn_agent.value_fn

        best_action = np.zeros(cfg.n_assets, dtype=int)
        best_q = self._eval_q(state, best_action)

        no_improve = 0
        while no_improve < self.patience:
            candidate = self.policy_network.sample_action(state)
            q = self._eval_q(state, candidate)
            if q < best_q:
                best_q = q
                best_action = candidate
                no_improve = 0
            else:
                no_improve += 1

        return best_action

    def update(self, transitions: list[Transition]) -> None:
        """
        1. Update DQN value function.
        2. Update policy via supervised imitation of local-search-optimal actions.
        """
        # Update critic
        self.dqn_agent.update(transitions)

        # Compute local-search-optimal actions for supervised imitation
        self._update_policy(transitions)

    def _eval_q(self, state: State, action: np.ndarray) -> float:
        env = self.dqn_agent.env
        cfg = env.config
        vf = self.dqn_agent.value_fn

        s_post = env.post_decision_state(state, action)
        cost = env.immediate_cost(state, action, s_post)
        v = vf.predict([s_post])[0]
        return cost + v

    def _update_policy(self, transitions: list[Transition]) -> None:
        """Cross-entropy loss against local-search-optimal actions (imitation)."""
        import torch
        import torch.nn.functional as F

        env = self.dqn_agent.env
        ag = self.dqn_agent.action_gen

        states = [t.state for t in transitions]
        # Compute optimal actions via local search (critic-guided)
        optimal_actions = []
        for s in states:
            a = ag.generate(s, self.dqn_agent.value_fn, env)
            optimal_actions.append(a)

        pnet = self.policy_network
        X = torch.tensor(
            np.stack([build_policy_input(s, pnet.n_assets, pnet.T, pnet.finite_horizon)
                      for s in states]),
            dtype=torch.float32,
        )  # (B, 5N) or (B, 5N+1)
        Y = torch.tensor(np.stack(optimal_actions), dtype=torch.long)  # (B, N)

        self.policy_network._model.train()
        logits = self.policy_network.forward(X)  # (B, N, 4)
        B, N, A = logits.shape
        loss = F.cross_entropy(logits.view(B * N, A), Y.view(B * N))

        self.policy_network._optimizer.zero_grad()
        loss.backward()
        self.policy_network._optimizer.step()

    def save(self, path: str) -> None:
        """Save value function and policy network (with optimizer state) to `path/`."""
        import os, torch
        self.dqn_agent.save(path)
        if self.policy_network._model is not None:
            torch.save({
                'state_dict': self.policy_network._model.state_dict(),
                'optimizer_state_dict': (self.policy_network._optimizer.state_dict()
                                         if self.policy_network._optimizer is not None else None),
            }, os.path.join(path, 'policy_net.pt'))

    def load(self, path: str) -> None:
        """Restore value function and policy network from `path/`."""
        import os, torch
        self.dqn_agent.load(path)
        policy_path = os.path.join(path, 'policy_net.pt')
        if os.path.exists(policy_path) and self.policy_network._model is not None:
            data = torch.load(policy_path)
            if isinstance(data, dict) and 'state_dict' in data:
                self.policy_network._model.load_state_dict(data['state_dict'])
                if (data.get('optimizer_state_dict') is not None
                        and self.policy_network._optimizer is not None):
                    self.policy_network._optimizer.load_state_dict(data['optimizer_state_dict'])
            else:
                # Legacy: plain state dict saved without optimizer
                self.policy_network._model.load_state_dict(data)
