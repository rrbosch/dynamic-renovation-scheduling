"""PPO agent for infrastructure maintenance scheduling."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Iterator, TYPE_CHECKING

from agents.base import Agent
from agents.fn.policy import PolicyNetwork
from agents.fn.value_net import ValueNetwork
from env.mdp import State

if TYPE_CHECKING:
    from env.mdp import InfraEnv


# ---------------------------------------------------------------------------
# RolloutBuffer
# ---------------------------------------------------------------------------

@dataclass
class _Batch:
    states: 'torch.Tensor'        # (B, input_dim)
    actions: 'torch.Tensor'       # (B, N)
    masks: 'torch.Tensor'         # (B, N, 4) bool
    old_log_probs: 'torch.Tensor' # (B,)
    advantages: 'torch.Tensor'    # (B,)
    returns: 'torch.Tensor'       # (B,)


class RolloutBuffer:
    """Stores one episode of on-policy experience."""

    def __init__(self) -> None:
        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []
        self.rewards: list[float] = []
        self.old_log_probs: list[float] = []
        self.values: list[float] = []
        self._advantages: np.ndarray | None = None
        self._returns: np.ndarray | None = None

    def append(
        self,
        state_feat: np.ndarray,
        action: np.ndarray,
        mask: np.ndarray,
        reward: float,
        log_prob: float,
        value: float,
    ) -> None:
        self.states.append(state_feat)
        self.actions.append(action.copy())
        self.masks.append(mask.copy())
        self.rewards.append(reward)
        self.old_log_probs.append(log_prob)
        self.values.append(value)

    def compute_gae(
        self, last_value: float, gamma: float, lam: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """GAE backward pass. Returns (advantages, returns), shape (T,)."""
        T = len(self.rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        values = np.array(self.values, dtype=np.float32)
        rewards = np.array(self.rewards, dtype=np.float32)

        for t in reversed(range(T)):
            next_val = last_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + gamma * next_val - values[t]
            gae = delta + gamma * lam * gae
            advantages[t] = gae

        returns = advantages + values
        self._advantages = advantages
        self._returns = returns
        return advantages, returns

    def mini_batches(self, batch_size: int) -> Iterator[_Batch]:
        """Yield shuffled mini-batches as _Batch namedtuples."""
        import torch
        T = len(self.states)
        idx = np.random.permutation(T)

        states_arr = np.stack(self.states).astype(np.float32)
        actions_arr = np.stack(self.actions)
        masks_arr = np.stack(self.masks)
        old_lp_arr = np.array(self.old_log_probs, dtype=np.float32)
        adv_arr = self._advantages.astype(np.float32)
        ret_arr = self._returns.astype(np.float32)

        # Normalize advantages
        adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)

        for start in range(0, T, batch_size):
            b = idx[start: start + batch_size]
            yield _Batch(
                states=torch.tensor(states_arr[b]),
                actions=torch.tensor(actions_arr[b], dtype=torch.long),
                masks=torch.tensor(masks_arr[b]),
                old_log_probs=torch.tensor(old_lp_arr[b]),
                advantages=torch.tensor(adv_arr[b]),
                returns=torch.tensor(ret_arr[b]),
            )

    def __len__(self) -> int:
        return len(self.states)


# ---------------------------------------------------------------------------
# PPOAgent
# ---------------------------------------------------------------------------

class PPOAgent(Agent):
    """
    On-policy PPO agent.

    Actor: PolicyNetwork (reused from actor_critic.py), outputs (B, N, 4) logits.
    Critic: ValueNetwork, outputs scalar V(s).
    Joint log-prob = sum of per-asset log-probs (actions are conditionally independent).
    """

    def __init__(
        self,
        env: 'InfraEnv',
        input_dim: int,
        n_assets: int,
        n_actions: int = 4,
        hidden_dims: list[int] | None = None,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        gae_lambda: float = 0.95,
        ppo_epochs: int = 4,
        mini_batch_size: int = 256,
        finite_horizon: bool = True,
    ):
        if hidden_dims is None:
            hidden_dims = [256, 256]
        self._env = env
        self.n_assets = n_assets
        self.n_actions = n_actions
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.gae_lambda = gae_lambda
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.finite_horizon = finite_horizon
        self.input_dim = input_dim

        self.actor = PolicyNetwork(
            input_dim=input_dim,
            n_assets=n_assets,
            n_actions=n_actions,
            hidden_dims=hidden_dims,
            lr=actor_lr,
        )
        self.critic = ValueNetwork(input_dim=input_dim, hidden_dims=hidden_dims)
        import torch
        self._critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    # ------------------------------------------------------------------
    # State → tensor
    # ------------------------------------------------------------------

    def _state_to_tensor(self, state: State) -> 'torch.Tensor':
        """Convert state to (1, input_dim) float tensor."""
        import torch
        feat = state.features().astype(np.float32)
        if self.finite_horizon:
            T = self._env.config.T
            t_norm = np.array([state.t / T], dtype=np.float32)
            feat = np.concatenate([feat, t_norm])
        return torch.tensor(feat).unsqueeze(0)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(self, state: State) -> np.ndarray:
        """Greedy (argmax) action for evaluation."""
        import torch
        with torch.no_grad():
            x = self._state_to_tensor(state)
            logits = self.actor.forward(x)[0]          # (N, 4)
            mask = self._env.feasible_actions(state)    # (N, 4) bool
            logits[~mask] = -1e9
            action = logits.argmax(dim=-1)
        return action.numpy()

    def act_with_info(
        self, state: State
    ) -> tuple[np.ndarray, float, float]:
        """Stochastic action + log-prob sum + value estimate for rollout collection."""
        import torch
        from torch.distributions import Categorical
        with torch.no_grad():
            x = self._state_to_tensor(state)
            logits = self.actor.forward(x)[0]           # (N, 4)
            value = self.critic.forward(x).item()       # scalar
            mask = self._env.feasible_actions(state)    # (N, 4) bool
            logits[~mask] = -1e9
            dist = Categorical(logits=logits)           # N independent
            action = dist.sample()                      # (N,)
            log_prob = dist.log_prob(action).sum()      # scalar
        return action.numpy(), log_prob.item(), value

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update_ppo(self, rollout: RolloutBuffer) -> dict:
        """Run ppo_epochs mini-batch passes and return mean loss stats."""
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.distributions import Categorical

        total_l_clip = 0.0
        total_l_value = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.ppo_epochs):
            for batch in rollout.mini_batches(self.mini_batch_size):
                logits = self.actor.forward(batch.states)   # (B, N, 4)
                logits = logits.masked_fill(~batch.masks, -1e9)

                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch.actions).sum(dim=-1)  # (B,)
                entropy = dist.entropy().sum(dim=-1).mean()               # scalar

                ratio = (new_log_probs - batch.old_log_probs).exp()
                adv = batch.advantages
                l_clip = torch.min(
                    ratio * adv,
                    ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv,
                ).mean()

                values = self.critic.forward(batch.states)  # (B,)
                l_value = F.mse_loss(values, batch.returns)

                loss = -l_clip + self.value_coef * l_value - self.entropy_coef * entropy

                self.actor._optimizer.zero_grad()
                self._critic_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor._model.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.critic._model.parameters(), 0.5)
                self.actor._optimizer.step()
                self._critic_opt.step()

                total_l_clip += l_clip.item()
                total_l_value += l_value.item()
                total_entropy += entropy.item()
                n_updates += 1

        denom = max(n_updates, 1)
        return {
            'l_clip': total_l_clip / denom,
            'l_value': total_l_value / denom,
            'entropy': total_entropy / denom,
        }

    def evaluate_action(
        self, state: State, action: np.ndarray, mask: np.ndarray
    ) -> tuple[float, float]:
        """
        Compute log_prob and value for an externally supplied action.
        Used in Phase 0 to train on heuristic trajectories.

        mask: (N, 4) bool from env.feasible_actions(state)
        Returns (log_prob, value) — scalars.
        """
        import torch
        from torch.distributions import Categorical
        with torch.no_grad():
            x = self._state_to_tensor(state)             # (1, input_dim)
            logits = self.actor.forward(x)[0]            # (N, 4)
            logits_masked = logits.clone()
            logits_masked[~torch.tensor(mask)] = -1e9
            dist = Categorical(logits=logits_masked)
            action_t = torch.tensor(action, dtype=torch.long)
            log_prob = dist.log_prob(action_t).sum().item()
            value = self.critic.forward(x).item()
        return log_prob, value

    # ------------------------------------------------------------------
    # Agent ABC
    # ------------------------------------------------------------------

    def update(self, transitions: list) -> None:
        """No-op — PPOTrainer calls update_ppo directly."""

    def save(self, path: str) -> None:
        import os, torch
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor._model.state_dict(), os.path.join(path, 'ppo_actor.pt'))
        torch.save(self.critic._model.state_dict(), os.path.join(path, 'ppo_critic.pt'))
