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
# Running mean/std (for return / value normalization)
# ---------------------------------------------------------------------------

class _RunningMeanStd:
    """Welford running mean/variance, updated one batch (rollout) at a time."""

    def __init__(self) -> None:
        self.mean: float = 0.0
        self.var: float = 1.0
        self.count: float = 1e-4  # tiny prior count avoids div-by-zero

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return
        b_mean = float(x.mean())
        b_var = float(x.var())
        b_count = x.size
        delta = b_mean - self.mean
        tot = self.count + b_count
        self.mean += delta * b_count / tot
        m_a = self.var * self.count
        m_b = b_var * b_count
        m2 = m_a + m_b + delta * delta * self.count * b_count / tot
        self.var = m2 / tot
        self.count = tot

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var)) + 1e-8


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
        self._critic_lr = critic_lr  # stored for curriculum_reset_critic in PPOTrainer
        self._critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # --- Normalization (see fix note) ---------------------------------
        # The critic regresses to discounted returns of magnitude ~1e10-1e11
        # (rewards are -cost in euros). A plain-MSE MLP cannot fit targets of
        # that scale (loss ~1e20, grad-clipped steps never move the net), so the
        # critic collapses to a near-constant -> GAE advantages are noise -> the
        # policy never learns. This is the same scale pathology that broke the
        # ADP NeuralValueFn. Fix:
        #   (1) Value normalization: the critic predicts a STANDARDIZED value
        #       v_norm; raw value = v_norm * ret_std + ret_mean. The running
        #       return stats are updated once per rollout. Raw values are used
        #       everywhere GAE needs them (collection, bootstrap); the critic
        #       MSE loss is computed in normalized space (O(1) targets).
        #   (2) Input scaling: state.features() mixes columns in [0,1] (d, h,
        #       ell, r) with n_fail in [0, T]. We rescale the n_fail block by
        #       1/T so every input column is ~O(1) (t is already normalized in
        #       _state_to_tensor). Static, stateless, deterministic.
        self._ret_rms = _RunningMeanStd()
        feat_scale = np.ones(input_dim, dtype=np.float32)
        T = max(int(self._env.config.T), 1)
        feat_scale[4 * n_assets:5 * n_assets] = 1.0 / T  # n_fail block
        self._feat_scale = feat_scale

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
        feat = feat * self._feat_scale  # rescale n_fail block to ~O(1)
        return torch.tensor(feat).unsqueeze(0)

    def _value_raw(self, x: 'torch.Tensor') -> 'torch.Tensor':
        """Critic value in RAW (euro) units. The critic head outputs a
        standardized value; we invert the return normalization here so that
        everything GAE consumes stays in raw reward units."""
        return self.critic.forward(x) * self._ret_rms.std + self._ret_rms.mean

    def predict_value(self, state: State) -> float:
        """Raw-unit value estimate for a single state (used for GAE bootstrap)."""
        import torch
        with torch.no_grad():
            return float(self._value_raw(self._state_to_tensor(state)).item())

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
            value = self._value_raw(x).item()           # scalar, raw euro units
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

        # Update running return stats once per rollout, then train the critic to
        # predict STANDARDIZED returns (O(1) targets). Stats are frozen for the
        # duration of this update so the critic head and the (already-collected)
        # raw value estimates remain mutually consistent.
        if rollout._returns is not None:
            self._ret_rms.update(rollout._returns)
        ret_mean, ret_std = self._ret_rms.mean, self._ret_rms.std

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

                # Critic predicts standardized values; compare to standardized
                # returns so the regression target is O(1) (see __init__ note).
                values = self.critic.forward(batch.states)  # (B,) normalized
                ret_target = (batch.returns - ret_mean) / ret_std
                l_value = F.mse_loss(values, ret_target)

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
            value = self._value_raw(x).item()
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
        # Return-normalization stats: needed to interpret the critic's
        # standardized output in raw units (e.g. for analysis / warm restart).
        torch.save({'ret_mean': self._ret_rms.mean, 'ret_var': self._ret_rms.var,
                    'ret_count': self._ret_rms.count},
                   os.path.join(path, 'ppo_norm.pt'))
