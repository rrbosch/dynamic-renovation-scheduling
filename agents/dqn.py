"""Value-based agents: ADPAgent (post-decision) and DQNAgent (next-state TD)."""
from __future__ import annotations

import abc
import copy
import numpy as np

from agents.base import Agent
from agents.fn.value_fn import ValueFn
from agents.action_gen import ActionGenerator
from env.mdp import State, InfraEnv
from training.buffer import Transition


class ValueBasedAgent(Agent, abc.ABC):
    """Abstract base for value-function agents with a shared action generator."""

    def __init__(
        self,
        value_fn: ValueFn,
        action_gen: ActionGenerator,
        env: InfraEnv,
        training_config,  # TrainingConfig — imported lazily to avoid circular
        finite_horizon: bool = True,
        init_action_mode: str = 'empty',
        warmstart_policy: 'Agent | None' = None,
    ):
        self.value_fn = value_fn
        self.action_gen = action_gen
        self.env = env
        self.training_config = training_config
        self.finite_horizon = finite_horizon
        # Action-search seed: 'empty' starts from do-nothing; 'policy' starts the
        # search from `warmstart_policy`'s action at each decision (set externally
        # in build_experiment — same heuristic used to warmstart the buffer).
        self.init_action_mode = init_action_mode
        self.warmstart_policy = warmstart_policy

    def act(self, state: State) -> np.ndarray:
        init_action = None
        if self.init_action_mode == 'policy' and self.warmstart_policy is not None:
            init_action = self.warmstart_policy.act(state)
        action = self.action_gen.generate(state, self.value_fn, self.env,
                                          init_action=init_action)
        self.step_metrics = dict(getattr(self.action_gen, 'last_metrics', {}))
        return action

    @abc.abstractmethod
    def update(self, transitions: list[Transition]) -> None: ...

    def save(self, path: str) -> None:
        """Save value function to `path/`."""
        import os
        os.makedirs(path, exist_ok=True)
        self.value_fn.save(os.path.join(path, 'value_fn'))

    def load(self, path: str) -> None:
        """Restore value function from `path/`."""
        import os
        self.value_fn.load(os.path.join(path, 'value_fn'))


class ADPAgent(ValueBasedAgent):
    """
    ADP-style agent using post-decision state bootstrapping.

    Training target: mc_return - cost  (future-only discounted return).
    Value function is trained on post-decision state features.
    Action generators evaluate Q(s,a) = C(s,a) + V'(s_post).
    """

    def update(self, transitions: list[Transition]) -> None:
        if len(transitions) == 0:
            return

        y = np.array([t.mc_return - t.cost for t in transitions])
        post_states = [t.post_state for t in transitions]
        # fit_targets fits on the post-decision states; when the value fn has the
        # per-epoch baseline enabled it learns the advantage y - b(t) and adds
        # b(t) back on predict (Q reconstruction in the action gens unchanged).
        self.value_fn.fit_targets(post_states, y)

        # Update prediction errors for buffer strategies
        if hasattr(self.value_fn, 'last_rank_errors') and self.value_fn.last_rank_errors is not None:
            for t, err in zip(transitions, self.value_fn.last_rank_errors):
                t.pred_error = float(err)
        else:
            v_pred = self.value_fn.predict([t.post_state for t in transitions])
            target_vals = np.array([t.mc_return - t.cost for t in transitions])
            for t, pred, tgt in zip(transitions, v_pred, target_vals):
                t.pred_error = abs(tgt - pred)


class DQNAgent(ValueBasedAgent):
    """
    Standard DQN agent using next-state TD(0) bootstrapping.

    Training target: cost + γ·V(s_next; θ_prev).
    Value function is trained on pre-decision state features.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._theta_prev: ValueFn | None = None

    def update(self, transitions: list[Transition]) -> None:
        if len(transitions) == 0:
            return

        cfg = self.env.config
        self._theta_prev = copy.deepcopy(self.value_fn)

        v_bootstrap = self._theta_prev.predict([t.next_state for t in transitions])
        y = np.array([
            t.cost + (0.0 if t.done else cfg.gamma * v)
            for t, v in zip(transitions, v_bootstrap)
        ])
        X = self.value_fn._feats([t.state for t in transitions])
        self.value_fn.fit(X, y)

        # Update prediction errors for buffer strategies
        if hasattr(self.value_fn, 'last_rank_errors') and self.value_fn.last_rank_errors is not None:
            for t, err in zip(transitions, self.value_fn.last_rank_errors):
                t.pred_error = float(err)
        else:
            v_pred = self.value_fn.predict([t.state for t in transitions])
            v_next_all = self._theta_prev.predict([t.next_state for t in transitions])
            costs = np.array([t.cost for t in transitions])
            dones = np.array([t.done for t in transitions], dtype=float)
            target_vals = costs + (1.0 - dones) * cfg.gamma * v_next_all
            for t, pred, tgt in zip(transitions, v_pred, target_vals):
                t.pred_error = abs(tgt - pred)

    def save(self, path: str) -> None:
        """Save value function and theta_prev snapshot to `path/`."""
        import os, pickle
        super().save(path)
        if self._theta_prev is not None:
            with open(os.path.join(path, 'theta_prev.pkl'), 'wb') as f:
                pickle.dump(self._theta_prev, f)

    def load(self, path: str) -> None:
        """Restore value function and theta_prev snapshot from `path/`."""
        import os, pickle
        super().load(path)
        theta_path = os.path.join(path, 'theta_prev.pkl')
        if os.path.exists(theta_path):
            with open(theta_path, 'rb') as f:
                self._theta_prev = pickle.load(f)
