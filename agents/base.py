"""Abstract base class for all agents."""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from env.mdp import State


class Agent(ABC):
    @abstractmethod
    def act(self, state: State) -> np.ndarray:
        """Returns action array shape (N,)."""

    def on_episode_start(self, phase: str, episode_idx: int, base_seed: int,
                         env, horizon: int) -> None:
        """Hook called by the evaluation harness at the start of each episode,
        immediately after env.begin_episode()/reset().

        No-op for almost all agents. Anticipative baselines (e.g.
        ``ClairvoyantAgent``) use it to reconstruct the episode's exact realized
        noise from ``(phase, base_seed, episode_idx)`` and pre-solve a per-seed
        plan. ``horizon`` is the evaluation length (T + tail_epochs)."""

    def update(self, transitions: list) -> None:
        """No-op for non-learning agents."""

    def save(self, path: str) -> None:
        """Persist learned parameters to directory `path`. No-op for non-learning agents."""

    def load(self, path: str) -> None:
        """Restore learned parameters from directory `path`. No-op for non-learning agents."""
