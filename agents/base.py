"""Abstract base class for all agents."""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from env.mdp import State


class Agent(ABC):
    @abstractmethod
    def act(self, state: State) -> np.ndarray:
        """Returns action array shape (N,)."""

    def update(self, transitions: list) -> None:
        """No-op for non-learning agents."""

    def save(self, path: str) -> None:
        """Persist learned parameters to directory `path`. No-op for non-learning agents."""

    def load(self, path: str) -> None:
        """Restore learned parameters from directory `path`. No-op for non-learning agents."""
