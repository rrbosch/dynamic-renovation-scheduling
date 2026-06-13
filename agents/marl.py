"""Multi-agent RL (CTDE) — stub."""
from __future__ import annotations

import numpy as np
from agents.base import Agent
from env.mdp import State


class CTDEAgent(Agent):
    """
    Centralized Training, Decentralized Execution (stub).

    Architecture intent (future):
    - N per-asset policy networks π_i(o_i) where o_i is local observation.
    - Centralized critic V(S_t) with access to full joint state during training.
    - GAT encoder for topology-aware local embeddings.
    """

    def act(self, state: State) -> np.ndarray:
        raise NotImplementedError("MARL CTDE not yet implemented.")

    def update(self, transitions: list) -> None:
        raise NotImplementedError("MARL CTDE not yet implemented.")
