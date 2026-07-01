"""MixtureAgent: a per-episode weighted mixture of behavior policies.

Used to diversify warmstart data collection. At each episode start it randomly selects
one of several weighted sub-policies (e.g. tuned reactive / sampled-threshold reactive /
do-nothing) and follows it for the whole episode. Mixing in do-nothing episodes covers
the sustained-failure (high-``n_fail``) states a purely reactive warmstart never reaches
— the fix for the ADP value function's noisy-regime under-prediction of failure cost
(see docs/adp_value_fn_improvements.md). This is the offline-RL "mixed-quality behavior
dataset" idea, built from config and wired in via `experiments/configs.py::_build_agent`.

COLLECTION-ONLY / NOT parallelism-invariant. The agent holds mutable per-episode state
(`self._current`) and detects episode starts by `state.t == 0`, drawing from an explicit
`np.random.Generator`. This reproduces for the *sequential* warmstart loop (the draw
sequence is a pure function of the seed), but it must NOT be used as an evaluated policy
across parallel workers — there the codebase's `(seed, phase, episode_idx)` keyed-noise
model applies instead (CLAUDE.md §8). The cleaner `Agent.begin_episode` hook that would
make this parallelism-invariant is deliberately out of scope.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from agents.base import Agent
from env.mdp import State


class MixtureAgent(Agent):
    """Per-episode weighted choice among behavior sub-policies.

    modes: list of (weight, factory), where ``factory(rng) -> Agent`` produces this
    episode's sub-policy. A factory may return a fixed prebuilt agent (ignoring ``rng``)
    or sample a fresh agent per episode (e.g. randomized reactive thresholds). Weights
    are normalized; they need not sum to 1.
    """

    def __init__(self, modes: list[tuple[float, Callable[[np.random.Generator], Agent]]],
                 rng: np.random.Generator):
        if not modes:
            raise ValueError("MixtureAgent requires at least one mode.")
        weights = np.asarray([w for w, _ in modes], dtype=float)
        if np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError(
                f"MixtureAgent weights must be non-negative with positive sum; got {weights}.")
        self._weights = weights / weights.sum()
        self._factories = [f for _, f in modes]
        self._rng = rng
        self._current: Agent | None = None

    def act(self, state: State) -> np.ndarray:
        # New episode (state.t == 0) or first-ever call: pick this episode's sub-policy.
        if self._current is None or getattr(state, "t", 0) == 0:
            idx = int(self._rng.choice(len(self._factories), p=self._weights))
            self._current = self._factories[idx](self._rng)
        return self._current.act(state)
