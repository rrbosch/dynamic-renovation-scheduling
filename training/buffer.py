"""Replay buffer implementations."""
from __future__ import annotations

import collections
import random
import numpy as np
from dataclasses import dataclass, field

from env.mdp import State


@dataclass
class Transition:
    state:       State
    action:      np.ndarray
    cost:        float
    next_state:  State
    post_state:  State   # state after action, before stochastic transitions
    done:        bool
    pred_error:  float = float('inf')  # for lowest_error strategy; inf ensures new transitions are retained
    mc_return:   float = 0.0  # discounted return G_t from this step to end of episode


class ReplayBuffer:
    """
    Replay buffer with pluggable eviction strategies.

    Strategies:
      'fifo'               — oldest dropped when full (collections.deque)
      'lowest_error'       — drop transition with lowest pred_error when full
      'stochastic_knockout'— parametrised knockout eviction
    """

    def __init__(
        self,
        capacity: int,
        strategy: str = 'fifo',
        y: int = 5,   # rounds
        z: int = 5,   # candidate sets per round
        knockout_fraction: float = 0.05,  # min eviction batch as a fraction of capacity
    ):
        self.capacity = capacity
        self.strategy = strategy
        self.y = y
        self.z = z
        self.knockout_fraction = knockout_fraction
        # Optional backstop callback (set by the Trainer). Invoked by
        # stochastic_knockout eviction when no prediction-error signal exists
        # yet (every pred_error == inf); it should train the value function so
        # that errors become finite. Not pickled with the buffer.
        self.refresh_errors_fn = None

        if strategy == 'fifo':
            self._data: collections.deque = collections.deque(maxlen=capacity)
        else:
            self._data: list = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add(self, transition: Transition) -> None:
        if self.strategy == 'fifo':
            self._data.append(transition)
        elif self.strategy == 'lowest_error':
            if len(self._data) < self.capacity:
                self._data.append(transition)
            else:
                # Drop the transition with the lowest prediction error
                min_idx = int(np.argmin([t.pred_error for t in self._data]))
                self._data.pop(min_idx)
                self._data.append(transition)
        elif self.strategy == 'stochastic_knockout':
            self._data.append(transition)
            if len(self._data) > self.capacity:
                self._knockout_evict()
        else:
            raise ValueError(f"Unknown buffer strategy: {self.strategy!r}")

    def sample(self, n: int) -> list[Transition]:
        n = min(n, len(self._data))
        # deque needs list() for O(1) indexed access; list is already fine
        seq = list(self._data) if isinstance(self._data, collections.deque) else self._data
        return random.sample(seq, n)

    def update_errors(self, errors: np.ndarray) -> None:
        """Update pred_error fields for all stored transitions (in insertion order)."""
        data = list(self._data)
        for i, err in enumerate(errors[: len(data)]):
            data[i].pred_error = float(err)

    def save(self, path: str) -> None:
        """Pickle all transitions to `path`."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(list(self._data), f)

    def load(self, path: str) -> None:
        """Restore transitions from `path`."""
        import pickle, collections
        with open(path, 'rb') as f:
            data = pickle.load(f)
        if self.strategy == 'fifo':
            self._data = collections.deque(data, maxlen=self.capacity)
        else:
            self._data = data

    def __len__(self) -> int:
        return len(self._data)

    # ------------------------------------------------------------------
    # Knockout eviction
    # ------------------------------------------------------------------

    def _has_finite_error(self) -> bool:
        """True if at least one stored transition has a finite pred_error.
        Short-circuits, so it is O(1) once any error has been populated."""
        return any(np.isfinite(t.pred_error) for t in self._data)

    def _knockout_evict(self) -> None:
        """
        Remove transitions via knockout.
        For x too many datapoints, y rounds, z candidate sets per round, x/y removals per winning set.
        Total model re-evaluations: y * z.

        x is at least `knockout_fraction` of capacity, so eviction removes a
        sizeable batch and is therefore triggered far less often than once per
        add (it would otherwise run on essentially every add once the buffer is
        full, which is pathologically slow on a large buffer).
        """
        # Backstop: knockout ranks transitions by prediction error, so it needs
        # a trained model. If none exists yet (every pred_error is still inf,
        # e.g. eviction triggered during warmstart before any fit), train one
        # now via the Trainer-supplied callback. If errors are STILL inf
        # afterwards, that is a genuine bug — fail loudly rather than silently
        # evicting at random.
        if not self._has_finite_error():
            if self.refresh_errors_fn is not None:
                self.refresh_errors_fn()
            if not self._has_finite_error():
                raise RuntimeError(
                    "stochastic_knockout eviction was triggered but every "
                    "pred_error is inf"
                    + (" even after refresh_errors_fn()" if self.refresh_errors_fn
                       else " and no refresh_errors_fn was set")
                    + ". The value function did not populate prediction errors."
                )

        overshoot = len(self._data) - self.capacity
        x = max(int(self.knockout_fraction * self.capacity), overshoot)
        x = min(x, len(self._data))  # never try to remove more than we hold
        per_round = max(1, x // self.y)
        data = self._data

        for _round in range(self.y):
            best_set = None
            best_impact = np.inf

            for _trial in range(self.z):
                # Sample a candidate set to remove
                if len(data) <= per_round:
                    break
                indices = random.sample(range(len(data)), per_round)
                # Impact = sum of pred_errors in this set (lower = less informative)
                impact = sum(data[i].pred_error for i in indices)
                # `best_set is None` ensures a set is always chosen on the first
                # trial — otherwise when every pred_error is inf (e.g. during
                # warmstart, before the VFA has scored anything) `impact < inf`
                # is always False, best_set stays None, and NOTHING is evicted,
                # so the buffer grows unbounded.
                if best_set is None or impact < best_impact:
                    best_impact = impact
                    best_set = indices

            if best_set is not None:
                # Rebuild excluding the winning set: O(n) per round regardless of
                # batch size, vs O(x*n) for repeated list.pop on a large batch.
                remove = set(best_set)
                data[:] = [d for j, d in enumerate(data) if j not in remove]
