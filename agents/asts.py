"""Asset-Sequential Tree Search (ASTS) agent."""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field

import numpy as np

from agents.base import Agent
from agents.dqn import ValueBasedAgent
from env.mdp import State


@dataclass
class Node:
    depth: int
    partial_action: np.ndarray   # shape (N,), unassigned assets = 0
    q_value: float               # vf.predict([s_post_partial])[0]
    children: list = field(default_factory=list)


class ASTSAgent(Agent):
    """
    Asset-Sequential Tree Search agent.

    Builds an explicit depth-N tree where layer d assigns the action for
    asset d. Partial actions are padded with zeros (ACTION_NONE) for
    unevaluated assets. Search is depth-first with best-first restarts
    via a global min-heap of unexplored siblings.
    """

    def __init__(self, dqn_agent: ValueBasedAgent, max_leaves: int = 20):
        self.dqn_agent = dqn_agent
        self.env = dqn_agent.env
        self.max_leaves = max_leaves

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def act(self, state: State) -> np.ndarray:
        n = self.env.config.n_assets
        zeros = np.zeros(n, dtype=int)

        try:
            return self._search(state, n, zeros)
        except Exception:
            # Cold start or unfitted value function — fall back to ACTION_NONE
            return zeros

    def update(self, transitions) -> None:
        self.dqn_agent.update(transitions)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _node_q(self, state: State, partial_action: np.ndarray) -> float:
        s_post = self.env.post_decision_state(state, partial_action)
        return self.dqn_agent.value_fn.predict([s_post])[0]

    def _search(self, state: State, n: int, zeros: np.ndarray) -> np.ndarray:
        feas = self.env.feasible_actions(state)   # (N, 4) bool

        root = Node(
            depth=0,
            partial_action=zeros.copy(),
            q_value=self._node_q(state, zeros),
        )

        heap: list = []                    # min-heap of (q_value, counter, node)
        counter = itertools.count()
        best_q = np.inf
        best_action: np.ndarray = zeros.copy()
        leaves_found = 0

        current = root

        while leaves_found < self.max_leaves:
            if current.depth == n:
                # Leaf: complete action assigned for all assets
                if current.q_value < best_q:
                    best_q = current.q_value
                    best_action = current.partial_action.copy()
                leaves_found += 1
                if not heap:
                    break
                _, _, current = heapq.heappop(heap)
                continue

            # Expand: iterate feasible sub-actions for asset `current.depth`
            depth = current.depth
            feasible_sub = np.where(feas[depth])[0]

            children: list[Node] = []
            for a in feasible_sub:
                child_action = current.partial_action.copy()
                child_action[depth] = a
                q = self._node_q(state, child_action)
                children.append(Node(depth=depth + 1, partial_action=child_action, q_value=q))

            current.children = children

            if not children:
                # No feasible sub-actions — treat as dead branch, pop heap
                if not heap:
                    break
                _, _, current = heapq.heappop(heap)
                continue

            # DFS: descend to best child; push remaining siblings to heap
            children.sort(key=lambda nd: nd.q_value)
            for sibling in children[1:]:
                heapq.heappush(heap, (sibling.q_value, next(counter), sibling))
            current = children[0]

        return best_action
