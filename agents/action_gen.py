"""Action generation strategies."""
from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod

from env.mdp import State, InfraEnv
from agents.fn.value_fn import ValueFn


class ActionGenerator(ABC):
    @abstractmethod
    def generate(self, state: State, value_fn: ValueFn, env: InfraEnv,
                 rng: np.random.Generator | None = None) -> np.ndarray:
        """Returns action array shape (N,)."""


# ---------------------------------------------------------------------------
# Local search
# ---------------------------------------------------------------------------

class LocalSearchGenerator(ActionGenerator):
    """
    Greedy local search over single-asset deviations.
    Ranks actions by Q(s,a) = c_maint + c_risk + c_travel + V'(s_post).
    """

    def __init__(self, n_restarts: int = 1, log_q_breakdown: bool = False):
        self.n_restarts = n_restarts
        # Opt-in diagnostic: when True, last_metrics carries the Q-component
        # decomposition (c_maint/c_risk/c_travel/V') for the chosen action vs the
        # do-nothing baseline at every decision. Off by default (zero overhead).
        self.log_q_breakdown = log_q_breakdown
        self.last_metrics: dict = {}

    def generate(self, state: State, value_fn: ValueFn, env: InfraEnv,
                 rng: np.random.Generator | None = None) -> np.ndarray:
        cfg = env.config
        feas = env.feasible_actions(state)  # (N, 4) bool — computed once
        self._n_candidates = 0

        def _q(s, a):
            self._n_candidates += 1
            return self._evaluate_q(s, a, value_fn, env)

        best_action = np.zeros(cfg.n_assets, dtype=int)
        best_q = _q(state, best_action)

        for _ in range(self.n_restarts):
            current_action = np.zeros(cfg.n_assets, dtype=int)
            current_q = _q(state, current_action)

            improved = True
            while improved:
                improved = False
                for i in range(cfg.n_assets):
                    for a in range(1, 4):  # try actions 1, 2, 3
                        if not feas[i, a]:
                            continue
                        candidate = current_action.copy()
                        candidate[i] = a
                        q = _q(state, candidate)
                        if q < current_q:
                            current_q = q
                            current_action = candidate
                            improved = True

            if current_q < best_q:
                best_q = current_q
                best_action = current_action

        self.last_metrics = {'n_candidates': self._n_candidates}
        if self.log_q_breakdown:
            self.last_metrics.update(
                _q_breakdown_metrics(state, best_action, value_fn, env))
        return best_action

    def _evaluate_q(self, state: State, action: np.ndarray,
                    value_fn: ValueFn, env: InfraEnv) -> float:
        """
        Q(s, a) = C(s, a) + V'(s_post_a)
        C(s, a) = c_maint + c_travel + c_risk  (all deterministic given s and a).
        V' is trained with future-only targets (mc_return - cost).
        """
        s_post = env.post_decision_state(state, action, check=False)
        return env.immediate_cost(state, action, s_post) + value_fn.predict([s_post])[0]


# ---------------------------------------------------------------------------
# Sequential generator
# ---------------------------------------------------------------------------

class SequentialGenerator(ActionGenerator):
    """
    Process assets in random order; for each, pick best action given committed choices.
    Ranks by Q(s,a) = c_maint + c_risk + c_travel + V'(s_post).
    """

    def __init__(self, log_q_breakdown: bool = False):
        self.log_q_breakdown = log_q_breakdown
        self.last_metrics: dict = {}

    def generate(self, state: State, value_fn: ValueFn, env: InfraEnv,
                 rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        cfg = env.config
        n = cfg.n_assets
        action = np.zeros(n, dtype=int)
        feas = env.feasible_actions(state)  # (N, 4) bool — computed once
        order = rng.permutation(n)
        count = 0

        for i in order:
            candidates = []
            for a in range(4):
                if not feas[i, a]:
                    continue
                count += 1
                candidate = action.copy()
                candidate[i] = a
                s_post = env.post_decision_state(state, candidate, check=False)
                candidates.append((a, candidate, s_post))

            if not candidates:
                continue
            # Batch predict
            post_states = [c[2] for c in candidates]
            v_preds = value_fn.predict(post_states)
            costs = [env.immediate_cost(state, c[1], c[2]) for c in candidates]
            q_vals = [c + v for c, v in zip(costs, v_preds)]
            best_idx = int(np.argmin(q_vals))
            action[i] = candidates[best_idx][0]

        self.last_metrics = {'n_candidates': count}
        if self.log_q_breakdown:
            self.last_metrics.update(
                _q_breakdown_metrics(state, action, value_fn, env))
        return action


# ---------------------------------------------------------------------------
# Diagnostic helper (opt-in): per-decision Q-component decomposition
# ---------------------------------------------------------------------------

def _q_breakdown_metrics(state: State, chosen: np.ndarray,
                         value_fn: ValueFn, env: InfraEnv) -> dict:
    """Decompose Q = c_maint + c_risk + c_travel + V'(s_post) for the chosen
    action vs the do-nothing baseline, plus a compact state summary.

    Reveals whether the greedy argmin is driven by the immediate-cost terms or
    by V', and whether V' actually discriminates acting from not-acting on the
    on-policy distribution. ~2 extra TAP calls per decision (usually cached) —
    negligible next to the dozens local search already performs.
    """
    none = np.zeros_like(chosen)
    sp_none = env.post_decision_state(state, none, check=False)
    sp_chosen = env.post_decision_state(state, chosen, check=False)
    m_n, r_n, t_n = env.immediate_cost_components(state, none, sp_none)
    m_c, r_c, t_c = env.immediate_cost_components(state, chosen, sp_chosen)
    v_n = float(value_fn.predict([sp_none])[0])
    v_c = float(value_fn.predict([sp_chosen])[0])
    return {
        # chosen-action Q components
        'q_chosen':    m_c + r_c + t_c + v_c,
        'cmaint_chosen': m_c, 'crisk_chosen': r_c, 'ctravel_chosen': t_c,
        'vpost_chosen': v_c,
        # do-nothing baseline Q components
        'q_none':      m_n + r_n + t_n + v_n,
        'crisk_none':  r_n, 'ctravel_none': t_n, 'vpost_none': v_n,
        # V' spread between acting and not (≈0 ⇒ V' cannot tell them apart)
        'dvpost_chosen_vs_none': v_c - v_n,
        # state summary at this decision
        'n_act':    int(np.count_nonzero(chosen)),
        'mean_d':   float(state.d.mean()),
        'max_d':    float(state.d.max()),
        'n_failed': int(np.count_nonzero((state.d >= env.config.d_fail) & (state.h == 0))),
    }


# ---------------------------------------------------------------------------
# BDQ generator (stub)
# ---------------------------------------------------------------------------

class BDQGenerator:
    """
    Branched DQN with N independent heads, shared trunk.
    use_gat=True path not yet implemented.
    """

    def __init__(self, use_gat: bool = False):
        if use_gat:
            raise NotImplementedError("GAT-based BDQGenerator not yet implemented.")
        self.use_gat = use_gat

    def generate(self, state: State, value_fn: ValueFn, env: InfraEnv,
                 rng: np.random.Generator | None = None) -> np.ndarray:
        raise NotImplementedError("BDQGenerator not yet implemented.")
