"""Monte Carlo Rollout Agent.

Selects actions by estimating Q(s, a) via forward rollouts under a base heuristic.
Serves as an anytime baseline / upper bound on heuristic policy quality.
"""
from __future__ import annotations

import numpy as np

from agents.base import Agent
from env.mdp import State, InfraEnv
from env.noise import keyed_philox


def rollout_noise(
    seed: int,
    root_state: State,
    decision_t: int,
    rollout_idx: int,
    n_steps: int,
    n_assets: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Deterministic per-rollout noise, keyed on (seed, root state, decision epoch,
    rollout index). Returns (u, eps), each shape (n_steps, n_assets):
      u   ~ Uniform[0, 1)  for the Gamma degradation increment
      eps ~ Normal(0, 1)   for the Wiener renovation step

    The key intentionally EXCLUDES the candidate action, so every candidate
    evaluated from the same root state during local search shares identical
    noise (common random numbers). The root state is hashed so that different
    decision states / episodes get independent scenarios, and `rollout_idx`
    separates the n_rollouts replicates. Fully reproducible with no stored
    rng state.

    The key is prefixed with the `"rollout"` phase tag so this stream is
    structurally independent from the real environment's `"training"` /
    `"evaluation"` transition streams (which use the same `keyed_philox`
    primitive) — an agent's internal rollouts can never replay the realized
    evaluation future. A single Philox generator produces the per-asset draws
    in one vectorized call.
    """
    g = keyed_philox(
        "rollout", seed, root_state.features().tobytes(), decision_t, rollout_idx
    )
    u = g.random((n_steps, n_assets))
    eps = g.standard_normal((n_steps, n_assets))
    return u, eps


class MonteCarloRolloutAgent(Agent):
    """
    Greedy local search where Q is estimated via Monte Carlo rollouts.

    Q(s, a) = immediate_cost(s, a) + E[discounted future cost under rollout_policy]

    Rollout noise is generated deterministically from a (seed, root state,
    decision epoch, rollout index) key — never from env.rng — so all candidate
    actions evaluated at a decision step share common random numbers. The agent
    holds no mutable rng state and never mutates env state or env.rng.
    """

    def __init__(
        self,
        rollout_policy: Agent,
        env: InfraEnv,
        n_rollouts: int = 30,
        rollout_horizon: int | None = None,
        seed: int = 0,
        action_threshold: float = 0.5,
        initial_action: str = 'policy',
        selection: str = 'adaptive',
        p_threshold: float = 0.02,
        min_rollouts: int = 20,
        max_rollouts: int | None = 100,
        rollout_batch: int = 5,
    ):
        self.rollout_policy = rollout_policy
        self.env = env
        self.n_rollouts = n_rollouts
        # Rollout lookahead is a FIXED window measured from the decision epoch.
        # When unspecified we fall back to the planning-horizon length T (a
        # constant, t-independent window), NOT `T - t_next`.
        #
        # The old `T - t_next` behaviour is a bug: evaluation runs T + tail_epochs
        # (the tail can be as long as T itself), and for every tail epoch
        # `t_next >= T` makes `max_steps = T - t_next <= 0`. With no rollout the
        # agent degenerates to a pure immediate-cost minimiser, which always
        # picks "none" (renovation/repair cost up front), lets assets fail, and
        # the escalating risk cost explodes — the rollout ends up ~10x worse than
        # its own base policy. A fixed window keeps the lookahead long enough to
        # value renovation at every epoch, including the tail.
        self.rollout_horizon = (
            int(rollout_horizon) if rollout_horizon is not None
            else int(env.config.T)
        )
        self.seed = seed
        self.action_threshold = action_threshold
        self.initial_action = initial_action

        # --- Adaptive (sequential Wilcoxon) rollout budgeting --------------
        # selection='adaptive' (default): incumbent-vs-challenger comparisons
        #   stop early on a one-sided paired Wilcoxon signed-rank test over the
        #   shared CRN rollout differences (§ docs/adaptive_rollout_literature.md).
        #   p_threshold / min_rollouts / max_rollouts mirror Optuna's
        #   WilcoxonPruner (p_threshold / n_startup_steps) plus a budget cap. The
        #   defaults (p=0.02, min=20, max=100) are the chosen Pareto operating
        #   point from the p_threshold x min_rollouts sweep on instance_10p
        #   (~sub-1% mean cost regret at ~62% rollouts saved).
        # selection='fixed': legacy behaviour — every candidate gets exactly
        #   n_rollouts rollouts, then argmin.
        if selection not in ('fixed', 'adaptive'):
            raise ValueError(f"selection must be 'fixed' or 'adaptive', got {selection!r}")
        self.selection = selection
        self.p_threshold = p_threshold
        self.min_rollouts = min_rollouts
        self.max_rollouts = max_rollouts if max_rollouts is not None else n_rollouts
        self.rollout_batch = max(1, rollout_batch)
        if self.min_rollouts > self.max_rollouts:
            raise ValueError(
                f"min_rollouts ({self.min_rollouts}) > max_rollouts ({self.max_rollouts})"
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def act(self, state: State) -> np.ndarray:
        """Greedy local search over single-asset deviations, using MC Q estimates."""
        if self.selection == 'adaptive':
            return self._act_adaptive(state)
        cfg = self.env.config
        n = cfg.n_assets
        feas = self.env.feasible_actions(state)  # (N, 4) bool
        # Assets below threshold may only do ACTION_NONE
        feas[state.d < self.action_threshold, 1:] = False
        _n_candidates = 0

        # Initial action: policy warm-start or empty (all zeros)
        if self.initial_action == 'policy':
            current_action = self.rollout_policy.act(state)
            current_action = np.where(
                feas[np.arange(n), current_action], current_action, self.env.ACTION_NONE
            )
        else:
            current_action = np.zeros(n, dtype=int)
        _n_candidates += 1
        current_q = self._estimate_q(state, current_action)

        improved = True
        while improved:
            improved = False
            for i in range(n):
                for a in range(4):
                    if a == current_action[i]:
                        continue
                    if not feas[i, a]:
                        continue
                    candidate = current_action.copy()
                    candidate[i] = a
                    _n_candidates += 1
                    q = self._estimate_q(state, candidate)
                    if q < current_q:
                        current_q = q
                        current_action = candidate
                        improved = True

        self.step_metrics = {'n_candidates': _n_candidates}
        return current_action

    # ------------------------------------------------------------------
    # Q estimation
    # ------------------------------------------------------------------

    def _estimate_q(self, state: State, action: np.ndarray) -> float:
        """
        Q(s, a) = immediate_cost(s, a) + mean future cost over N rollouts.

        immediate_cost = c_maint + c_risk + c_travel (deterministic given s, a).
        future_cost    = mean over rollouts of _single_rollout(s_post, t+1, max_steps).
        """
        cfg = self.env.config
        s_post = self.env.post_decision_state(state, action)
        immediate_cost = self.env.immediate_cost(state, action, s_post)

        # Max remaining steps for rollout
        t_next = state.t + 1
        if self.rollout_horizon is not None:
            max_steps = self.rollout_horizon
        else:
            max_steps = cfg.T - t_next

        if max_steps <= 0:
            return immediate_cost

        # Monte Carlo future cost estimate. Noise is keyed on the root `state`
        # (shared across candidate actions) and the rollout index, giving CRN.
        future_cost = float(np.mean([
            self._single_rollout(s_post, t_next, max_steps, state, r)
            for r in range(self.n_rollouts)
        ]))

        return immediate_cost + future_cost

    # ------------------------------------------------------------------
    # Adaptive (sequential Wilcoxon) action selection
    # ------------------------------------------------------------------

    def _act_adaptive(self, state: State) -> np.ndarray:
        """Greedy local search where each incumbent-vs-challenger comparison
        stops early via a paired Wilcoxon signed-rank test over shared CRN
        rollouts. Identical search structure to `act`; only the Q comparison
        differs. Fully deterministic (same CRN keying, deterministic test)."""
        cfg = self.env.config
        n = cfg.n_assets
        feas = self.env.feasible_actions(state)  # (N, 4) bool
        feas[state.d < self.action_threshold, 1:] = False

        # Per-decision-epoch cache of computed Q samples, keyed by action bytes.
        # The incumbent's samples are reused across comparisons (CRN keys on the
        # root state, not the action), mirroring the fixed path's cached current_q.
        cache: dict[bytes, dict] = {}
        self._sim_count = 0
        _n_candidates = 0

        if self.initial_action == 'policy':
            current_action = self.rollout_policy.act(state)
            current_action = np.where(
                feas[np.arange(n), current_action], current_action, self.env.ACTION_NONE
            )
        else:
            current_action = np.zeros(n, dtype=int)
        _n_candidates += 1

        # Round cap: unlike the fixed path (where current_q strictly decreases
        # so termination is guaranteed), the statistical accept decision is not
        # guaranteed transitive — a noisy accept could in principle cycle. Cap
        # the improving passes defensively. n full sweeps is far more than a
        # greedy local search ever needs in practice.
        max_rounds = max(2 * n, 8)
        improved = True
        rounds = 0
        while improved and rounds < max_rounds:
            improved = False
            rounds += 1
            for i in range(n):
                for a in range(4):
                    if a == current_action[i]:
                        continue
                    if not feas[i, a]:
                        continue
                    candidate = current_action.copy()
                    candidate[i] = a
                    _n_candidates += 1
                    if self._challenger_wins(state, candidate, current_action, cache):
                        current_action = candidate
                        improved = True

        self.step_metrics = {
            'n_candidates': _n_candidates,
            'n_rollout_sims': self._sim_count,
            'local_search_rounds': rounds,
        }
        return current_action

    def _q_samples(
        self, state: State, action: np.ndarray, n: int, cache: dict
    ) -> np.ndarray:
        """Return the first `n` per-rollout Q samples for (state, action),
        computing (and caching) any not yet drawn. Sample r uses CRN rollout
        index r, so samples are paired across candidate actions."""
        cfg = self.env.config
        key = action.tobytes()
        entry = cache.get(key)
        if entry is None:
            s_post = self.env.post_decision_state(state, action)
            immediate_cost = self.env.immediate_cost(state, action, s_post)
            t_next = state.t + 1
            if self.rollout_horizon is not None:
                max_steps = self.rollout_horizon
            else:
                max_steps = cfg.T - t_next
            entry = {
                's_post': s_post,
                'immediate': immediate_cost,
                't_next': t_next,
                'max_steps': max_steps,
                'q': [],
            }
            cache[key] = entry

        q = entry['q']
        max_steps = entry['max_steps']
        while len(q) < n:
            r = len(q)
            if max_steps <= 0:
                q.append(entry['immediate'])
            else:
                q.append(entry['immediate'] + self._single_rollout(
                    entry['s_post'], entry['t_next'], max_steps, state, r
                ))
                self._sim_count += 1
        return np.asarray(q[:n], dtype=float)

    def _challenger_wins(
        self, state: State, challenger: np.ndarray, incumbent: np.ndarray, cache: dict
    ) -> bool:
        """Sequentially test whether `challenger` has strictly lower Q than
        `incumbent` using a one-sided paired Wilcoxon signed-rank stop on the
        shared-CRN differences. Returns True iff the challenger should replace
        the incumbent. Caps at max_rollouts; at the cap, decides by sign of the
        mean paired difference (ties favour the incumbent)."""
        from scipy.stats import wilcoxon

        n = min(self.min_rollouts, self.max_rollouts)
        while True:
            qc = self._q_samples(state, challenger, n, cache)
            qi = self._q_samples(state, incumbent, n, cache)
            diffs = qc - qi  # < 0 ⇒ challenger cheaper (better)

            mean_diff = float(diffs.mean())
            if np.allclose(diffs, 0.0):
                # Exact tie across all scenarios — never displace the incumbent.
                return False

            decided = False
            challenger_better = False
            # One-sided test "challenger better" (diff < 0), guarded by the mean
            # (Optuna's average-is-best safety check) so a near-zero or wrong-sign
            # mean can't trigger a spurious early decision.
            if mean_diff < 0:
                p = wilcoxon(diffs, alternative='less', zero_method='zsplit').pvalue
                if p < self.p_threshold:
                    decided, challenger_better = True, True
            elif mean_diff > 0:
                p = wilcoxon(diffs, alternative='greater', zero_method='zsplit').pvalue
                if p < self.p_threshold:
                    decided, challenger_better = True, False

            if decided:
                return challenger_better
            if n >= self.max_rollouts:
                # Budget exhausted: decide by mean sign (strict ⇒ ties keep incumbent).
                return mean_diff < 0
            n = min(n + self.rollout_batch, self.max_rollouts)

    # ------------------------------------------------------------------
    # Single rollout
    # ------------------------------------------------------------------

    def _single_rollout(
        self, s_post: State, t_start: int, max_steps: int,
        root_state: State, rollout_idx: int,
    ) -> float:
        """
        Simulate forward from s_post for max_steps epochs under rollout_policy.

        Noise is drawn once for this rollout from a (seed, root_state, decision
        epoch, rollout_idx) key, so all candidate actions evaluated at this
        decision step replay the identical scenario (common random numbers).
        Discount starts at gamma^1 (future relative to the current epoch).
        Returns total discounted future cost for one trajectory.
        """
        cfg = self.env.config
        current = s_post.copy()
        current.t = t_start - 1  # t_start is next epoch; current.t is the s_post epoch

        u_bank, eps_bank = rollout_noise(
            self.seed, root_state, root_state.t, rollout_idx, max_steps, cfg.n_assets
        )

        discount = cfg.gamma
        total = 0.0
        t = t_start

        for k in range(max_steps):
            # Get heuristic action; mask infeasible choices to ACTION_NONE
            raw_action = self.rollout_policy.act(current)
            feas = self.env.feasible_actions(current)
            action = np.where(
                feas[np.arange(cfg.n_assets), raw_action],
                raw_action,
                InfraEnv.ACTION_NONE,
            )

            # Post-decision state
            s_post_step = self.env.post_decision_state(current, action)
            cost = self.env.immediate_cost(current, action, s_post_step)

            total += discount * cost
            discount *= cfg.gamma

            # Stochastic transition to next state using this rollout's CRN noise
            current, t = self.env.stochastic_transition(
                s_post_step, t, (u_bank[k], eps_bank[k])
            )

        return total

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """No mutable state: rollout noise is fully determined by (seed, state, t)."""
        import os
        os.makedirs(path, exist_ok=True)

    def load(self, path: str) -> None:
        """No mutable state to restore (rollout noise is key-derived)."""
        pass


class SequentialMCRolloutAgent(MonteCarloRolloutAgent):
    """
    Sequential per-asset action selection with MC rollout Q estimates.

    Processes assets in index order (0, 1, ..., N-1), picking the best
    action for each given already-committed choices for earlier assets.
    Mirrors SequentialGenerator logic but with MC rollout Q estimation.
    """

    def act(self, state: State) -> np.ndarray:
        """Sequential per-asset action selection using MC Q estimates."""
        if self.selection == 'adaptive':
            return self._act_adaptive_sequential(state)
        cfg = self.env.config
        n = cfg.n_assets
        feas = self.env.feasible_actions(state)  # (N, 4) bool
        feas[state.d < self.action_threshold, 1:] = False

        # Initial action: policy warm-start or empty (all zeros)
        if self.initial_action == 'policy':
            action = self.rollout_policy.act(state)
            action = np.where(
                feas[np.arange(n), action], action, self.env.ACTION_NONE
            )
        else:
            action = np.zeros(n, dtype=int)
        n_candidates = 0

        for i in range(n):
            # Baseline = asset i's current action (0 under 'empty' init, the
            # warm-start policy's choice under 'policy' init). best_a must track
            # the action that produced best_q — initialising it to 0 while
            # measuring best_q at a non-zero current action[i] silently commits
            # ACTION_NONE whenever no deviation improves, which corrupted the
            # 'policy' init variant.
            best_a = int(action[i])
            best_q = self._estimate_q(state, action)
            n_candidates += 1
            for a in range(4):
                if a == best_a or not feas[i, a]:
                    continue
                candidate = action.copy()
                candidate[i] = a
                n_candidates += 1
                q = self._estimate_q(state, candidate)
                if q < best_q:
                    best_q = q
                    best_a = a
            action[i] = best_a

        self.step_metrics = {'n_candidates': n_candidates}
        return action

    def _act_adaptive_sequential(self, state: State) -> np.ndarray:
        """Adaptive counterpart of `act`: per-asset, each deviation challenges
        the asset's current best action via the same paired-Wilcoxon early stop
        as the local-search agent. The incumbent for asset i is the best action
        committed so far; a challenger replaces it only if it wins the test."""
        cfg = self.env.config
        n = cfg.n_assets
        feas = self.env.feasible_actions(state)  # (N, 4) bool
        feas[state.d < self.action_threshold, 1:] = False

        cache: dict[bytes, dict] = {}
        self._sim_count = 0

        if self.initial_action == 'policy':
            action = self.rollout_policy.act(state)
            action = np.where(
                feas[np.arange(n), action], action, self.env.ACTION_NONE
            )
        else:
            action = np.zeros(n, dtype=int)
        n_candidates = 0

        for i in range(n):
            best_a = int(action[i])
            n_candidates += 1
            for a in range(4):
                if a == best_a or not feas[i, a]:
                    continue
                candidate = action.copy()
                candidate[i] = a
                n_candidates += 1
                # Compare candidate against the asset's current best (incumbent
                # = `action` with asset i set to best_a).
                incumbent = action.copy()
                incumbent[i] = best_a
                if self._challenger_wins(state, candidate, incumbent, cache):
                    best_a = a
            action[i] = best_a

        self.step_metrics = {
            'n_candidates': n_candidates,
            'n_rollout_sims': self._sim_count,
        }
        return action
