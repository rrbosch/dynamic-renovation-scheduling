"""MDP environment for infrastructure maintenance scheduling."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional

from env.network import NetworkData
from env.degradation import gamma_step, wiener_step
from env.noise import keyed_philox
from env.tap import TAPSolver


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class State:
    d:      np.ndarray  # (N,) condition levels in [0,1]
    h:      np.ndarray  # (N,) remaining renovation work (>0 = under renovation)
    ell:    np.ndarray  # (N,) load restriction indicator (bool-like float)
    r:      np.ndarray  # (N,) repair-used indicator (bool-like float)
    n_fail: np.ndarray  # (N,) consecutive failed steps
    t:      int = 0     # current epoch index (populated by InfraEnv)

    def features(self) -> np.ndarray:
        """Flat concatenation, shape (5N,). Input to value function."""
        return np.concatenate([self.d, self.h, self.ell, self.r, self.n_fail])

    def copy(self) -> 'State':
        return State(self.d.copy(), self.h.copy(),
                     self.ell.copy(), self.r.copy(), self.n_fail.copy(), self.t)


# ---------------------------------------------------------------------------
# EnvConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EnvConfig:
    n_assets: int
    gamma: float
    mu_h: float | np.ndarray
    sigma_h: float | np.ndarray
    delta_repair: float
    alpha0: np.ndarray        # (N,) baseline shape rates
    beta: np.ndarray          # (N,) Gamma rate parameters
    c_ren: np.ndarray         # (N,) renovation costs (€/asset)
    c_rep: np.ndarray         # (N,) repair costs (€/asset)
    asset_lengths_m: np.ndarray  # (N,) metres per asset
    T: int = 120
    dt: float = 0.5
    d_fail: float = 1.0
    eta_ren: float = 0.05
    eta_load: float = 0.50
    restrict_degrad_multiplier: float = 0.5   # degradation rate multiplier reduction under restriction
    vot: float = 10.76           # value of time (euros per vehicle-hour)
    traffic_cost_factor: float = 1.0  # scales raw traffic cost
    risk_base: float = 10_000.0       # €/m/year per failure step
    d_init: Optional[np.ndarray] = None
    allow_repair:   bool = True   # if False, repair action is always infeasible
    allow_restrict: bool = True   # if False, load-restriction action is always infeasible

    class Config:
        # Allow numpy arrays in frozen dataclass
        arbitrary_types_allowed = True


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class InfraEnv:
    """
    MDP environment for infrastructure maintenance scheduling.

    Actions: 0=none, 1=repair, 2=renovate, 3=restrict
    """

    ACTION_NONE = 0
    ACTION_REPAIR = 1
    ACTION_RENOVATE = 2
    ACTION_RESTRICT = 3

    def __init__(
        self,
        network: NetworkData,
        tap_fn: TAPSolver,
        config: EnvConfig,
        rng_seed: int = 0,
    ):
        self.network = network
        self.tap_fn = tap_fn
        self.config = config
        # Stateless, phase-keyed randomness (see env/noise.py). The env no longer
        # holds a mutable stepping RNG; each episode derives a fresh generator
        # from (phase, base_seed, episode_idx) via begin_episode(). `rng_seed`
        # becomes the default base seed used when a caller does not pass one.
        self._base_seed = int(rng_seed)
        self._noise_gen: Optional[np.random.Generator] = None  # transition noise (per episode)
        self._reset_gen: Optional[np.random.Generator] = None  # d_init sampling (per episode)
        self._episode_ready = False   # True after begin_episode(), until consumed by reset()
        self._auto_ep = 0             # counter for the auto-begin fallback
        self._t = 0

        # Pre-compute baseline travel cost: full nominal capacities, no ongoing projects
        net = network
        bl_flows = tap_fn.solve(net.nominal_capacities)
        bl_tt = net.free_flow_tt * (
            1.0 + net.bpr_beta * (bl_flows / np.maximum(net.nominal_capacities, 1e-9)) ** net.bpr_nu
        )
        self._c_travel_baseline = float(np.sum(bl_flows * bl_tt))
        self.last_cost_breakdown = (0.0, 0.0, 0.0)  # (c_travel, c_maint, c_risk)

    # ------------------------------------------------------------------
    # Episode randomness keying
    # ------------------------------------------------------------------

    def begin_episode(self, phase: str, episode_idx: int,
                      base_seed: int | None = None) -> None:
        """Arm the next episode with a stateless, phase-keyed noise stream.

        All randomness for the upcoming episode (the `d_init` sampling in
        reset() and the per-step transition noise in step()) is derived
        deterministically from ``(phase, seed, episode_idx)`` — independent of
        any prior draws, of how many episodes ran before, of parallelism, and of
        resumes. Different ``phase`` tags ("training" / "evaluation" /
        "rollout" / ...) yield structurally independent streams.

        Call this immediately before reset() for each episode. `base_seed`
        defaults to the env's construction seed.
        """
        seed = self._base_seed if base_seed is None else int(base_seed)
        self._noise_gen = keyed_philox("transition", phase, seed, int(episode_idx))
        self._reset_gen = keyed_philox("reset", phase, seed, int(episode_idx))
        self._episode_ready = True

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> State:
        """
        Sample initial state:
          d ~ Uniform(0, 0.5) per asset
          h = 0, ell = 0, r = 0

        If begin_episode() was not called since the last reset, an auto-begin
        fallback keys the episode on ("default", base_seed, auto_counter) so
        ad-hoc env use (tests, scripts) stays deterministic.
        """
        cfg = self.config
        if not self._episode_ready:
            self.begin_episode("default", self._auto_ep)
            self._auto_ep += 1
        self._episode_ready = False  # consumed; explicit callers re-arm each episode

        d = (cfg.d_init.copy() if cfg.d_init is not None
             else self._reset_gen.uniform(0.0, 0.5, size=cfg.n_assets))
        h = np.zeros(cfg.n_assets)
        ell = np.zeros(cfg.n_assets)
        r = np.zeros(cfg.n_assets)
        n_fail = np.zeros(cfg.n_assets)
        self._t = 0
        return State(d, h, ell, r, n_fail)

    def set_state(self, state: State, t: int) -> None:
        """
        Restore the environment to a previously visited (state, t) without
        copying any expensive objects (network, tap_fn, config).
        Allows rollout agents to fork from a checkpoint using env.step() if
        needed, instead of deepcopying the environment.
        """
        self._t = t

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, state: State, action: np.ndarray) -> tuple[State, float, bool]:
        """
        action: (N,) int array, values in {0,1,2,3}.
        1. Assert feasibility (raises ValueError if action is infeasible).
        2. Compute effective capacities.
        3. Solve TAP -> flows.
        4. Compute cost.
        5. Apply degradation (gamma_step) and renovation progress (wiener_step).
        6. Apply renovation completions: where h<=0, reset d=0, ell=0, r=0.
        Returns (next_state, cost, done).
        """
        cfg = self.config
        net = self.network

        # 1. Feasibility check
        self.assert_feasible(state, action)

        # 2. Apply action effects on state (for s_post)
        state_post = self.post_decision_state(state, action, check=False)

        # 3. Effective capacities based on state AFTER action
        caps = self._effective_capacities(state_post)

        # 4. TAP
        flows = self.tap_fn.solve(caps)

        # 5. Cost (use post-action capacities already computed above)
        total, c_travel, c_maint, c_risk = self._compute_cost(state, action, flows, caps)
        cost = total
        self.last_cost_breakdown = (c_travel, c_maint, c_risk)

        # 6 & 7. Stochastic transitions from post-decision state.
        # Noise comes from the episode-scoped, phase-keyed generator (armed in
        # begin_episode / reset); the same inverse-CDF transform is used as in
        # rollouts (see degradation.py). Lazily auto-begin if step() is somehow
        # reached without a reset (e.g. set_state forking).
        if self._noise_gen is None:
            self.begin_episode("default", self._auto_ep)
            self._auto_ep += 1
            self._episode_ready = False
        n = cfg.n_assets
        noise = (self._noise_gen.random(n), self._noise_gen.standard_normal(n))
        next_state, _ = self.stochastic_transition(state_post, self._t + 1, noise)
        self._t += 1
        next_state.t = self._t
        done = self._t >= cfg.T
        return next_state, cost, done

    # ------------------------------------------------------------------
    # Post-decision state
    # ------------------------------------------------------------------

    def post_decision_state(self, state: State, action: np.ndarray,
                            check: bool = True) -> State:
        """
        State after applying action but before stochastic transitions.
        Used for value function targets: v = cost + gamma * V(s_post).

        check=False skips assert_feasible; callers that pre-checked feasibility
        (action generators, step()) should pass check=False to avoid the redundant
        feasible_actions() call.
        """
        cfg = self.config
        if check:
            self.assert_feasible(state, action)

        renovate = action == self.ACTION_RENOVATE
        repair = action == self.ACTION_REPAIR
        restrict = action == self.ACTION_RESTRICT

        # np.where creates new arrays naturally — no upfront .copy() needed
        h = np.where(renovate, 1.0, state.h)
        d = np.where(repair, np.maximum(0.0, state.d - cfg.delta_repair), state.d)
        r = np.where(repair, 1.0, state.r)
        ell = np.where(restrict, 1.0, state.ell)

        s_post = State(d, h, ell, r, state.n_fail.copy())
        s_post.t = state.t              # same epoch as input state (pre-transition)
        return s_post

    # ------------------------------------------------------------------
    # Effective capacities
    # ------------------------------------------------------------------

    def _effective_capacities(self, state: State) -> np.ndarray:
        """
        Shape (E,). Each asset is a bidirectional link: renovation/restriction
        reduces the capacity of *all* directed edges of the asset (both
        directions). Assets under renovation get eta_ren * c_e, assets under
        restriction get eta_load * c_e, others unchanged. Renovation takes
        precedence over restriction.
        """
        cfg = self.config
        net = self.network
        nominal = net.nominal_capacities
        caps = nominal.copy()

        under_ren = state.h > 0      # (N,)
        under_load = state.ell > 0   # (N,)

        # Per-asset capacity factor (renovation precedence over restriction).
        factor = np.ones(net.n_assets)
        factor[under_load & ~under_ren] = cfg.eta_load
        factor[under_ren] = cfg.eta_ren

        # Apply to every directed edge of each asset's bidirectional link.
        # Assignment from nominal (not *=) is robust to any repeated edge index.
        asset_edges = net.asset_edges
        if asset_edges is None:                      # legacy single-edge fallback
            asset_edges = net.asset_indices[:, None]
        for col in range(asset_edges.shape[1]):
            e = asset_edges[:, col]
            caps[e] = nominal[e] * factor

        return caps

    # ------------------------------------------------------------------
    # Travel cost helper (for action generators)
    # ------------------------------------------------------------------

    def travel_cost(self, s_post: State) -> float:
        """
        Compute c_travel for a given post-decision state by calling TAP.
        Used by action generators to include traffic cost in Q(s, a).
        """
        cfg = self.config
        net = self.network
        caps = self._effective_capacities(s_post)
        flows = self.tap_fn.solve(caps)
        tt = net.free_flow_tt * (
            1.0 + net.bpr_beta * (flows / np.maximum(caps, 1e-9)) ** net.bpr_nu
        )
        extra_veh_hours = (float(np.sum(flows * tt)) - self._c_travel_baseline) / 60.0
        return cfg.traffic_cost_factor * cfg.vot * extra_veh_hours * cfg.dt * 365

    def stochastic_transition(
        self, s_post: State, t: int, noise: tuple[np.ndarray, np.ndarray]
    ) -> tuple[State, int]:
        """
        Apply stochastic transitions from a post-decision state.

        noise = (u, eps), each shape (N,): u are uniforms for the Gamma
        degradation increment, eps are standard normals for the Wiener
        renovation step. The caller supplies the noise so rollout agents can
        feed state/seed-derived common random numbers (CRN) and the real
        environment can feed its own rng draws.
        Returns (next_state, t+1).
        """
        cfg = self.config
        u, eps = noise

        d_new = gamma_step(s_post.d, cfg.alpha0, cfg.beta, s_post.ell, cfg.dt, u,
                           restrict_degrad_multiplier=cfg.restrict_degrad_multiplier)
        d_new = np.where(s_post.h > 0, s_post.d, d_new)
        h_new = wiener_step(s_post.h, cfg.mu_h, cfg.sigma_h, cfg.dt, eps)

        ell_new = s_post.ell.copy()
        r_new = s_post.r.copy()

        completed = (s_post.h > 0) & (h_new <= 0)
        d_new = np.where(completed, 0.0, d_new)
        ell_new = np.where(completed, 0.0, ell_new)
        r_new = np.where(completed, 0.0, r_new)
        h_new = np.where(h_new <= 0, 0.0, h_new)

        still_failed = (d_new >= cfg.d_fail) & (h_new == 0)
        n_fail_new = np.where(still_failed, s_post.n_fail + 1, 0.0)

        next_state = State(d_new, h_new, ell_new, r_new, n_fail_new)
        next_state.t = t
        return next_state, t + 1

    def immediate_cost(self, state: State, action: np.ndarray, s_post: State) -> float:
        """
        C(s, a) = c_maint + c_risk + c_travel.
        s_post must already be computed (post_decision_state(state, action)).
        Callers pass s_post explicitly to avoid recomputing it when it is
        already needed for value function prediction.
        """
        cfg = self.config
        c_maint = float(
            np.sum(cfg.c_ren * (action == self.ACTION_RENOVATE)) +
            np.sum(cfg.c_rep * (action == self.ACTION_REPAIR))
        )
        failed = (state.d >= cfg.d_fail) & (state.h == 0)
        c_risk = float(np.sum(
            cfg.risk_base * state.n_fail * cfg.asset_lengths_m * cfg.dt * failed
        ))
        return c_maint + c_risk + self.travel_cost(s_post)

    def immediate_cost_components(
        self, state: State, action: np.ndarray, s_post: State
    ) -> tuple[float, float, float]:
        """Same as immediate_cost but returns (c_maint, c_risk, c_travel) separately.

        Used by opt-in diagnostic logging (action generators) to decompose
        Q(s,a) = c_maint + c_risk + c_travel + V'(s_post). Not on the hot path.
        """
        cfg = self.config
        c_maint = float(
            np.sum(cfg.c_ren * (action == self.ACTION_RENOVATE)) +
            np.sum(cfg.c_rep * (action == self.ACTION_REPAIR))
        )
        failed = (state.d >= cfg.d_fail) & (state.h == 0)
        c_risk = float(np.sum(
            cfg.risk_base * state.n_fail * cfg.asset_lengths_m * cfg.dt * failed
        ))
        return c_maint, c_risk, self.travel_cost(s_post)

    # ------------------------------------------------------------------
    # Cost
    # ------------------------------------------------------------------

    def _compute_cost(self, state: State, action: np.ndarray,
                      flows: np.ndarray, caps: np.ndarray) -> tuple[float, float, float, float]:
        """Travel time cost + maintenance costs + escalating risk."""
        cfg = self.config
        net = self.network

        # Traffic cost using already-computed post-decision capacities
        tt = net.free_flow_tt * (
            1.0 + net.bpr_beta * (flows / np.maximum(caps, 1e-9)) ** net.bpr_nu
        )
        extra_veh_hours = (float(np.sum(flows * tt)) - self._c_travel_baseline) / 60.0
        c_travel = cfg.traffic_cost_factor * cfg.vot * extra_veh_hours * cfg.dt * 365

        # Construction costs (c_ren = 50_000 * length, c_rep = 25_000 * length)
        c_maint = float(
            np.sum(cfg.c_ren * (action == self.ACTION_RENOVATE)) +
            np.sum(cfg.c_rep * (action == self.ACTION_REPAIR))
        )

        # Escalating risk for failed assets not yet under renovation
        failed = (state.d >= cfg.d_fail) & (state.h == 0)
        c_risk = float(np.sum(
            cfg.risk_base * state.n_fail * cfg.asset_lengths_m * cfg.dt * failed
        ))

        return c_travel + c_maint + c_risk, c_travel, c_maint, c_risk

    # ------------------------------------------------------------------
    # Feasibility
    # ------------------------------------------------------------------

    def feasible_actions(self, state: State) -> np.ndarray:
        """
        Shape (N, 4) bool. Column i = feasibility of action i.
        """
        n = self.config.n_assets
        mask = np.ones((n, 4), dtype=bool)

        under_ren = state.h > 0

        # repair: not under renovation AND repair not yet used AND repair allowed
        mask[:, self.ACTION_REPAIR] = (~under_ren) & (state.r == 0) & self.config.allow_repair

        # renovate: not already renovating
        mask[:, self.ACTION_RENOVATE] = ~under_ren

        # restrict: not under renovation AND no restriction active AND restrict allowed
        mask[:, self.ACTION_RESTRICT] = (~under_ren) & (state.ell == 0) & self.config.allow_restrict

        return mask

    def assert_feasible(self, state: State, action: np.ndarray) -> None:
        """Raise ValueError if any action is infeasible for the given state."""
        feas = self.feasible_actions(state)
        valid = feas[np.arange(self.config.n_assets), action]
        if valid.all():
            return
        violations = np.where(~valid)[0].tolist()
        if violations:
            details = ', '.join(
                f"asset {i}: action={action[i]} not in {np.where(feas[i])[0].tolist()}"
                for i in violations
            )
            raise ValueError(f"Infeasible actions at assets: {details}")

    @property
    def t(self) -> int:
        return self._t
