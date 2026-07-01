"""Clairvoyant (perfect-information) baseline.

A *clairvoyant* agent is allowed to see the future. For a given evaluation
episode it reconstructs the **exact** realized noise path (degradation
increments and renovation durations) and then solves a deterministic
finite-horizon control problem against that single realization — i.e. the
"wait-and-see" / perfect-information solution from stochastic programming. The
mean clairvoyant cost over the evaluation seeds is a **lower bound** on the
expected cost of *any* non-anticipative policy (an information-relaxation bound;
Brown, Smith & Sun 2010). The gap between a learned policy and the clairvoyant
is the price of not knowing the future (the EVPI).

Why this is well-defined here
-----------------------------
Environment randomness is a pure function of ``(phase, seed, episode_idx)``
(see ``.claude/rules/randomness-and-crn.md``): ``env.begin_episode`` arms a
``keyed_philox("transition", phase, seed, episode_idx)`` generator and
``env.step`` pulls ``u = g.random(N)`` then ``eps = g.standard_normal(N)`` every
epoch. ``replay_episode_noise`` reproduces that *exact* draw sequence, so the
clairvoyant solves and acts against the same future the evaluation harness will
realize — fully reproducible, no clairvoyance leak into other agents.

Solver (per the design decision: "metaheuristic + per-asset DP")
----------------------------------------------------------------
The only cross-asset coupling in this MDP is the network travel cost (TAP);
degradation, renovation, maintenance cost, escalating risk cost and feasibility
are all per-asset. So:

1. **Per-asset DP** (`perasset_dp`): for each asset, an exact (up to a condition
   grid) finite-horizon DP over all four actions against the fixed noise, taking
   an optional per-epoch travel penalty (3 capacity classes × H). Used both as a
   warm start (zero travel penalty) and as the exact subproblem below.
2. **Block-coordinate descent** (`_bcd`): the coordination metaheuristic. Each
   sweep, for each asset, holds the others fixed, builds the asset's *exact* 3-way
   travel lookup (`_build_travel_lut`), and replaces its plan with the per-asset DP
   best response if that lowers the exact joint cost. Because the lookup is the
   asset's true travel given the others, every accepted move lowers the true joint
   objective → monotone convergence to a coordinate-optimal point. Warm-started by
   the best of {do-nothing, DP, optional heuristic}; the returned cost is clamped
   to ``<=`` that best start, so the lower-bound-on-policies property is robust.

Bounds (`_lower_bounds`): alongside the solution (an upper bound ``UB`` on the
clairvoyant optimum) we report a lower bound: **LB0** = travel-free per-asset
decomposition (rigorous floor, since travel ≥ 0), optionally tightened to
**LB-cache** by adding the cached single-asset travels as a separable
underestimating penalty — guarded by a super-additivity validity check on the
solution's own (cached) configs, falling back to LB0 on any violation. The gap
``(UB−LB)/UB`` certifies how close the clairvoyant solution is to optimal.

The reported clairvoyant cost is always taken from the exact simulator, so it is
faithful to ``env.step`` regardless of any grid approximation in the DP.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.special import gammaincinv

from agents.base import Agent
from env.mdp import State, InfraEnv
from env.noise import keyed_philox


# ---------------------------------------------------------------------------
# Exact noise replay
# ---------------------------------------------------------------------------

def replay_episode_noise(
    env: InfraEnv, phase: str, episode_idx: int, base_seed: int, horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct the exact ``(u_bank, eps_bank, d_init)`` for one episode.

    Mirrors ``env.begin_episode`` + ``env.step``'s per-epoch draw order EXACTLY:
    a single ``keyed_philox("transition", phase, seed, episode_idx)`` generator
    yields, per epoch ``t``, ``u = g.random(N)`` followed by
    ``eps = g.standard_normal(N)``. Initial conditions follow ``env.reset``:
    fixed ``cfg.d_init`` when present, else the independent ``"reset"`` stream's
    ``uniform(0, 0.5, N)``.

    Returns ``(u_bank (H, N), eps_bank (H, N), d_init (N,))``.
    """
    cfg = env.config
    n = cfg.n_assets
    g = keyed_philox("transition", phase, int(base_seed), int(episode_idx))
    u_bank = np.empty((horizon, n))
    eps_bank = np.empty((horizon, n))
    for t in range(horizon):
        u_bank[t] = g.random(n)
        eps_bank[t] = g.standard_normal(n)
    if cfg.d_init is not None:
        d_init = np.asarray(cfg.d_init, dtype=float).copy()
    else:
        gr = keyed_philox("reset", phase, int(base_seed), int(episode_idx))
        d_init = gr.uniform(0.0, 0.5, size=n)
    return u_bank, eps_bank, d_init


# ---------------------------------------------------------------------------
# Exact deterministic evaluator (matches env.step on the replayed noise)
# ---------------------------------------------------------------------------

def simulate_intended(
    env: InfraEnv,
    u_bank: np.ndarray,
    eps_bank: np.ndarray,
    d_init: np.ndarray,
    intended: np.ndarray,
    horizon: int,
) -> tuple[float, np.ndarray]:
    """Faithfully roll an *intended* plan forward against replayed noise.

    For each epoch the intended joint action is projected to the feasible set of
    the realized state (infeasible entries → ACTION_NONE), exactly as the
    evaluation harness's states would require. Cost and dynamics reuse the env's
    own pure methods, so the total equals what ``env.step`` accumulates:

        cost_t = immediate_cost(state, a, post)   # c_maint + c_risk + c_travel
        total  = Σ_t γ^t · cost_t

    Returns ``(total_discounted_cost, executed_plan (H, N) int)``. The executed
    plan is feasible by construction, so replaying it through ``env.step`` (same
    noise) never raises.
    """
    cfg = env.config
    n = cfg.n_assets
    gamma = cfg.gamma
    state = State(d_init.copy(), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n))
    state.t = 0
    executed = np.zeros((horizon, n), dtype=int)
    total = 0.0
    disc = 1.0
    idx = np.arange(n)
    for t in range(horizon):
        feas = env.feasible_actions(state)            # (N, 4) bool
        a = np.asarray(intended[t], dtype=int).copy()
        bad = ~feas[idx, a]
        if bad.any():
            a[bad] = InfraEnv.ACTION_NONE
        executed[t] = a
        s_post = env.post_decision_state(state, a, check=False)
        total += disc * env.immediate_cost(state, a, s_post)
        disc *= gamma
        state, _ = env.stochastic_transition(s_post, t + 1, (u_bank[t], eps_bank[t]))
        state.t = t + 1
    return total, executed


# ---------------------------------------------------------------------------
# Per-asset renovation-timing DP (warm start)
# ---------------------------------------------------------------------------

def _asset_params(cfg, asset: int) -> dict:
    """Scalar per-asset parameters pulled out of EnvConfig (so the per-asset DP /
    stepper never touch the joint env)."""
    bc = lambda v: float(np.broadcast_to(np.asarray(v, dtype=float), cfg.n_assets)[asset])
    return {
        'alpha0': float(np.asarray(cfg.alpha0)[asset]),
        'beta': float(np.asarray(cfg.beta)[asset]),
        'mu_h': bc(cfg.mu_h),
        'sigma_h': bc(cfg.sigma_h),
        'delta': float(cfg.delta_repair),
        'dt': float(cfg.dt),
        'd_fail': float(cfg.d_fail),
        'eta_ren': float(cfg.eta_ren),
        'eta_load': float(cfg.eta_load),
        'f': float(cfg.restrict_degrad_multiplier),
        'c_ren': float(np.asarray(cfg.c_ren)[asset]),
        'c_rep': float(np.asarray(cfg.c_rep)[asset]),
        'L': float(np.asarray(cfg.asset_lengths_m)[asset]),
        'risk_base': float(cfg.risk_base),
        'allow_repair': bool(cfg.allow_repair),
        'allow_restrict': bool(cfg.allow_restrict),
    }


# Capacity classes (index into a (3, H) travel-penalty array).
_CLS_NORMAL, _CLS_RESTRICT, _CLS_RENOVATE = 0, 1, 2


def _asset_step(p: dict, state, a_int: int, u_t: float, eps_t: float):
    """One epoch of EXACT single-asset dynamics, mirroring InfraEnv.step's
    per-asset arithmetic (feasibility projection, post-decision, cost, Gamma/Wiener
    transition). ``state`` and the returned next state are ``(d, h, ell, r, nf)``
    tuples. Returns ``(executed_action, capacity_class, pa_cost, next_state)`` where
    ``pa_cost`` is the per-asset cost (c_maint + c_risk; travel is a joint quantity
    handled separately). Used for the DP's exact-deploy reconstruction, for tests,
    and (later) by the metaheuristic's coordination."""
    d, h, ell, r, nf = state
    under_ren = h > 0.0
    a = int(a_int)
    if a == InfraEnv.ACTION_RENOVATE:
        if under_ren:
            a = InfraEnv.ACTION_NONE
    elif a == InfraEnv.ACTION_REPAIR:
        if under_ren or r != 0 or not p['allow_repair']:
            a = InfraEnv.ACTION_NONE
    elif a == InfraEnv.ACTION_RESTRICT:
        if under_ren or ell != 0 or not p['allow_restrict']:
            a = InfraEnv.ACTION_NONE

    # Post-decision
    h_post = 1.0 if a == InfraEnv.ACTION_RENOVATE else h
    if a == InfraEnv.ACTION_REPAIR:
        d_post = max(0.0, d - p['delta']); r_post = 1.0
    else:
        d_post = d; r_post = r
    ell_post = 1.0 if a == InfraEnv.ACTION_RESTRICT else ell

    cls = (_CLS_RENOVATE if h_post > 0.0
           else _CLS_RESTRICT if ell_post > 0.0 else _CLS_NORMAL)

    # Cost (pre-decision state, as in InfraEnv._compute_cost)
    c_maint = (p['c_ren'] if a == InfraEnv.ACTION_RENOVATE
               else p['c_rep'] if a == InfraEnv.ACTION_REPAIR else 0.0)
    failed = (d >= p['d_fail']) and (h == 0.0)
    c_risk = p['risk_base'] * nf * p['L'] * p['dt'] if failed else 0.0
    pa_cost = c_maint + c_risk

    # Stochastic transition (Gamma degradation + Wiener renovation), env-faithful.
    if h_post > 0.0:
        d_new = d_post                                 # condition frozen under renovation
    else:
        alpha = (1.0 - (1.0 - p['f']) * ell_post) * p['alpha0']
        ai = alpha * p['dt']
        if ai > 0.0:
            d_new = min(1.0, d_post + gammaincinv(max(ai, 1e-12), u_t) / p['beta'])
        else:
            d_new = d_post
    if h_post > 0.0:
        h_new = h_post - p['mu_h'] * p['dt'] + p['sigma_h'] * (p['dt'] ** 0.5) * eps_t
    else:
        h_new = h_post
    completed = (h_post > 0.0) and (h_new <= 0.0)
    if completed:
        d_new = 0.0; ell_new = 0.0; r_new = 0.0
    else:
        ell_new = ell_post; r_new = r_post
    if h_new <= 0.0:
        h_new = 0.0
    still_failed = (d_new >= p['d_fail']) and (h_new == 0.0)
    nf_new = (nf + 1.0) if still_failed else 0.0
    return a, cls, pa_cost, (d_new, h_new, ell_new, r_new, nf_new)


def _renovation_completion(p: dict, eps_col: np.ndarray, horizon: int) -> np.ndarray:
    """``c[t]`` = epoch a renovation started at ``t`` completes (asset free again at
    ``c[t]``), deterministic given the fixed Wiener noise. ``horizon`` if it never
    completes within the horizon."""
    c = np.full(horizon, horizon, dtype=int)
    mu, sig, dt = p['mu_h'], p['sigma_h'], p['dt']
    sdt = dt ** 0.5
    for s in range(horizon):
        h = 1.0
        for t in range(s, horizon):
            h = h - mu * dt + sig * sdt * eps_col[t]
            if h <= 0.0:
                c[s] = t + 1
                break
    return c


def asset_plan_cost(
    p: dict, d0: float, intended, u_col: np.ndarray, eps_col: np.ndarray,
    gamma: float, travel_penalty: np.ndarray | None = None,
) -> float:
    """Exact discounted per-asset cost of an intended action column against the
    fixed noise: ``Σ_t γ^t (c_maint + c_risk + travel_penalty[class_t, t])``. This
    is precisely the objective ``perasset_dp`` minimises (with the same
    ``travel_penalty`` convention), so it is the reference for testing the DP and
    the per-asset evaluator for the coordination step."""
    H = len(intended)
    tp = np.zeros((3, H)) if travel_penalty is None else np.asarray(travel_penalty, float)
    state = (float(d0), 0.0, 0.0, 0.0, 0.0)
    total = 0.0
    disc = 1.0
    for t in range(H):
        _, cls, pa, state = _asset_step(p, state, int(intended[t]), u_col[t], eps_col[t])
        total += disc * (pa + tp[cls, t])
        disc *= gamma
    return total


def perasset_dp(
    p: dict, d0: float, u_col: np.ndarray, eps_col: np.ndarray,
    horizon: int, gamma: float, travel_penalty: np.ndarray | None = None,
    n_grid: int = 128, nf_max: int = 24, return_value: bool = False,
):
    """Exact (up to a condition grid) per-asset DP over all four actions.

    Minimises the discounted per-asset cost ``Σ_t γ^t (c_maint + c_risk +
    travel_penalty[class_t, t])`` against the asset's fixed noise. ``travel_penalty``
    (shape ``(3, H)``, classes {normal, restricted, renovating}) is the per-epoch
    cost of the asset's capacity state; ``None`` ⇒ zeros (standalone warm start,
    ignores the network). The same array is the exact per-asset subproblem objective
    when supplied with the 3-way TAP lookup (the metaheuristic's coordination).

    State ``(d_bin, ℓ, r, nf)`` on a ``n_grid`` condition grid; renovation is a
    deterministic macro ``t → c(t)`` so ``h`` leaves the state; ℓ/r are within-cycle
    latches; ``nf`` (failed counter) is capped at ``nf_max``. Increments are
    condition-independent, so "degrade one epoch" is a fixed grid shift. Returns the
    intended action sequence ``(H,)`` reconstructed by deploying the DP policy on the
    EXACT (un-snapped) trajectory. Grid snapping affects only plan quality, never the
    reported cost (taken from the exact simulator) nor the lower-bound guarantee.
    """
    H = horizon
    NF1 = nf_max + 1
    G = n_grid
    d_fail = p['d_fail']
    if travel_penalty is None:
        tp = np.zeros((3, H))
    else:
        tp = np.asarray(travel_penalty, dtype=float)

    d_grid = np.linspace(0.0, 1.0, G)
    failed = d_grid >= d_fail                                   # (G,) bool

    def snap(x):
        return np.clip(np.rint(x * (G - 1)).astype(int), 0, G - 1)

    # Condition-independent per-epoch increments (ell=0 and ell=1), as grid shifts.
    u = np.clip(u_col, 0.0, 1.0)
    a0 = max(p['alpha0'] * p['dt'], 1e-12)
    incr0 = gammaincinv(a0, u) / p['beta']                      # (H,)
    alpha1 = p['f'] * p['alpha0']
    if alpha1 * p['dt'] > 0.0:
        incr1 = gammaincinv(max(alpha1 * p['dt'], 1e-12), u) / p['beta']
    else:
        incr1 = np.zeros(H)
    shift0 = snap(np.minimum(1.0, d_grid[None, :] + incr0[:, None]))   # (H, G)
    shift1 = snap(np.minimum(1.0, d_grid[None, :] + incr1[:, None]))   # (H, G)
    rep_idx = snap(np.maximum(0.0, d_grid - p['delta']))              # (G,)

    c_arr = _renovation_completion(p, eps_col, H)
    gpow = gamma ** np.arange(H + 1)
    P2 = gpow[:H] * tp[_CLS_RENOVATE]
    cumP = np.concatenate([[0.0], np.cumsum(P2)])               # (H+1,)

    risk_col = np.where(failed, p['risk_base'] * p['L'] * p['dt'], 0.0)  # (G,) per nf=1
    nf_vec = np.arange(NF1)
    capped = np.minimum(nf_vec + 1, nf_max)                     # (NF1,)
    risk_pre = risk_col[:, None] * nf_vec[None, :]              # (G, NF1)

    # V[t]: cost-to-go (absolute discount γ^t) from a free state at epoch t.
    V = np.zeros((H + 1, G, 2, 2, NF1))
    policy = np.zeros((H, G, 2, 2, NF1), dtype=np.int8)

    def gather(Vn, nb, ell_n, r_n):
        """Continuation V[t+1] for next-bin ``nb`` (G,), fixed next ℓ/r, with the
        nf transition (nf+1 capped if next bin failed, else 0)."""
        nfn = np.where(failed[nb][:, None], capped[None, :], 0)   # (G, NF1)
        return Vn[nb[:, None], ell_n, r_n, nfn]                   # (G, NF1)

    for t in range(H - 1, -1, -1):
        gt = gpow[t]
        Vn = V[t + 1]
        c = c_arr[t]
        lock = cumP[min(c, H)] - cumP[min(t + 1, H)]
        Vc = V[c, 0, 0, 0, 0] if c < H else 0.0
        ren_const = gt * (p['c_ren'] + tp[_CLS_RENOVATE, t]) + lock + Vc

        for ell in (0, 1):
            shift_e = shift1[t] if ell == 1 else shift0[t]        # increment under current ℓ
            cls0 = _CLS_RESTRICT if ell == 1 else _CLS_NORMAL
            for r in (0, 1):
                # NONE
                val = gt * (risk_pre + tp[cls0, t]) + gather(Vn, shift_e, ell, r)
                best_val = val
                best_act = np.zeros((G, NF1), dtype=np.int8)     # NONE
                # RENOVATE (always feasible when free)
                val = ren_const + gt * risk_pre
                better = val < best_val
                best_val = np.where(better, val, best_val)
                best_act = np.where(better, InfraEnv.ACTION_RENOVATE, best_act)
                # REPAIR
                if r == 0 and p['allow_repair']:
                    nb = shift_e[rep_idx]                         # repair then degrade
                    val = gt * (p['c_rep'] + risk_pre + tp[cls0, t]) + gather(Vn, nb, ell, 1)
                    better = val < best_val
                    best_val = np.where(better, val, best_val)
                    best_act = np.where(better, InfraEnv.ACTION_REPAIR, best_act)
                # RESTRICT (latches ℓ→1; degrades restricted from this epoch)
                if ell == 0 and p['allow_restrict']:
                    val = gt * (risk_pre + tp[_CLS_RESTRICT, t]) + gather(Vn, shift1[t], 1, r)
                    better = val < best_val
                    best_val = np.where(better, val, best_val)
                    best_act = np.where(better, InfraEnv.ACTION_RESTRICT, best_act)
                V[t, :, ell, r, :] = best_val
                policy[t, :, ell, r, :] = best_act

    # Reconstruct by deploying the policy on the EXACT (un-snapped) trajectory.
    intended = np.zeros(H, dtype=int)
    state = (float(d0), 0.0, 0.0, 0.0, 0.0)
    for t in range(H):
        d, h, ell, r, nf = state
        if h <= 0.0:
            b = int(np.clip(round(d * (G - 1)), 0, G - 1))
            a = int(policy[t, b, int(ell), int(r), min(int(nf), nf_max)])
        else:
            a = InfraEnv.ACTION_NONE
        a_exec, _, _, state = _asset_step(p, state, a, u_col[t], eps_col[t])
        intended[t] = a_exec
    if return_value:
        b0 = int(np.clip(round(float(d0) * (G - 1)), 0, G - 1))
        return intended, float(V[0, b0, 0, 0, 0])     # grid-optimal cost-to-go from the start
    return intended


def _dp_warm_start(env, u_bank, eps_bank, d_init, horizon,
                   n_grid: int = 128, nf_max: int = 24) -> np.ndarray:
    """Per-asset DP for every asset (no travel penalty) → joint intended plan (H, N)."""
    n = env.config.n_assets
    intended = np.zeros((horizon, n), dtype=int)
    for i in range(n):
        p = _asset_params(env.config, i)
        intended[:, i] = perasset_dp(
            p, float(d_init[i]), u_bank[:, i], eps_bank[:, i],
            horizon, env.config.gamma, travel_penalty=None,
            n_grid=n_grid, nf_max=nf_max,
        )
    return intended


# ---------------------------------------------------------------------------
# Multi-start local search over the joint plan (exact objective)
# ---------------------------------------------------------------------------

def _heuristic_intended(env, agent, u_bank, eps_bank, d_init, horizon) -> np.ndarray:
    """Roll a non-anticipative heuristic forward against the replayed noise to
    obtain its intended action plan (used as a local-search start)."""
    cfg = env.config
    n = cfg.n_assets
    state = State(d_init.copy(), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n))
    state.t = 0
    intended = np.zeros((horizon, n), dtype=int)
    for t in range(horizon):
        a = np.asarray(agent.act(state), dtype=int)
        feas = env.feasible_actions(state)
        a = np.where(feas[np.arange(n), a], a, InfraEnv.ACTION_NONE)
        intended[t] = a
        s_post = env.post_decision_state(state, a, check=False)
        state, _ = env.stochastic_transition(s_post, t + 1, (u_bank[t], eps_bank[t]))
        state.t = t + 1
    return intended


# ---------------------------------------------------------------------------
# Joint travel + per-asset capacity classes (the only cross-asset coupling)
# ---------------------------------------------------------------------------

def _travel_from_caps(env, caps: np.ndarray) -> float:
    """C_travel for a capacity vector, identical to InfraEnv.travel_cost's body
    (TAP solve is served from the env's bounded cache, so repeated configs are
    free)."""
    net = env.network
    cfg = env.config
    flows = env.tap_fn.solve(caps)
    tt = net.free_flow_tt * (
        1.0 + net.bpr_beta * (flows / np.maximum(caps, 1e-9)) ** net.bpr_nu
    )
    extra = (float(np.sum(flows * tt)) - env._c_travel_baseline) / 60.0
    return cfg.traffic_cost_factor * cfg.vot * extra * cfg.dt * 365


def _asset_classes(p, d0, intended, u_col, eps_col, horizon) -> np.ndarray:
    """Per-epoch capacity class (0 normal / 1 restricted / 2 renovating) realized
    by one asset's intended plan."""
    cls = np.zeros(horizon, dtype=int)
    state = (float(d0), 0.0, 0.0, 0.0, 0.0)
    for t in range(horizon):
        _, c, _, state = _asset_step(p, state, int(intended[t]), u_col[t], eps_col[t])
        cls[t] = c
    return cls


def _build_travel_lut(env, classes, i, edges, nominal, factors) -> np.ndarray:
    """``lut[k, t]`` = travel with asset ``i`` in class ``k`` and every other asset
    held at its current class — the EXACT travel asset ``i`` sees given the others.
    ``3·H`` TAP solves (cached). ``edges`` is (N, K): each asset is a bidirectional
    link, so its K directed edges all change capacity together. Assets map to
    distinct edges."""
    H, N = classes.shape
    lut = np.empty((3, H))
    mask = np.ones(N, dtype=bool); mask[i] = False
    other_edges = edges[mask]              # (N-1, K)
    ei = edges[i]                          # (K,)
    for t in range(H):
        caps_others = nominal.copy()
        caps_others[other_edges] = nominal[other_edges] * factors[classes[t, mask]][:, None]
        for k in range(3):
            caps = caps_others.copy()
            caps[ei] = nominal[ei] * factors[k]
            lut[k, t] = _travel_from_caps(env, caps)
    return lut


def _joint_travel(env, classes, edges, nominal, factors) -> np.ndarray:
    """Per-epoch joint travel (H,) of a full solution's capacity configuration.
    ``edges`` is (N, K) — both directions of each asset's link change together."""
    H = classes.shape[0]
    out = np.empty(H)
    for t in range(H):
        caps = nominal.copy()
        caps[edges] = nominal[edges] * factors[classes[t]][:, None]
        out[t] = _travel_from_caps(env, caps)
    return out


def _single_asset_travel(env, i, edges, nominal, factors) -> np.ndarray:
    """``τ_i = [0, travel(i restricted), travel(i renovating)]`` with all other
    assets at nominal — epoch-independent (constant demand). ``edges[i]`` is the
    (K,) directed edges of asset ``i``'s bidirectional link."""
    tau = np.zeros(3)
    ei = edges[i]                          # (K,)
    for k in (1, 2):
        caps = nominal.copy()
        caps[ei] = nominal[ei] * factors[k]
        tau[k] = _travel_from_caps(env, caps)
    return tau


# ---------------------------------------------------------------------------
# Block-coordinate descent (the coordination metaheuristic)
# ---------------------------------------------------------------------------

def _bcd(env, u, eps, d_init, horizon, intended_start, params,
        edges, nominal, factors, gamma, max_sweeps, deadline,
        n_grid, nf_max) -> tuple[np.ndarray, int, bool]:
    """Block-coordinate descent: each sweep, for each asset, hold the others fixed,
    build its exact 3-way travel lookup, and replace its plan with the per-asset DP
    best response if that strictly lowers the EXACT joint cost. Because the lookup
    is the asset's true travel given the others, every accepted move strictly lowers
    the true joint objective, so BCD is monotone and provably terminates at a
    coordinate-optimal point on its own.

    ``max_sweeps=None`` runs until that local optimum (the intended mode); a finite
    cap is only a runtime guard. ``deadline`` (wall-clock) is the hard stop. Returns
    ``(intended (H,N), n_sweeps, converged)`` where ``converged`` means a full sweep
    accepted nothing (a true local optimum) rather than hitting the cap/deadline."""
    N = env.config.n_assets
    intended = np.asarray(intended_start, dtype=int).copy()
    classes = np.stack(
        [_asset_classes(params[i], d_init[i], intended[:, i], u[:, i], eps[:, i], horizon)
         for i in range(N)], axis=1)                                # (H, N)

    sweep = 0
    while max_sweeps is None or sweep < max_sweeps:
        sweep += 1
        improved = False
        for i in range(N):
            if deadline is not None and time.monotonic() > deadline:
                return intended, sweep, False          # wall-clock stop, not converged
            lut = _build_travel_lut(env, classes, i, edges, nominal, factors)
            new_i = perasset_dp(params[i], float(d_init[i]), u[:, i], eps[:, i],
                                horizon, gamma, travel_penalty=lut,
                                n_grid=n_grid, nf_max=nf_max)
            # Accept on the EXACT asset-i objective under the lookup (= the part of
            # the joint cost that changes when only asset i moves).
            new_c = asset_plan_cost(params[i], float(d_init[i]), new_i,
                                    u[:, i], eps[:, i], gamma, travel_penalty=lut)
            cur_c = asset_plan_cost(params[i], float(d_init[i]), intended[:, i],
                                    u[:, i], eps[:, i], gamma, travel_penalty=lut)
            if new_c < cur_c - 1e-3 * max(1.0, abs(cur_c)):
                intended[:, i] = new_i
                classes[:, i] = _asset_classes(params[i], d_init[i], new_i,
                                               u[:, i], eps[:, i], horizon)
                improved = True
        if not improved:
            return intended, sweep, True               # local optimum reached
    return intended, sweep, False                      # finite cap exhausted


# ---------------------------------------------------------------------------
# Lower bounds (LB0 + cache-served tightener)
# ---------------------------------------------------------------------------

def _dp_value_sum(env, u, eps, d_init, horizon, params, n_grid, nf_max,
                  travel_penalties=None) -> float:
    """Σ_i (per-asset DP optimal value). ``travel_penalties`` is None (LB0,
    travel-free) or a list of (3,H) per-asset penalties."""
    total = 0.0
    for i in range(len(params)):
        tp = None if travel_penalties is None else travel_penalties[i]
        _, v = perasset_dp(params[i], float(d_init[i]), u[:, i], eps[:, i],
                           horizon, env.config.gamma, travel_penalty=tp,
                           n_grid=n_grid, nf_max=nf_max, return_value=True)
        total += v
    return total


def _lower_bounds(env, u, eps, d_init, horizon, params, classes_solution,
                  edges, nominal, factors, n_grid, nf_max, ub) -> dict:
    """LB0 (travel-free decomposition) + LB-cache (LB0 tightened with the cached
    single-asset travels τ_i as a separable underestimating travel penalty).

    LB0 is a rigorous floor (it drops the non-negative travel term). LB-cache adds
    ``tpen_i[k,t] = γ^t·τ_i(k)``; this is only a valid underestimate if congestion
    is super-additive, so we *verify* ``Σ_i τ_i(class_i) ≤ joint_travel`` on the
    solution's own (already-cached) per-epoch configs and fall back to LB0 on any
    violation. Both are grid-based estimates; reported LB is clamped to ≤ UB."""
    N = env.config.n_assets
    gamma = env.config.gamma
    gpow = gamma ** np.arange(horizon)

    lb0 = _dp_value_sum(env, u, eps, d_init, horizon, params, n_grid, nf_max)

    tau = np.stack([_single_asset_travel(env, i, edges, nominal, factors)
                    for i in range(N)])                            # (N, 3)
    joint = _joint_travel(env, classes_solution, edges, nominal, factors)   # (H,)
    sum_tau = tau[np.arange(N)[None, :], classes_solution].sum(axis=1)      # (H,)
    valid = bool(np.all(sum_tau <= joint + 1e-6 * np.maximum(1.0, np.abs(joint))))

    lb_cache = None
    if valid:
        tps = [np.outer(tau[i], gpow) for i in range(N)]           # tpen_i[k,t]=τ_i(k)·γ^t
        lb_cache = _dp_value_sum(env, u, eps, d_init, horizon, params,
                                 n_grid, nf_max, travel_penalties=tps)

    lb = lb_cache if (valid and lb_cache is not None) else lb0
    lb = min(lb, ub)                                               # never report LB above UB
    return {'lb0': float(lb0),
            'lb_cache': (float(lb_cache) if lb_cache is not None else None),
            'lb_cache_valid': valid,
            'lb': float(lb), 'ub': float(ub),
            'gap': float((ub - lb) / max(abs(ub), 1.0))}


# ---------------------------------------------------------------------------
# Top-level solve
# ---------------------------------------------------------------------------

def solve_clairvoyant(
    env: InfraEnv,
    u_bank: np.ndarray,
    eps_bank: np.ndarray,
    d_init: np.ndarray,
    horizon: int,
    *,
    use_dp: bool = True,
    warm_start_agent=None,
    max_sweeps: int | None = None,
    time_budget_s: float | None = None,
    n_grid: int = 128,
    nf_max: int = 24,
    compute_bounds: bool = True,
) -> tuple[float, np.ndarray, dict]:
    """Solve one perfect-information episode. Returns ``(cost, executed_plan, bounds)``.

    Picks the best of {do-nothing, per-asset DP, optional heuristic} warm starts,
    refines it with block-coordinate descent (per-asset DP best responses coupled
    through the exact travel lookup), and returns the exact simulated cost (UB).
    ``bounds`` carries LB0 / LB-cache / the reported LB / the optimality gap, plus
    ``bcd_sweeps`` / ``bcd_converged`` (empty if ``compute_bounds`` is False). The
    returned cost is always ≤ the best warm start, so the lower-bound-on-policies
    property holds.

    ``max_sweeps=None`` (default) runs BCD to its **local optimum** (a coordinate-
    optimal fixed point — BCD is monotone and terminates on its own); a finite cap
    is only a runtime guard. ``time_budget_s`` is the wall-clock hard stop. Because
    BCD converges to a deterministic fixed point well before any sane budget, the
    result is reproducible; the budget only matters as a safety net.
    """
    cfg = env.config
    n = cfg.n_assets
    edges = env.network.asset_edges
    if edges is None:                              # legacy single-edge fallback
        edges = env.network.asset_indices[:, None]
    nominal = env.network.nominal_capacities
    factors = np.array([1.0, cfg.eta_load, cfg.eta_ren])
    params = [_asset_params(cfg, i) for i in range(n)]
    deadline = (time.monotonic() + time_budget_s) if time_budget_s else None

    starts: list[np.ndarray] = [np.zeros((horizon, n), dtype=int)]   # do-nothing
    if use_dp:
        starts.append(_dp_warm_start(env, u_bank, eps_bank, d_init, horizon, n_grid, nf_max))
    if warm_start_agent is not None:
        starts.append(
            _heuristic_intended(env, warm_start_agent, u_bank, eps_bank, d_init, horizon)
        )
    start_costs = [simulate_intended(env, u_bank, eps_bank, d_init, s, horizon)[0]
                   for s in starts]
    bi = int(np.argmin(start_costs))
    best_start, best_start_cost = starts[bi], start_costs[bi]

    intended, n_sweeps, converged = _bcd(
        env, u_bank, eps_bank, d_init, horizon, best_start, params,
        edges, nominal, factors, cfg.gamma, max_sweeps, deadline, n_grid, nf_max)
    ub, executed = simulate_intended(env, u_bank, eps_bank, d_init, intended, horizon)
    if best_start_cost < ub:                       # safety net: never worse than the start
        ub = best_start_cost
        _, executed = simulate_intended(env, u_bank, eps_bank, d_init, best_start, horizon)
        intended = best_start

    bounds: dict = {}
    if compute_bounds:
        classes_sol = np.stack(
            [_asset_classes(params[i], d_init[i], intended[:, i], u_bank[:, i],
                            eps_bank[:, i], horizon) for i in range(n)], axis=1)
        bounds = _lower_bounds(env, u_bank, eps_bank, d_init, horizon, params,
                               classes_sol, edges, nominal, factors,
                               n_grid, nf_max, ub)
        bounds['bcd_sweeps'] = int(n_sweeps)
        bounds['bcd_converged'] = bool(converged)
    return ub, executed, bounds


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ClairvoyantAgent(Agent):
    """Perfect-information baseline. Solves a per-seed deterministic plan in
    ``on_episode_start`` (replaying that seed's exact future) and replays it via
    ``act``. Non-learning; produces a lower-bound reference for any policy.

    Holds no env reference (so it pickles cleanly for parallel evaluation); the
    env is supplied per episode by the evaluation harness. ``warm_start_spec`` is
    an optional ``{agent_type, extra}`` heuristic spec used as a local-search
    start (built lazily on the worker's env).
    """

    def __init__(
        self,
        *,
        use_dp: bool = True,
        warm_start_spec: dict | None = None,
        max_sweeps: int | None = None,
        time_budget_s: float | None = None,
        n_grid: int = 128,
        nf_max: int = 24,
        seed: int = 0,
    ):
        self.use_dp = bool(use_dp)
        self.warm_start_spec = warm_start_spec
        self.max_sweeps = (int(max_sweeps) if max_sweeps is not None else None)
        self.time_budget_s = time_budget_s
        self.n_grid = int(n_grid)
        self.nf_max = int(nf_max)
        self.seed = int(seed)
        self._executed: np.ndarray | None = None
        self.last_solution_cost: float | None = None
        self.last_bounds: dict = {}
        self.step_metrics: dict = {}

    def on_episode_start(self, phase, episode_idx, base_seed, env, horizon):
        """Replay this episode's exact future and solve the perfect-information plan."""
        u_bank, eps_bank, d_init = replay_episode_noise(
            env, phase, int(episode_idx), int(base_seed), int(horizon)
        )
        warm = None
        if self.warm_start_spec is not None:
            from experiments.configs import AgentConfig, _build_agent
            warm = _build_agent(AgentConfig.from_dict(self.warm_start_spec), env, self.seed)
        cost, executed, bounds = solve_clairvoyant(
            env, u_bank, eps_bank, d_init, int(horizon),
            use_dp=self.use_dp, warm_start_agent=warm, max_sweeps=self.max_sweeps,
            time_budget_s=self.time_budget_s, n_grid=self.n_grid, nf_max=self.nf_max,
        )
        self._executed = executed
        self.last_solution_cost = float(cost)
        self.last_bounds = bounds
        # Logged per step into agent_metrics.csv (constant within an episode).
        self.step_metrics = {'clairvoyant_cost': float(cost),
                             'lb': bounds.get('lb'), 'gap': bounds.get('gap'),
                             'bcd_sweeps': bounds.get('bcd_sweeps'),
                             'bcd_converged': bounds.get('bcd_converged')}

    def act(self, state: State) -> np.ndarray:
        if self._executed is None:
            raise RuntimeError(
                "ClairvoyantAgent.act called before on_episode_start; the "
                "evaluation harness must call on_episode_start(phase, episode_idx, "
                "base_seed, env, horizon) at the start of each episode."
            )
        t = int(state.t)
        if t >= self._executed.shape[0]:
            # Beyond the solved horizon (should not happen in normal eval): idle.
            return np.zeros(self._executed.shape[1], dtype=int)
        return self._executed[t].copy()
