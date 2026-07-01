"""Tests for the per-asset dynamic program of the clairvoyant baseline.

  * `_asset_step` reproduces InfraEnv's per-asset dynamics and (maint+risk) cost
    exactly (validated against env.step on a single-asset env);
  * `perasset_dp` matches a brute-force optimum on small horizons (the DP is
    optimal up to the condition grid);
  * the travel-penalty interface plumbs through (huge penalties suppress the
    penalised actions);
  * determinism.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from env.mdp import InfraEnv, EnvConfig
from env.network import load_sioux_falls
from env.tap import make_tap
from agents.clairvoyant import (
    _asset_params, _asset_step, asset_plan_cost, perasset_dp, replay_episode_noise,
)


def _params(alpha0=0.8, beta=4.0, c_ren=1e7, c_rep=4e6, mu_h=1.5, d_fail=1.0,
            allow_repair=True, allow_restrict=True, risk_base=1e4, L=200.0,
            delta=0.1, eta_load=0.5, f=0.9):
    return dict(alpha0=alpha0, beta=beta, mu_h=mu_h, sigma_h=0.3, delta=delta,
                dt=0.5, d_fail=d_fail, eta_ren=0.05, eta_load=eta_load, f=f,
                c_ren=c_ren, c_rep=c_rep, L=L, risk_base=risk_base,
                allow_repair=allow_repair, allow_restrict=allow_restrict)


# ---------------------------------------------------------------------------
# _asset_step vs env (the linchpin for the DP being env-faithful)
# ---------------------------------------------------------------------------

def test_asset_step_matches_env_single_asset():
    """On a 1-asset env, _asset_step reproduces env.step's per-asset (maint+risk)
    cost and full state trajectory for an arbitrary action sequence."""
    network = load_sioux_falls(n_assets=1)
    n = network.n_assets
    cfg = EnvConfig(
        n_assets=n, gamma=0.97, mu_h=np.full(n, 1.5), sigma_h=np.full(n, 0.3),
        delta_repair=0.1, alpha0=np.full(n, 0.8), beta=np.full(n, 4.0),
        c_ren=np.full(n, 1e7), c_rep=np.full(n, 5e6), asset_lengths_m=np.full(n, 200.0),
        risk_base=1e4, T=16, d_init=np.array([0.62]),
        restrict_degrad_multiplier=0.9, eta_load=0.5,
    )
    env = InfraEnv(network, make_tap(network, backend='null'), cfg, rng_seed=4)
    H = 16
    rng = np.random.default_rng(0)
    plan = rng.integers(0, 4, size=(H, n))

    env.begin_episode("evaluation", 0, 4)
    state = env.reset()
    env_costs, env_states, env_actions = [], [], []
    for t in range(H):
        feas = env.feasible_actions(state)              # env asserts feasibility;
        a = np.where(feas[np.arange(n), plan[t]], plan[t], 0)   # project like the harness
        ns, _, _ = env.step(state, a)
        c_travel, c_maint, c_risk = env.last_cost_breakdown
        env_costs.append(c_maint + c_risk)
        env_states.append((ns.d[0], ns.h[0], ns.ell[0], ns.r[0], ns.n_fail[0]))
        env_actions.append(int(a[0]))
        state = ns

    u, eps, d0 = replay_episode_noise(env, "evaluation", 0, 4, H)
    p = _asset_params(cfg, 0)
    st = (float(d0[0]), 0.0, 0.0, 0.0, 0.0)
    for t in range(H):
        a_exec, _, pa, st = _asset_step(p, st, int(plan[t, 0]), u[t, 0], eps[t, 0])
        assert a_exec == env_actions[t]                 # projection matches env
        assert pa == pytest.approx(env_costs[t], rel=1e-9, abs=1e-3)
        assert np.allclose(st, env_states[t], atol=1e-9)


# ---------------------------------------------------------------------------
# DP optimality vs brute force
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_dp_matches_bruteforce(seed):
    """On a short horizon the DP's plan is (near-)optimal: its exact cost is within
    grid tolerance of the brute-force minimum over all action sequences."""
    H, gamma = 7, 0.97
    p = _params()
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 1, H)
    eps = rng.standard_normal(H)
    d0 = 0.6

    # Brute force over all 4^H sequences (exact per-asset dynamics).
    best = np.inf
    for seq in itertools.product(range(4), repeat=H):
        c = asset_plan_cost(p, d0, np.array(seq), u, eps, gamma)
        if c < best:
            best = c

    plan = perasset_dp(p, d0, u, eps, H, gamma, n_grid=256, nf_max=12)
    dp_cost = asset_plan_cost(p, d0, plan, u, eps, gamma)

    assert dp_cost >= best - 1e-3                      # brute force is the true min
    assert dp_cost <= best * (1 + 1e-3) + 1.0          # DP optimal up to grid slack


def test_dp_travel_penalty_suppresses_renovation():
    """A prohibitive renovating-class penalty makes the DP avoid renovating
    (vs. renovating with zero penalty on a failing asset)."""
    H, gamma = 30, 0.97
    p = _params()
    rng = np.random.default_rng(3)
    u = rng.uniform(0, 1, H)
    eps = rng.standard_normal(H)
    d0 = 0.7

    plan0 = perasset_dp(p, d0, u, eps, H, gamma, travel_penalty=None, n_grid=128)
    assert (plan0 == InfraEnv.ACTION_RENOVATE).any()   # zero penalty ⇒ renovates

    tp = np.zeros((3, H))
    tp[2] = 1e15                                        # renovating is ruinous
    plan1 = perasset_dp(p, d0, u, eps, H, gamma, travel_penalty=tp, n_grid=128)
    assert not (plan1 == InfraEnv.ACTION_RENOVATE).any()


def test_dp_beats_donothing():
    """On a failing asset the DP's plan is strictly cheaper than do-nothing."""
    H, gamma = 40, 0.97
    p = _params()
    rng = np.random.default_rng(5)
    u = rng.uniform(0, 1, H)
    eps = rng.standard_normal(H)
    d0 = 0.65
    plan = perasset_dp(p, d0, u, eps, H, gamma, n_grid=128)
    dp_cost = asset_plan_cost(p, d0, plan, u, eps, gamma)
    dn_cost = asset_plan_cost(p, d0, np.zeros(H, int), u, eps, gamma)
    assert dp_cost < dn_cost


def test_dp_determinism():
    H, gamma = 20, 0.97
    p = _params()
    rng = np.random.default_rng(9)
    u = rng.uniform(0, 1, H); eps = rng.standard_normal(H)
    a = perasset_dp(p, 0.6, u, eps, H, gamma, n_grid=128)
    b = perasset_dp(p, 0.6, u, eps, H, gamma, n_grid=128)
    assert np.array_equal(a, b)
