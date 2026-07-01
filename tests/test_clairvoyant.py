"""Tests for the clairvoyant (perfect-information) baseline.

Covers the properties that make it a valid lower-bound reference:

  * replay faithfulness — `simulate_intended` on the replayed noise reproduces
    `env.step`'s discounted cost and executed actions EXACTLY (the linchpin: if
    this holds, the reconstructed future matches the realized one);
  * lower-bound property — clairvoyant cost <= do-nothing and <= a reactive
    heuristic on the same (seed, episode) realization;
  * act/solve consistency — the plan returned by `act`, when actually executed
    by `env.step` on the same noise, reproduces the solver's reported cost and
    never raises an infeasibility error;
  * determinism — identical (seed, episode) ⇒ identical plan and cost.

NullTAP is used (no numba locally); travel cost is then zero, so the instance is
purely per-asset, which is the regime that most sharply exercises the DP +
search (do-nothing pays the escalating failure risk the clairvoyant must avoid).
"""
from __future__ import annotations

import numpy as np
import pytest

from env.mdp import InfraEnv, EnvConfig, State
from env.network import load_sioux_falls
from env.tap import make_tap
from agents.heuristics import ReactiveAgent, DoNothingAgent
from agents.clairvoyant import (
    ClairvoyantAgent,
    replay_episode_noise,
    simulate_intended,
    solve_clairvoyant,
    _heuristic_intended,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_env(seed: int = 3, T: int = 14) -> InfraEnv:
    """Small, fast-degrading env where assets fail within the horizon."""
    network = load_sioux_falls(n_assets=4)
    n = network.n_assets
    cfg = EnvConfig(
        n_assets=n,
        gamma=0.97,
        mu_h=np.full(n, 1.5),       # renovation ~2 epochs
        sigma_h=np.full(n, 0.3),
        delta_repair=0.1,
        alpha0=np.full(n, 0.8),
        beta=np.full(n, 4.0),       # mean increment alpha0*dt/beta = 0.1 / epoch
        c_ren=np.full(n, 1.0e7),
        c_rep=np.full(n, 5.0e6),
        asset_lengths_m=np.full(n, 200.0),
        risk_base=10_000.0,
        T=T,
        d_init=np.array([0.70, 0.60, 0.65, 0.55])[:n],
    )
    return InfraEnv(network, make_tap(network, backend='null'), cfg, rng_seed=seed)


def _run_agent_cost(env, agent, seed, ep, horizon, use_hook=False):
    """Run `agent` through the real env (env.step) for one episode; return
    (discounted_total, executed_actions (H, N))."""
    env.begin_episode("evaluation", ep, seed)
    state = env.reset()
    if use_hook:
        agent.on_episode_start("evaluation", ep, seed, env, horizon)
    gamma = env.config.gamma
    total = 0.0
    executed = np.zeros((horizon, env.config.n_assets), dtype=int)
    for t in range(horizon):
        a = np.asarray(agent.act(state), dtype=int)
        executed[t] = a
        next_state, cost, _ = env.step(state, a)
        total += (gamma ** t) * cost
        state = next_state
    return total, executed


# ---------------------------------------------------------------------------
# Replay faithfulness
# ---------------------------------------------------------------------------

def test_replay_matches_env_step_exactly():
    """simulate_intended on replayed noise reproduces env.step's cost and the
    executed actions for the same policy, epoch-for-epoch."""
    seed, ep, H = 11, 2, 14
    env = _make_env(seed=seed, T=H)
    heur = ReactiveAgent(threshold=0.7, env_config=env.config,
                         repair_threshold=0.9, restrict_threshold=0.6)

    env_cost, env_exec = _run_agent_cost(env, heur, seed, ep, H)

    # Replay the same episode's noise and roll the heuristic's intended plan.
    u, eps, d_init = replay_episode_noise(env, "evaluation", ep, seed, H)
    np.testing.assert_allclose(d_init, env.config.d_init)
    intended = _heuristic_intended(env, heur, u, eps, d_init, H)
    sim_cost, sim_exec = simulate_intended(env, u, eps, d_init, intended, H)

    assert np.array_equal(sim_exec, env_exec)
    assert sim_cost == pytest.approx(env_cost, rel=1e-9, abs=1e-3)


def test_replay_initial_condition_sampled_when_no_d_init():
    """When d_init is sampled (cfg.d_init is None) the replay must reproduce the
    env's reset draw, so simulate_intended still matches env.step."""
    seed, ep, H = 5, 1, 10
    env = _make_env(seed=seed, T=H)
    # Drop the fixed initial condition so reset() samples it.
    import dataclasses
    env.config = dataclasses.replace(env.config, d_init=None)

    heur = ReactiveAgent(threshold=0.7, env_config=env.config)
    env_cost, env_exec = _run_agent_cost(env, heur, seed, ep, H)

    u, eps, d_init = replay_episode_noise(env, "evaluation", ep, seed, H)
    intended = _heuristic_intended(env, heur, u, eps, d_init, H)
    sim_cost, sim_exec = simulate_intended(env, u, eps, d_init, intended, H)

    assert np.array_equal(sim_exec, env_exec)
    assert sim_cost == pytest.approx(env_cost, rel=1e-9, abs=1e-3)


# ---------------------------------------------------------------------------
# Lower-bound property
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ep", [0, 1, 2])
def test_clairvoyant_is_lower_bound(ep):
    """Clairvoyant cost <= do-nothing and <= a reactive heuristic on the same
    realization (a perfect-information policy can match either by construction)."""
    seed, H = 7, 14
    env = _make_env(seed=seed, T=H)

    donothing = DoNothingAgent(env.config)
    reactive = ReactiveAgent(threshold=0.7, env_config=env.config,
                             repair_threshold=0.9, restrict_threshold=0.6)
    dn_cost, _ = _run_agent_cost(env, donothing, seed, ep, H)
    re_cost, _ = _run_agent_cost(env, reactive, seed, ep, H)

    u, eps, d_init = replay_episode_noise(env, "evaluation", ep, seed, H)
    cv_cost, _, _ = solve_clairvoyant(env, u, eps, d_init, H, use_dp=True, max_sweeps=3)

    tol = 1e-3 * max(1.0, abs(dn_cost))
    assert cv_cost <= dn_cost + tol
    assert cv_cost <= re_cost + tol
    # On this failing instance the clairvoyant should strictly beat do-nothing.
    assert cv_cost < dn_cost


def test_dp_warm_start_helps_or_matches():
    """Including the per-asset DP start never hurts (multi-start keeps the best)."""
    seed, ep, H = 7, 0, 14
    env = _make_env(seed=seed, T=H)
    u, eps, d_init = replay_episode_noise(env, "evaluation", ep, seed, H)
    with_dp, _, _ = solve_clairvoyant(env, u, eps, d_init, H, use_dp=True, max_sweeps=3)
    no_dp, _, _ = solve_clairvoyant(env, u, eps, d_init, H, use_dp=False, max_sweeps=3)
    assert with_dp <= no_dp + 1e-3


@pytest.mark.parametrize("ep", [0, 1])
def test_bounds_sane(ep):
    """LB0 <= LB <= UB and gap >= 0; the reported LB is a valid lower bound on the
    clairvoyant solution (and never exceeds it)."""
    seed, H = 7, 14
    env = _make_env(seed=seed, T=H)
    u, eps, d_init = replay_episode_noise(env, "evaluation", ep, seed, H)
    ub, _, b = solve_clairvoyant(env, u, eps, d_init, H, use_dp=True, max_sweeps=3)
    assert b['lb0'] <= b['lb'] + 1e-6
    assert b['lb'] <= ub + 1e-6
    assert b['gap'] >= -1e-9
    assert b['ub'] == pytest.approx(ub)
    if b['lb_cache_valid'] and b['lb_cache'] is not None:
        assert b['lb_cache'] >= b['lb0'] - 1e-6      # tightener never below the floor


# ---------------------------------------------------------------------------
# act / solve consistency through the real env
# ---------------------------------------------------------------------------

def test_agent_plan_executes_faithfully():
    """The plan act() returns, executed by env.step on the same noise, reproduces
    the solver's reported cost and raises no infeasibility error."""
    seed, ep, H = 7, 1, 14
    env = _make_env(seed=seed, T=H)
    agent = ClairvoyantAgent(use_dp=True, max_sweeps=3, seed=seed)

    realized_cost, _ = _run_agent_cost(env, agent, seed, ep, H, use_hook=True)
    assert agent.last_solution_cost is not None
    assert realized_cost == pytest.approx(agent.last_solution_cost, rel=1e-9, abs=1e-3)


def test_determinism():
    """Same (seed, episode) ⇒ identical plan and cost."""
    seed, ep, H = 7, 2, 14
    env = _make_env(seed=seed, T=H)
    a1 = ClairvoyantAgent(use_dp=True, max_sweeps=3, seed=seed)
    a2 = ClairvoyantAgent(use_dp=True, max_sweeps=3, seed=seed)
    a1.on_episode_start("evaluation", ep, seed, env, H)
    a2.on_episode_start("evaluation", ep, seed, env, H)
    assert np.array_equal(a1._executed, a2._executed)
    assert a1.last_solution_cost == pytest.approx(a2.last_solution_cost, rel=1e-12)


def test_act_before_solve_raises():
    env = _make_env()
    agent = ClairvoyantAgent()
    with pytest.raises(RuntimeError):
        agent.act(State(np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)))
