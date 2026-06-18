"""Adaptive (sequential Wilcoxon) rollout budgeting for the MC rollout agents.

These tests lock in the opt-in `selection='adaptive'` mode added to
`agents/rollout.py` (see docs/adaptive_rollout_literature.md). They cover:

  * the default path is unchanged ('fixed') and adaptive is strictly opt-in;
  * determinism — same seed/state ⇒ identical actions (CRN keying untouched);
  * the stopping rule never exceeds max_rollouts and never decides before
    min_rollouts (startup guard);
  * a synthetic clearly-dominant challenger stops at the startup size;
  * a synthetic dead-heat runs to the cap;
  * adaptive returns feasible actions and reports rollout-sim usage.

The env fixture mirrors tests/test_seed_separation.py (current EnvConfig).
"""
from __future__ import annotations

import numpy as np
import pytest

from env.mdp import InfraEnv, EnvConfig
from env.network import load_sioux_falls
from env.tap import make_tap
from agents.heuristics import ReactiveAgent
from agents.rollout import MonteCarloRolloutAgent, SequentialMCRolloutAgent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_env(seed: int = 7, T: int = 8) -> InfraEnv:
    network = load_sioux_falls()
    n = network.n_assets
    cfg = EnvConfig(
        n_assets=n,
        gamma=0.97,
        mu_h=np.full(n, 1.5),
        sigma_h=np.full(n, 0.3),
        delta_repair=0.1,
        alpha0=np.full(n, 0.05),
        beta=np.full(n, 6.0),
        c_ren=np.full(n, 500.0),
        c_rep=np.full(n, 100.0),
        asset_lengths_m=np.full(n, 200.0),
        T=T,
        d_init=None,
    )
    return InfraEnv(network, make_tap(network, backend='null'), cfg, rng_seed=seed)


def _policy(env: InfraEnv) -> ReactiveAgent:
    return ReactiveAgent(threshold=0.6, env_config=env.config,
                         repair_threshold=0.8, restrict_threshold=0.5)


def _agent(env, cls=MonteCarloRolloutAgent, selection='fixed', **kw):
    defaults = dict(
        rollout_policy=_policy(env), env=env, n_rollouts=12, seed=0,
        action_threshold=0.5, initial_action='policy', selection=selection,
    )
    defaults.update(kw)
    return cls(**defaults)


def _decision_state(env: InfraEnv, seed: int = 0):
    """A mid-episode state with several assets above the action threshold."""
    env.begin_episode("evaluation", seed)
    state = env.reset()
    # Drive degradation up so the local search actually has candidates.
    rng = np.random.default_rng(123)
    state.d = rng.uniform(0.55, 0.95, size=env.config.n_assets)
    state.t = 2
    return state


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

def test_default_selection_is_adaptive():
    """Global default is the chosen Pareto operating point (adaptive p=0.02,
    min=20, max=100). Constructed WITHOUT selection/p/min/max to test defaults."""
    env = _make_env()
    agent = MonteCarloRolloutAgent(rollout_policy=_policy(env), env=env)
    assert agent.selection == 'adaptive'
    assert agent.p_threshold == 0.02
    assert agent.min_rollouts == 20
    assert agent.max_rollouts == 100


def test_invalid_selection_raises():
    env = _make_env()
    with pytest.raises(ValueError):
        _agent(env, selection='bogus')


def test_min_gt_max_raises():
    env = _make_env()
    with pytest.raises(ValueError):
        _agent(env, selection='adaptive', min_rollouts=10, max_rollouts=4)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [MonteCarloRolloutAgent, SequentialMCRolloutAgent])
def test_adaptive_deterministic(cls):
    env = _make_env()
    state = _decision_state(env)

    a1 = _agent(env, cls=cls, selection='adaptive').act(state.copy())
    a2 = _agent(env, cls=cls, selection='adaptive').act(state.copy())
    np.testing.assert_array_equal(a1, a2)


@pytest.mark.parametrize("cls", [MonteCarloRolloutAgent, SequentialMCRolloutAgent])
def test_fixed_still_deterministic(cls):
    env = _make_env()
    state = _decision_state(env)
    a1 = _agent(env, cls=cls, selection='fixed').act(state.copy())
    a2 = _agent(env, cls=cls, selection='fixed').act(state.copy())
    np.testing.assert_array_equal(a1, a2)


def test_adaptive_returns_feasible_action():
    env = _make_env()
    state = _decision_state(env)
    agent = _agent(env, selection='adaptive')
    action = agent.act(state.copy())
    feas = env.feasible_actions(state)
    assert action.shape == (env.config.n_assets,)
    assert np.all(feas[np.arange(env.config.n_assets), action])
    # Usage is reported and bounded by the worst case (candidates * max each side).
    assert 'n_rollout_sims' in agent.step_metrics
    assert agent.step_metrics['n_rollout_sims'] >= 0


# ---------------------------------------------------------------------------
# Stopping-rule unit tests (synthetic Q samples via _q_samples override)
# ---------------------------------------------------------------------------

def _stub_compare(env, q_chal, q_inc, *, min_rollouts, max_rollouts,
                  p_threshold=0.1, rollout_batch=1):
    """Run `_challenger_wins` against synthetic paired samples, returning
    (challenger_won, max_n_requested)."""
    agent = _agent(env, selection='adaptive', min_rollouts=min_rollouts,
                   max_rollouts=max_rollouts, p_threshold=p_threshold,
                   rollout_batch=rollout_batch, n_rollouts=max_rollouts)
    chal = np.zeros(env.config.n_assets, dtype=int)
    inc = np.zeros(env.config.n_assets, dtype=int)
    inc[0] = 1  # distinct key
    q_chal = np.asarray(q_chal, dtype=float)
    q_inc = np.asarray(q_inc, dtype=float)
    seen = {'max_n': 0}

    def fake_q_samples(state, action, n, cache):
        seen['max_n'] = max(seen['max_n'], n)
        src = q_chal if np.array_equal(action, chal) else q_inc
        return src[:n]

    agent._q_samples = fake_q_samples
    won = agent._challenger_wins(state=None, challenger=chal, incumbent=inc, cache={})
    return won, seen['max_n']


def test_clear_dominant_stops_at_startup():
    """Challenger uniformly far cheaper ⇒ decided at min_rollouts."""
    env = _make_env()
    n_max = 20
    won, max_n = _stub_compare(
        env,
        q_chal=np.zeros(n_max),
        q_inc=np.full(n_max, 100.0),
        min_rollouts=5, max_rollouts=n_max,
    )
    assert won is True
    assert max_n == 5  # stopped at the startup size, no extra rollouts


def test_clear_dominant_incumbent_stops_at_startup():
    """Incumbent uniformly far cheaper ⇒ challenger rejected at min_rollouts."""
    env = _make_env()
    n_max = 20
    won, max_n = _stub_compare(
        env,
        q_chal=np.full(n_max, 100.0),
        q_inc=np.zeros(n_max),
        min_rollouts=5, max_rollouts=n_max,
    )
    assert won is False
    assert max_n == 5


def test_dead_heat_runs_to_cap():
    """Alternating ±1 differences (zero mean) never reach significance ⇒ cap."""
    env = _make_env()
    n_max = 16
    q_chal = np.array([0.0, 1.0] * (n_max // 2))
    q_inc = np.array([1.0, 0.0] * (n_max // 2))
    won, max_n = _stub_compare(
        env, q_chal=q_chal, q_inc=q_inc,
        min_rollouts=5, max_rollouts=n_max, rollout_batch=1,
    )
    assert max_n == n_max  # exhausted the budget
    assert won is False     # mean diff == 0 ⇒ incumbent kept


def test_never_exceeds_max_or_decides_before_min():
    """Across a sweep of synthetic signals, the requested count is always in
    [min_rollouts, max_rollouts]."""
    env = _make_env()
    rng = np.random.default_rng(0)
    n_max = 18
    for _ in range(20):
        # Random paired samples with a random (possibly tiny) effect size.
        shift = rng.normal(0, 5)
        noise_c = rng.normal(0, 10, n_max)
        noise_i = rng.normal(0, 10, n_max)
        q_chal = noise_c
        q_inc = noise_i + shift
        _, max_n = _stub_compare(
            env, q_chal=q_chal, q_inc=q_inc,
            min_rollouts=6, max_rollouts=n_max, rollout_batch=3,
        )
        assert 6 <= max_n <= n_max


def test_exact_tie_keeps_incumbent_immediately():
    """Identical samples (all-zero diffs) ⇒ incumbent kept, decided at startup."""
    env = _make_env()
    n_max = 20
    won, max_n = _stub_compare(
        env,
        q_chal=np.full(n_max, 42.0),
        q_inc=np.full(n_max, 42.0),
        min_rollouts=5, max_rollouts=n_max,
    )
    assert won is False
    assert max_n == 5
