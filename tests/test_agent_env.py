"""Agent-environment compatibility tests.

For each registered agent, runs one full episode and asserts:
  1. act() returns a valid action array (correct shape, dtype, value range)
  2. Every action is feasible according to env.feasible_actions() BEFORE being
     passed to step() — i.e. the agent must handle feasibility itself.
  3. env.step() never raises (infeasible actions now raise ValueError).
  4. State invariants hold at every timestep:
       d  in [0, 1]
       h  >= 0
       ell in {0, 1}
       r   in {0, 1}
  5. Episode terminates with done=True after exactly T steps.

To add a new agent: append a fixture to AGENT_FIXTURES below.
"""
from __future__ import annotations

import numpy as np
import pytest

from env.mdp import InfraEnv, EnvConfig, State
from env.network import load_sioux_falls
from env.tap import make_tap


# ---------------------------------------------------------------------------
# Shared environment fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def env():
    network = load_sioux_falls()
    tap = make_tap(network, backend='fast')
    cfg = EnvConfig(
        n_assets=network.n_assets,
        gamma=0.97,
        c_fail=1000.0,
        mu_h=np.full(network.n_assets, 1.73),   # ~30 weeks
        sigma_h=np.full(network.n_assets, 0.35),
        delta_repair=0.1,
        alpha0=np.full(network.n_assets, 0.05),
        kappa=np.full(network.n_assets, 0.001),
        beta=np.full(network.n_assets, 6.0),
        c_ren=np.full(network.n_assets, 500.0),
        c_rep=np.full(network.n_assets, 100.0),
        T=10,   # short episode so tests run fast
    )
    return InfraEnv(network, tap, cfg, rng_seed=0)


# ---------------------------------------------------------------------------
# Agent registry
# Each entry: (agent_id, factory_fn(env) -> Agent)
# Add new agents here to include them in the compatibility check.
# ---------------------------------------------------------------------------

def _make_reactive(env):
    from agents.heuristics import ReactiveAgent
    return ReactiveAgent(threshold=0.7, env_config=env.config)


def _make_paced(env):
    from agents.heuristics import PacedAgent
    return PacedAgent(threshold=0.7, env_config=env.config)


def _make_asts(env):
    from agents.fn.value_fn import XGBoostValueFn
    from agents.action_gen import LocalSearchGenerator
    from agents.dqn import ADPAgent
    from agents.asts import ASTSAgent
    from training.trainer import TrainingConfig
    adp = ADPAgent(XGBoostValueFn(), LocalSearchGenerator(), env, TrainingConfig())
    return ASTSAgent(adp, max_leaves=5)


AGENT_FIXTURES = [
    pytest.param(_make_reactive, id='reactive'),
    pytest.param(_make_paced,    id='paced'),
    pytest.param(_make_asts,     id='asts'),
]


# ---------------------------------------------------------------------------
# Compatibility test
# ---------------------------------------------------------------------------

def _assert_state_invariants(state: State, n: int, t: int) -> None:
    assert state.d.shape == (n,),   f"t={t}: d shape {state.d.shape}"
    assert state.h.shape == (n,),   f"t={t}: h shape {state.h.shape}"
    assert state.ell.shape == (n,), f"t={t}: ell shape {state.ell.shape}"
    assert state.r.shape == (n,),   f"t={t}: r shape {state.r.shape}"
    assert np.all(state.d >= 0) and np.all(state.d <= 1), \
        f"t={t}: d out of [0,1]: {state.d}"
    assert np.all(state.h >= 0), \
        f"t={t}: h < 0: {state.h}"
    assert np.all(np.isin(state.ell, [0.0, 1.0])), \
        f"t={t}: ell not in {{0,1}}: {state.ell}"
    assert np.all(np.isin(state.r, [0.0, 1.0])), \
        f"t={t}: r not in {{0,1}}: {state.r}"


@pytest.mark.parametrize('make_agent', AGENT_FIXTURES)
def test_agent_episode(make_agent, env):
    agent = make_agent(env)
    n = env.config.n_assets
    T = env.config.T

    state = env.reset()
    _assert_state_invariants(state, n, t=0)

    done = False
    steps = 0
    while not done:
        action = agent.act(state)

        # --- contract checks on the action itself ---
        assert isinstance(action, np.ndarray), \
            f"act() must return np.ndarray, got {type(action)}"
        assert action.shape == (n,), \
            f"action shape {action.shape} != ({n},)"
        assert action.dtype.kind in ('i', 'u'), \
            f"action dtype {action.dtype} is not integer"
        assert np.all(action >= 0) and np.all(action <= 3), \
            f"action values out of {{0,1,2,3}}: {action}"

        # --- agent must produce feasible actions ---
        feas = env.feasible_actions(state)
        infeasible = [i for i in range(n) if not feas[i, action[i]]]
        assert not infeasible, (
            f"Agent produced infeasible actions at assets {infeasible}. "
            f"Actions: {action[infeasible]}, "
            f"Feasible masks: {[feas[i].tolist() for i in infeasible]}"
        )

        # --- step must not raise ---
        state, cost, done = env.step(state, action)

        assert np.isfinite(cost), f"Non-finite cost at t={steps}: {cost}"
        _assert_state_invariants(state, n, t=steps + 1)
        steps += 1

    assert steps == T, f"Episode ended after {steps} steps, expected {T}"
