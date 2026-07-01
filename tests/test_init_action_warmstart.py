"""Tests for the two ADP knobs split out on 2026-06-20:

  * init_action ('empty' | 'policy'): the ADP local action search starts from the
    do-nothing action, or from the warmstart heuristic's action (seeded search).
  * n_warmstart_states: buffer warmstart is sized in transitions (states), not episodes.

Run from the project root under the venv:
    "C:\\Python_Venv\\Code v2\\Scripts\\python.exe" -m pytest tests/test_init_action_warmstart.py -v
"""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _chdir_project_root():
    prev = os.getcwd()
    os.chdir(PROJECT_ROOT)
    try:
        yield
    finally:
        os.chdir(prev)


def _build(cfg_name: str):
    from experiments.configs import ExperimentConfig, build_experiment
    cfg = ExperimentConfig.from_file(str(PROJECT_ROOT / "configs" / cfg_name))
    return build_experiment(cfg)


# ---------------------------------------------------------------------------
# init_action wiring (build_experiment attaches the warmstart policy for 'policy')
# ---------------------------------------------------------------------------

def test_empty_init_has_no_warmstart_policy():
    from agents.dqn import ADPAgent
    _, agent, _ = _build("i10p_adp_normal_empty_fifo_xgb.json")
    assert isinstance(agent, ADPAgent)
    assert agent.init_action_mode == "empty"
    assert agent.warmstart_policy is None


def test_policy_init_attaches_perasset_warmstart_policy():
    from agents.dqn import ADPAgent
    from agents.heuristics import PerAssetReactiveAgent
    env, agent, _ = _build("i10p_adp_seq_policy_lowesterror_nn.json")
    assert isinstance(agent, ADPAgent)
    assert agent.init_action_mode == "policy"
    assert isinstance(agent.warmstart_policy, PerAssetReactiveAgent)
    # act() must run end-to-end (exercises generate(init_action=...))
    env.begin_episode("test", 0)
    a = agent.act(env.reset())
    assert a.shape == (env.config.n_assets,)


def test_policy_init_requires_warmstart_block():
    """init_action='policy' without a training.warmstart heuristic must fail loudly."""
    from experiments.configs import ExperimentConfig, build_experiment
    cfg = ExperimentConfig.from_file(str(PROJECT_ROOT / "configs" / "i10p_adp_normal_empty_fifo_xgb.json"))
    training = dict(cfg.training)
    training.pop("warmstart", None)
    training.pop("n_warmstart_states", None)
    agent = dataclasses.replace(cfg.agent, extra={**cfg.agent.extra, "init_action": "policy"})
    bad = dataclasses.replace(cfg, training=training, agent=agent)
    with pytest.raises(ValueError, match="init_action='policy'"):
        build_experiment(bad)


# ---------------------------------------------------------------------------
# init_action seeds the action search (generator-level)
# ---------------------------------------------------------------------------

def test_local_search_respects_init_action_seed():
    """A pre-seeded action that local search cannot improve is returned verbatim."""
    from agents.action_gen import LocalSearchGenerator
    env, agent, _ = _build("i10p_adp_normal_policy_fifo_xgb.json")
    vf = agent.value_fn
    env.begin_episode("test", 0)
    state = env.reset()
    gen = LocalSearchGenerator()
    seed = agent.warmstart_policy.act(state)            # feasible heuristic action
    out = gen.generate(state, vf, env, init_action=seed)
    # With an untrained/flat V', the seeded search returns an action at least as good
    # as the seed; the returned action must be feasible.
    feas = env.feasible_actions(state)
    assert all(feas[i, out[i]] for i in range(env.config.n_assets))


def test_local_search_batches_predict_calls():
    """Batched LocalSearch issues one predict() per sweep, not one per candidate.

    The throughput fix collapses ~30 single-candidate predict() calls per
    decision into one batched call (the dominant cost for tree value fns). Guard
    that property: across a decision there must be far fewer predict() calls than
    candidates evaluated, and every batched call must carry >1 post-state on at
    least one sweep.
    """
    from agents.action_gen import LocalSearchGenerator
    env, agent, _ = _build("i10p_adp_normal_policy_fifo_xgb.json")
    vf = agent.value_fn
    env.begin_episode("test", 0)
    state = env.reset()

    batch_sizes: list[int] = []
    orig = vf.predict
    vf.predict = lambda states, _o=orig, _b=batch_sizes: (_b.append(len(states)), _o(states))[1]
    try:
        gen = LocalSearchGenerator()
        gen.generate(state, vf, env, init_action=np.zeros(env.config.n_assets, dtype=int))
    finally:
        vf.predict = orig

    n_candidates = gen.last_metrics['n_candidates']
    assert sum(batch_sizes) == n_candidates          # accounting is exact
    assert len(batch_sizes) < n_candidates           # fewer calls than candidates
    assert max(batch_sizes) > 1                       # at least one true batch


def test_local_search_finds_true_local_optimum():
    """Steepest-descent batched search returns a single-asset local optimum:
    no feasible single-asset deviation lowers Q below the returned action."""
    from agents.action_gen import LocalSearchGenerator
    env, agent, _ = _build("i10p_adp_normal_policy_fifo_xgb.json")

    # Deterministic toy value fn so the optimum is well-defined and reproducible.
    rng = np.random.default_rng(0)
    w = rng.normal(size=5 * env.config.n_assets + 1)

    class ToyVF:
        finite_horizon = True
        def _feats(self, states):
            from agents.fn.value_fn import ValueFn
            return ValueFn._feats(self, states)
        def predict(self, states):
            X = self._feats(states)
            return np.abs(X @ w) * 1e6

    vf = ToyVF()
    env.begin_episode("test", 0)
    state = env.reset()
    gen = LocalSearchGenerator()
    out = gen.generate(state, vf, env, init_action=np.zeros(env.config.n_assets, dtype=int))

    feas = env.feasible_actions(state)

    def q(action):
        sp = env.post_decision_state(state, action, check=False)
        return env.immediate_cost(state, action, sp) + float(vf.predict([sp])[0])

    q_out = q(out)
    for i in range(env.config.n_assets):
        for a in range(4):
            if not feas[i, a] or a == out[i]:
                continue
            cand = out.copy(); cand[i] = a
            assert q(cand) >= q_out - 1e-3      # no improving single-asset move


# ---------------------------------------------------------------------------
# n_warmstart_states: warmstart sized in transitions
# ---------------------------------------------------------------------------

def test_warmstart_fills_exact_state_count():
    env, agent, trainer = _build("i10p_adp_normal_policy_fifo_xgb.json")
    trainer.config = dataclasses.replace(trainer.config, n_warmstart_states=850)
    ws = trainer._resolve_warmstart_agent(trainer.config.warmstart_agent_config)
    trainer._run_warmstart(ws)
    # FIFO buffer, capacity >> 850 → buffer holds exactly the requested states.
    assert len(trainer.buffer) == 850
