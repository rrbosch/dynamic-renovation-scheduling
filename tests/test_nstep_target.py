"""Tests for the n-step ADP target (training.trainer._assign_mc_returns).

n_step=0 (or >= episode length) must reproduce the full-horizon MC return bit-identically;
n_step=n must bootstrap off V' at post_{i+n} with the exact n-step formula and fall back to
the full return for the tail. Also checks the config→agent wiring.

Run: "C:\\Python_Venv\\Code v2\\Scripts\\python.exe" -m pytest tests/test_nstep_target.py -v
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from training.trainer import _assign_mc_returns
from training.buffer import Transition

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GAMMA = 0.9
COSTS = [10.0, 20.0, 30.0, 40.0, 50.0]
POSTV = [1.0, 2.0, 3.0, 4.0, 5.0]   # V'(post_i) tag values


class _FakeState:
    def __init__(self, v): self.v = v; self.t = 0
    def copy(self):
        s = _FakeState(self.v); s.t = self.t; return s


class _FakeVF:
    def predict(self, posts):
        return np.array([p.v for p in posts], dtype=float)


class _FakeAgent:
    def __init__(self, n): self.value_fn = _FakeVF(); self.n_step = n


def _make():
    return [
        Transition(state=_FakeState(0), action=np.zeros(3, int), cost=c,
                   next_state=_FakeState(0), post_state=_FakeState(v), done=False)
        for c, v in zip(COSTS, POSTV)
    ]


def _full_ref():
    G, out = 0.0, [0.0] * len(COSTS)
    for i in range(len(COSTS) - 1, -1, -1):
        G = COSTS[i] + GAMMA * G
        out[i] = G
    return out


def test_nstep_zero_is_full_horizon_bit_identical():
    trs = _make()
    _assign_mc_returns(trs, GAMMA, 'horizon_rollout', _FakeAgent(0))
    assert [t.mc_return for t in trs] == _full_ref()


def test_nstep_ge_length_is_full_horizon():
    trs = _make()
    _assign_mc_returns(trs, GAMMA, 'horizon_rollout', _FakeAgent(99))
    assert [t.mc_return for t in trs] == _full_ref()


def test_nstep_matches_bootstrap_formula_and_full_tail():
    n = 2
    trs = _make()
    _assign_mc_returns(trs, GAMMA, 'horizon_rollout', _FakeAgent(n))
    got = [t.mc_return for t in trs]
    ref = _full_ref()
    L = len(COSTS)
    for i in range(L):
        if i + n < L:
            exp = sum(GAMMA**k * COSTS[i + k] for k in range(n + 1)) + GAMMA**n * POSTV[i + n]
            assert got[i] == pytest.approx(exp)
        else:
            assert got[i] == ref[i]      # tail falls back to the exact full return


def test_nstep_falls_back_when_value_fn_unfitted():
    class _UnfitVF:
        def predict(self, posts): raise RuntimeError("not fitted")
    class _A:
        def __init__(self): self.value_fn = _UnfitVF(); self.n_step = 2
    trs = _make()
    _assign_mc_returns(trs, GAMMA, 'horizon_rollout', _A())
    assert [t.mc_return for t in trs] == _full_ref()   # graceful full-horizon fallback


def test_n_step_wired_from_config_to_agent():
    """build_experiment must propagate agent.extra.n_step onto the ADPAgent."""
    import json, dataclasses
    from experiments.configs import ExperimentConfig, build_experiment
    from agents.dqn import ADPAgent
    prev = os.getcwd(); os.chdir(PROJECT_ROOT)
    try:
        cfg = ExperimentConfig.from_file(str(PROJECT_ROOT / "configs" / "i10p_adp_normal_policy_fifo_xgb.json"))
        d = json.loads(cfg.to_json())
        d['agent']['extra'] = {**d['agent']['extra'], 'n_step': 4}
        cfg2 = ExperimentConfig.from_json(json.dumps(d))
        _, agent, _ = build_experiment(cfg2)
        assert isinstance(agent, ADPAgent)
        assert agent.n_step == 4
    finally:
        os.chdir(prev)
