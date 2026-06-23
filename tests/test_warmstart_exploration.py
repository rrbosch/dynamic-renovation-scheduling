"""Tests for fix (b): d-dependent acting exploration in the warmstart buffer.

The warmstart heuristic is ~98% do-nothing, so V' never sees the acting
post-states the greedy search must rank. The new exploration flips a per-asset
action with a probability that ramps with condition d, biased toward acting
(esp. renovate) at high d. These tests guard:
  * reproducibility (same seed -> identical buffer),
  * that stronger exploration raises the acting fraction in the buffer,
  * that disabling it (p_high == p_base, act_bias == 0) recovers the old
    uniform-flip behaviour.
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


def _build():
    from experiments.configs import ExperimentConfig, build_experiment
    cfg = ExperimentConfig.from_file(str(PROJECT_ROOT / "configs" / "i10p_adp_normal_empty_fifo_xgb.json"))
    return build_experiment(cfg)


def _acting_fraction(trainer, n_states=3000):
    trainer.config = dataclasses.replace(trainer.config, n_warmstart_states=n_states)
    ws = trainer._resolve_warmstart_agent(trainer.config.warmstart_agent_config)
    trainer._run_warmstart(ws)
    acts = np.array([t.action for t in trainer.buffer._data])
    return (acts != 0).mean(), (acts == 2).mean()


def test_warmstart_is_reproducible():
    env1, _, t1 = _build()
    env2, _, t2 = _build()
    t1.config = dataclasses.replace(t1.config, n_warmstart_states=1500)
    t2.config = dataclasses.replace(t2.config, n_warmstart_states=1500)
    ws1 = t1._resolve_warmstart_agent(t1.config.warmstart_agent_config)
    ws2 = t2._resolve_warmstart_agent(t2.config.warmstart_agent_config)
    t1._run_warmstart(ws1)
    t2._run_warmstart(ws2)
    a1 = np.array([t.action for t in t1.buffer._data])
    a2 = np.array([t.action for t in t2.buffer._data])
    np.testing.assert_array_equal(a1, a2)


def test_stronger_exploration_raises_acting_fraction():
    """Default (b) exploration must buffer more acting transitions than the old
    uniform p_flip = 1/T behaviour."""
    # old behaviour: ramp disabled, no acting bias
    _, _, t_old = _build()
    t_old.config = dataclasses.replace(
        t_old.config,
        warmstart_explore_p_base=1.0 / 120,
        warmstart_explore_p_high=1.0 / 120,
        warmstart_explore_act_bias=0.0,
    )
    act_old, ren_old = _acting_fraction(t_old)

    # new behaviour: defaults from the config (d-ramped, acting-biased)
    _, _, t_new = _build()
    act_new, ren_new = _acting_fraction(t_new)

    assert act_new > act_old
    assert ren_new > ren_old


def test_disabled_exploration_matches_uniform_flip():
    """p_high == p_base and act_bias == 0 reproduces a uniform-feasible flip:
    the acting fraction stays low (close to the heuristic's own ~1-3%)."""
    _, _, t = _build()
    t.config = dataclasses.replace(
        t.config,
        warmstart_explore_p_base=1.0 / 120,
        warmstart_explore_p_high=1.0 / 120,
        warmstart_explore_act_bias=0.0,
    )
    act, _ = _acting_fraction(t)
    assert act < 0.10
