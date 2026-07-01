"""Warmstart buffer reproducibility.

NOTE: the old in-trainer `warmstart_explore_*` acting-exploration knobs were removed
(2026-06-24) — acting coverage is now an opt-in composable heuristic (`explore_flip`
/ `FlipWrapperAgent`, tested via the agent's own unit tests). The two tests that
exercised those removed knobs were dropped; this file now only guards that the
warmstart buffer is a pure function of the seed.
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
