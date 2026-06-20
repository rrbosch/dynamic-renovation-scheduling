"""Tests for the network-/lifetime-aware heuristic baselines.

Covers the five heuristics added alongside reactive/paced/reactiveperasset:
  leadtime, netconcurrency, holding, valuedensity, worstfirst.

Each is checked for:
  1. direct build via experiments.configs._build_agent (fixed-param form),
  2. feasibility — a full rollout through InfraEnv.step (which asserts feasibility),
  3. save/load round-trip of its JSON params,
and the Optuna wiring (heuristic_type → default param_space + agent_factory) is
checked end-to-end through build_experiment.

Run from the project root under the venv:
    "C:\\Python_Venv\\Code v2\\Scripts\\python.exe" -m pytest tests/test_new_heuristics.py -v
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Fixed-param `extra` blocks exercising every action channel of each heuristic.
DIRECT_AGENTS = {
    "leadtime":       {"lead_epochs": 6.0, "repair_lead": 10.0, "restrict_lead": 14.0},
    "netconcurrency": {"threshold": 0.6, "max_concurrent": 3, "spread_penalty": 0.5},
    "holding":        {"threshold": 0.6, "max_concurrent": 3, "defer_window": 8.0,
                       "restrict_flow_quantile": 0.5},
    "valuedensity":   {"max_concurrent": 3, "risk_weight": 1.0, "degrad_weight": 1.0,
                       "threshold": 0.2},
    "worstfirst":     {"max_concurrent": 3, "threshold": 0.4, "use_length": True},
}

EXPECTED_CLASS = {
    "leadtime":       "LeadTimeAgent",
    "netconcurrency": "NetConcurrencyAgent",
    "holding":        "HoldingAgent",
    "valuedensity":   "ValueDensityAgent",
    "worstfirst":     "WorstFirstAgent",
}


def _make_env(n: int = 10, T: int = 30):
    from env.network import load_sioux_falls
    from env.tap import make_tap
    from env.mdp import InfraEnv, EnvConfig

    net = load_sioux_falls(n_assets=n)
    rng = np.random.default_rng(0)
    L = rng.uniform(100.0, 400.0, size=n)
    cfg = EnvConfig(
        n_assets=n, gamma=0.95, mu_h=1.0, sigma_h=0.2, delta_repair=0.3,
        alpha0=rng.uniform(0.03, 0.08, size=n), beta=rng.uniform(2.0, 4.0, size=n),
        c_ren=50_000.0 * L, c_rep=25_000.0 * L, asset_lengths_m=L,
        T=T, dt=0.5,
    )
    return InfraEnv(net, make_tap(net, backend="null"), cfg, rng_seed=0)


@pytest.fixture(autouse=True)
def _chdir_project_root():
    prev = os.getcwd()
    os.chdir(PROJECT_ROOT)
    try:
        yield
    finally:
        os.chdir(prev)


# ---------------------------------------------------------------------------
# Direct build + feasibility rollout + save/load
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("agent_type", list(DIRECT_AGENTS))
def test_direct_build_and_feasible_rollout(agent_type, tmp_path):
    from experiments.configs import _build_agent, AgentConfig

    env = _make_env()
    agent = _build_agent(
        AgentConfig(agent_type=agent_type, extra=DIRECT_AGENTS[agent_type]),
        env, seed=0,
    )
    assert type(agent).__name__ == EXPECTED_CLASS[agent_type]

    env.begin_episode("test", 0)
    state = env.reset()
    total = 0.0
    done = False
    while not done:
        action = agent.act(state)
        assert action.shape == (env.config.n_assets,)
        # env.step asserts feasibility internally — an infeasible action raises.
        state, cost, done = env.step(state, action)
        total += cost
    assert np.isfinite(total)


@pytest.mark.parametrize("agent_type", list(DIRECT_AGENTS))
def test_save_load_roundtrip(agent_type, tmp_path):
    from experiments.configs import _build_agent, AgentConfig

    env = _make_env()
    cfg = AgentConfig(agent_type=agent_type, extra=DIRECT_AGENTS[agent_type])
    agent = _build_agent(cfg, env, seed=0)
    before = agent._heuristic_params()

    agent.save(str(tmp_path))
    fresh = _build_agent(cfg, env, seed=0)
    fresh.load(str(tmp_path))
    assert fresh._heuristic_params() == before


def test_unknown_extra_key_rejected():
    from experiments.configs import _build_agent, AgentConfig

    env = _make_env()
    with pytest.raises(ValueError, match="Unknown key"):
        _build_agent(
            AgentConfig(agent_type="holding", extra={"defer_windwo": 4.0}),  # typo
            env, seed=0,
        )


# ---------------------------------------------------------------------------
# Optuna wiring: heuristic_type → default param_space + agent_factory
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("heuristic_type", list(DIRECT_AGENTS))
def test_optuna_wiring(heuristic_type, tmp_path):
    from experiments.configs import ExperimentConfig, build_experiment
    from experiments.optuna_heuristic_search import OptunaHeuristicTrainer

    run_name = f"_tmp_test/{heuristic_type}_optuna"
    cfg = {
        "network": "sioux_falls",
        "tap_backend": "null",
        "seed": 42,
        "run_name": run_name,
        "instance": "instances/instance_10p.json",
        "training": {"time_budget": 1, "n_eval_episodes": 2,
                     "early_stopping_seconds": 1, "n_workers": 1},
        "agent": {
            "agent_type": "optuna_heuristic",
            "extra": {"heuristic_type": heuristic_type, "n_tuning_episodes": 2},
        },
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    try:
        _, agent, trainer = build_experiment(ExperimentConfig.from_file(str(cfg_path)))
        assert isinstance(trainer, OptunaHeuristicTrainer)
        assert trainer.param_space, "default param_space should be non-empty"
        # The prototype agent is the right class and the factory builds the same.
        assert type(agent).__name__ == EXPECTED_CLASS[heuristic_type]
        sample = {}
        for name, spec in trainer.param_space.items():
            if spec["type"] == "categorical":
                sample[name] = spec["choices"][0]
            elif spec["type"] == "int":
                sample[name] = int(spec["low"])
            else:
                sample[name] = float(spec["low"])
        built = trainer.agent_factory(sample)
        assert type(built).__name__ == EXPECTED_CLASS[heuristic_type]
    finally:
        shutil.rmtree(PROJECT_ROOT / "results" / "_tmp_test", ignore_errors=True)
