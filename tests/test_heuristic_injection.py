"""End-to-end tests for tuned-heuristic injection into Exp 0B agent configs.

Covers the generalized build path that lets ANY tuned heuristic
(`reactive`, `paced`, `reactiveperasset`) be used as:
  * an ADP warmstart agent          (training.warmstart block)
  * a Monte-Carlo-rollout base policy (agent.extra.rollout_policy block)

Each test:
  1. copies a real config into tmp_path (never mutates configs/),
  2. runs the REAL injector (experiments/apply_optuna_params.py) on the copy,
  3. build_experiment()s the patched config,
  4. asserts the constructed warmstart agent / rollout base policy is the right
     class and (for per-asset) that its (N,3) thresholds match the injected params.

Run from the project root under the venv:
    "C:\\Python_Venv\\Code v2\\Scripts\\python.exe" -m pytest tests/test_heuristic_injection.py -v
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INJECTOR = PROJECT_ROOT / "experiments" / "apply_optuna_params.py"

# Bundled fixtures (self-contained — do NOT depend on gitignored results/ run outputs).
# Parent dir names end with the heuristic keyword so apply_optuna_params._infer_heuristic
# resolves reactive/paced/reactiveperasset from the path.
_FIX = PROJECT_ROOT / "tests" / "fixtures"
PERASSET_PARAMS = _FIX / "i10p_optuna_perasset" / "best_params.json"
REACTIVE_PARAMS = _FIX / "i10p_optuna_reactive" / "best_params.json"
PACED_PARAMS    = _FIX / "i10p_optuna_paced" / "best_params.json"

ADP_CONFIG     = PROJECT_ROOT / "configs" / "i10p_adp_normal_policy_fifo_xgb.json"
ROLLOUT_CONFIG = PROJECT_ROOT / "configs" / "i10p_rollout_policy.json"


@pytest.fixture(autouse=True)
def _chdir_project_root():
    """build_experiment resolves the instance path relative to cwd."""
    prev = os.getcwd()
    os.chdir(PROJECT_ROOT)
    try:
        yield
    finally:
        os.chdir(prev)


def _run_injector(params: Path, target: Path, block: str) -> None:
    res = subprocess.run(
        [sys.executable, str(INJECTOR),
         "--params", str(params), "--targets", str(target), "--block", block],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True,
    )
    assert res.returncode == 0, f"injector failed:\n{res.stdout}\n{res.stderr}"
    assert "patched 1 config" in res.stdout, res.stdout


def _heuristic_params(params_path: Path) -> dict:
    """Injected params = best_params.json minus the meta key(s)."""
    with open(params_path, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if k != "best_value"}


def _copy(src: Path, tmp_path: Path) -> Path:
    dst = tmp_path / src.name
    shutil.copy(src, dst)
    return dst


def _assert_perasset_thresholds(agent, params: dict, n: int = 10):
    from agents.heuristics import PerAssetReactiveAgent
    assert isinstance(agent, PerAssetReactiveAgent)
    thr = agent._thr  # (N, 3): [repair, restrict, renovate]
    assert thr.shape == (n, 3)
    for i in range(n):
        assert thr[i, 0] == pytest.approx(params[f"repair_threshold_{i}"])
        assert thr[i, 1] == pytest.approx(params[f"restrict_threshold_{i}"])
        assert thr[i, 2] == pytest.approx(params[f"renovate_threshold_{i}"])


# ---------------------------------------------------------------------------
# Per-asset: the new capability (blocks the 0B re-run)
# ---------------------------------------------------------------------------

def test_perasset_warmstart_builds(tmp_path):
    from experiments.configs import ExperimentConfig, build_experiment

    cfg_path = _copy(ADP_CONFIG, tmp_path)
    _run_injector(PERASSET_PARAMS, cfg_path, "warmstart")

    # block written correctly: agent_type + nested extra of 30 keys
    patched = json.loads(cfg_path.read_text())
    ws = patched["training"]["warmstart"]
    assert ws["agent_type"] == "reactiveperasset"
    assert len(ws["extra"]) == 30

    _, _, trainer = build_experiment(ExperimentConfig.from_file(str(cfg_path)))
    ws_agent = trainer._resolve_warmstart_agent(trainer.config.warmstart_agent_config)
    _assert_perasset_thresholds(ws_agent, _heuristic_params(PERASSET_PARAMS))


def test_perasset_rollout_policy_builds(tmp_path):
    from experiments.configs import ExperimentConfig, build_experiment

    cfg_path = _copy(ROLLOUT_CONFIG, tmp_path)
    _run_injector(PERASSET_PARAMS, cfg_path, "rollout_policy")

    # block written correctly: flat agent_type + 30 keys
    patched = json.loads(cfg_path.read_text())
    rp = patched["agent"]["extra"]["rollout_policy"]
    assert rp["agent_type"] == "reactiveperasset"
    assert len({k for k in rp if k != "agent_type"}) == 30

    _, agent, _ = build_experiment(ExperimentConfig.from_file(str(cfg_path)))
    _assert_perasset_thresholds(agent.rollout_policy, _heuristic_params(PERASSET_PARAMS))


# ---------------------------------------------------------------------------
# Regression: reactive / paced still inject + build correctly
# ---------------------------------------------------------------------------

def test_reactive_warmstart_builds(tmp_path):
    from experiments.configs import ExperimentConfig, build_experiment
    from agents.heuristics import ReactiveAgent

    cfg_path = _copy(ADP_CONFIG, tmp_path)
    _run_injector(REACTIVE_PARAMS, cfg_path, "warmstart")

    p = _heuristic_params(REACTIVE_PARAMS)
    _, _, trainer = build_experiment(ExperimentConfig.from_file(str(cfg_path)))
    agent = trainer._resolve_warmstart_agent(trainer.config.warmstart_agent_config)
    assert isinstance(agent, ReactiveAgent)
    assert agent.threshold == pytest.approx(p["threshold"])
    assert agent.repair_threshold == pytest.approx(p["repair_threshold"])
    assert agent.restrict_threshold == pytest.approx(p["restrict_threshold"])


def test_reactive_rollout_policy_builds(tmp_path):
    from experiments.configs import ExperimentConfig, build_experiment
    from agents.heuristics import ReactiveAgent

    cfg_path = _copy(ROLLOUT_CONFIG, tmp_path)
    _run_injector(REACTIVE_PARAMS, cfg_path, "rollout_policy")

    p = _heuristic_params(REACTIVE_PARAMS)
    _, agent, _ = build_experiment(ExperimentConfig.from_file(str(cfg_path)))
    assert isinstance(agent.rollout_policy, ReactiveAgent)
    assert agent.rollout_policy.threshold == pytest.approx(p["threshold"])


def test_paced_warmstart_builds(tmp_path):
    from experiments.configs import ExperimentConfig, build_experiment
    from agents.heuristics import PacedAgent

    cfg_path = _copy(ADP_CONFIG, tmp_path)
    _run_injector(PACED_PARAMS, cfg_path, "warmstart")

    p = _heuristic_params(PACED_PARAMS)
    _, _, trainer = build_experiment(ExperimentConfig.from_file(str(cfg_path)))
    agent = trainer._resolve_warmstart_agent(trainer.config.warmstart_agent_config)
    assert isinstance(agent, PacedAgent)
    assert agent.threshold == pytest.approx(p["threshold"])
    assert agent.pace_threshold == pytest.approx(p["pace_threshold"])


# ---------------------------------------------------------------------------
# Sanity: a bad per-asset key is rejected by strict validation
# ---------------------------------------------------------------------------

def test_perasset_typo_key_rejected():
    from experiments.configs import _build_agent, AgentConfig
    from env.network import load_sioux_falls
    from env.tap import make_tap
    from env.mdp import InfraEnv, EnvConfig

    net = load_sioux_falls(n_assets=10)
    env = InfraEnv(net, make_tap(net, backend="null"),
                   EnvConfig(n_assets=10, gamma=0.95,
                             mu_h=1.0, sigma_h=0.2, delta_repair=0.3,
                             alpha0=np.full(10, 0.05), beta=np.full(10, 3.0),
                             c_ren=np.full(10, 1.0), c_rep=np.full(10, 1.0),
                             asset_lengths_m=np.full(10, 200.0)),
                   rng_seed=0)
    extra = {f"repair_threshold_{i}": 0.5 for i in range(10)}
    extra.update({f"restrict_threshold_{i}": 0.5 for i in range(10)})
    extra.update({f"renovate_threshold_{i}": 0.5 for i in range(10)})
    extra["renovate_threshld_3"] = 0.5  # typo
    with pytest.raises(ValueError, match="Unknown key"):
        _build_agent(AgentConfig(agent_type="reactiveperasset", extra=extra), env, seed=0)
