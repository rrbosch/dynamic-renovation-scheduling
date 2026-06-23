"""Faithful DCL (agents/dcl.py, training/dcl_trainer.py).

Dependency-free tests (numpy/scipy + NullTAP + heuristic base policy):
  * Sequential Halving picks the best arm and is reproducible;
  * rollout CRN is deterministic and rollout-index-independent;
  * a partial joint action yields a valid post-decision State and the sequential
    decomposition builds one row per asset with the label as target;
  * the labelling oracle returns feasible, deterministic joint actions for every
    (action_search x rollout_selection) combination;
  * an unfitted DCLAgent falls back to its base heuristic.

xgboost-gated end-to-end test (skipped when xgboost is unavailable): one DCL
round trains the classifier and produces a finite, reproducible evaluation.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from env.mdp import InfraEnv, EnvConfig, State
from env.network import load_sioux_falls
from env.tap import make_tap
from agents.heuristics import ReactiveAgent
from agents.rollout import rollout_noise, sequential_halving
from agents.dcl import DCLAgent, build_decomposition
from training.dcl_trainer import DCLConfig


# ---------------------------------------------------------------------------
# Fixtures (mirror tests/test_adaptive_rollout.py)
# ---------------------------------------------------------------------------

def _make_env(seed: int = 7, T: int = 8) -> InfraEnv:
    network = load_sioux_falls(n_assets=6)      # small ⇒ fast oracle in tests
    n = network.n_assets
    cfg = EnvConfig(
        n_assets=n, gamma=0.97, mu_h=np.full(n, 1.5), sigma_h=np.full(n, 0.3),
        delta_repair=0.1, alpha0=np.full(n, 0.05), beta=np.full(n, 6.0),
        c_ren=np.full(n, 500.0), c_rep=np.full(n, 100.0),
        asset_lengths_m=np.full(n, 200.0), T=T, d_init=None,
    )
    return InfraEnv(network, make_tap(network, backend='null'), cfg, rng_seed=seed)


def _heuristic(env):
    return ReactiveAgent(threshold=0.6, env_config=env.config,
                         repair_threshold=0.8, restrict_threshold=0.5)


def _decision_state(env, seed: int = 0):
    env.begin_episode("evaluation", seed)
    state = env.reset()
    rng = np.random.default_rng(123)
    state.d = rng.uniform(0.55, 0.95, size=env.config.n_assets)
    state.t = 2
    return state


def _dcl_cfg(**kw):
    base = dict(n_rollouts=4, rollout_horizon=4, rollout_selection='fixed',
                action_threshold=0.5, initial_action='policy', sh_budget_per_arm=4)
    base.update(kw)
    return DCLConfig(**base)


# ---------------------------------------------------------------------------
# Sequential Halving
# ---------------------------------------------------------------------------

def test_sequential_halving_picks_best():
    true_means = [10.0, 7.0, 5.0, 2.0, 8.0, 6.0, 9.0]      # arm 3 is best
    rng = np.random.default_rng(0)
    banks = {i: true_means[i] + rng.standard_normal(2000) for i in range(len(true_means))}
    qfn = lambda arm, n: banks[arm][:n]
    winner = sequential_halving(list(range(len(true_means))), qfn,
                                total_budget=30 * len(true_means))
    assert winner == 3


def test_sequential_halving_reproducible_and_single_arm():
    banks = {i: float(i) + np.random.default_rng(i).standard_normal(500)
             for i in range(5)}
    qfn = lambda arm, n: banks[arm][:n]
    w1 = sequential_halving([0, 1, 2, 3, 4], qfn, 100)
    w2 = sequential_halving([0, 1, 2, 3, 4], qfn, 100)
    assert w1 == w2 == 0
    assert sequential_halving(['only'], qfn, 100) == 'only'   # k == 1 short-circuit


# ---------------------------------------------------------------------------
# CRN
# ---------------------------------------------------------------------------

def test_rollout_noise_crn():
    s = State(np.array([0.3, 0.5]), np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2))
    s.t = 4
    u1, e1 = rollout_noise(7, s, s.t, 2, 6, 2)
    u2, e2 = rollout_noise(7, s, s.t, 2, 6, 2)
    assert np.array_equal(u1, u2) and np.array_equal(e1, e2)
    u3, _ = rollout_noise(7, s, s.t, 3, 6, 2)            # different rollout index
    assert not np.array_equal(u1, u3)


# ---------------------------------------------------------------------------
# Partial post-decision states / dataset construction
# ---------------------------------------------------------------------------

def test_partial_post_decision_and_sequential_rows():
    env = _make_env()
    n = env.config.n_assets
    state = _decision_state(env)
    label = np.zeros(n, dtype=int)
    label[0] = InfraEnv.ACTION_RENOVATE
    if n > 1:
        label[1] = InfraEnv.ACTION_RESTRICT

    # A partial joint action is a valid post-decision State.
    s_post = env.post_decision_state(state, label, check=False)
    assert s_post.h[0] > 0 and s_post.d.shape == (n,)
    assert np.all(np.isfinite(s_post.d))

    dec = build_decomposition('sequential', env, estimator_kind='xgboost')
    rows = list(dec.build_rows(state, label))
    assert len(rows) == n
    assert [t for _, t in rows] == list(label)            # targets == label
    assert all(np.asarray(f).ndim == 1 for f, _ in rows)


# ---------------------------------------------------------------------------
# Oracle labelling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("search", ["sequential", "independent", "local_search"])
@pytest.mark.parametrize("selection", ["fixed", "wilcoxon", "sequential_halving"])
def test_oracle_label_feasible_and_deterministic(search, selection):
    env = _make_env()
    n = env.config.n_assets
    state = _decision_state(env)
    cfg = _dcl_cfg(rollout_selection=selection)

    def label_once():
        dec = build_decomposition(search, env, estimator_kind='xgboost')
        agent = DCLAgent(dec, _heuristic(env), env, action_search=search)
        oracle = dec.make_oracle(agent, cfg, seed=0, value_fn=None)
        return oracle.act(state.copy())

    a1 = label_once()
    a2 = label_once()
    feas = env.feasible_actions(state)
    assert a1.shape == (n,)
    assert all(feas[i, a1[i]] for i in range(n))          # feasible
    assert np.array_equal(a1, a2)                          # deterministic


def test_unfitted_agent_falls_back_to_heuristic():
    env = _make_env()
    state = _decision_state(env)
    dec = build_decomposition('sequential', env, estimator_kind='xgboost')
    heur = _heuristic(env)
    agent = DCLAgent(dec, heur, env, action_search='sequential')
    assert not dec._fitted
    assert np.array_equal(agent.act(state.copy()), heur.act(state.copy()))


# ---------------------------------------------------------------------------
# End-to-end round (needs xgboost; nn cases also need torch)
# ---------------------------------------------------------------------------

def _trainer(env, search, policy, tmp_path, **cfg_kw):
    from utils.logging import RunLogger
    from training.dcl_trainer import DCLTrainer
    dec = build_decomposition(search, env, estimator_kind=policy,
                              epochs=5, batch_size=64)
    agent = DCLAgent(dec, _heuristic(env), env, action_search=search)
    cfg = dict(n_rounds=1, samples_per_round=24, warmup_steps=1, collect_steps=4,
               rollout_horizon=3, n_rollouts=3, rollout_selection='fixed',
               n_eval_episodes=2, time_budget=0, T_tail=1.0, seed=0, config_hash='h')
    cfg.update(cfg_kw)
    logger = RunLogger(str(tmp_path / "dcl_test"))
    return DCLTrainer(agent, env, DCLConfig(**cfg), logger), dec


@pytest.mark.parametrize("search", ["sequential", "independent", "local_search"])
@pytest.mark.parametrize("policy", ["xgboost", "nn"])
def test_one_round_trains_classifier(search, policy, tmp_path):
    pytest.importorskip("xgboost")
    if policy == "nn":
        pytest.importorskip("torch")
    env = _make_env(T=6)
    trainer, dec = _trainer(env, search, policy, tmp_path)
    trainer.train()
    assert dec._fitted                                     # classifier trained
    assert np.isfinite(trainer.evaluate(n_episodes=2)['mean_cost'])


def test_vfa_built_when_horizon_set_and_fits(tmp_path):
    pytest.importorskip("xgboost")
    env = _make_env(T=6)
    trainer, dec = _trainer(env, "sequential", "xgboost", tmp_path,
                            rollout_horizon=2, value_fn_kind="xgboost")
    assert trainer.value_fn is not None                    # VFA built (opt-in)
    trainer.train()
    assert dec._fitted and trainer.value_fn._fitted        # both fit


def test_no_vfa_when_horizon_none(tmp_path):
    env = _make_env(T=6)
    trainer, _ = _trainer(env, "independent", "xgboost", tmp_path,
                          rollout_horizon=None)
    assert trainer.value_fn is None                        # faithful: no VFA


def test_checkpoint_resume(tmp_path):
    pytest.importorskip("xgboost")
    import os
    env = _make_env(T=6)
    trainer, dec = _trainer(env, "sequential", "xgboost", tmp_path, n_rounds=2)
    trainer.train()
    assert dec._fitted
    ckpt = os.path.join(str(trainer.logger.run_dir), "checkpoints", "ep_1")
    assert os.path.isdir(ckpt)

    # Fresh trainer resumes from the checkpoint.
    trainer2, dec2 = _trainer(env, "sequential", "xgboost", tmp_path / "b", n_rounds=2)
    assert not dec2._fitted
    start = trainer2.load_checkpoint(ckpt)
    assert start == 2 and dec2._fitted                     # restored classifier
