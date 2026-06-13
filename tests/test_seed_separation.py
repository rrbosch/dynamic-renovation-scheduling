"""Phase-keyed environment noise: train/eval/rollout separation + reproducibility.

These tests lock in the guarantees of the stateless, phase-keyed RNG
(`env.noise.keyed_philox` + `InfraEnv.begin_episode`):

  * different "phase" tags produce structurally independent streams
    (no leakage between training, evaluation, and rollout);
  * a given (phase, seed, episode_idx) reproduces the identical trajectory,
    independent of how many other episodes ran before (parallelism- and
    resume-invariant);
  * evaluation uses shared common random numbers (same noise regardless of the
    acting agent), enabling paired comparisons;
  * rollout noise excludes the candidate action (CRN across candidates).

The fixture builds the env from the CURRENT EnvConfig (do NOT copy the stale
fixture in tests/test_agent_env.py, which uses removed fields).
"""
from __future__ import annotations

import numpy as np

from env.mdp import InfraEnv, EnvConfig
from env.network import load_sioux_falls
from env.noise import keyed_philox
from env.tap import make_tap


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_env(seed: int = 42, T: int = 6) -> InfraEnv:
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
        d_init=None,   # force reset() to sample d (exercises the reset stream)
    )
    return InfraEnv(network, make_tap(network, backend='null'), cfg, rng_seed=seed)


def _run(env: InfraEnv, phase: str, ep: int, n_steps: int = 6,
         action_fn=None) -> list[np.ndarray]:
    """Begin a keyed episode and roll it out with a scripted policy.
    Returns the trajectory of `d` arrays (initial + after each step)."""
    if action_fn is None:
        action_fn = lambda state, t: np.zeros(env.config.n_assets, dtype=int)  # ACTION_NONE
    env.begin_episode(phase, ep)
    state = env.reset()
    traj = [state.d.copy()]
    for t in range(n_steps):
        state, _cost, done = env.step(state, action_fn(state, t))
        traj.append(state.d.copy())
        if done:
            break
    return traj


def _equal_traj(a, b) -> bool:
    return len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# 1. keyed_philox primitive
# ---------------------------------------------------------------------------

def test_keyed_philox_deterministic():
    a = keyed_philox("transition", "evaluation", 42, 3).random(10)
    b = keyed_philox("transition", "evaluation", 42, 3).random(10)
    assert np.array_equal(a, b)


def test_keyed_philox_phase_independent():
    k = lambda phase: keyed_philox("transition", phase, 42, 3).random(64)
    train, ev, roll = k("training"), k("evaluation"), k("rollout")
    assert not np.array_equal(train, ev)
    assert not np.array_equal(train, roll)
    assert not np.array_equal(ev, roll)


def test_keyed_philox_no_key_collision():
    # ("ab", 1) must not collide with ("a", "b1") etc.
    assert not np.array_equal(
        keyed_philox("ab", 1).random(8),
        keyed_philox("a", "b1").random(8),
    )


# ---------------------------------------------------------------------------
# 2. Env trajectory determinism & no mutable cross-episode state
# ---------------------------------------------------------------------------

def test_same_key_same_trajectory():
    env = _make_env()
    t1 = _run(env, "evaluation", 3)
    t2 = _run(env, "evaluation", 3)          # same env, run again
    assert _equal_traj(t1, t2)

    env2 = _make_env()                        # fresh env, same seed
    assert _equal_traj(t1, _run(env2, "evaluation", 3))


def test_eval_independent_of_training_history():
    """Eval episode k is identical regardless of intervening training episodes
    (the core parallelism/resume-invariance property)."""
    env = _make_env()
    baseline = _run(env, "evaluation", 5)
    for ep in range(7):                       # churn arbitrary training episodes
        _run(env, "training", ep)
    assert _equal_traj(baseline, _run(env, "evaluation", 5))


# ---------------------------------------------------------------------------
# 3. Phase disjointness in the real env (no train/eval/rollout leakage)
# ---------------------------------------------------------------------------

def test_training_and_evaluation_diverge():
    env = _make_env()
    assert not _equal_traj(_run(env, "training", 3), _run(env, "evaluation", 3))


def test_reset_initial_conditions_keyed():
    env = _make_env()
    env.begin_episode("evaluation", 9)
    d_eval = env.reset().d.copy()
    env.begin_episode("training", 9)
    d_train = env.reset().d.copy()
    assert not np.array_equal(d_eval, d_train)
    # ...but evaluation/9 is reproducible:
    env.begin_episode("evaluation", 9)
    assert np.array_equal(d_eval, env.reset().d)


# ---------------------------------------------------------------------------
# 4. Shared-CRN evaluation across agents
# ---------------------------------------------------------------------------

def test_eval_crn_shared_across_agents():
    """Two different policies on the same eval episode see identical underlying
    randomness: same sampled initial conditions, and a matching transition when
    they happen to take the same action."""
    env = _make_env()
    zeros = lambda s, t: np.zeros(env.config.n_assets, dtype=int)

    # Agent A: always NONE. Agent B: identical at t=0, one renovate at t=1.
    def agent_b(state, t):
        a = np.zeros(env.config.n_assets, dtype=int)
        if t == 1:
            a[0] = InfraEnv.ACTION_RENOVATE
        return a

    traj_a = _run(env, "evaluation", 1, action_fn=zeros)
    traj_b = _run(env, "evaluation", 1, action_fn=agent_b)

    # Identical initial conditions (shared reset stream) ...
    assert np.array_equal(traj_a[0], traj_b[0])
    # ... and identical first transition (same action at t=0, same noise).
    assert np.array_equal(traj_a[1], traj_b[1])


# ---------------------------------------------------------------------------
# 5. Rollout CRN: noise excludes the candidate action; independent of real eval
# ---------------------------------------------------------------------------

def test_rollout_noise_excludes_action_and_is_independent():
    from agents.rollout import rollout_noise
    env = _make_env()
    env.begin_episode("evaluation", 2)
    root = env.reset()

    # Same (seed, root_state, t, rollout_idx) -> identical noise (CRN across the
    # candidate actions evaluated at this decision, which are not in the key).
    u1, e1 = rollout_noise(42, root, root.t, 0, n_steps=4, n_assets=env.config.n_assets)
    u2, e2 = rollout_noise(42, root, root.t, 0, n_steps=4, n_assets=env.config.n_assets)
    assert np.array_equal(u1, u2) and np.array_equal(e1, e2)

    # The agent's internal "rollout" stream is independent of the real
    # "evaluation"/"transition" stream for the same nominal indices -> the agent
    # can never replay its realized evaluation future (no clairvoyance).
    u_eval = keyed_philox("transition", "evaluation", 42, 2).random((4, env.config.n_assets))
    assert not np.array_equal(u1, u_eval)
