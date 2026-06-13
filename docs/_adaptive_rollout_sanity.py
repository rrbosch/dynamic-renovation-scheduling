"""Tiny fixed-vs-adaptive rollout sanity check on instance_10p.

NOT an experiment runner — a few decision epochs of one short episode, to report
(1) action agreement between fixed and adaptive selection and (2) rollout-sim
budget saved. Run: python docs/_adaptive_rollout_sanity.py
"""
from __future__ import annotations

import json
import time
import numpy as np

from env.mdp import InfraEnv, EnvConfig
from env.network import load_sioux_falls
from env.tap import make_tap
from agents.heuristics import ReactiveAgent
from agents.rollout import MonteCarloRolloutAgent, SequentialMCRolloutAgent

INSTANCE = "instances/instance_10p.json"
N_EPOCHS = 5          # decision epochs to probe
N_ROLLOUTS = 30       # fixed budget / adaptive cap
P_THRESHOLD = 0.1
MIN_ROLLOUTS = 5


def _arr(v, n):
    return np.full(n, float(v)) if isinstance(v, (int, float)) else np.array(v, float)


def build_env(seed=0):
    with open(INSTANCE) as f:
        inst = json.load(f)
    net = load_sioux_falls(n_assets=inst["n_assets"])
    n = net.n_assets
    cfg = EnvConfig(
        n_assets=n, dt=inst["dt"], T=round(inst["years"] / inst["dt"]),
        gamma=inst["gamma"] ** inst["dt"], d_fail=inst["d_fail"],
        eta_ren=inst["eta_ren"], eta_load=inst["eta_load"],
        restrict_degrad_multiplier=float(inst.get("restrict_degrad_multiplier", 0.5)),
        mu_h=_arr(inst["mu_h"], n), sigma_h=_arr(inst["sigma_h"], n),
        delta_repair=inst["delta_repair"], alpha0=_arr(inst["alpha0"], n),
        beta=_arr(inst["beta"], n), c_ren=_arr(inst["c_ren"], n),
        c_rep=_arr(inst["c_rep"], n), asset_lengths_m=_arr(inst["asset_lengths_m"], n),
        vot=float(inst.get("vot", 10.76)),
        traffic_cost_factor=float(inst.get("traffic_cost_factor", 1.0)),
        risk_base=float(inst.get("risk_base", 10_000.0)),
        d_init=np.array(inst["d_init"], float) if inst.get("d_init") is not None else None,
    )
    return InfraEnv(net, make_tap(net, backend="fast"), cfg, rng_seed=seed)


def make_agent(env, selection):
    pol = ReactiveAgent(threshold=0.7, env_config=env.config)
    return MonteCarloRolloutAgent(
        rollout_policy=pol, env=env, n_rollouts=N_ROLLOUTS, seed=0,
        action_threshold=0.5, initial_action="policy", selection=selection,
        p_threshold=P_THRESHOLD, min_rollouts=MIN_ROLLOUTS, max_rollouts=N_ROLLOUTS,
        rollout_batch=5,
    )


def main():
    env = build_env()
    fixed = make_agent(env, "fixed")
    adapt = make_agent(env, "adaptive")

    env.begin_episode("evaluation", 0)
    state = env.reset()

    print(f"\n{'epoch':>5} | {'diff':>9} | {'fixed sims':>10} | {'adapt sims':>10} | "
          f"{'saved':>7} | {'fixed s':>8} | {'adapt s':>8}")
    print("-" * 75)

    agree = 0
    tot_fixed_sims = tot_adapt_sims = 0
    for t in range(N_EPOCHS):
        t0 = time.perf_counter(); a_fix = fixed.act(state); t_fix = time.perf_counter() - t0
        t0 = time.perf_counter(); a_adp = adapt.act(state); t_adp = time.perf_counter() - t0

        # fixed mode doesn't track n_rollout_sims; reconstruct the worst-case count:
        # n_candidates * n_rollouts (every candidate fully evaluated).
        fixed_sims = fixed.step_metrics["n_candidates"] * N_ROLLOUTS
        adapt_sims = adapt.step_metrics["n_rollout_sims"]
        tot_fixed_sims += fixed_sims; tot_adapt_sims += adapt_sims

        n_diff = int(np.sum(a_fix != a_adp))
        match = n_diff == 0
        agree += int(match)
        saved = 1.0 - adapt_sims / max(1, fixed_sims)
        tag = "same" if match else f"{n_diff} asset(s)"
        print(f"{t:>5} | {tag:>9} | {fixed_sims:>10,} | {adapt_sims:>10,} | "
              f"{saved:>6.1%} | {t_fix:>8.2f} | {t_adp:>8.2f}")

        # advance with the fixed action (reference trajectory)
        state, _, _ = env.step(state, a_fix)

    print("-" * 72)
    print(f"action agreement: {agree}/{N_EPOCHS} epochs")
    print(f"total rollout sims  fixed={tot_fixed_sims:,}  adaptive={tot_adapt_sims:,}  "
          f"saved={1 - tot_adapt_sims / max(1, tot_fixed_sims):.1%}")


if __name__ == "__main__":
    main()
