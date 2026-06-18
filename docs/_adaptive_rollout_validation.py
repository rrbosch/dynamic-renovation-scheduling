"""Larger fixed-vs-adaptive rollout validation on instance_10p (single-pass).

Methodology (printed at runtime too):

  STATES: simulate instance_10p forward under the ReactiveAgent base heuristic
  (the same policy the rollout agents roll out internally) for several episodes
  on the "evaluation" CRN phase, collect every pre-decision state across epochs,
  then sample N_STATES of them with a fixed RNG, STRATIFIED across the planning
  horizon (early/mid/late) so we don't oversample pristine early states. These
  are on-(base-)policy operational states — a proxy for "reasonably common
  states to encounter."

  SINGLE-PASS PER STATE (the compute-saving trick):
    1. Run the FIXED MCRollout local search at n_rollouts=N_FULL and RECORD,
       for every action it evaluates, the full vector of N_FULL per-rollout Q
       samples in a shared "bank" (keyed by action). a_opt = its result.
    2. RECONSTRUCT what adaptive MCRollout would have done by running the real
       adaptive local search (`_act_adaptive` + `_challenger_wins`, the shipped
       code) but feeding it samples FROM THE BANK. Because rollout noise is
       CRN-keyed on (root state, rollout index) and EXCLUDES the action, the
       sample values adaptive needs are byte-identical to the ones fixed already
       computed — so no rollout is recomputed except for any actions on a
       divergent adaptive path (computed live, same CRN). a_adp = its result.

  KPIs:
    - action correctness: exact match + per-asset Hamming distance vs a_opt
    - fractional cost deviation: re-evaluate BOTH final actions with the full
      N_FULL-rollout Q (from the bank), report
          frac_dev = (Q_full(a_adp) - Q_full(a_opt)) / |Q_full(a_opt)|
    - % ROLLOUTS SAVED: fixed logical sims = (#fixed Q-evals incl. repeats)*N_FULL
      (what the shipped fixed agent actually computes); adaptive logical sims =
      sum over distinct actions of the high-water rollout count adaptive needed
      (what the shipped adaptive agent, with per-epoch caching, would compute).
      saved = 1 - adaptive_sims / fixed_sims.

Throwaway diagnostic. Bounded rollout_horizon keeps it tractable; the SAME
horizon is used for the fixed pass and the adaptive reconstruction.
Run: python docs/_adaptive_rollout_validation.py [N_STATES] [ROLLOUT_HORIZON]
"""
from __future__ import annotations

import sys
import json
import time
import numpy as np

from env.mdp import InfraEnv, EnvConfig
from env.network import load_sioux_falls
from env.tap import make_tap
from agents.heuristics import ReactiveAgent
from agents.rollout import MonteCarloRolloutAgent

INSTANCE = "instances/instance_10p.json"
N_STATES = int(sys.argv[1]) if len(sys.argv) > 1 else 100
ROLLOUT_HORIZON = int(sys.argv[2]) if len(sys.argv) > 2 else 30
N_FULL = 100          # fixed budget / adaptive cap
P_THRESHOLD = 0.1
MIN_ROLLOUTS = 5
BATCH = 5
N_GEN_EPISODES = 8
GEN_SEED = 12345
SAMPLE_SEED = 2024


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


# ---------------------------------------------------------------------------
# Recording agent: shared sample bank, faithful fixed + adaptive accounting
# ---------------------------------------------------------------------------

class RecordingRolloutAgent(MonteCarloRolloutAgent):
    """Overrides `_q_samples` to draw from a per-state shared bank so the fixed
    pass and the adaptive reconstruction never recompute the same CRN rollout.
    Tracks faithful logical sim counts for both modes."""

    def new_state(self):
        self._bank = {}        # action.tobytes() -> {'sp','ic','tn','ms','q':[...]}
        self._mode = None      # 'fixed' | 'adaptive' | None
        self._phys = 0         # physical rollouts actually computed
        self._fixed_evals = 0  # # full-budget Q-evaluations in the fixed search
        self._adp_hw = {}      # action -> high-water rollout count adaptive needed

    def _q_samples(self, state, action, n, cache):
        key = action.tobytes()
        e = self._bank.get(key)
        if e is None:
            s_post = self.env.post_decision_state(state, action)
            ic = self.env.immediate_cost(state, action, s_post)
            t_next = state.t + 1
            if self.rollout_horizon is not None:
                ms = self.rollout_horizon
            else:
                ms = self.env.config.T - t_next
            e = {"sp": s_post, "ic": ic, "tn": t_next, "ms": ms, "q": []}
            self._bank[key] = e
        q, ms = e["q"], e["ms"]
        while len(q) < n:
            r = len(q)
            if ms <= 0:
                q.append(e["ic"])
            else:
                q.append(e["ic"] + self._single_rollout(e["sp"], e["tn"], ms, state, r))
                self._phys += 1
        if self._mode == "adaptive":
            self._adp_hw[key] = max(self._adp_hw.get(key, 0), n)
        return np.asarray(q[:n], dtype=float)

    # -- faithful fixed local search (mirrors MonteCarloRolloutAgent.act) ----
    def _qmean_full(self, state, action):
        self._fixed_evals += 1
        return float(self._q_samples(state, action, N_FULL, None).mean())

    def fixed_act(self, state):
        self._mode = "fixed"
        n = self.env.config.n_assets
        feas = self.env.feasible_actions(state)
        feas[state.d < self.action_threshold, 1:] = False
        if self.initial_action == "policy":
            cur = self.rollout_policy.act(state)
            cur = np.where(feas[np.arange(n), cur], cur, self.env.ACTION_NONE)
        else:
            cur = np.zeros(n, dtype=int)
        cur_q = self._qmean_full(state, cur)
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for a in range(4):
                    if a == cur[i] or not feas[i, a]:
                        continue
                    cand = cur.copy(); cand[i] = a
                    q = self._qmean_full(state, cand)
                    if q < cur_q:
                        cur_q, cur, improved = q, cand, True
        return cur

    def adaptive_act(self, state):
        self._mode = "adaptive"
        self._adp_hw = {}
        action = self._act_adaptive(state)   # shipped adaptive logic, bank-backed
        return action

    def q_full(self, state, action):
        self._mode = None
        return float(self._q_samples(state, action, N_FULL, None).mean())


def harvest_states(env, policy):
    visited = []
    for ep in range(N_GEN_EPISODES):
        env.begin_episode("evaluation", ep)
        state = env.reset()
        for t in range(env.config.T):
            visited.append((t, state.copy()))
            state, _, done = env.step(state, policy.act(state))
            if done:
                break
    return visited


def stratified_sample(visited, n_states, T):
    rng = np.random.default_rng(SAMPLE_SEED)
    thirds = [[], [], []]
    for idx, (t, _) in enumerate(visited):
        thirds[min(2, int(3 * t / T))].append(idx)
    per = [n_states // 3, n_states // 3, n_states - 2 * (n_states // 3)]
    chosen = []
    for bucket, k in zip(thirds, per):
        if bucket:
            chosen.extend(rng.choice(bucket, size=min(k, len(bucket)),
                                     replace=False).tolist())
    remaining = [i for i in range(len(visited)) if i not in set(chosen)]
    while len(chosen) < n_states and remaining:
        chosen.append(remaining.pop(rng.integers(len(remaining))))
    return [visited[i] for i in chosen[:n_states]]


def main():
    env = build_env()
    pol = ReactiveAgent(threshold=0.7, env_config=env.config)
    print(f"# Adaptive-rollout validation (single-pass) | N_STATES={N_STATES} "
          f"N_FULL={N_FULL} horizon={ROLLOUT_HORIZON} p={P_THRESHOLD} "
          f"min={MIN_ROLLOUTS} batch={BATCH}")

    visited = harvest_states(env, pol)
    states = stratified_sample(visited, N_STATES, env.config.T)
    print(f"# harvested {len(visited)} base-policy states, sampled {len(states)}")

    agent = RecordingRolloutAgent(
        rollout_policy=pol, env=env, n_rollouts=N_FULL, seed=0,
        rollout_horizon=ROLLOUT_HORIZON, action_threshold=0.5,
        initial_action="policy", selection="adaptive",
        p_threshold=P_THRESHOLD, min_rollouts=MIN_ROLLOUTS, max_rollouts=N_FULL,
        rollout_batch=BATCH,
    )

    exact = 0
    pa_disagree, frac_devs, budget_fracs = [], [], []
    sims_fixed_tot = sims_adp_tot = phys_tot = 0
    t0 = time.perf_counter()

    for k, (t, s) in enumerate(states):
        agent.new_state()
        a_opt = agent.fixed_act(s)
        fixed_sims = agent._fixed_evals * N_FULL
        a_adp = agent.adaptive_act(s)
        adp_sims = int(sum(agent._adp_hw.values()))

        q_opt = agent.q_full(s, a_opt)
        q_adp = agent.q_full(s, a_adp)
        frac = (q_adp - q_opt) / max(abs(q_opt), 1e-9)

        n_diff = int(np.sum(a_opt != a_adp))
        exact += int(n_diff == 0)
        pa_disagree.append(n_diff)
        frac_devs.append(frac)
        budget_fracs.append(adp_sims / max(1, fixed_sims))
        sims_fixed_tot += fixed_sims
        sims_adp_tot += adp_sims
        phys_tot += agent._phys

        if (k + 1) % 10 == 0:
            print(f"  [{k+1:>3}/{len(states)}] epoch={t:>3} match={n_diff==0} "
                  f"frac_dev={frac:+.3%} fixed_sims={fixed_sims:>5} "
                  f"adp_sims={adp_sims:>4} elapsed={time.perf_counter()-t0:6.1f}s")

    fd = np.array(frac_devs); pa = np.array(pa_disagree); bf = np.array(budget_fracs)
    N = len(states); na = env.config.n_assets

    print("\n" + "=" * 64)
    print("RESULTS")
    print("=" * 64)
    print(f"states evaluated            : {N}")
    print(f"exact action match          : {exact}/{N} ({exact/N:.1%})")
    print(f"per-asset agreement         : {1 - pa.sum()/(na*N):.2%} "
          f"({pa.sum()} disagreeing asset-decisions of {na*N})")
    print(f"  states 0 / 1 / >=2 diff    : {int(np.sum(pa==0))} / "
          f"{int(np.sum(pa==1))} / {int(np.sum(pa>=2))}   (max {int(pa.max())})")
    print("-" * 64)
    print("fractional cost deviation (Q_adp - Q_opt)/|Q_opt|, full-budget Q:")
    print(f"  mean / median             : {fd.mean():+.4%} / {np.median(fd):+.4%}")
    print(f"  p90 / p99 / max           : {np.percentile(fd,90):+.4%} / "
          f"{np.percentile(fd,99):+.4%} / {fd.max():+.4%}")
    print(f"  within 0.1% / 1% of opt   : {np.mean(fd<=0.001):.1%} / "
          f"{np.mean(fd<=0.01):.1%}")
    print(f"  strictly worse / better   : {int(np.sum(fd>1e-9))} / "
          f"{int(np.sum(fd<-1e-9))}")
    print("-" * 64)
    print("ROLLOUTS SAVED:")
    print(f"  fixed logical sims (total): {sims_fixed_tot:,}")
    print(f"  adaptive logical sims     : {sims_adp_tot:,}")
    print(f"  ROLLOUTS SAVED            : {1 - sims_adp_tot/max(1,sims_fixed_tot):.1%}")
    print(f"  per-state budget used     : mean {bf.mean():.1%} "
          f"median {np.median(bf):.1%} max {bf.max():.1%}")
    print(f"  physical rollouts run     : {phys_tot:,} "
          f"(vs {sims_fixed_tot + sims_adp_tot:,} if both run separately)")
    print(f"total wallclock             : {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
