"""Build a congestion-synergy instance with a CONTINUOUS (sorted-uniform) d_init.

Instead of hand-set d_init "waves", this:
  1. computes the 38x38 bidirectional-link congestion-synergy matrix (link_synergy),
  2. greedily selects ``n`` links that are mutually highly synergistic,
  3. orders them on a maximum-synergy Hamiltonian path (so each link's path-neighbours
     are its strongest synergy partners),
  4. draws d_init ~ Uniform(lo, hi), sorts it, and assigns it along the path.

Because degradation is homogeneous (e_fail_cv=0, alpha0_sigma=0) the sorted-d_init order
IS the failure-time order, so assets that fail at adjacent times are the congestion-coupled
ones: a single-threshold reactive must bunch a catastrophic cut while a clairvoyant staggers
it — but the failure timeline is now a smooth continuous spread (no visible cohorts).

``lo`` is auto-chosen so every asset has > ``fail_target`` probability of failing within the
horizon; ``lengths_mean`` is sized so the portfolio sustains > ``target_avgsim`` simultaneous
renovations. Fully reproducible from the CLI (seeded; no post-hoc JSON edits).

Example:
    python experiments/build_synergy_instance.py --n 15 --seed 0 \
        --output instances/instance_sf24_synergy_cont.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
from scipy.stats import gamma as gamma_dist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.link_synergy import compute_synergy
from experiments.generate_instance import generate_instance, assemble_instance, print_instance_stats
from env.tap import make_tap


# ---------------------------------------------------------------------------
# Selection + ordering on the synergy matrix
# ---------------------------------------------------------------------------

def link_v2c(net) -> np.ndarray:
    """Per-link baseline volume^2/capacity, summed over both directed edges (BPR delay
    ~ (v/c)^4, so v^2/c is a simple congestion-importance / volume proxy)."""
    tap = make_tap(net, backend="fast")
    flows = np.asarray(tap.solve(net.nominal_capacities))
    cap = net.nominal_capacities
    return np.array([sum(flows[e] ** 2 / cap[e] for e in net.link_edges[li])
                     for li in range(len(net.link_edges))])


def select_vol2cap(net, n: int) -> list[int]:
    """Select the n links with the highest baseline V^2/C (busiest roads)."""
    return list(np.argsort(-link_v2c(net))[:n].astype(int))


def greedy_select(syn: np.ndarray, n: int, v2c: np.ndarray | None = None,
                  bias: float = 0.0) -> list[int]:
    """Greedy mutual-synergy cluster: seed at the hottest pair, then repeatedly add the
    link with the highest summed synergy to the set.

    ``bias`` (0..1) tilts the selection toward high-V^2/C (busy) roads *without* abandoning
    synergy: both the seed pair and each candidate are scored by a geometric blend
    ``synergy^(1-bias) * (V^2/C)^bias`` (scores min-max normalised across the live pool).
    bias=0 ⇒ pure synergy (low-volume cuts allowed); bias=1 ⇒ pure volume; ~0.2-0.4 ⇒ a
    slight lean toward busier roads while keeping the super-additive cuts that drive the gap.
    Path-ordering still uses pure synergy, so d_init-neighbours remain strongest-synergy pairs."""
    L = len(syn)
    if n > L:
        raise ValueError(f"n={n} exceeds number of links {L}")
    use_bias = v2c is not None and bias > 0.0
    g = (v2c / v2c.max()) if use_bias and v2c.max() > 0 else None   # normalized V^2/C in [0,1]

    if not use_bias:
        i, j = np.unravel_index(int(np.argmax(syn)), syn.shape)
    else:
        W = syn * np.power(np.outer(g, g), bias)        # bias the seed too
        i, j = np.unravel_index(int(np.argmax(W)), W.shape)
    S = [int(i), int(j)]

    while len(S) < n:
        rest = [v for v in range(L) if v not in S]
        ssyn = np.array([float(np.sum(syn[v, S])) for v in rest])
        if not use_bias:
            score = ssyn
        else:
            sn = ssyn / ssyn.max() if ssyn.max() > 0 else np.zeros_like(ssyn)
            gn = g[rest]
            score = np.power(np.maximum(sn, 1e-9), 1 - bias) * np.power(np.maximum(gn, 1e-9), bias)
        S.append(rest[int(np.argmax(score))])
    return S


def order_max_synergy_path(syn: np.ndarray, S: list[int]) -> list[int]:
    """Order S as a maximum-synergy Hamiltonian path (greedy nearest-neighbour from
    the hottest internal edge, then 2-opt). Deterministic."""
    Sset = list(S)
    # start from the hottest internal edge
    best = (-np.inf, None, None)
    for a in Sset:
        for b in Sset:
            if a < b and syn[a, b] > best[0]:
                best = (syn[a, b], a, b)
    _, a, b = best
    path, used = [a, b], {a, b}
    while len(path) < len(Sset):
        rem = [v for v in Sset if v not in used]
        candL = max(rem, key=lambda v: syn[path[0], v])
        candR = max(rem, key=lambda v: syn[path[-1], v])
        if syn[path[0], candL] >= syn[path[-1], candR]:
            path.insert(0, candL); used.add(candL)
        else:
            path.append(candR); used.add(candR)

    def total(p):
        return float(sum(syn[p[k], p[k + 1]] for k in range(len(p) - 1)))

    improved = True
    while improved:
        improved = False
        for i in range(len(path) - 1):
            for j in range(i + 2, len(path)):
                cand = path[:i + 1] + path[i + 1:j + 1][::-1] + path[j + 1:]
                if total(cand) > total(path) + 1e-9:
                    path = cand; improved = True
    return path


# ---------------------------------------------------------------------------
# Auto-size renovation length for a target avg simultaneous renovations
# ---------------------------------------------------------------------------

def proxy_avgsim(d_init, alpha0, beta, e_ren_years, dt, years, tail, K, seed) -> float:
    """Mean simultaneous renovations under a renovate-at-failure policy — the intrinsic
    maintenance *workload* (agent-independent): degrade each asset from d_init via Gamma
    increments until d>=1, occupy e_ren years of renovation, average the under-renovation
    count over the full eval (T + tail). Captures both non-failure and end-of-horizon
    truncation, so it matches the env's renovate-at-failure measure."""
    rng = np.random.default_rng(seed)
    d_init = np.asarray(d_init, float); alpha0 = np.asarray(alpha0, float); beta = np.asarray(beta, float)
    n = len(d_init)
    H = int(round((years + tail) / dt))
    e_ep = max(1, int(round(e_ren_years / dt)))
    shape = alpha0 * dt
    scale = 1.0 / beta
    tot = 0.0
    for _ in range(K):
        inc = rng.gamma(shape=shape[None, :], scale=scale[None, :], size=(H, n))
        cum = d_init[None, :] + np.cumsum(inc, axis=0)
        failed = cum >= 1.0
        ft = np.where(failed.any(axis=0), failed.argmax(axis=0), H)   # first-failure epoch
        cnt = np.zeros(H)
        for i in range(n):
            if ft[i] < H:
                cnt[ft[i]:ft[i] + e_ep] += 1.0
        tot += cnt.mean()
    return tot / K


def calibrate_e_ren(target, d_init, alpha0, beta, dt, years, tail,
                    K=300, seed=0, lo=2.0, hi=45.0) -> float:
    """Binary-search renovation duration (years) so the workload avg-sim == target."""
    for _ in range(34):
        mid = 0.5 * (lo + hi)
        if proxy_avgsim(d_init, alpha0, beta, mid, dt, years, tail, K, seed) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# d_init range from the fail-in-horizon target
# ---------------------------------------------------------------------------

def auto_lo(fail_target: float, years: float, alpha0_mean: float, e_fail_mean: float) -> float:
    """Smallest d_init s.t. P(fail within horizon) >= fail_target.

    Cumulative degradation over the horizon ~ Gamma(shape=years*alpha0, rate=alpha0*e_fail)
    (sum of per-epoch Gamma increments). P(fail | d_init) = P(cumulative >= 1 - d_init), so
    lo = 1 - ppf(1 - fail_target)."""
    shape = years * alpha0_mean
    scale = 1.0 / (alpha0_mean * e_fail_mean)
    q = gamma_dist.ppf(1.0 - fail_target, a=shape, scale=scale)
    return float(max(0.02, 1.0 - q))


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', required=True)
    p.add_argument('--n', type=int, default=15, help='number of assets (links).')
    p.add_argument('--select-rule', choices=['synergy', 'vol2cap'], default='synergy',
                   help="how to pick the n links: 'synergy' = greedy mutual-synergy cluster; "
                        "'vol2cap' = top-n by baseline volume^2/capacity (busiest roads).")
    p.add_argument('--v2c-bias', type=float, default=0.0,
                   help="(synergy rule only) 0..1 geometric tilt of the selection toward high-V^2/C "
                        "roads: score = synergy^(1-bias) * (V^2/C)^bias. 0=pure synergy, "
                        "~0.3=slight volume lean, 1=pure volume.")
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--network', default='sioux_falls')
    # d_init range
    p.add_argument('--fail-target', type=float, default=0.82,
                   help='min P(asset fails within horizon); sets lo.')
    p.add_argument('--hi', type=float, default=0.85, help='upper end of the d_init draw.')
    p.add_argument('--lo', type=float, default=None, help='override auto lo.')
    # avg-sim sizing
    p.add_argument('--target-avgsim', type=float, default=3.3,
                   help='target avg simultaneous renovations (full-eval) -> sizes lengths_mean.')
    p.add_argument('--lengths-mean-m', type=float, default=None, help='override auto length.')
    p.add_argument('--lengths-cv', type=float, default=0.3)
    # horizon / physics
    p.add_argument('--years', type=float, default=80.0)
    p.add_argument('--dt', type=float, default=0.5)
    p.add_argument('--t-tail-years', type=float, default=15.0)
    p.add_argument('--e-fail-mean', type=float, default=100.0)
    p.add_argument('--alpha0-mean', type=float, default=0.8)
    p.add_argument('--gamma', type=float, default=0.97)
    p.add_argument('--d-fail', type=float, default=1.0)
    p.add_argument('--eta-ren', type=float, default=0.05)
    p.add_argument('--eta-load', type=float, default=0.5)
    p.add_argument('--restrict-degrad-multiplier', type=float, default=0.9)
    p.add_argument('--delta-repair', type=float, default=0.1)
    p.add_argument('--vot', type=float, default=10.76)
    p.add_argument('--traffic-cost-factor', type=float, default=1.0)
    p.add_argument('--risk-base', type=float, default=10_000.0)
    return p.parse_args()


def main():
    a = _parse_args()
    rng = np.random.default_rng(a.seed)

    # 1. synergy matrix
    print("computing 38x38 congestion-synergy matrix ...", flush=True)
    dT, syn, dTij, links, net = compute_synergy(network=a.network, eta_ren=a.eta_ren)

    # 2-3. select + order
    if a.select_rule == 'vol2cap':
        S = select_vol2cap(net, a.n)
    else:
        v2c = link_v2c(net) if a.v2c_bias > 0 else None
        S = greedy_select(syn, a.n, v2c=v2c, bias=a.v2c_bias)
    path = order_max_synergy_path(syn, S)   # d_init-neighbours = strongest synergy partners

    # 4. d_init range + draw + assign along the path
    lo = a.lo if a.lo is not None else auto_lo(a.fail_target, a.years, a.alpha0_mean, a.e_fail_mean)
    if not (0.0 < lo < a.hi):
        raise ValueError(f"bad d_init range: lo={lo:.3f}, hi={a.hi}")
    d_sorted = np.sort(rng.uniform(lo, a.hi, a.n))            # ascending
    d_init_list = [float(d_sorted[k]) for k in range(a.n)]    # position k -> path[k]

    # 6. auto-size renovation length so the renovate-at-failure WORKLOAD avg-sim matches
    #    --target-avgsim, given n, e_fail, horizon and the drawn d_init (deterministic
    #    binary search on e_ren; lengths only affect renovation duration, not failure timing).
    if a.lengths_mean_m is not None:
        lengths_mean = a.lengths_mean_m
        e_ren_years = (10.0 + lengths_mean / 5.0) / 52.0
    else:
        alpha0_arr = np.full(a.n, a.alpha0_mean)
        beta_arr = np.full(a.n, a.alpha0_mean * a.e_fail_mean)
        e_ren_years = calibrate_e_ren(
            a.target_avgsim, np.array(d_init_list), alpha0_arr, beta_arr,
            a.dt, a.years, a.t_tail_years, K=300, seed=a.seed)
        lengths_mean = 5.0 * (e_ren_years * 52.0 - 10.0)
    workload_avgsim = proxy_avgsim(
        np.array(d_init_list), np.full(a.n, a.alpha0_mean),
        np.full(a.n, a.alpha0_mean * a.e_fail_mean), e_ren_years,
        a.dt, a.years, a.t_tail_years, K=400, seed=a.seed + 1)

    # 7. core per-asset arrays (homogeneous degradation; explicit asset_links + d_init)
    core = generate_instance(
        a.n, a.network, a.seed,
        lengths_mean_m=lengths_mean, lengths_cv=a.lengths_cv,
        alpha0_mean=a.alpha0_mean, alpha0_sigma=0.0,
        e_fail_mean=a.e_fail_mean, e_fail_cv=0.0, ren_noise_cv=0.2,
        restrict_degrad_multiplier=a.restrict_degrad_multiplier,
        asset_links=path, d_init_override=d_init_list,
    )
    args_ns = types.SimpleNamespace(
        years=a.years, dt=a.dt, t_tail_years=a.t_tail_years, gamma=a.gamma,
        d_fail=a.d_fail, eta_ren=a.eta_ren, eta_load=a.eta_load,
        restrict_degrad_multiplier=a.restrict_degrad_multiplier, delta_repair=a.delta_repair,
        vot=a.vot, traffic_cost_factor=a.traffic_cost_factor, risk_base=a.risk_base,
        lengths_cv=a.lengths_cv, avg_ongoing_projects=None,
        alpha0_mean=a.alpha0_mean, alpha0_sigma=0.0,
        e_fail_mean=a.e_fail_mean, e_fail_cv=0.0, ren_noise_cv=0.2,
    )
    inst, _ = assemble_instance(core, args_ns)

    os.makedirs(os.path.dirname(a.output) or '.', exist_ok=True)
    with open(a.output, 'w') as f:
        json.dump(inst, f, indent=2)

    # ---- diagnostics ----
    def road(li):
        u, v = net.edge_list[links[li, 0]]
        return f"{u+1}-{v+1}"
    adj_syn = [syn[path[k], path[k + 1]] for k in range(len(path) - 1)]
    rng2 = np.random.default_rng(123)
    rand_pairs = [syn[i, j] for i, j in
                  zip(rng2.integers(0, len(syn), 2000), rng2.integers(0, len(syn), 2000)) if i != j]
    print(f"\nGenerated -> {a.output}")
    print(f"  n={a.n}  d_init range=[{lo:.3f}, {a.hi:.3f}]  lengths_mean={lengths_mean:.0f} m "
          f"(e_ren~{e_ren_years:.1f} yr)")
    print(f"  target workload avg-sim={a.target_avgsim:.2f} -> calibrated workload avg-sim={workload_avgsim:.2f}")
    print(f"  path order (link: road @ d_init):")
    for k, li in enumerate(path):
        print(f"    {li:2d}  {road(li):>7}  d_init={d_init_list[k]:.3f}")
    print(f"  mean synergy of d_init-ADJACENT pairs : {np.mean(adj_syn):8.1f}%")
    print(f"  mean synergy of RANDOM link pairs     : {np.mean(rand_pairs):8.1f}%")
    print_instance_stats(inst)


if __name__ == '__main__':
    main()
