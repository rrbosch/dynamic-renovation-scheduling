"""Bidirectional-link congestion-synergy analysis for the road network.

Each asset is one bidirectional link (a road, both directions). When two links
are renovated at the same time the extra congestion can be *super-additive*: the
realized travel-time increase of closing both exceeds the sum of closing each
alone (e.g. the two links form a cut around a node/region). Those are exactly the
roads where bunching maintenance is catastrophic and *staggering* (anticipation)
pays — so an instance that makes high-synergy links fail together creates a large
gap between a foresighted policy (clairvoyant) and a reactive single-threshold one.

For every bidirectional link i let

    dT_i  = total network travel time with link i renovated (both directions at
            eta_ren) minus the nominal-capacity baseline,
    dT_ij = the same with links i and j both renovated.

The synergy of a pair is the percentage by which the joint impact exceeds the
naive sum:

    synergy(i, j) = (dT_ij - (dT_i + dT_j)) / (dT_i + dT_j) * 100 %.

Cost constants (vot, traffic_cost_factor, dt) cancel in the ratio, so we work in
raw vehicle-hours. Run:

    python experiments/link_synergy.py            # print analysis
    python experiments/link_synergy.py --save out.npz   # also save matrices
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.network import load_sioux_falls
from env.tap import make_tap

ETA_REN = 0.05


def _travel_time(tap, net, caps: np.ndarray) -> float:
    flows = tap.solve(caps)
    tt = net.free_flow_tt * (1.0 + net.bpr_beta * (flows / np.maximum(caps, 1e-9)) ** net.bpr_nu)
    return float(np.sum(flows * tt))


def compute_synergy(network: str = "sioux_falls", eta_ren: float = ETA_REN):
    """Return (dT, synergy, dTij, link_edges) for all bidirectional links."""
    net = load_sioux_falls(n_assets=1)            # only need topology + link_edges
    links = net.link_edges                        # (L, 2)
    L = len(links)
    tap = make_tap(net, backend="fast")
    nom = net.nominal_capacities
    base = _travel_time(tap, net, nom)

    def caps_with(link_ids):
        c = nom.copy()
        for li in link_ids:
            e = links[li]
            c[e] = nom[e] * eta_ren
        return c

    dT = np.array([_travel_time(tap, net, caps_with([i])) - base for i in range(L)])
    syn = np.zeros((L, L))
    dTij = np.zeros((L, L))
    for i in range(L):
        for j in range(i + 1, L):
            tij = _travel_time(tap, net, caps_with([i, j])) - base
            dTij[i, j] = dTij[j, i] = tij
            denom = dT[i] + dT[j]
            s = (tij - denom) / denom * 100.0 if denom > 1e-9 else 0.0
            syn[i, j] = syn[j, i] = s
    return dT, syn, dTij, links, net


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--save", default=None, help="optional .npz path to save matrices")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    dT, syn, dTij, links, net = compute_synergy()
    L = len(links)

    def road(i):
        a, b = net.edge_list[links[i, 0]]
        return f"{a + 1}<->{b + 1}"

    order = np.argsort(-dT)
    print(f"=== Per-link individual congestion impact dT_i (top {args.top} of {L}) ===")
    for r in order[:args.top]:
        print(f"link {r:2d}  road {road(r):>8}  dT={dT[r]:12.1f}")

    sig = dT > 0.02 * dT.max()
    pairs = sorted(
        ((syn[i, j], i, j) for i in range(L) for j in range(i + 1, L) if sig[i] and sig[j]),
        reverse=True)
    print(f"\n=== Top {args.top} super-additive (cut) pairs ===")
    for s, i, j in pairs[:args.top]:
        print(f"  synergy={s:8.1f}%  link{i:2d}({road(i)}) & link{j:2d}({road(j)})  "
              f"dTij={dTij[i, j]:12.1f}")

    if args.save:
        np.savez(args.save, dT=dT, synergy=syn, dTij=dTij, link_edges=links)
        print(f"\nsaved -> {args.save}")


if __name__ == "__main__":
    main()
