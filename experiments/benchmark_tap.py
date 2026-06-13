"""Benchmark FastTAP on 100 randomised capacity vectors."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from env.network import load_sioux_falls
from env.tap import FastTAP

ETA_REN = 0.05
N_SOLVES = 100
N_ASSETS = 20
RNG_SEED = 42


def generate_capacity_vectors(net, rng: np.random.Generator, n: int) -> list[np.ndarray]:
    """Generate n unique capacity vectors with 2-6 asset edges reduced."""
    caps = []
    seen = set()
    while len(caps) < n:
        cap = net.nominal_capacities.copy()
        k = rng.integers(2, 7)  # 2 to 6 inclusive
        chosen = rng.choice(net.asset_indices, size=k, replace=False)
        cap[chosen] *= ETA_REN
        key = cap.tobytes()
        if key not in seen:
            seen.add(key)
            caps.append(cap)
    return caps


def time_backend(tap, capacity_vectors: list[np.ndarray]) -> list[float]:
    """Run all solves and return per-solve wall times in seconds."""
    times = []
    for cap in capacity_vectors:
        t0 = time.perf_counter()
        tap.solve(cap)
        times.append(time.perf_counter() - t0)
    return times


def print_stats(label: str, times: list[float]) -> None:
    arr = np.array(times)
    print(f"  {label:<20s}  mean={arr.mean()*1e3:7.1f}ms  "
          f"min={arr.min()*1e3:7.1f}ms  max={arr.max()*1e3:7.1f}ms  "
          f"total={arr.sum():.2f}s")


def main():
    rng = np.random.default_rng(RNG_SEED)

    print(f"Loading Sioux Falls (n_assets={N_ASSETS})...")
    net = load_sioux_falls(n_assets=N_ASSETS)

    print(f"Generating {N_SOLVES} unique capacity vectors...")
    cap_vectors = generate_capacity_vectors(net, rng, N_SOLVES)

    print("\nInitialising FastTAP (triggers numba JIT compilation)...")
    fast = FastTAP(net)
    fast._cache._d.clear()
    fast._last_flows = None

    print(f"Timing FastTAP ({N_SOLVES} solves)...")
    fast_times = time_backend(fast, cap_vectors)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print_stats("FastTAP", fast_times)
    print("=" * 70)


if __name__ == '__main__':
    main()
