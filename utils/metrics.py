"""Metrics computation utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd


def discounted_cost(costs: list[float], gamma: float) -> float:
    """Sum of γ^t * cost_t."""
    return float(sum(gamma ** t * c for t, c in enumerate(costs)))


def cvar(costs, alpha: float = 0.1) -> float:
    """CVaR / expected shortfall at level ``alpha``: mean of the worst (upper) ``alpha``
    fraction of per-episode costs. Cost is a loss, so the *right* tail is the bad tail.
    ``alpha=0.1`` → mean of the worst 10% of episodes."""
    a = np.sort(np.asarray(costs, dtype=float))          # ascending
    if a.size == 0:
        return float('nan')
    k = max(1, int(np.ceil(alpha * a.size)))
    return float(a[-k:].mean())


def cost_summary(costs, cvar_alpha: float = 0.1) -> dict:
    """Distributional summary of per-episode costs — mean plus the tail statistics that
    the sf15/sf20 analysis showed matter (median wins vs mean/tail losses). Keys:
    n, mean, p50, p90, cvar (worst-``cvar_alpha`` mean), std, cv, max."""
    a = np.asarray(costs, dtype=float)
    if a.size == 0:
        return {'n': 0, 'mean': float('nan'), 'p50': float('nan'), 'p90': float('nan'),
                'cvar': float('nan'), 'std': float('nan'), 'cv': float('nan'), 'max': float('nan')}
    mean = float(a.mean())
    return {
        'n': int(a.size),
        'mean': mean,
        'p50': float(np.percentile(a, 50)),
        'p90': float(np.percentile(a, 90)),
        'cvar': cvar(a, cvar_alpha),
        'std': float(a.std()),
        'cv': float(a.std() / mean) if mean else float('nan'),
        'max': float(a.max()),
    }


def per_asset_stats(episodes: list) -> pd.DataFrame:
    """
    Given a list of episode trajectories, return DataFrame with per-asset stats:
    mean condition, renovation frequency, failure rate, mean cost contribution.

    episodes: list of episode dicts, each with list of step dicts.
    Each step dict has keys: 't', 'state', 'action', 'cost'.
    """
    if not episodes:
        return pd.DataFrame()

    # Infer n_assets from first episode
    n_assets = len(episodes[0][0]['state'].d)
    records = []

    for ep in episodes:
        for step in ep:
            s = step['state']
            a = step['action']
            for i in range(n_assets):
                records.append({
                    'asset': i,
                    'condition': s.d[i],
                    'renovating': int(s.h[i] > 0),
                    'failed': int(s.d[i] >= 1.0),
                    'action': int(a[i]),
                })

    df = pd.DataFrame(records)
    stats = df.groupby('asset').agg(
        mean_condition=('condition', 'mean'),
        renovation_frequency=('renovating', 'mean'),
        failure_rate=('failed', 'mean'),
    ).reset_index()

    return stats
