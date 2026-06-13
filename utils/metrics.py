"""Metrics computation utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd


def discounted_cost(costs: list[float], gamma: float) -> float:
    """Sum of γ^t * cost_t."""
    return float(sum(gamma ** t * c for t, c in enumerate(costs)))


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
