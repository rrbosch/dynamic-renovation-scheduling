"""Hyperparameter grid search over a sweep spec JSON.

Usage:
    python experiments/hparam_search.py configs/sweeps/my_sweep.json
    python experiments/hparam_search.py configs/sweeps/my_sweep.json --workers 2

Sweep spec format (configs/sweeps/<name>.json):
    {
      "base_config": "configs/exp1_dqn_localsearch.json",
      "n_workers": 4,
      "search_space": {
        "training.update_interval": [5, 10, 20],
        "training.buffer_capacity": [50000, 200000],
        "agent.extra.threshold": [0.6, 0.7, 0.8]
      }
    }
"""
from __future__ import annotations

import copy
import itertools
import json
import multiprocessing as mp
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_nested(d: dict, dotted_key: str, value) -> None:
    """Set a value in a nested dict using a dot-notation key (in-place)."""
    parts = dotted_key.split('.')
    node = d
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def _make_run_name(sweep_name: str, combo: dict) -> str:
    """Build a readable unique trial id from parameter overrides.

    Format: ``param-name=value__param-name=value``
    - hyphens replace underscores in parameter names
    - ``=`` separates name from value
    - ``__`` separates parameters
    Example: ``threshold=0.7__pace-threshold=0.05``
    """
    parts = []
    for key, val in combo.items():
        short_key = key.split('.')[-1].replace('_', '-')
        # Format floats without trailing zeros
        if isinstance(val, float) and val == int(val):
            val_str = str(int(val))
        else:
            val_str = str(val)
        parts.append(f"{short_key}={val_str}")
    trial_id = '__'.join(parts)
    return f"sweep_{sweep_name}/{trial_id}"


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _run_trial(args: tuple) -> dict:
    """
    Worker: apply overrides to base config dict, build experiment, train, evaluate.
    Returns a dict with trial_id and mean_cost.
    """
    base_dict, combo, run_name = args

    from experiments.configs import ExperimentConfig, build_experiment

    config_dict = copy.deepcopy(base_dict)
    config_dict['run_name'] = run_name

    for key, val in combo.items():
        _set_nested(config_dict, key, val)

    config = ExperimentConfig.from_json(json.dumps(config_dict))
    print(f"[Worker] Starting {run_name} (pid={os.getpid()})")
    if combo:
        print(f"  Overrides: {combo}")
    print(config.to_json())

    env, agent, trainer = build_experiment(config)
    trainer.train()

    results = trainer.evaluate()
    trainer.logger.save_episodes(results['episodes'])
    trainer.logger.save_agent(agent)

    mean_cost = results['mean_cost']
    print(f"[Worker] Done {run_name}: mean_cost={mean_cost:.2f}")
    return {'run_name': run_name, 'mean_cost': mean_cost, 'combo': combo}


# ---------------------------------------------------------------------------
# Main search driver
# ---------------------------------------------------------------------------

def run_hparam_search(spec_path: str, n_workers: int | None = None) -> None:
    """Load a sweep spec, generate the grid, run all trials, print ranked summary."""
    with open(spec_path) as f:
        spec = json.load(f)

    base_config_path = spec['base_config']
    workers = n_workers if n_workers is not None else spec.get('n_workers', 4)
    search_space: dict = spec.get('search_space', {})

    # Load base config as plain dict
    with open(base_config_path) as f:
        base_dict = json.load(f)

    sweep_name = os.path.splitext(os.path.basename(spec_path))[0]

    # Generate all grid combinations
    keys = list(search_space.keys())
    value_lists = [search_space[k] for k in keys]
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*value_lists)]

    print(f"Sweep '{sweep_name}': {len(combos)} trials, {workers} workers")
    print(f"Base config: {base_config_path}")
    if keys:
        print("Search space:")
        for k, v in search_space.items():
            print(f"  {k}: {v}")
    print()

    # Build worker args
    worker_args = []
    for combo in combos:
        run_name = _make_run_name(sweep_name, combo)
        worker_args.append((base_dict, combo, run_name))

    # Run in parallel
    with mp.Pool(processes=workers) as pool:
        trial_results = pool.map(_run_trial, worker_args)

    # Print ranked summary
    _print_summary(trial_results, sweep_name)


def _print_summary(trial_results: list[dict], sweep_name: str) -> None:
    """Print a table of all trials sorted by mean_cost (best first)."""
    sorted_results = sorted(trial_results, key=lambda r: r['mean_cost'])

    print()
    print(f"=== Sweep '{sweep_name}' results (ranked by mean_cost) ===")

    if not sorted_results:
        print("No results.")
        return

    # Determine column widths
    all_keys = list(sorted_results[0]['combo'].keys()) if sorted_results else []
    header_parts = ['rank', 'mean_cost'] + [k.split('.')[-1] for k in all_keys] + ['run_name']
    col_widths = [max(len(h), 6) for h in header_parts]

    # Collect row strings for alignment
    rows = []
    for rank, r in enumerate(sorted_results, 1):
        row = [str(rank), f"{r['mean_cost']:.4f}"]
        for k in all_keys:
            v = r['combo'][k]
            if isinstance(v, float) and v == int(v):
                row.append(str(int(v)))
            else:
                row.append(str(v))
        row.append(r['run_name'])
        rows.append(row)

    # Update column widths based on data
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(cell))
            else:
                col_widths.append(len(cell))

    # Print header
    header = '  '.join(h.ljust(col_widths[i]) for i, h in enumerate(header_parts))
    sep = '  '.join('-' * col_widths[i] for i in range(len(header_parts)))
    print(header)
    print(sep)
    for row in rows:
        print('  '.join(cell.ljust(col_widths[i]) for i, cell in enumerate(row)))
    print()
    print(f"Best: {sorted_results[0]['run_name']}  (mean_cost={sorted_results[0]['mean_cost']:.4f})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Hyperparameter grid search')
    parser.add_argument('spec', help='Path to sweep spec JSON (configs/sweeps/<name>.json)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (overrides spec n_workers)')
    args = parser.parse_args()

    run_hparam_search(args.spec, n_workers=args.workers)
