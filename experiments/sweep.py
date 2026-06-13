"""Parallel experiment sweep over multiple config files."""
from __future__ import annotations

import os
import sys
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run_one(config_path: str) -> None:
    """Worker function: load config, build experiment, train, save."""
    from experiments.configs import ExperimentConfig, build_experiment

    config = ExperimentConfig.from_file(config_path)
    print(f"[Worker] Starting {config.run_name} (pid={os.getpid()})")
    print(config.to_json())

    env, agent, trainer = build_experiment(config)
    trainer.train()

    results = trainer.evaluate()
    trainer.logger.save_episodes(results['episodes'])
    trainer.logger.save_agent(agent)
    print(f"[Worker] Done {config.run_name}: mean_cost={results['mean_cost']:.2f}")


def run_sweep(config_paths: list[str], n_workers: int = 4) -> None:
    """
    Run one experiment per config file in parallel using multiprocessing.Pool.
    """
    print(f"Sweep: {len(config_paths)} configs, {n_workers} workers")
    with mp.Pool(processes=n_workers) as pool:
        pool.map(_run_one, config_paths)
    print("Sweep complete.")


if __name__ == '__main__':
    import argparse
    import glob

    parser = argparse.ArgumentParser(description='Run a sweep of experiments')
    parser.add_argument('configs', nargs='+', help='Config JSON files or glob patterns')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    args = parser.parse_args()

    paths = []
    for pattern in args.configs:
        expanded = glob.glob(pattern)
        paths.extend(expanded if expanded else [pattern])

    run_sweep(paths, n_workers=args.workers)
