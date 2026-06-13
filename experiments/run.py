"""CLI entry point for running experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# Ensure project root is on path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))


def _is_sweep_spec(d: dict) -> bool:
    return "base_config" in d and "search_space" in d


def _run_status(cfg: dict) -> str:
    """Return 'not started' | 'in progress' | 'finished' for a non-sweep config."""
    run_name = cfg.get('run_name')
    if not run_name:
        return 'not started'
    run_dir = _project_root / 'results' / run_name
    if not run_dir.exists():
        return 'not started'
    for meta_path in run_dir.glob('checkpoints/**/metadata.json'):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get('complete'):
                return 'finished'
        except (json.JSONDecodeError, OSError):
            return 'corrupted'
    return 'in progress'


def _pick_config() -> list[str]:
    """Interactively list all JSON configs under configs/ and return the chosen path(s)."""
    configs_dir = _project_root / "configs"
    candidates = sorted(configs_dir.rglob("*.json")) if configs_dir.exists() else []
    if not candidates:
        print("No config files found under 'configs/'. Pass --config <path> manually.")
        sys.exit(1)
    print("Available configs:")
    for i, p in enumerate(candidates):
        rel = p.relative_to(_project_root)
        with open(p) as f:
            try:
                d = json.load(f)
            except json.JSONDecodeError:
                d = {}
        if _is_sweep_spec(d):
            suffix = "   [sweep]"
        else:
            status = _run_status(d)
            suffix = f"   [{status}]"
        print(f"  [{i}] {rel}{suffix}")
    print("  [all] Run all non-sweep, non-finished configs in sequence")
    choice = input("Select a config (number, comma-separated numbers, or 'all'): ").strip().lower()
    if choice == 'all':
        return ['__all__']
    if ',' in choice:
        indices = [int(x.strip()) for x in choice.split(',')]
        return [str(candidates[i]) for i in indices]
    return [str(candidates[int(choice)])]


def _compute_config_hash(config_dict: dict) -> str:
    """SHA-256 (first 16 hex chars) of config JSON with 'seed' excluded."""
    d = {k: v for k, v in config_dict.items() if k != 'seed'}
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def _find_incomplete_optuna_checkpoint(config_hash: str) -> str | None:
    """
    Scan results/*/checkpoints/optuna/metadata.json for an incomplete Optuna study
    with a matching config_hash.  Returns the checkpoint directory path or None.
    """
    results_dir = _project_root / 'results'
    if not results_dir.exists():
        return None
    for meta_path in results_dir.rglob('checkpoints/optuna/metadata.json'):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception:
            continue
        if meta.get('config_hash') != config_hash:
            continue
        if meta.get('complete'):
            continue
        return str(meta_path.parent)
    return None


def _find_incomplete_checkpoint(config_hash: str) -> str | None:
    """
    Scan results/*/checkpoints/ep_*/metadata.json for a matching config_hash
    without 'complete': true.  Returns the path of the latest matching checkpoint
    directory, or None.
    """
    results_dir = _project_root / 'results'
    if not results_dir.exists():
        return None
    best = None
    best_ep = -1
    for meta_path in results_dir.rglob('checkpoints/ep_*/metadata.json'):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception:
            continue
        if meta.get('config_hash') != config_hash:
            continue
        if meta.get('complete'):
            continue
        ep = meta.get('episode', -1)
        if ep > best_ep:
            best_ep = ep
            best = str(meta_path.parent)
    return best


def _run_one(config_path: str, args, auto_resume: bool = False) -> None:
    """Run a single experiment config. auto_resume=True skips interactive prompts."""
    with open(config_path) as f:
        raw = json.load(f)

    if _is_sweep_spec(raw):
        from experiments.hparam_search import run_hparam_search
        run_hparam_search(config_path, n_workers=args.workers)
        return

    from experiments.configs import ExperimentConfig, build_experiment

    config = ExperimentConfig.from_json(json.dumps(raw))

    # Override run name if specified
    if args.run_name is not None:
        import dataclasses
        config = dataclasses.replace(config, run_name=args.run_name)

    # Determine checkpoint to resume from
    checkpoint_dir = args.resume
    config_hash = _compute_config_hash(json.loads(config.to_json()))

    if checkpoint_dir is None:
        # Auto-detect an incomplete run with the same config
        found = _find_incomplete_checkpoint(config_hash)
        if found:
            checkpoint_dir = found
            print(f"Resuming from checkpoint: {checkpoint_dir}")

    print(f"Starting experiment: {config.run_name}")
    print(config.to_json())

    env, agent, trainer = build_experiment(config)

    print(f"Environment: {env.config.n_assets} assets, T={env.config.T}")

    start_ep = 0
    already_elapsed = 0.0

    # Optuna trainer: auto-detect incomplete study (trainer self-resumes via
    # _load_optuna_checkpoint; we only need to pass already_elapsed)
    from experiments.optuna_heuristic_search import OptunaHeuristicTrainer
    if isinstance(trainer, OptunaHeuristicTrainer) and checkpoint_dir is None:
        optuna_ckpt = _find_incomplete_optuna_checkpoint(config_hash)
        if optuna_ckpt:
            try:
                with open(os.path.join(optuna_ckpt, 'metadata.json')) as f:
                    ometa = json.load(f)
                already_elapsed = float(ometa.get('elapsed_seconds', 0.0))
            except Exception:
                pass

    if checkpoint_dir is not None:
        print(f"Resuming from checkpoint: {checkpoint_dir}")
        start_ep = trainer.load_checkpoint(checkpoint_dir)
        # Recover previously elapsed time from metadata to honour remaining time_budget
        try:
            with open(os.path.join(checkpoint_dir, 'metadata.json')) as f:
                meta = json.load(f)
            already_elapsed = float(meta.get('elapsed_seconds', 0.0))
        except Exception:
            pass

    print("Training...")
    trainer.train(start_ep=start_ep, already_elapsed=already_elapsed)

    # Save final evaluation and agent (episodes saved incrementally during evaluate)
    results = trainer.evaluate(n_episodes=50, resume=True, save_episodes=True)
    trainer.logger.save_agent(agent)
    if results['episodes'] and hasattr(agent, 'value_fn') and agent.value_fn._fitted:
        trainer.logger.save_buffer_predictions(trainer.buffer, agent)
        trainer.logger.save_eval_predictions(results['episodes'], agent, env)
    print(f"\nFinal evaluation: mean_cost={results['mean_cost']:.2f} ± {results['std_cost']:.2f}")
    print(f"Results saved to: results/{config.run_name}/")


def main():
    from utils.logging import enable_timestamped_stdout
    enable_timestamped_stdout()

    parser = argparse.ArgumentParser(description='Run RL infrastructure maintenance experiment')
    parser.add_argument('--config', default=None, help='Path to ExperimentConfig JSON file')
    parser.add_argument('--run-name', default=None, help='Override run name from config')
    parser.add_argument('--workers', type=int, default=None,
                        help='Parallel workers for sweep (overrides spec n_workers)')
    parser.add_argument('--resume', default=None, metavar='CHECKPOINT_DIR',
                        help='Resume training from an explicit checkpoint directory')
    args = parser.parse_args()

    if args.config is None:
        selected = _pick_config()
        if selected == ['__all__']:
            args.config = '__all__'
        elif len(selected) == 1:
            args.config = selected[0]
        else:
            for config_path in selected:
                print(f"\n{'='*60}")
                print(f"Config: {Path(config_path).relative_to(_project_root)}")
                print('='*60)
                _run_one(config_path, args, auto_resume=True)
            return

    if args.config == '__all__':
        configs_dir = _project_root / "configs"
        candidates = sorted(configs_dir.rglob("*.json")) if configs_dir.exists() else []
        to_run = []
        for p in candidates:
            with open(p) as f:
                try:
                    d = json.load(f)
                except json.JSONDecodeError:
                    d = {}
            if _is_sweep_spec(d):
                print(f"Skipping sweep spec: {p.relative_to(_project_root)}")
                continue
            status = _run_status(d)
            if status == 'finished':
                print(f"Skipping finished:   {p.relative_to(_project_root)}")
                continue
            to_run.append(str(p))
        print(f"\nRunning {len(to_run)} config(s) in sequence.\n")
        for config_path in to_run:
            print(f"\n{'='*60}")
            print(f"Config: {Path(config_path).relative_to(_project_root)}")
            print('='*60)
            _run_one(config_path, args, auto_resume=True)
        return

    _run_one(args.config, args, auto_resume=False)


if __name__ == '__main__':
    main()
