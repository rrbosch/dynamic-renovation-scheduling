"""Post-mortem evaluation of training snapshots (item 6: decouple eval from training).

A learner run with `training.snapshot_interval_seconds > 0` writes lightweight agent-only
snapshots to `results/<run>/snapshots/t_<elapsed>s/` during training (see Trainer._save_snapshot).
This script reconstructs the env + agent from the config, loads each snapshot's artifacts, runs the
shared-CRN evaluation offline (so eval never competes with the training budget), and writes a
cost-vs-wallclock curve with mean + tail statistics (CVaR/P90 — item 7).

Usage:
  python experiments/evaluate_checkpoints.py --config configs/sf20_adp.json [--n-episodes 50]
      [--workers 16] [--run-name exp0/sf20_adp] [--only-final]

Outputs (under results/<run>/):
  eval_curve.csv                      one row per snapshot: elapsed, episode, mean/p50/p90/cvar/...
  snapshots/t_<E>s/eval_summary.csv   per-episode discounted cost + component split for that snapshot
"""
from __future__ import annotations
import argparse, csv, json, os, sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from experiments.configs import ExperimentConfig, build_experiment
from utils.metrics import cost_summary


def _episode_costs(episodes, gamma):
    """Per-episode discounted total + component splits from evaluate()'s step dicts."""
    rows = []
    for ep in episodes:
        D = Dt = Dm = Dr = 0.0
        for step in ep:
            t = int(step['t']); g = gamma ** t
            D += g * float(step['cost'])
            Dt += g * float(step.get('c_travel', 0.0))
            Dm += g * float(step.get('c_maint', 0.0))
            Dr += g * float(step.get('c_risk', 0.0))
        rows.append((D, Dt, Dm, Dr))
    return rows


def _find_snapshots(run_dir: Path):
    snap_root = run_dir / 'snapshots'
    if not snap_root.exists():
        return []
    snaps = []
    for d in snap_root.glob('t_*s'):
        meta_p = d / 'meta.json'
        if not (d / 'agent').exists() or not meta_p.exists():
            continue
        meta = json.loads(meta_p.read_text())
        snaps.append((float(meta.get('elapsed_seconds', 0.0)), int(meta.get('episode', 0)), d))
    return sorted(snaps, key=lambda x: x[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--n-episodes', type=int, default=50)
    ap.add_argument('--workers', type=int, default=None, help='override n_workers for eval')
    ap.add_argument('--run-name', default=None, help='override run_name from config')
    ap.add_argument('--cvar-alpha', type=float, default=0.1)
    ap.add_argument('--only-final', action='store_true', help='evaluate only the last snapshot')
    args = ap.parse_args()

    raw = json.loads(Path(args.config).read_text())
    config = ExperimentConfig.from_json(json.dumps(raw))
    if args.run_name is not None:
        import dataclasses
        config = dataclasses.replace(config, run_name=args.run_name)
    if args.workers is not None:
        config.training['n_workers'] = int(args.workers)

    env, agent, trainer = build_experiment(config)
    if not hasattr(agent, 'load') or not callable(getattr(agent, 'load')):
        print(f"Agent {type(agent).__name__} has no load(); nothing to post-mortem-evaluate.")
        return
    run_dir = Path(trainer.logger.run_dir)
    gamma = env.config.gamma

    snaps = _find_snapshots(run_dir)
    if not snaps:
        print(f"No snapshots found under {run_dir/'snapshots'}. "
              f"Was the run trained with training.snapshot_interval_seconds > 0?")
        return
    if args.only_final:
        snaps = snaps[-1:]
    print(f"{type(agent).__name__}: evaluating {len(snaps)} snapshot(s) x {args.n_episodes} CRN eps "
          f"(workers={config.training.get('n_workers', 1)}) -> {run_dir/'eval_curve.csv'}")

    curve_path = run_dir / 'eval_curve.csv'
    with open(curve_path, 'w', newline='') as cf:
        cw = csv.writer(cf)
        cw.writerow(['elapsed_seconds', 'episode', 'n', 'mean', 'p50', 'p90',
                     f'cvar{int(args.cvar_alpha*100)}', 'std', 'cv', 'max',
                     'mean_travel', 'mean_maint', 'mean_risk'])
        for elapsed, episode, snap_dir in snaps:
            agent.load(str(snap_dir / 'agent'))
            res = trainer.evaluate(n_episodes=args.n_episodes)
            rows = _episode_costs(res['episodes'], gamma)
            totals = [r[0] for r in rows]
            s = cost_summary(totals, cvar_alpha=args.cvar_alpha)
            mt = float(np.mean([r[1] for r in rows])); mm = float(np.mean([r[2] for r in rows]))
            mr = float(np.mean([r[3] for r in rows]))
            cw.writerow([elapsed, episode, s['n'], s['mean'], s['p50'], s['p90'], s['cvar'],
                         s['std'], s['cv'], s['max'], mt, mm, mr])
            cf.flush()
            # per-episode detail for this snapshot (feeds the CVaR/P90 view in the dashboard)
            with open(snap_dir / 'eval_summary.csv', 'w', newline='') as sf:
                sw = csv.writer(sf); sw.writerow(['episode', 'disc_cost', 'c_travel', 'c_maint', 'c_risk'])
                for i, r in enumerate(rows):
                    sw.writerow([i, *r])
            print(f"  t={elapsed:8.0f}s ep={episode:6d}: mean={s['mean']/1e6:8.0f}M "
                  f"p50={s['p50']/1e6:8.0f}M p90={s['p90']/1e6:8.0f}M "
                  f"cvar{int(args.cvar_alpha*100)}={s['cvar']/1e6:8.0f}M (CV {s['cv']:.2f})")
    print(f"\nWrote {curve_path}")


if __name__ == '__main__':
    main()
