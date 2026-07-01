"""Generate hpc/registry.json — flat list of experiments for HyperQueue dispatch.

Conventions for n_workers (parallelism) and replications (seeds) — including why Optuna
tuning is NOT seed-replicated — are documented in hpc/registry_conventions.md. Read that
before changing the --seeds expansion or the per-config worker counts.

Write named per-experiment registries under hpc/registries/ (one file per batch); dispatch
each with `bash hpc/submit.sh <registry> <array>` (job name derived from the filename).

Usage examples
--------------
# A named batch registry (single run per config, seed from JSON):
python hpc/generate_registry.py --configs configs/sf15_optuna_*.json \
    --output hpc/registries/sf15_0a.json

# Specific configs, 5 seeds each:
python hpc/generate_registry.py --configs configs/exp1_*.json --seeds 0 1 2 3 4 \
    --output hpc/registries/exp1.json

# Specific configs, no seed expansion:
python hpc/generate_registry.py --configs configs/exp1_reactive.json configs/exp1_paced.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from experiments.run import _run_status, _is_sweep_spec  # noqa: E402


def _base_run_name(config_path: str) -> str:
    """Extract the stem of the config file as a base run name."""
    return Path(config_path).stem


def build_registry(config_paths: list[str], seeds: list[int] | None) -> list[dict]:
    entries: list[dict] = []
    for cp in config_paths:
        try:
            with open(cp) as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"  [skip] Cannot read {cp}: {e}")
            continue

        if _is_sweep_spec(raw):
            print(f"  [skip] Sweep spec (not supported for HPC registry): {cp}")
            continue

        if seeds:
            for s in seeds:
                base = raw.get("run_name") or _base_run_name(cp)
                # Nest seeds under a per-experiment folder (results/<base>/s<seed>/)
                # so the comparison dashboard lists one entry per experiment and
                # the existing algorithm_id grouping aggregates the seeds.
                run_name = f"{base}/s{s}"
                # Check if this specific seeded run is already finished
                seeded_raw = dict(raw)
                seeded_raw["run_name"] = run_name
                if _run_status(seeded_raw) == "finished":
                    print(f"  [skip] Already finished: {run_name}")
                    continue
                entries.append({"config": Path(cp).resolve().relative_to(_project_root).as_posix(),
                                 "run_name": run_name,
                                 "seed": s})
        else:
            status = _run_status(raw)
            if status == "finished":
                print(f"  [skip] Already finished: {raw.get('run_name', cp)}")
                continue
            entries.append({"config": Path(cp).resolve().relative_to(_project_root).as_posix(),
                             "run_name": None,
                             "seed": None})
    return entries


def _interactive_pick(seeds_from_cli: list[int] | None) -> tuple[list[str], list[int] | None]:
    """Interactively select configs and seeds when no CLI args are given."""
    configs_dir = _project_root / "configs"
    candidates = sorted(configs_dir.rglob("*.json")) if configs_dir.exists() else []

    # Filter out sweep specs
    valid: list[Path] = []
    for p in candidates:
        try:
            with open(p) as f:
                d = json.load(f)
            if not _is_sweep_spec(d):
                valid.append(p)
        except (OSError, json.JSONDecodeError):
            pass

    if not valid:
        print("No config files found under configs/.")
        sys.exit(1)

    print("\nAvailable configs:")
    for i, p in enumerate(valid):
        rel = p.relative_to(_project_root)
        try:
            with open(p) as f:
                d = json.load(f)
            status = _run_status(d)
        except Exception:
            status = "?"
        print(f"  [{i:2d}] {rel}  [{status}]")

    raw = input("\nSelect configs (comma-separated indices): ").strip()
    try:
        indices = [int(x.strip()) for x in raw.split(",")]
        config_paths = [str(valid[i]) for i in indices]
    except (ValueError, IndexError) as e:
        print(f"Invalid selection: {e}")
        sys.exit(1)

    if seeds_from_cli is not None:
        return config_paths, seeds_from_cli

    start_raw = input("Starting seed: ").strip()
    n_raw = input("Number of seeds: ").strip()
    try:
        start = int(start_raw)
        n = int(n_raw)
    except ValueError as e:
        print(f"Invalid seed input: {e}")
        sys.exit(1)
    seeds = list(range(start, start + n))
    print(f"  -> seeds: {seeds}")
    return config_paths, seeds


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate hpc/registry.json")
    parser.add_argument(
        "--configs", nargs="*", default=None,
        help="Config JSON files or glob patterns. Default: all non-sweep configs under configs/",
    )
    parser.add_argument(
        "--seeds", nargs="*", type=int, default=None,
        help="Seeds to expand each config over. If omitted, seed from each config file is used.",
    )
    parser.add_argument(
        "--output", default=str(_project_root / "hpc" / "registries" / "registry.json"),
        help="Output path for the registry (default: hpc/registries/registry.json). "
             "Name it <experiment>_<appendix>.json so the job name becomes rl_<experiment>_<appendix>.",
    )
    parser.add_argument(
        "--include-finished", action="store_true",
        help="Include already-finished experiments (skip the status check)",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Append to the existing --output registry instead of overwriting it. "
             "Lets multiple invocations (e.g. different seed counts per config) build "
             "one combined registry.",
    )
    args = parser.parse_args()

    # Resolve config paths — interactive if not supplied via CLI
    if args.configs is None:
        config_paths, args.seeds = _interactive_pick(args.seeds)
    else:
        config_paths = []
        for pattern in args.configs:
            expanded = glob.glob(pattern)
            config_paths.extend(expanded if expanded else [pattern])
        config_paths = sorted(set(config_paths))

    print(f"Scanning {len(config_paths)} config file(s)...")

    if args.include_finished:
        # Monkey-patch _run_status to never skip
        import experiments.run as _run_mod
        _orig = _run_mod._run_status
        _run_mod._run_status = lambda cfg: "not started"

    entries = build_registry(config_paths, args.seeds)

    if args.include_finished:
        _run_mod._run_status = _orig  # restore

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Append mode: keep the existing entries and add the new ones after them.
    if args.append and out_path.exists():
        try:
            with open(out_path, encoding="utf-8") as f:
                existing = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"  [warn] Cannot read existing registry to append ({e}); overwriting.")
            existing = []
        entries = existing + entries

    # Force UNIX (LF) line endings regardless of platform so the registry is
    # consumed cleanly on the HPC side; newline="\n" disables Windows CRLF
    # translation. Trailing newline follows POSIX text-file convention.
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(entries, f, indent=2)
        f.write("\n")

    n = len(entries)
    print(f"\n{n} experiment(s) written to {out_path}")
    if n > 0:
        print(f"  -> submit with:  bash hpc/submit.sh {out_path.as_posix()} 0-{n - 1}")
    else:
        print("  (nothing to run)")


if __name__ == "__main__":
    main()
