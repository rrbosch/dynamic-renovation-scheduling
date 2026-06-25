"""Inject Optuna-tuned heuristic parameters (Exp 0A) into 0B agent configs (Exp 0B).

Exp 0A tunes a heuristic and writes results/<run>/best_params.json. Exp 0B agents
(ADP warmstart, MC-rollout base policy, PPO curriculum imitation target) must then
*consume* those tuned parameters. This script performs that handoff reproducibly
instead of hand-editing configs.

The consumer blocks have different schemas (see experiments/configs.py):
  * warmstart            (training.warmstart):            {"agent_type": H, "extra": {<params>}}
  * curriculum_heuristic (training.curriculum_heuristic): {"agent_type": H, "extra": {<params>}}
  * rollout_policy       (agent.extra.rollout_policy):    {"agent_type": H, <params...>}   (flat)

Usage
-----
# ADP warmstart (ALL 24 cells now carry a warmstart block) <- tuned heuristic
python experiments/apply_optuna_params.py \
    --params results/exp0/i10p_optuna_perasset/best_params.json \
    --targets "configs/i10p_adp_*.json" \
    --block warmstart

# PPO phase-0 imitation target <- tuned heuristic
python experiments/apply_optuna_params.py \
    --params results/exp0/i10p_optuna_perasset/best_params.json \
    --targets "configs/i10p_ppo_curriculum.json" \
    --block curriculum_heuristic

# MC-rollout base policy <- tuned reactive heuristic
python experiments/apply_optuna_params.py \
    --params results/i10p_optuna_reactive/best_params.json \
    --targets "configs/i10p_rollout_*.json" "configs/i10p_seq_rollout_*.json" \
    --block rollout_policy

# Per-asset works identically: the dir name '..._perasset' is mapped to
# agent_type 'reactiveperasset' and the 30 flat per-asset keys
# (repair/restrict/renovate_threshold_i) are injected verbatim into either block.
python experiments/apply_optuna_params.py \
    --params results/exp0/i10p_optuna_perasset/best_params.json \
    --targets "configs/i10p_adp_*_policy_*.json" \
    --block warmstart

Add --dry-run to preview without writing. Targets that don't already contain the
requested block are skipped (so the empty-init ADP cells are left untouched).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

# Keys written by the Optuna trainer that are NOT heuristic hyperparameters.
_META_KEYS = {"best_value"}

# Map an Optuna run/heuristic name to the agent_type the heuristic is built as.
_HEURISTIC_ALIASES = {
    "reactive": "reactive",
    "paced": "paced",
    "perasset": "reactiveperasset",
    "reactiveperasset": "reactiveperasset",
}


def _infer_heuristic(params_path: Path) -> str:
    """Infer the heuristic agent_type from a path like results/i10p_optuna_reactive/..."""
    stem = params_path.parent.name.lower()  # e.g. 'i10p_optuna_reactive'
    for key, agent_type in _HEURISTIC_ALIASES.items():
        if stem.endswith(key):
            return agent_type
    raise ValueError(
        f"Cannot infer heuristic from {params_path.parent.name!r}; pass --heuristic explicitly."
    )


def _load_params(params_path: Path) -> dict:
    with open(params_path, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if k not in _META_KEYS}


def _patch_config(cfg: dict, block: str, heuristic: str, params: dict) -> bool:
    """Patch cfg in place. Returns True if the block existed and was updated."""
    if block in ("warmstart", "curriculum_heuristic"):
        training = cfg.get("training", {})
        if block not in training:
            return False
        training[block] = {"agent_type": heuristic, "extra": dict(params)}
        return True
    if block == "rollout_policy":
        extra = cfg.get("agent", {}).get("extra", {})
        if "rollout_policy" not in extra:
            return False
        extra["rollout_policy"] = {"agent_type": heuristic, **params}
        return True
    if block == "heuristic_policy":
        # DCL base policy (agent.extra.heuristic_policy): nested {agent_type, extra}
        # schema, same shape as warmstart but under the agent block.
        extra = cfg.get("agent", {}).get("extra", {})
        if "heuristic_policy" not in extra:
            return False
        extra["heuristic_policy"] = {"agent_type": heuristic, "extra": dict(params)}
        return True
    raise ValueError(f"Unknown block: {block!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params", required=True,
                        help="Path to an Optuna best_params.json")
    parser.add_argument("--targets", nargs="+", required=True,
                        help="Target config files or globs to patch")
    parser.add_argument("--block", required=True,
                        choices=["warmstart", "curriculum_heuristic", "rollout_policy",
                                 "heuristic_policy"],
                        help="Which consumer block to overwrite")
    parser.add_argument("--heuristic", default=None,
                        help="Heuristic agent_type (default: inferred from --params dir name)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print intended changes without writing")
    args = parser.parse_args()

    params_path = Path(args.params)
    params = _load_params(params_path)
    heuristic = args.heuristic or _infer_heuristic(params_path)

    print(f"Source:    {params_path}")
    print(f"Heuristic: {heuristic}")
    print(f"Params:    {params}")
    print(f"Block:     {args.block}\n")

    target_paths: list[str] = []
    for pat in args.targets:
        hits = glob.glob(pat)
        target_paths.extend(hits if hits else [pat])
    target_paths = sorted(set(target_paths))

    patched = skipped = 0
    for tp in target_paths:
        path = Path(tp)
        try:
            with open(path, encoding="utf-8") as f:
                cfg = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"  [skip] cannot read {path.name}: {e}")
            skipped += 1
            continue

        if _patch_config(cfg, args.block, heuristic, params):
            if args.dry_run:
                print(f"  [would patch] {path.name}")
            else:
                with open(path, "w", encoding="utf-8", newline="\n") as f:
                    json.dump(cfg, f, indent=2)
                    f.write("\n")
                print(f"  [patched]     {path.name}")
            patched += 1
        else:
            print(f"  [skip] no '{args.block}' block in {path.name}")
            skipped += 1

    verb = "would patch" if args.dry_run else "patched"
    print(f"\n{verb} {patched} config(s), skipped {skipped}.")


if __name__ == "__main__":
    main()
