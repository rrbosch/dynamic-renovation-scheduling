"""Generate the 24-cell Phase-1 ADP grid configs on instance_10p.

Grid dimensions (see EXPERIMENTS.md, Phase 1 / 1b-i):
    action_gen : normal (local_search) | seq (sequential)
    init       : empty                  | policy
    buffer     : fifo | lowesterror (lowest_error) | knockout (stochastic_knockout)
    vfa        : xgb (xgboost)          | nn (neural)

2 x 2 x 3 x 2 = 24 files named  i10p_adp_{ag}_{init}_{buf}_{vfa}.json
Run:  python configs/gen_phase1_adp_grid.py

Warmstart vs init_action — two SEPARATE knobs (changed 2026-06-20):
  * Warmstart (training.warmstart + n_warmstart_states): ALL 24 cells seed the
    replay buffer with 200k heuristic transitions before training. This is now a
    CONSTANT across the grid, not a grid axis.
  * init_action (agent.extra.init_action): the `empty`/`policy` axis. 'empty' starts
    the ADP local action search from do-nothing; 'policy' starts it from the warmstart
    heuristic's action at each decision. The same heuristic is used for both buffer
    warmstart and the policy-init seed (build_experiment wires it from training.warmstart).
"""
import json
from pathlib import Path

OUT = Path(__file__).parent

ACTION_GEN = {"normal": "local_search", "seq": "sequential"}
BUFFER = {"fifo": "fifo", "lowesterror": "lowest_error", "knockout": "stochastic_knockout"}
VFA = {"xgb": "xgboost", "nn": "neural"}
INIT = ["empty", "policy"]   # -> agent.extra.init_action

# Buffer is warmstarted with this many heuristic transitions (states) for EVERY cell.
# == buffer_capacity, so the buffer starts full of heuristic data.
WARMSTART_STATES = 200000
# Warmstart heuristic — PLACEHOLDER (reactive 0.95). The real run uses the Exp-0A
# tuned per-asset heuristic, injected post-hoc via experiments/apply_optuna_params.py.
# Re-running this generator OVERWRITES the configs and reverts that injection, so after
# any regeneration you MUST re-inject into ALL adp cells (both empty and policy now
# carry a warmstart block):
#   python experiments/apply_optuna_params.py \
#       --params results/exp0/i10p_optuna_perasset/best_params.json \
#       --targets "configs/i10p_adp_*.json" --block warmstart
WARMSTART = {"agent_type": "reactive", "extra": {"threshold": 0.95}}


def make_config(ag, init, buf, vfa):
    run_name = f"exp0/i10p_adp_{ag}_{init}_{buf}_{vfa}"  # exp0/ groups results under results/exp0/
    training = {
        "time_budget": 86400,
        "eval_interval": 1000000,          # episode-based eval effectively off; use time-based below
        "eval_interval_seconds": 3600,     # hourly policy eval -> training_log.csv (policy curve)
        "update_interval": 50,
        "truncation_mode": "horizon_rollout",  # simulate the tail in training (non-circular target);
                                               # requires the trainer fix that doesn't break at done=T
        "buffer_capacity": 200000,
        "buffer_strategy": BUFFER[buf],
        "n_eval_episodes": 10,
        "n_workers": 16,  # Snellius default: fills the 16-core min slot (see hpc/registry_conventions.md)
        # Warmstart the buffer for ALL cells (constant across the grid).
        "n_warmstart_states": WARMSTART_STATES,
        "warmstart": WARMSTART,
    }
    return {
        "network": "sioux_falls",
        "tap_backend": "fast",
        "seed": 42,
        "run_name": run_name,
        "instance": "instances/instance_10p.json",
        "training": training,
        "agent": {
            "agent_type": "adp",
            "value_fn": VFA[vfa],
            "action_gen": ACTION_GEN[ag],
            "extra": {
                "init_action": init,   # 'empty' | 'policy'
                # (c) per-epoch advantage baseline — now enabled for ALL VFAs
                # (was NN-only). The raw cost-to-go target's variance is ~94%
                # epoch-trend b(t), which swamps the act-vs-wait advantage signal;
                # subtracting b(t) focuses the fit on the controllable signal. This
                # is a TARGET-variance argument, independent of XGBoost's feature
                # scale-invariance — see docs/adp_value_fn_improvements.md
                # (calibration analysis). Pairs with fix (b) acting warmstart
                # coverage (on by default via training.warmstart_explore_*).
                "advantage_baseline": True,
            },
        },
    }


def main():
    n = 0
    for ag in ACTION_GEN:
        for init in INIT:
            for buf in BUFFER:
                for vfa in VFA:
                    cfg = make_config(ag, init, buf, vfa)
                    path = OUT / f"{cfg['run_name'].split('/')[-1]}.json"  # filename = stem, no exp0/ prefix
                    path.write_text(json.dumps(cfg, indent=2) + "\n")
                    n += 1
    print(f"Wrote {n} ADP grid configs to {OUT}")


if __name__ == "__main__":
    main()
