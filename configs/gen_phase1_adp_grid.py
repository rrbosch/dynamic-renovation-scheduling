"""Generate the 24-cell Phase-1 ADP grid configs on instance_10p.

Grid dimensions (see EXPERIMENTS.md, Phase 1 / 1b-i):
    action_gen : normal (local_search) | seq (sequential)
    init       : empty                  | policy (reactive warmstart, thr=0.95)
    buffer     : fifo | lowesterror (lowest_error) | knockout (stochastic_knockout)
    vfa        : xgb (xgboost)          | nn (neural)

2 x 2 x 3 x 2 = 24 files named  i10p_adp_{ag}_{init}_{buf}_{vfa}.json
Run:  python configs/gen_phase1_adp_grid.py
"""
import json
from pathlib import Path

OUT = Path(__file__).parent

ACTION_GEN = {"normal": "local_search", "seq": "sequential"}
BUFFER = {"fifo": "fifo", "lowesterror": "lowest_error", "knockout": "stochastic_knockout"}
VFA = {"xgb": "xgboost", "nn": "neural"}
INIT = ["empty", "policy"]

# Warmstart episodes for the 'policy' (heuristic-initialised) variants.
WARMSTART_EPISODES = 1667
WARMSTART = {"agent_type": "reactive", "extra": {"threshold": 0.95}}


def make_config(ag, init, buf, vfa):
    run_name = f"i10p_adp_{ag}_{init}_{buf}_{vfa}"
    training = {
        "time_budget": 86400,
        "eval_interval": 1000000,
        "update_interval": 50,
        "truncation_mode": "bootstrap",
        "buffer_capacity": 200000,
        "buffer_strategy": BUFFER[buf],
        "n_eval_episodes": 10,
        "n_workers": 8,  # Snellius: parallel episode collection (see hpc/registry_conventions.md)
    }
    if init == "policy":
        training["n_warmstart_episodes"] = WARMSTART_EPISODES
        training["warmstart"] = WARMSTART
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
        },
    }


def main():
    n = 0
    for ag in ACTION_GEN:
        for init in INIT:
            for buf in BUFFER:
                for vfa in VFA:
                    cfg = make_config(ag, init, buf, vfa)
                    path = OUT / f"{cfg['run_name']}.json"
                    path.write_text(json.dumps(cfg, indent=2) + "\n")
                    n += 1
    print(f"Wrote {n} ADP grid configs to {OUT}")


if __name__ == "__main__":
    main()
