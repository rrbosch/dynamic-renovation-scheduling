"""Generate the Exp-0B **re-run** ADP grid (12 cells) on instance_10p.

Replaces the original 24-cell factorial (mostly dead cells) with a focused grid aimed at
ADP's ceiling, informed by the completed run + the value-fn/degradation deep-dives:

Held constant (best-known): action_gen=local_search, init_action=policy, buffer=knockout,
warmstart = flip-wrapped per-asset (explore_flip, NO neglect — flips help / neglect hurts on
exp0), 200k warmstart states, horizon_rollout.

Core sweep (8) = value_fn {xgboost, neural} × advantage_baseline {off, on} × n_step {full(0), 4}.
  - n_step=4 is the best within-epoch target on exp0 (full-horizon is anti-predictive); see
    docs/adp_nstep_target_plan.md / docs/adp_value_fn_improvements.md.
  - advantage_baseline is swept (not default-on) because it hurt the NN full-run last time.

Controls (4), one-factor-off from the reference (xgb / ab=off / n_step=4 / normal / policy /
knockout / flip): empty init_action; sequential action_gen; lowesterror buffer; plain (no-flip)
warmstart. These support the paper's OFAT ablation story.

Run:  python configs/gen_adp_nstep_grid.py    (reads the tuned per-asset best_params)
"""
import json
from pathlib import Path

OUT = Path(__file__).parent
ROOT = OUT.parent
PERASSET = ROOT / "results" / "exp0" / "i10p_optuna_perasset" / "best_params.json"


def _perasset_extra() -> dict:
    raw = json.loads(PERASSET.read_text())
    return {k: v for k, v in raw.items() if k != "best_value"}


PA = _perasset_extra()
# Flip-wrapped per-asset warmstart (flip defaults in configs.py ARE the flip-rich recipe:
# p_base=1/120, p_high=0.5, d_ref=0.5, act_bias=0.9, renovate_bias=0.7). No neglect.
WS_FLIP = {"agent_type": "explore_flip",
           "extra": {"base": {"agent_type": "reactiveperasset", "extra": PA}}}
# Plain per-asset (no-flip control).
WS_PLAIN = {"agent_type": "reactiveperasset", "extra": PA}


def make(stem, *, vfa="xgboost", ab=False, n_step=4, ag="local_search",
         init="policy", buf="stochastic_knockout", ws=WS_FLIP):
    run_name = f"exp0/i10p_adp2_{stem}"
    return run_name, {
        "network": "sioux_falls", "tap_backend": "fast", "seed": 42,
        "run_name": run_name, "instance": "instances/instance_10p.json",
        "training": {
            "time_budget": 86400, "eval_interval": 1000000, "eval_interval_seconds": 3600,
            "update_interval": 50, "truncation_mode": "horizon_rollout",
            "buffer_capacity": 200000, "buffer_strategy": buf, "n_eval_episodes": 10,
            "n_workers": 16, "n_warmstart_states": 200000, "warmstart": ws,
        },
        "agent": {
            "agent_type": "adp", "value_fn": vfa, "action_gen": ag,
            "extra": {"init_action": init, "advantage_baseline": ab, "n_step": n_step},
        },
    }


def main():
    cfgs = []
    # Core sweep: vfa × advantage_baseline × n_step
    for vfa, vt in (("xgboost", "xgb"), ("neural", "nn")):
        for ab, at in ((False, "aboff"), (True, "abon")):
            for ns, nt in ((0, "nsfull"), (4, "ns4")):
                cfgs.append(make(f"{vt}_{at}_{nt}", vfa=vfa, ab=ab, n_step=ns))
    # Controls (one-factor-off from xgb / ab=off / n_step=4 / normal / policy / knockout / flip)
    cfgs.append(make("ctrl_emptyinit",   init="empty"))
    cfgs.append(make("ctrl_seq",         ag="sequential"))
    cfgs.append(make("ctrl_lowesterror", buf="lowest_error"))
    cfgs.append(make("ctrl_noflip",      ws=WS_PLAIN))

    for run_name, cfg in cfgs:
        path = OUT / f"{run_name.split('/')[-1]}.json"
        path.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"Wrote {len(cfgs)} ADP-grid configs to {OUT}")
    for rn, _ in cfgs:
        print("  ", rn)


if __name__ == "__main__":
    main()
