"""Generate the full sf20 Exp-0 config set (0A heuristics + 0B learners) + HPC registries.

sf20 = instance_sf20.json (N=20, ~3-yr renovations, ~14-yr lifespans; the item-1 redesign of sf15,
see docs/rl_underperformance_action_plan.md §1). This mirrors the sf15 experiment structure and wires
in the items 4/5/6 infra changes:

  * item 6 (post-mortem eval):   ADP + DCL learners set defer_eval + snapshot_interval_seconds; the
                                 authoritative eval runs offline via experiments/evaluate_checkpoints.py.
                                 Rollout stays eval-only (no training to protect; its eval is resumable).
  * item 5 (early stopping):     ADP learners get a cheap patience early-stop.
  * item 4 (PPO curriculum):     PPO uses curriculum_additive_travel + curriculum_phase_budget_frac.

0A must run first: it tunes the heuristics and writes results/exp0/sf20_optuna_perasset/best_params.json.
This generator bakes the tuned per-asset params into every 0B consumer block, so RE-RUN IT after 0A
finishes on Snellius to refresh the 0B configs with the final params. Until then it falls back to the
laptop tune (results/cal/sf20_perasset) so the 0B configs are runnable immediately.

Run:  python configs/gen_sf20_configs.py
"""
import json
from pathlib import Path

OUT = Path(__file__).parent
ROOT = OUT.parent
REG = ROOT / "hpc" / "registries"
INSTANCE = "instances/instance_sf20.json"

# ---- 0B base policy = the 0A WINNER (lowest held-out cost), not hardcoded per-asset ---------
# sf20 favours network-aware heuristics (netconcurrency/holding) where sf15 favoured per-asset,
# so 0B must seed from whatever 0A actually wins. Filled by resolve_base() in main().
BASE_NESTED: dict = {}   # {agent_type, extra}       — warmstart base / dcl / ppo curriculum
BASE_FLAT: dict = {}     # {agent_type, **params}    — rollout_policy (flat schema)
WS_FLIP: dict = {}       # explore_flip-wrapped base — ADP warmstart
WS_PLAIN: dict = {}      # bare base                 — ADP ctrl_noflip


def _gpe() -> float:
    inst = json.loads((ROOT / INSTANCE).read_text())
    return inst["gamma"] ** inst["dt"]


def _held_out_cost(run_dir: Path, gpe: float):
    """Mean discounted held-out cost from a 0A run's eval_episodes.csv (None if absent)."""
    import csv
    from collections import defaultdict
    f = run_dir / "eval_episodes.csv"
    if not f.exists():
        return None
    eps = defaultdict(float)
    with open(f, newline="") as fh:
        for row in csv.DictReader(fh):
            eps[int(row["episode"])] += (gpe ** int(row["t"])) * float(row["cost"])
    return (sum(eps.values()) / len(eps)) if eps else None


def _agent_type(heuristic: str) -> str:
    return "reactiveperasset" if heuristic == "perasset" else heuristic


def resolve_base(results_glob: str, force: str | None) -> str:
    """Set BASE_NESTED/BASE_FLAT/WS_FLIP/WS_PLAIN from the 0A winner (lowest held-out cost)
    found under `results_glob`. `--base <h>` forces a heuristic; if no rankable 0A results
    exist, fall back to the laptop per-asset tune so the pipeline stays runnable. Returns a
    short label for logging."""
    import glob as _glob
    global BASE_NESTED, BASE_FLAT, WS_FLIP, WS_PLAIN
    gpe = _gpe()
    ranked = []
    for d in sorted(_glob.glob(str(ROOT / results_glob))):
        d = Path(d)
        if not (d / "best_params.json").exists():
            continue
        h = d.name.split("optuna_")[-1]
        ranked.append((h, _held_out_cost(d, gpe), d))

    chosen = None
    if force:
        for h, cost, d in ranked:
            if h == force:
                chosen = (h, cost, d)
                break
        if chosen is None:
            raise SystemExit(f"--base {force!r} not found under {results_glob}")
    else:
        rankable = sorted([r for r in ranked if r[1] is not None], key=lambda r: r[1])
        chosen = rankable[0] if rankable else None

    if chosen is None:
        lap = ROOT / "results" / "cal" / "sf20_perasset" / "best_params.json"
        params = {k: v for k, v in json.loads(lap.read_text()).items() if k != "best_value"}
        atype, label = "reactiveperasset", "perasset (laptop fallback — re-run after 0A)"
    else:
        h, cost, d = chosen
        params = {k: v for k, v in json.loads((d / "best_params.json").read_text()).items()
                  if k != "best_value"}
        atype = _agent_type(h)
        label = f"{h} ({'forced' if force else 'winner'}" + \
                (f", held-out {cost/1e6:.0f}M" if cost is not None else ", no eval") + ")"

    BASE_NESTED = {"agent_type": atype, "extra": params}
    BASE_FLAT = {"agent_type": atype, **params}
    WS_FLIP = {"agent_type": "explore_flip", "extra": {"base": BASE_NESTED}}
    WS_PLAIN = BASE_NESTED
    print(f"0B base policy: {label}  [{len(params)} params, agent_type={atype}]")
    return label

# ---- item 5/6 shared blocks (learners that TRAIN) ------------------------------------------
POSTMORTEM = {"defer_eval": True, "snapshot_interval_seconds": 3600}
EARLYSTOP = {"early_stop_patience": 4, "early_stop_tol": 0.01, "early_stop_min_seconds": 14400,
             "early_stop_episodes": 10, "early_stop_interval_seconds": 3600}


def _base(run_name: str) -> dict:
    return {"network": "sioux_falls", "tap_backend": "fast", "seed": 42,
            "run_name": run_name, "instance": INSTANCE}


def write(run_name: str, cfg: dict) -> str:
    stem = run_name.split("/")[-1]
    (OUT / f"{stem}.json").write_text(json.dumps(cfg, indent=2) + "\n")
    return f"configs/{stem}.json"


# ============================================================================================
# 0A — Optuna-tuned heuristics (8). Standalone; tune + write best_params.json.
# ============================================================================================
def gen_0a() -> list[str]:
    heuristics = ["reactive", "paced", "perasset", "leadtime", "netconcurrency",
                  "holding", "valuedensity", "worstfirst"]
    # perasset/reactive get the param space auto-generated by configs.py (no param_space key).
    paths = []
    for h in heuristics:
        run_name = f"exp0/sf20_optuna_{h}"
        cfg = _base(run_name)
        cfg["training"] = {"time_budget": 86400, "n_eval_episodes": 50,
                           "early_stopping_seconds": 7200, "n_workers": 1}
        extra = {"heuristic_type": "reactiveperasset" if h == "perasset" else h,
                 "n_tuning_episodes": 100}
        cfg["agent"] = {"agent_type": "optuna_heuristic", "extra": extra}
        paths.append(write(run_name, cfg))
    return paths


# ============================================================================================
# 0B-i — ADP grid (12): vfa×ab×nstep (8) + one-factor-off controls (4). Mirrors gen_adp_nstep_grid.
# ============================================================================================
def _adp(stem, *, vfa="xgboost", ab=False, n_step=4, ag="local_search",
         init="policy", buf="stochastic_knockout", ws=None) -> tuple[str, dict]:
    ws = WS_FLIP if ws is None else ws   # resolve at call time (after resolve_base sets it)
    run_name = f"exp0/sf20_adp2_{stem}"
    cfg = _base(run_name)
    cfg["training"] = {"time_budget": 86400, "update_interval": 50,
                       "truncation_mode": "horizon_rollout", "buffer_capacity": 200000,
                       "buffer_strategy": buf, "n_eval_episodes": 50, "n_workers": 16,
                       "n_warmstart_states": 200000, "warmstart": ws, **POSTMORTEM, **EARLYSTOP}
    cfg["agent"] = {"agent_type": "adp", "value_fn": vfa, "action_gen": ag,
                    "extra": {"init_action": init, "advantage_baseline": ab, "n_step": n_step}}
    return run_name, cfg


def gen_adp() -> list[str]:
    cells = []
    for vfa, vt in (("xgboost", "xgb"), ("neural", "nn")):
        for ab, at in ((False, "aboff"), (True, "abon")):
            for ns, nt in ((0, "nsfull"), (4, "ns4")):
                cells.append(_adp(f"{vt}_{at}_{nt}", vfa=vfa, ab=ab, n_step=ns))
    cells.append(_adp("ctrl_emptyinit", init="empty"))
    cells.append(_adp("ctrl_seq", ag="sequential"))
    cells.append(_adp("ctrl_lowesterror", buf="lowest_error"))
    cells.append(_adp("ctrl_noflip", ws=WS_PLAIN))
    return [write(rn, c) for rn, c in cells]


# ============================================================================================
# 0B-ii — MC rollout (4): {normal,seq} × {empty,policy} init. Eval-only (no defer_eval).
# ============================================================================================
def gen_rollout() -> list[str]:
    paths = []
    for seq, prefix, atype in ((False, "rollout", "rollout"),
                               (True, "seq_rollout", "sequential_rollout")):
        for init in ("empty", "policy"):
            run_name = f"exp0/sf20_{prefix}_{init}"
            cfg = _base(run_name)
            cfg["training"] = {"n_episodes": 5, "n_eval_episodes": 50, "n_workers": 16}
            cfg["agent"] = {"agent_type": atype, "extra": {
                "n_rollouts": 20, "action_threshold": 0.5, "rollout_selection": "adaptive",
                "p_threshold": 0.02, "min_rollouts": 20, "max_rollouts": 100,
                "initial_action": init, "rollout_policy": BASE_FLAT, "rollout_horizon": 100}}
            paths.append(write(run_name, cfg))
    return paths


# ============================================================================================
# 0B-iii — PPO curriculum (1): item-4 additive travel + force-graduation gate.
# ============================================================================================
def gen_ppo() -> list[str]:
    run_name = "exp0/sf20_ppo_curriculum"
    cfg = _base(run_name)
    cfg["training"] = {"time_budget": 86400, "n_episodes": 10000, "eval_interval": 1000,
                       "n_eval_episodes": 50, "n_workers": 16,
                       "curriculum_phase0_episodes": 500,
                       "curriculum_phase0_mode": "bc",          # item: BC actor + normal critic
                       "curriculum_phase1_plateau_window": 5, "curriculum_phase1_plateau_tol": 0.01,
                       "curriculum_reset_critic": False,
                       "curriculum_additive_travel": True,      # item 4
                       "curriculum_phase_budget_frac": 0.5,     # item 4: guarantee Phase 2 runs
                       "curriculum_phase1_max_episodes": 0,
                       "curriculum_heuristic": BASE_NESTED}
    cfg["agent"] = {"agent_type": "ppo", "extra": {
        "hidden_dims": [64, 64],
        "ppo_kwargs": {"actor_lr": 0.0003, "critic_lr": 0.001, "clip_eps": 0.2,
                       "entropy_coef": 0.02, "value_coef": 0.5, "gae_lambda": 0.95,
                       "ppo_epochs": 4, "mini_batch_size": 64}}}
    return [write(run_name, cfg)]


# ============================================================================================
# 0B-iv — DCL (10): architecture sweep + rollout-elimination + VFA. item-6 post-mortem eval.
# ============================================================================================
_DCL_NN = {"finite_horizon": True, "hidden_dims": [256, 256], "policy_lr": 0.001,
           "policy_epochs": 40, "policy_batch_size": 256}
_DCL_XGB = {"use_global_context": True}


def _dcl(stem, *, action_search="sequential", policy_type="xgboost", rollout_selection="fixed",
         rollout_horizon=None, n_rollouts=40, extra_sel=None) -> tuple[str, dict]:
    run_name = f"exp0/sf20_dcl_{stem}"
    cfg = _base(run_name)
    cfg["training"] = {"eval_interval": 1, "n_eval_episodes": 50, "time_budget": 86400,
                       "n_workers": 16, **POSTMORTEM}
    ex = {"class_weight": "balanced_sqrt", "action_search": action_search,
          "policy_type": policy_type, "n_rounds": 100, "samples_per_round": 5000,
          "collect_steps": 0, "rollout_horizon": rollout_horizon}
    if rollout_horizon is not None:
        ex["value_fn"] = "xgboost"
    ex.update({"n_rollouts": n_rollouts, "rollout_selection": rollout_selection})
    if extra_sel:
        ex.update(extra_sel)
    ex.update({"action_threshold": 0.5, "initial_action": "policy"})
    ex.update(_DCL_NN if policy_type == "nn" else _DCL_XGB)
    ex["heuristic_policy"] = BASE_NESTED
    cfg["agent"] = {"agent_type": "dcl", "extra": ex}
    return run_name, cfg


def gen_dcl() -> list[str]:
    SH = {"sh_budget_per_arm": 25}
    WILC = {"p_threshold": 0.02, "min_rollouts": 20, "max_rollouts": 100, "rollout_batch": 5}
    cells = [
        _dcl("independent_nn", action_search="independent", policy_type="nn"),
        _dcl("independent_xgb", action_search="independent"),
        _dcl("localsearch_nn", action_search="local_search", policy_type="nn"),
        _dcl("localsearch_xgb", action_search="local_search"),
        _dcl("seq_nn", policy_type="nn"),
        _dcl("seq_xgb"),
        _dcl("seq_xgb_sh", rollout_selection="sequential_halving", n_rollouts=20, extra_sel=SH),
        _dcl("seq_xgb_wilcoxon", rollout_selection="wilcoxon", n_rollouts=20, extra_sel=WILC),
        _dcl("seq_xgb_vfa", rollout_horizon=20),
        _dcl("seq_xgb_vfa_sh", rollout_horizon=20, rollout_selection="sequential_halving",
             n_rollouts=20, extra_sel=SH),
    ]
    return [write(rn, c) for rn, c in cells]


def _registry(paths: list[str]) -> list[dict]:
    return [{"config": p, "run_name": None, "seed": None} for p in paths]


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-glob", default="results/exp0/sf20_optuna_*",
                    help="where to find 0A runs (best_params.json + eval_episodes.csv) to pick the "
                         "winner from. Default = the Snellius 0A output.")
    ap.add_argument("--base", default=None,
                    help="force a specific 0A heuristic as the 0B base (e.g. netconcurrency) "
                         "instead of the auto-selected winner.")
    ap.add_argument("--only-0b", action="store_true",
                    help="regenerate only the 0B configs (0A already dispatched/unchanged).")
    args = ap.parse_args()

    resolve_base(args.results_glob, args.base)   # sets BASE_NESTED/BASE_FLAT/WS_FLIP/WS_PLAIN

    a0 = [] if args.only_0b else gen_0a()
    # 0B order (registry array indexing): ppo, rollout(4), adp(12), dcl(10) = 27
    b0 = gen_ppo() + gen_rollout() + gen_adp() + gen_dcl()
    REG.mkdir(parents=True, exist_ok=True)
    if a0:
        (REG / "sf20_0a.json").write_text(json.dumps(_registry(a0), indent=2) + "\n")
    (REG / "sf20_0b.json").write_text(json.dumps(_registry(b0), indent=2) + "\n")
    if a0:
        print(f"\n0A: {len(a0)} configs -> hpc/registries/sf20_0a.json")
    print(f"0B: {len(b0)} configs -> hpc/registries/sf20_0b.json")
    for p in a0 + b0:
        print("  ", p)


if __name__ == "__main__":
    main()
