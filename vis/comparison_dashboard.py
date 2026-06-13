"""Multi-run comparison dashboard for RL infrastructure maintenance experiments.

Launch (with explicit results folder):
    python vis/comparison_dashboard.py -- --results results/
    python vis/comparison_dashboard.py -- --results results/sweep_paced_sweep

Launch without arguments (interactive folder picker in terminal):
    python vis/comparison_dashboard.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__" and not os.environ.get("_COMPARISON_DASHBOARD_LAUNCHED"):
    if "--results" not in sys.argv:
        # Search for results/ relative to this script, then relative to cwd
        _script_dir = Path(__file__).resolve().parent
        for _candidate_root in (_script_dir.parent / "results", Path("results").resolve()):
            if _candidate_root.exists():
                results_root = _candidate_root
                break
        else:
            results_root = Path("results")  # will fail gracefully below

        candidates = sorted([d for d in results_root.iterdir() if d.is_dir()]) if results_root.exists() else []
        if not candidates:
            print(f"No subdirectories found under '{results_root}'. Pass --results <path> manually.")
            sys.exit(1)
        print("Available results folders:")
        for i, d in enumerate(candidates):
            print(f"  [{i}] {d}")
        raw = input("Select results folder(s) (number, comma-separated numbers, or path): ").strip()
        parts = [p.strip() for p in raw.split(",")]
        if all(p.isdigit() for p in parts):
            selected_paths = [str(candidates[int(p)]) for p in parts]
        else:
            selected_paths = [raw]   # single path typed directly
        results_args = []
        for p in selected_paths:
            results_args += ["--results", p]
    else:
        results_args = ["--results", sys.argv[sys.argv.index("--results") + 1]]
    env = {**os.environ, "_COMPARISON_DASHBOARD_LAUNCHED": "1"}
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", __file__, "--", *results_args],
        env=env,
    )
    sys.exit()

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from vis._charts import (  # noqa: E402
    ACTION_NAMES,
    get_n_assets, get_episode,
    make_cost_chart, make_heatmap, make_asset_detail,
    make_cross_episode_charts, make_training_curve, make_pred_residual_chart,
    make_state_diversity_charts, make_cost_breakdown_chart,
    make_degradation_fan, make_renovation_fan,
)
from experiments.algorithm_meta import (  # noqa: E402
    compute_algorithm_class, compute_algorithm_id, compute_algorithm_label,
)

# ---------------------------------------------------------------------------
# CLI argument
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--results", action="append", default=None)
args, _ = parser.parse_known_args()
all_results_folders: list[str] = args.results or ["results"]

# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------

def discover_runs(folder: str) -> list[Path]:
    """Return all directories up to 4 levels deep that contain eval_episodes.csv."""
    root = Path(folder)
    if not root.exists():
        return []

    if (root / "eval_episodes.csv").exists():
        return [root]

    hits: list[Path] = []

    def _scan(directory: Path, depth: int):
        if depth > 4:
            return
        try:
            children = sorted(directory.iterdir())
        except PermissionError:
            return
        for child in children:
            if not child.is_dir():
                continue
            if (child / "eval_episodes.csv").exists():
                hits.append(child)
            else:
                _scan(child, depth + 1)

    _scan(root, 1)
    return hits


def run_label(root: Path, run_path: Path) -> str:
    """Return a short display label relative to root."""
    try:
        return str(run_path.relative_to(root))
    except ValueError:
        return run_path.name


def parse_trial_params(label: str) -> dict[str, str]:
    """Parse parameter values from a sweep trial label.

    Expects the directory name (last path component) to follow the convention
    ``param-name=value__param-name=value``.  Returns an empty dict for labels
    that don't match (e.g. flat non-sweep runs).

    Example:
        >>> parse_trial_params("sweep_paced/threshold=0.7__pace-threshold=0.05")
        {'threshold': '0.7', 'pace-threshold': '0.05'}
    """
    trial_dir = label.split("/")[-1].split("\\")[-1]
    params: dict[str, str] = {}
    for part in trial_dir.split("__"):
        if "=" in part:
            name, _, value = part.partition("=")
            params[name] = value
    return params


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_run(path_str: str) -> tuple[
        pd.DataFrame | None, dict, pd.DataFrame | None,
        pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None,
        dict | None, pd.DataFrame | None]:
    p = Path(path_str)
    episodes_path = p / "eval_episodes.csv"
    if not episodes_path.exists():
        return None, {}, None, None, None, None, None, None
    df = pd.read_csv(episodes_path)
    config: dict = {}
    config_path = p / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    log = None
    log_path = p / "training_log.csv"
    if log_path.exists():
        log = pd.read_csv(log_path)
    buf_pred = None
    bp = p / 'vf_buffer_predictions.csv'
    if bp.exists():
        buf_pred = pd.read_csv(bp)
    eval_pred = None
    ep2 = p / 'vf_eval_predictions.csv'
    if ep2.exists():
        eval_pred = pd.read_csv(ep2)
    agent_metrics = None
    am = p / 'agent_metrics.csv'
    if am.exists():
        agent_metrics = pd.read_csv(am)
    best_params = None
    bp_path = p / 'best_params.json'
    if bp_path.exists():
        with open(bp_path) as f:
            best_params = json.load(f)
    optuna_trials = None
    ot_path = p / 'optuna_trials.csv'
    if ot_path.exists():
        optuna_trials = pd.read_csv(ot_path)
    return df, config, log, buf_pred, eval_pred, agent_metrics, best_params, optuna_trials


# ---------------------------------------------------------------------------
# Grouping logic
# ---------------------------------------------------------------------------

def _ensure_algorithm_metadata(config: dict) -> None:
    """Recompute algorithm_class/label/id from the stored agent config in place.

    Old runs predate these fields (shown as "?"), and pre-existing rollout runs
    carry a stale label that doesn't distinguish empty/policy. The compute_*
    functions are pure and deterministic, so recomputed == baked for every
    healthy run, while old runs gain correct, consistent grouping metadata
    without re-running. Seeds of one agent differ only by seed/run_name (both
    stripped in compute_algorithm_id) -> identical id -> still aggregate.
    """
    if not config.get("agent"):
        return
    config["algorithm_class"] = compute_algorithm_class(config)
    config["algorithm_label"] = compute_algorithm_label(config)
    config["algorithm_id"] = compute_algorithm_id(config)


def _class_display(config: dict) -> str:
    """Algorithm class string, marked '(optuna)' for Optuna-tuned heuristics."""
    alg_class = config.get("algorithm_class", "?")
    if config.get("agent", {}).get("agent_type") == "optuna_heuristic":
        return f"{alg_class} (optuna)"
    return alg_class


def _get_group_key(rd: dict, level: str) -> str:
    """Return grouping key for a run dict based on level."""
    config = rd.get("config", {})
    if level == "config":
        return config.get("algorithm_id", rd["label"])
    elif level == "class":
        return config.get("algorithm_class", rd["label"])
    return rd["label"]


def _get_group_display(rd: dict, level: str) -> str:
    """Return human-readable group name."""
    config = rd.get("config", {})
    if level == "config":
        alg_label = config.get("algorithm_label", "?")
        return f"{_class_display(config)}/{alg_label}"
    elif level == "class":
        return _class_display(config) if config.get("algorithm_class") else rd["label"]
    return rd["label"]


def group_runs(run_data: list[dict], level: str) -> list[dict]:
    """Group runs into aggregated groups.

    Returns list of dicts with keys:
        group_key, group_label, runs (list of run dicts), n_seeds
    """
    from collections import OrderedDict
    groups: OrderedDict[str, dict] = OrderedDict()
    for rd in run_data:
        key = _get_group_key(rd, level)
        if key not in groups:
            groups[key] = {
                "group_key": key,
                "group_label": _get_group_display(rd, level),
                "runs": [],
            }
        groups[key]["runs"].append(rd)
    for g in groups.values():
        g["n_seeds"] = len(g["runs"])

    # Safeguard: ensure display labels are unique so charts that use group_label
    # as a plotly category don't silently merge/stack two distinct groups.
    from collections import Counter
    label_counts = Counter(g["group_label"] for g in groups.values())
    for g in groups.values():
        if label_counts[g["group_label"]] > 1:
            g["group_label"] = f"{g['group_label']} [{str(g['group_key'])[:4]}]"

    return list(groups.values())


def _has_grouping_metadata(run_data: list[dict]) -> bool:
    """Check if any run has algorithm_id in config (new-style metadata)."""
    return any(rd.get("config", {}).get("algorithm_id") for rd in run_data)


# ---------------------------------------------------------------------------
# Comparison chart builders
# ---------------------------------------------------------------------------

# Distinct colours for up to ~12 runs
_PALETTE = px.colors.qualitative.Plotly


def _run_color(idx: int) -> str:
    return _PALETTE[idx % len(_PALETTE)]


def make_mean_cost_chart(stats: list[dict]) -> go.Figure:
    """Horizontal bar chart: mean total cost ± std, sorted best->worst."""
    stats_sorted = sorted(stats, key=lambda s: s["mean_cost"])
    labels = [s["label"] for s in stats_sorted]
    means = [s["mean_cost"] for s in stats_sorted]
    stds = [s["std_cost"] for s in stats_sorted]
    colors = [_run_color(i) for i in range(len(stats_sorted))]

    fig = go.Figure(go.Bar(
        x=means, y=labels, orientation="h",
        error_x=dict(type="data", array=stds, visible=True),
        marker_color=colors,
        hovertemplate="<b>%{y}</b><br>mean cost=%{x:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title="Mean total cost per episode +/- std (sorted best->worst)",
        xaxis_title="Mean total cost",
        xaxis_type="log",
        yaxis_title="Run",
        height=max(300, 60 + 40 * len(stats_sorted)),
        margin=dict(t=50, b=40, l=200),
    )
    return fig


def make_cost_box_chart(run_data: list[dict]) -> go.Figure:
    """Box plot of total cost per episode for each run."""
    fig = go.Figure()
    for idx, rd in enumerate(run_data):
        ep_costs = _ep_discounted_costs(rd["df"], rd.get("gamma_per_epoch", 1.0))
        fig.add_trace(go.Box(
            y=ep_costs.values,
            name=rd["label"],
            marker_color=_run_color(idx),
            boxmean="sd",
            hovertemplate=f"<b>{rd['label']}</b><br>cost=%{{y:.2f}}<extra></extra>",
        ))
    fig.update_layout(
        title="Cost distribution per episode",
        yaxis_title="Total cost per episode",
        yaxis_type="log",
        height=400,
        margin=dict(t=50, b=40),
    )
    return fig


def make_mean_condition_chart(run_data: list[dict]) -> go.Figure:
    """Line chart: mean condition across all assets & episodes over time."""
    fig = go.Figure()
    for idx, rd in enumerate(run_data):
        df = rd["df"]
        n = get_n_assets(df)
        d_cols = [f"d_{i}" for i in range(n)]
        # Mean over all asset columns at each timestep (averaged over episodes)
        mean_cond = df.groupby("t")[d_cols].mean().mean(axis=1).reset_index()
        mean_cond.columns = ["t", "mean_d"]
        fig.add_trace(go.Scatter(
            x=mean_cond["t"], y=mean_cond["mean_d"],
            name=rd["label"],
            mode="lines",
            line=dict(color=_run_color(idx), width=2),
            hovertemplate=f"<b>{rd['label']}</b><br>t=%{{x}}<br>mean d=%{{y:.3f}}<extra></extra>",
        ))
    fig.update_layout(
        title="Mean asset condition over time (all assets & episodes)",
        xaxis_title="Timestep",
        yaxis_title="Mean d",
        height=380,
        margin=dict(t=50, b=40),
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def make_action_frequency_chart(run_data: list[dict]) -> go.Figure:
    """Grouped bar chart: action frequency per run for Repair/Renovate/Restrict."""
    action_codes = [1, 2, 3]
    fig = go.Figure()
    for idx, rd in enumerate(run_data):
        df = rd["df"]
        n = get_n_assets(df)
        freqs = []
        for code in action_codes:
            a_cols = [f"a_{i}" for i in range(n) if f"a_{i}" in df.columns]
            if not a_cols:
                freqs.append(0.0)
                continue
            total_steps = len(df) * len(a_cols)
            hits = sum((df[c] == code).sum() for c in a_cols)
            freqs.append(hits / total_steps if total_steps > 0 else 0.0)
        fig.add_trace(go.Bar(
            name=rd["label"],
            x=[ACTION_NAMES[c] for c in action_codes],
            y=freqs,
            marker_color=_run_color(idx),
            hovertemplate=f"<b>{rd['label']}</b><br>action=%{{x}}<br>freq=%{{y:.3f}}<extra></extra>",
        ))
    fig.update_layout(
        title="Action frequency by type",
        xaxis_title="Action",
        yaxis_title="Fraction of steps",
        barmode="group",
        height=380,
        margin=dict(t=50, b=40),
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def make_near_failure_chart(run_data: list[dict]) -> go.Figure:
    """Bar chart: fraction of episode-steps with any asset d >= 0.95."""
    labels, rates = [], []
    colors = []
    for idx, rd in enumerate(run_data):
        df = rd["df"]
        n = get_n_assets(df)
        d_cols = [f"d_{i}" for i in range(n)]
        near_fail = (df[d_cols].max(axis=1) >= 0.95).mean()
        labels.append(rd["label"])
        rates.append(near_fail)
        colors.append(_run_color(idx))
    fig = go.Figure(go.Bar(
        x=labels, y=rates,
        marker_color=colors,
        hovertemplate="<b>%{x}</b><br>near-failure rate=%{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="Near-failure rate (fraction of steps with any d >= 0.95)",
        xaxis_title="Run",
        yaxis_title="Near-failure rate",
        height=350,
        margin=dict(t=50, b=80),
        xaxis=dict(tickangle=-30),
    )
    return fig


def make_training_overlay(run_data: list[dict], mwa_window: int = 10) -> go.Figure:
    """Overlay training curves for all runs that have a training log."""
    fig = go.Figure()
    skipped = []
    for idx, rd in enumerate(run_data):
        log = rd["log"]
        if log is None or "mean_cost" not in log.columns:
            skipped.append(rd["label"])
            continue
        y = log["mean_cost"]
        x = log["episode"] if "episode" in log.columns else log.index
        color = _run_color(idx)
        label = rd["label"]
        fig.add_trace(go.Scatter(
            x=x, y=y, name=label,
            mode="lines",
            line=dict(color=color, width=2),
            hovertemplate=f"<b>{label}</b><br>ep=%{{x}}<br>mean cost=%{{y:.2f}}<extra></extra>",
        ))
        if "std_cost" in log.columns:
            std = log["std_cost"]
            rgba = _hex_to_rgba(color, 0.15)
            fig.add_trace(go.Scatter(
                x=list(x) + list(x)[::-1],
                y=list(y + std) + list((y - std).clip(lower=0))[::-1],
                fill="toself",
                fillcolor=rgba,
                line=dict(color="rgba(0,0,0,0)"),
                name=f"{label} +/-std",
                showlegend=False,
                hoverinfo="skip",
            ))
        # Moving-window average
        mwa = y.rolling(mwa_window, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=x, y=mwa, name=f"{label} (MWA-{mwa_window})",
            mode="lines",
            line=dict(color=color, dash="dot", width=3),
            hovertemplate=f"<b>{label} MWA</b><br>ep=%{{x}}<br>mwa=%{{y:.2f}}<extra></extra>",
        ))
    fig.update_layout(
        title="Training curves (all runs)",
        xaxis_title="Episode",
        yaxis_title="Mean cost",
        height=400,
        margin=dict(t=50, b=40),
        legend=dict(orientation="h", y=1.12),
    )
    return fig, skipped


def _hex_to_rgba(color: str, alpha: float) -> str:
    """Convert a Plotly color string to rgba(...) with the given alpha."""
    if color.startswith("#") and len(color) == 7:
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        return f"rgba({r},{g},{b},{alpha})"
    return f"rgba(100,100,200,{alpha})"


def _ep_discounted_costs(df: pd.DataFrame, gamma_per_epoch: float) -> pd.Series:
    """Discounted total cost per episode: sum_t gamma^t * cost_t."""
    if gamma_per_epoch >= 1.0:
        return df.groupby("episode")["cost"].sum()
    return df.groupby("episode")[["t", "cost"]].apply(
        lambda g: float((gamma_per_epoch ** g["t"].values * g["cost"].values).sum())
    ).rename("cost")


# ---------------------------------------------------------------------------
# Aggregated chart builders (for grouped views)
# ---------------------------------------------------------------------------

def _group_ep_costs(group: dict) -> tuple[list[float], list[float]]:
    """Return (per_seed_means, all_episode_costs) for a group."""
    per_seed_means = []
    all_costs = []
    for rd in group["runs"]:
        ep_costs = _ep_discounted_costs(rd["df"], rd.get("gamma_per_epoch", 1.0))
        per_seed_means.append(float(ep_costs.mean()))
        all_costs.extend(ep_costs.values.tolist())
    return per_seed_means, all_costs


def make_grouped_mean_cost_chart(groups: list[dict]) -> go.Figure:
    """Horizontal bar: mean of per-seed means, error bar = 95% CI across seeds."""
    stats = []
    for g in groups:
        seed_means, _ = _group_ep_costs(g)
        arr = np.array(seed_means)
        mean = float(arr.mean())
        n = len(arr)
        if n > 1:
            se = float(arr.std(ddof=1) / np.sqrt(n))
            ci95 = 1.96 * se
        else:
            ci95 = 0.0
        stats.append({"label": f"{g['group_label']} ({n} seeds)", "mean": mean, "ci95": ci95})
    stats.sort(key=lambda s: s["mean"])
    fig = go.Figure(go.Bar(
        x=[s["mean"] for s in stats],
        y=[s["label"] for s in stats],
        orientation="h",
        error_x=dict(type="data", array=[s["ci95"] for s in stats], visible=True),
        marker_color=[_run_color(i) for i in range(len(stats))],
        hovertemplate="<b>%{y}</b><br>mean=%{x:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title="Mean cost (aggregated across seeds) with 95% CI",
        xaxis_title="Mean total cost", xaxis_type="log",
        height=max(300, 60 + 50 * len(stats)),
        margin=dict(t=50, b=40, l=250),
    )
    return fig


def make_grouped_cost_box_chart(groups: list[dict]) -> go.Figure:
    """One box per group, pooling all episodes from all seeds."""
    fig = go.Figure()
    for idx, g in enumerate(groups):
        _, all_costs = _group_ep_costs(g)
        label = f"{g['group_label']} ({g['n_seeds']} seeds)"
        fig.add_trace(go.Box(
            y=all_costs, name=label,
            marker_color=_run_color(idx), boxmean="sd",
            hovertemplate=f"<b>{label}</b><br>cost=%{{y:.2f}}<extra></extra>",
        ))
    fig.update_layout(
        title="Cost distribution (pooled across seeds)",
        yaxis_title="Total cost per episode", yaxis_type="log",
        height=400, margin=dict(t=50, b=40),
    )
    return fig


def make_grouped_condition_chart(groups: list[dict]) -> go.Figure:
    """One line per group (mean across seeds) with shaded std band."""
    fig = go.Figure()
    for idx, g in enumerate(groups):
        # Collect per-seed mean condition series, then average across seeds
        seed_series = []
        for rd in g["runs"]:
            df = rd["df"]
            n = get_n_assets(df)
            d_cols = [f"d_{i}" for i in range(n)]
            mc = df.groupby("t")[d_cols].mean().mean(axis=1)
            seed_series.append(mc)
        combined = pd.DataFrame(seed_series)
        t_vals = combined.columns.values
        mean_vals = combined.mean(axis=0).values
        std_vals = combined.std(axis=0).values if len(seed_series) > 1 else np.zeros_like(mean_vals)

        color = _run_color(idx)
        label = f"{g['group_label']} ({g['n_seeds']} seeds)"
        fig.add_trace(go.Scatter(
            x=t_vals, y=mean_vals, name=label, mode="lines",
            line=dict(color=color, width=2),
        ))
        if len(seed_series) > 1:
            rgba = _hex_to_rgba(color, 0.15)
            fig.add_trace(go.Scatter(
                x=list(t_vals) + list(t_vals)[::-1],
                y=list(mean_vals + std_vals) + list(np.maximum(mean_vals - std_vals, 0))[::-1],
                fill="toself", fillcolor=rgba,
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False, hoverinfo="skip",
            ))
    fig.update_layout(
        title="Mean condition over time (aggregated across seeds)",
        xaxis_title="Timestep", yaxis_title="Mean d",
        height=380, margin=dict(t=50, b=40),
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def make_grouped_action_frequency_chart(groups: list[dict]) -> go.Figure:
    """Grouped bars: mean frequency across seeds with error bars."""
    action_codes = [1, 2, 3]
    fig = go.Figure()
    for idx, g in enumerate(groups):
        seed_freqs = {c: [] for c in action_codes}
        for rd in g["runs"]:
            df = rd["df"]
            n = get_n_assets(df)
            a_cols = [f"a_{i}" for i in range(n) if f"a_{i}" in df.columns]
            total_steps = len(df) * len(a_cols) if a_cols else 1
            for code in action_codes:
                hits = sum((df[c] == code).sum() for c in a_cols) if a_cols else 0
                seed_freqs[code].append(hits / total_steps)
        means = [float(np.mean(seed_freqs[c])) for c in action_codes]
        stds = [float(np.std(seed_freqs[c])) for c in action_codes] if g["n_seeds"] > 1 else [0.0] * 3
        label = f"{g['group_label']} ({g['n_seeds']} seeds)"
        fig.add_trace(go.Bar(
            name=label,
            x=[ACTION_NAMES[c] for c in action_codes],
            y=means,
            error_y=dict(type="data", array=stds, visible=True) if g["n_seeds"] > 1 else None,
            marker_color=_run_color(idx),
        ))
    fig.update_layout(
        title="Action frequency (aggregated across seeds)",
        xaxis_title="Action", yaxis_title="Fraction of steps",
        barmode="group", height=380, margin=dict(t=50, b=40),
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def make_grouped_near_failure_chart(groups: list[dict]) -> go.Figure:
    """One bar per group: mean near-failure rate across seeds with error bar."""
    labels, means, errors, colors = [], [], [], []
    for idx, g in enumerate(groups):
        seed_rates = []
        for rd in g["runs"]:
            df = rd["df"]
            n = get_n_assets(df)
            d_cols = [f"d_{i}" for i in range(n)]
            seed_rates.append(float((df[d_cols].max(axis=1) >= 0.95).mean()))
        arr = np.array(seed_rates)
        label = f"{g['group_label']} ({g['n_seeds']} seeds)"
        labels.append(label)
        means.append(float(arr.mean()))
        errors.append(float(arr.std()) if len(arr) > 1 else 0.0)
        colors.append(_run_color(idx))
    fig = go.Figure(go.Bar(
        x=labels, y=means,
        error_y=dict(type="data", array=errors, visible=True),
        marker_color=colors,
    ))
    fig.update_layout(
        title="Near-failure rate (aggregated across seeds)",
        xaxis_title="Group", yaxis_title="Near-failure rate",
        height=350, margin=dict(t=50, b=80),
        xaxis=dict(tickangle=-30),
    )
    return fig


def make_grouped_training_overlay(groups: list[dict], mwa_window: int = 10) -> tuple[go.Figure, list[str]]:
    """One band per group (mean +/- std across seeds)."""
    fig = go.Figure()
    skipped = []
    for idx, g in enumerate(groups):
        logs_data = []
        for rd in g["runs"]:
            log = rd["log"]
            if log is not None and "mean_cost" in log.columns:
                ep = log["episode"].values if "episode" in log.columns else np.arange(len(log))
                logs_data.append(pd.Series(log["mean_cost"].values, index=ep, dtype=float))
        if not logs_data:
            skipped.append(g["group_label"])
            continue

        # Align on common episode index (concat aligns properly on index)
        combined = pd.concat(logs_data, axis=1)
        mean_y = combined.mean(axis=1)
        std_y = combined.std(axis=1) if combined.shape[1] > 1 else pd.Series(0.0, index=mean_y.index)
        x = mean_y.index.values
        color = _run_color(idx)
        label = f"{g['group_label']} ({g['n_seeds']} seeds)"

        fig.add_trace(go.Scatter(
            x=x, y=mean_y.values, name=label, mode="lines",
            line=dict(color=color, width=2),
        ))
        if combined.shape[1] > 1:
            rgba = _hex_to_rgba(color, 0.15)
            upper = (mean_y + std_y).values
            lower = (mean_y - std_y).clip(lower=0).values
            fig.add_trace(go.Scatter(
                x=list(x) + list(x)[::-1],
                y=list(upper) + list(lower)[::-1],
                fill="toself", fillcolor=rgba,
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False, hoverinfo="skip",
            ))
        # MWA
        mwa = mean_y.rolling(mwa_window, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=x, y=mwa.values, name=f"{label} (MWA-{mwa_window})",
            mode="lines", line=dict(color=color, dash="dot", width=3),
        ))
    fig.update_layout(
        title="Training curves (aggregated across seeds)",
        xaxis_title="Episode", yaxis_title="Mean cost",
        height=400, margin=dict(t=50, b=40),
        legend=dict(orientation="h", y=1.12),
    )
    return fig, skipped


def make_summary_table(groups: list[dict]) -> pd.DataFrame:
    """Build summary statistics DataFrame for grouped runs."""
    rows = []
    for g in groups:
        seed_means, all_costs = _group_ep_costs(g)
        arr = np.array(seed_means)
        n = len(arr)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if n > 1 else 0.0
        se = std / np.sqrt(n) if n > 1 else 0.0
        ci95 = 1.96 * se
        rows.append({
            "Algorithm": g["group_label"],
            "Seeds": n,
            "Mean Cost": f"{mean:,.0f}",
            "Std (seeds)": f"{std:,.0f}" if n > 1 else "-",
            "95% CI": f"+/- {ci95:,.0f}" if n > 1 else "-",
            "Best Seed": f"{min(seed_means):,.0f}",
            "Worst Seed": f"{max(seed_means):,.0f}",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Optuna / best params helpers
# ---------------------------------------------------------------------------

_PER_ASSET_PARAM_THRESHOLD = 10  # if more params than this, treat as per-asset


def make_best_params_table(run_data: list[dict]) -> pd.DataFrame | None:
    """Build a table of best Optuna parameters for runs that have best_params.json."""
    rows = []
    for rd in run_data:
        bp = rd.get("best_params")
        if bp is None:
            continue
        row: dict = {"Run": rd["label"]}
        params = {k: v for k, v in bp.items() if k != "best_value"}
        if len(params) > _PER_ASSET_PARAM_THRESHOLD:
            row["Parameters"] = f"{len(params)} per-asset params"
        else:
            for k, v in params.items():
                row[k] = f"{v:.4f}" if isinstance(v, float) else str(v)
        if "best_value" in bp:
            row["Best Cost"] = f"{bp['best_value']:.4e}"
        rows.append(row)
    return pd.DataFrame(rows) if rows else None


def make_grouped_best_params_table(groups: list[dict]) -> pd.DataFrame | None:
    """Aggregated best-params table: mean +/- std across seeds per group."""
    rows = []
    for g in groups:
        runs_with_bp = [rd for rd in g["runs"] if rd.get("best_params")]
        if not runs_with_bp:
            continue
        row: dict = {"Algorithm": g["group_label"], "Seeds": len(runs_with_bp)}
        # Collect all param names (excluding best_value)
        all_params: dict[str, list[float]] = {}
        best_values: list[float] = []
        for rd in runs_with_bp:
            bp = rd["best_params"]
            for k, v in bp.items():
                if k == "best_value":
                    best_values.append(float(v))
                elif isinstance(v, (int, float)):
                    all_params.setdefault(k, []).append(float(v))
        if len(all_params) > _PER_ASSET_PARAM_THRESHOLD:
            row["Parameters"] = f"{len(all_params)} per-asset params"
        else:
            for k, vals in all_params.items():
                arr = np.array(vals)
                if len(arr) > 1:
                    row[k] = f"{arr.mean():.4f} \u00b1 {arr.std():.4f}"
                else:
                    row[k] = f"{arr[0]:.4f}"
        if best_values:
            arr = np.array(best_values)
            row["Mean Best Cost"] = f"{arr.mean():.4e}"
            if len(arr) > 1:
                row["Std"] = f"{arr.std():.4e}"
        rows.append(row)
    return pd.DataFrame(rows) if rows else None


def _get_per_asset_params_df(best_params: dict) -> pd.DataFrame:
    """Expand per-asset best_params into a tidy DataFrame for display."""
    params = {k: v for k, v in best_params.items() if k != "best_value"}
    # Parse param_name_idx pattern
    import re
    parsed: dict[int, dict[str, float]] = {}
    for k, v in params.items():
        m = re.match(r"(.+)_(\d+)$", k)
        if m:
            name, idx = m.group(1), int(m.group(2))
            parsed.setdefault(idx, {})["asset"] = idx
            parsed[idx][name] = float(v) if isinstance(v, (int, float)) else v
        else:
            parsed.setdefault(-1, {})[k] = v
    if not parsed:
        return pd.DataFrame()
    rows = [parsed[k] for k in sorted(parsed)]
    df = pd.DataFrame(rows)
    if "asset" in df.columns:
        df = df.set_index("asset").sort_index()
        # Format floats
        for col in df.columns:
            if df[col].dtype == float:
                df[col] = df[col].map(lambda x: f"{x:.4f}")
    return df


def make_optuna_convergence_chart(run_data: list[dict]) -> go.Figure | None:
    """Optuna tuning convergence: running minimum of mean_cost per trial."""
    fig = go.Figure()
    has_data = False
    for idx, rd in enumerate(run_data):
        trials = rd.get("optuna_trials")
        if trials is None or "mean_cost" not in trials.columns:
            continue
        has_data = True
        # Sort by trial number
        t = trials.sort_values("trial_number") if "trial_number" in trials.columns else trials
        cost_col = t["mean_cost"].values
        running_min = np.minimum.accumulate(cost_col)
        x = t["trial_number"].values if "trial_number" in t.columns else np.arange(len(t))
        color = _run_color(idx)
        fig.add_trace(go.Scatter(
            x=x, y=running_min, name=rd["label"],
            mode="lines", line=dict(color=color, width=2),
            hovertemplate=f"<b>{rd['label']}</b><br>trial=%{{x}}<br>best cost=%{{y:.4e}}<extra></extra>",
        ))
    if not has_data:
        return None
    fig.update_layout(
        title="Optuna tuning convergence (running best cost)",
        xaxis_title="Trial number",
        yaxis_title="Best cost so far",
        height=400, margin=dict(t=50, b=40),
        legend=dict(orientation="h", y=1.12),
    )
    return fig


def make_grouped_optuna_convergence_chart(groups: list[dict]) -> go.Figure | None:
    """Optuna convergence aggregated across seeds: mean +/- std of running-min curves."""
    fig = go.Figure()
    has_data = False
    for idx, g in enumerate(groups):
        curves = []
        for rd in g["runs"]:
            trials = rd.get("optuna_trials")
            if trials is None or "mean_cost" not in trials.columns:
                continue
            t = trials.sort_values("trial_number") if "trial_number" in trials.columns else trials
            running_min = np.minimum.accumulate(t["mean_cost"].values)
            x = t["trial_number"].values if "trial_number" in t.columns else np.arange(len(t))
            curves.append(pd.Series(running_min, index=x, dtype=float))
        if not curves:
            continue
        has_data = True
        combined = pd.concat(curves, axis=1)
        mean_y = combined.mean(axis=1)
        std_y = combined.std(axis=1) if combined.shape[1] > 1 else pd.Series(0.0, index=mean_y.index)
        x = mean_y.index.values
        color = _run_color(idx)
        label = f"{g['group_label']} ({g['n_seeds']} seeds)"
        fig.add_trace(go.Scatter(
            x=x, y=mean_y.values, name=label, mode="lines",
            line=dict(color=color, width=2),
        ))
        if combined.shape[1] > 1:
            rgba = _hex_to_rgba(color, 0.15)
            upper = (mean_y + std_y).values
            lower = (mean_y - std_y).clip(lower=0).values
            fig.add_trace(go.Scatter(
                x=list(x) + list(x)[::-1],
                y=list(upper) + list(lower)[::-1],
                fill="toself", fillcolor=rgba,
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False, hoverinfo="skip",
            ))
    if not has_data:
        return None
    fig.update_layout(
        title="Optuna tuning convergence (aggregated across seeds)",
        xaxis_title="Trial number",
        yaxis_title="Best cost so far",
        height=400, margin=dict(t=50, b=40),
        legend=dict(orientation="h", y=1.12),
    )
    return fig


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Comparison Dashboard", layout="wide")
st.title("RL Infrastructure Maintenance -- Comparison Dashboard")

# ---- Sidebar ----
with st.sidebar:
    st.header("Settings")
    results_input = st.text_input(
        "Results folder(s) -- comma-separated",
        value=", ".join(all_results_folders),
    )
    all_results_folders = [p.strip() for p in results_input.split(",") if p.strip()]

all_run_paths: list[Path] = []
_seen: set[str] = set()
for _folder in all_results_folders:
    for _p in discover_runs(_folder):
        _key = str(_p.resolve())
        if _key not in _seen:
            _seen.add(_key)
            all_run_paths.append(_p)

if not all_run_paths:
    st.error(f"No runs (directories with `eval_episodes.csv`) found under: {', '.join(all_results_folders)}")
    st.stop()

root_path = Path(os.path.commonpath([str(p.resolve()) for p in all_run_paths]))
all_labels = [run_label(root_path, p) for p in all_run_paths]
label_to_path = dict(zip(all_labels, all_run_paths))

with st.sidebar:
    selected_labels = st.multiselect(
        "Runs to compare",
        options=all_labels,
        default=all_labels,
    )
    if not selected_labels:
        st.warning("Select at least one run.")
        st.stop()

# Load data for all selected runs (before primary selectbox so we can default to best)
run_data: list[dict] = []
for lbl in selected_labels:
    path = label_to_path[lbl]
    df, config, log, buf_pred, eval_pred, agent_metrics, best_params, optuna_trials = load_run(str(path))
    if df is not None:
        _ensure_algorithm_metadata(config)
        # Compute per-epoch gamma from instance file
        _gamma_per_epoch = 1.0
        _inst_path = config.get("instance", "")
        if _inst_path and os.path.exists(_inst_path):
            with open(_inst_path) as _f:
                _inst = json.load(_f)
                _gamma_per_epoch = float(_inst.get("gamma", 0.99)) ** float(_inst.get("dt", 0.5))
        run_data.append({"label": lbl, "path": path, "df": df, "config": config,
                         "log": log, "buf_pred": buf_pred, "eval_pred": eval_pred,
                         "agent_metrics": agent_metrics,
                         "best_params": best_params, "optuna_trials": optuna_trials,
                         "gamma_per_epoch": _gamma_per_epoch})

if not run_data:
    st.error("None of the selected runs could be loaded.")
    st.stop()

# --- Instance mismatch warning ---
_id_to_labels: dict[str, list[str]] = {}
for rd in run_data:
    iid = rd["config"].get("instance_id") or "unknown"
    _id_to_labels.setdefault(iid, []).append(rd["label"])

if len(_id_to_labels) > 1:
    _parts = [
        f"{', '.join(lbls)} -- instance {iid[:8]}..."
        for iid, lbls in _id_to_labels.items()
    ]
    st.warning(
        "**Instance mismatch** -- selected runs were evaluated on different instances:\n\n"
        + "\n\n".join(f"- {p}" for p in _parts)
    )

# --- Grouping sidebar ---
has_metadata = _has_grouping_metadata(run_data)

# Check if any group would have >1 run
_test_groups_config = group_runs(run_data, "config") if has_metadata else []
_any_multi_seed = any(g["n_seeds"] > 1 for g in _test_groups_config)

with st.sidebar:
    st.markdown("---")
    grouping_options = ["None", "Config (aggregate seeds)", "Class (aggregate all)"]
    default_group_idx = 1 if (has_metadata and _any_multi_seed) else 0
    grouping_choice = st.radio(
        "Group by",
        grouping_options,
        index=default_group_idx,
        help="Config: same hyperparameters, aggregate seeds. Class: same algorithm family, aggregate everything.",
    )

grouping_level: str | None = None
if grouping_choice == "Config (aggregate seeds)":
    grouping_level = "config"
elif grouping_choice == "Class (aggregate all)":
    grouping_level = "class"

grouped_data: list[dict] | None = None
if grouping_level and has_metadata:
    grouped_data = group_runs(run_data, grouping_level)

# Default primary run = lowest average episode cost
_mean_costs = {rd["label"]: _ep_discounted_costs(rd["df"], rd["gamma_per_epoch"]).mean() for rd in run_data}
_best_label = min(_mean_costs, key=_mean_costs.__getitem__)
_primary_default_idx = selected_labels.index(_best_label) if _best_label in selected_labels else 0

with st.sidebar:
    primary_label = st.selectbox(
        "Primary run (episode viewer & drill-down)",
        selected_labels,
        index=_primary_default_idx,
    )

# Primary run
primary_rd = next((r for r in run_data if r["label"] == primary_label), run_data[0])
primary_df = primary_rd["df"]
primary_config = primary_rd["config"]
primary_log = primary_rd["log"]
primary_buf_pred = primary_rd["buf_pred"]
primary_eval_pred = primary_rd["eval_pred"]
primary_agent_metrics = primary_rd["agent_metrics"]
n_assets = get_n_assets(primary_df)
episodes = sorted(primary_df["episode"].unique())

dt = 0.5  # default
gamma_per_epoch = 1.0
instance_path = primary_config.get("instance", "")
if instance_path and os.path.exists(instance_path):
    with open(instance_path) as f_inst:
        _inst_data = json.load(f_inst)
        dt = _inst_data.get("dt", 0.5)
        gamma_per_epoch = float(_inst_data.get("gamma", 0.99)) ** float(dt)
n_episodes = len(episodes)

with st.sidebar:
    ep_idx = st.slider("Episode (primary run)", min_value=0, max_value=n_episodes - 1, value=0)
    selected_ep = episodes[ep_idx]
    asset_idx = st.slider("Asset (detail panel)", min_value=0, max_value=n_assets - 1, value=0)
    default_gamma = float(primary_config.get("gamma", 0.99))
    gamma = st.number_input("Gamma (discount)", min_value=0.0, max_value=1.0,
                            value=default_gamma, step=0.01, format="%.3f")
    st.markdown("---")
    st.markdown(
        f"**Primary run:** {primary_label}  \n"
        f"**Episodes:** {n_episodes}  \n"
        f"**Assets:** {n_assets}  \n"
        f"**Timesteps/ep:** {len(primary_df[primary_df['episode'] == selected_ep])}"
    )

# ---- Tabs ----
tabs = st.tabs(["Agent comparison", "Episode rollout", "Cross-episode summary",
                "Training curves", "Value function", "State diversity", "Problem structure"])

# ======================================================================
# Tab 1 -- Agent comparison
# ======================================================================
with tabs[0]:
    if len(run_data) < 2:
        st.info("Select at least 2 runs in the sidebar to see a comparison.")
    elif grouped_data is not None and len(grouped_data) >= 1:
        # --- Aggregated view ---
        st.subheader("Summary statistics (aggregated)")
        st.dataframe(make_summary_table(grouped_data), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Mean total cost (aggregated)")
        st.plotly_chart(make_grouped_mean_cost_chart(grouped_data), use_container_width=True)

        st.markdown("---")
        st.subheader("Cost distribution (aggregated)")
        st.plotly_chart(make_grouped_cost_box_chart(grouped_data), use_container_width=True)

        st.markdown("---")
        st.subheader("Mean condition over time (aggregated)")
        st.plotly_chart(make_grouped_condition_chart(grouped_data), use_container_width=True)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Action frequency (aggregated)")
            st.plotly_chart(make_grouped_action_frequency_chart(grouped_data), use_container_width=True)
        with col2:
            st.subheader("Near-failure rate (aggregated)")
            st.plotly_chart(make_grouped_near_failure_chart(grouped_data), use_container_width=True)

        # Tuned parameters (grouped)
        _gp_table = make_grouped_best_params_table(grouped_data)
        if _gp_table is not None:
            st.markdown("---")
            st.subheader("Tuned Parameters (aggregated)")
            st.dataframe(_gp_table, use_container_width=True, hide_index=True)
            # Per-asset drill-down for groups with many params
            for g in grouped_data:
                runs_with_bp = [rd for rd in g["runs"] if rd.get("best_params")]
                if runs_with_bp and len({k for k in runs_with_bp[0]["best_params"] if k != "best_value"}) > _PER_ASSET_PARAM_THRESHOLD:
                    with st.expander(f"Per-asset params: {g['group_label']}"):
                        for rd in runs_with_bp:
                            st.caption(rd["label"])
                            st.dataframe(_get_per_asset_params_df(rd["best_params"]), use_container_width=True)

        # Drill-down: per-seed detail
        with st.expander("Per-seed detail (individual runs)"):
            stats = []
            for rd in run_data:
                ep_costs = _ep_discounted_costs(rd["df"], rd["gamma_per_epoch"])
                stats.append({
                    "label": rd["label"],
                    "mean_cost": ep_costs.mean(),
                    "std_cost": ep_costs.std(),
                })
            st.plotly_chart(make_mean_cost_chart(stats), use_container_width=True)
            st.plotly_chart(make_cost_box_chart(run_data), use_container_width=True)
    else:
        # --- Original per-run view ---
        stats = []
        for rd in run_data:
            ep_costs = _ep_discounted_costs(rd["df"], rd["gamma_per_epoch"])
            stats.append({
                "label": rd["label"],
                "mean_cost": ep_costs.mean(),
                "std_cost": ep_costs.std(),
            })

        st.subheader("Mean total cost +/- std")
        st.plotly_chart(make_mean_cost_chart(stats), use_container_width=True)

        st.markdown("---")
        st.subheader("Cost distribution")
        st.plotly_chart(make_cost_box_chart(run_data), use_container_width=True)

        st.markdown("---")
        st.subheader("Mean condition over time")
        st.plotly_chart(make_mean_condition_chart(run_data), use_container_width=True)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Action frequency")
            st.plotly_chart(make_action_frequency_chart(run_data), use_container_width=True)
        with col2:
            st.subheader("Near-failure rate")
            st.plotly_chart(make_near_failure_chart(run_data), use_container_width=True)

        # Tuned parameters (ungrouped)
        _bp_table = make_best_params_table(run_data)
        if _bp_table is not None:
            st.markdown("---")
            st.subheader("Tuned Parameters")
            st.dataframe(_bp_table, use_container_width=True, hide_index=True)
            # Per-asset drill-down
            for rd in run_data:
                bp = rd.get("best_params")
                if bp and len({k for k in bp if k != "best_value"}) > _PER_ASSET_PARAM_THRESHOLD:
                    with st.expander(f"Per-asset params: {rd['label']}"):
                        st.dataframe(_get_per_asset_params_df(bp), use_container_width=True)

        _runs_with_metrics = [rd for rd in run_data if rd["agent_metrics"] is not None]
        if _runs_with_metrics:
            st.markdown("---")
            st.subheader("Search efficiency (mean candidates per timestep)")
            _eff_labels, _eff_means, _eff_colors = [], [], []
            for idx, rd in enumerate(run_data):
                am = rd["agent_metrics"]
                if am is not None and "n_candidates" in am.columns:
                    _eff_labels.append(rd["label"])
                    _eff_means.append(float(am["n_candidates"].mean()))
                    _eff_colors.append(_run_color(idx))
            if _eff_labels:
                _eff_fig = go.Figure(go.Bar(
                    x=_eff_means, y=_eff_labels, orientation="h",
                    marker_color=_eff_colors,
                    hovertemplate="<b>%{y}</b><br>mean candidates=%{x:.1f}<extra></extra>",
                ))
                _eff_fig.update_layout(
                    xaxis_title="Mean n_candidates per timestep",
                    yaxis_title="Run",
                    height=max(250, 60 + 40 * len(_eff_labels)),
                    margin=dict(t=20, b=40, l=200),
                )
                st.plotly_chart(_eff_fig, use_container_width=True)

# ======================================================================
# Tab 2 -- Episode rollout (primary run)
# ======================================================================
with tabs[1]:
    st.subheader(f"Episode {selected_ep} -- {primary_label}")
    ep_df = get_episode(primary_df, selected_ep)

    log_y = st.checkbox("Log-scale y-axis (cost chart)", value=True)
    st.plotly_chart(make_cost_chart(ep_df, gamma, log_y, dt=dt), use_container_width=True)

    st.markdown("---")
    st.plotly_chart(make_heatmap(ep_df, n_assets, dt=dt), use_container_width=True)

    st.markdown("---")
    st.plotly_chart(make_asset_detail(ep_df, asset_idx, dt=dt), use_container_width=True)

    if primary_agent_metrics is not None and "n_candidates" in primary_agent_metrics.columns:
        st.markdown("---")
        st.subheader("Search depth: candidates evaluated per timestep")
        _am_ep = primary_agent_metrics[primary_agent_metrics["episode"] == selected_ep]
        if not _am_ep.empty:
            _cand_fig = go.Figure(go.Scatter(
                x=_am_ep["t"], y=_am_ep["n_candidates"],
                mode="lines+markers",
                line=dict(width=2),
                hovertemplate="t=%{x}<br>n_candidates=%{y}<extra></extra>",
            ))
            _cand_fig.update_layout(
                xaxis_title="Timestep",
                yaxis_title="n_candidates",
                height=300,
                margin=dict(t=20, b=40),
            )
            st.plotly_chart(_cand_fig, use_container_width=True)

# ======================================================================
# Tab 3 -- Cross-episode summary (primary run)
# ======================================================================
with tabs[2]:
    st.subheader(f"Cross-episode summary -- {primary_label}")
    with st.spinner("Computing cross-episode charts..."):
        hist_fig, act_fig, fail_fig, cond_fig = make_cross_episode_charts(
            primary_df, n_assets, selected_ep, dt=dt, gamma_per_epoch=gamma_per_epoch
        )
    st.plotly_chart(hist_fig, use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(act_fig, use_container_width=True)
    with col2:
        st.plotly_chart(fail_fig, use_container_width=True)
    st.plotly_chart(cond_fig, use_container_width=True)
    st.markdown("---")
    breakdown_fig = make_cost_breakdown_chart(primary_df, dt=dt)
    st.plotly_chart(breakdown_fig, use_container_width=True)

# ======================================================================
# Tab 4 -- Training curves overlay
# ======================================================================
with tabs[3]:
    st.subheader("Training curves (all selected runs)")
    mwa_window = st.slider("Moving average window (episodes)", min_value=1,
                           max_value=100, value=10, step=1)
    if grouped_data is not None:
        curve_fig, skipped = make_grouped_training_overlay(grouped_data, mwa_window=mwa_window)
        st.plotly_chart(curve_fig, use_container_width=True)
        if skipped:
            st.info(f"No training log found for: {', '.join(skipped)}")
        with st.expander("Per-seed training curves"):
            per_seed_fig, per_seed_skipped = make_training_overlay(run_data, mwa_window=mwa_window)
            st.plotly_chart(per_seed_fig, use_container_width=True)
    else:
        curve_fig, skipped = make_training_overlay(run_data, mwa_window=mwa_window)
        st.plotly_chart(curve_fig, use_container_width=True)
        if skipped:
            st.info(f"No training log found for: {', '.join(skipped)}")

    # Optuna tuning convergence
    _any_optuna = any(rd.get("optuna_trials") is not None for rd in run_data)
    if _any_optuna:
        st.markdown("---")
        st.subheader("Optuna tuning convergence")
        if grouped_data is not None:
            _optuna_fig = make_grouped_optuna_convergence_chart(grouped_data)
        else:
            _optuna_fig = make_optuna_convergence_chart(run_data)
        if _optuna_fig is not None:
            st.plotly_chart(_optuna_fig, use_container_width=True)

# ======================================================================
# Tab 5 -- Value function diagnostics (primary run)
# ======================================================================
with tabs[4]:
    st.subheader(f"Value function diagnostics -- {primary_label}")
    if primary_buf_pred is None and primary_eval_pred is None:
        st.info("No VF prediction data found. Run with a learning agent.")
    else:
        if primary_buf_pred is not None:
            st.subheader("Buffer: predicted V(s_post) vs residual")
            st.plotly_chart(
                make_pred_residual_chart(primary_buf_pred, 'v_pred', 'mc_return',
                                         'Buffer predictions'),
                use_container_width=True,
            )
        if primary_eval_pred is not None:
            st.subheader("Eval episodes: predicted V(s_post) vs realized return")
            st.plotly_chart(
                make_pred_residual_chart(primary_eval_pred, 'v_pred', 'realized_return',
                                         'Eval predictions', color_col='episode'),
                use_container_width=True,
            )

# ======================================================================
# Tab 6 -- State diversity (primary run)
# ======================================================================
with tabs[5]:
    st.subheader(f"State diversity -- {primary_label}")
    with st.spinner("Computing state diversity charts..."):
        div_charts = make_state_diversity_charts(primary_df, n_assets)

    st.plotly_chart(div_charts['coverage'], use_container_width=True)
    st.markdown("---")
    st.plotly_chart(div_charts['violin'], use_container_width=True)
    st.markdown("---")
    pca_note = div_charts.get('pca_note', '')
    if pca_note:
        st.caption(f"PCA note: {pca_note} -- color = training progress (dark = early, light = late)")
    st.plotly_chart(div_charts['pca'], use_container_width=True)

# ======================================================================
# Tab 7 -- Problem structure (fan charts)
# ======================================================================
with tabs[6]:
    st.subheader(f"Problem structure -- {primary_label}")

    ps_asset = st.selectbox("Asset", range(n_assets), index=0, key="ps_asset")
    show_lines = st.checkbox("Show individual episode lines", value=True, key="ps_show_lines")

    st.subheader("Degradation path spread")
    with st.spinner("Computing degradation fan..."):
        deg_fan = make_degradation_fan(primary_df, ps_asset, dt=dt, show_lines=show_lines)
    st.plotly_chart(deg_fan, use_container_width=True)

    st.subheader("Renovation path spread")
    with st.spinner("Computing renovation fans..."):
        h_fan, dur_hist = make_renovation_fan(primary_df, ps_asset, dt=dt, show_lines=show_lines)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(h_fan, use_container_width=True)
    with col2:
        st.plotly_chart(dur_hist, use_container_width=True)
