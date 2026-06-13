"""Interactive episode visualizer for RL infrastructure maintenance experiments.

Launch:
    streamlit run vis/episode_viewer.py -- --results results/exp1_paced
Or simply run this file directly from PyCharm.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__" and not os.environ.get("_EPISODE_VIEWER_LAUNCHED"):
    if "--results" not in sys.argv:
        results_root = Path("results")
        candidates = sorted([d for d in results_root.iterdir() if d.is_dir()]) if results_root.exists() else []
        if not candidates:
            print("No results directories found under 'results/'. Pass --results <path> manually.")
            sys.exit(1)
        print("Available results:")
        for i, d in enumerate(candidates):
            print(f"  [{i}] {d}")
        choice = input("Select a results directory (number): ").strip()
        results_path = str(candidates[int(choice)])
    else:
        results_path = sys.argv[sys.argv.index("--results") + 1]
    env = {**os.environ, "_EPISODE_VIEWER_LAUNCHED": "1"}
    subprocess.run([sys.executable, "-m", "streamlit", "run", __file__, "--", "--results", results_path], env=env)
    sys.exit()

import pandas as pd
import streamlit as st

from vis._charts import (  # noqa: E402
    ACTION_NAMES, ACTION_SYMBOLS, ACTION_COLORS,
    get_n_assets, get_episode,
    make_cost_chart, make_heatmap, make_asset_detail,
    make_cross_episode_charts, make_training_curve,
    make_state_diversity_charts,
)

# ---------------------------------------------------------------------------
# CLI argument (results directory passed after --)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--results", default="results")
args, _ = parser.parse_known_args()

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_data(results_dir: str):
    p = Path(results_dir)
    episodes_path = p / "eval_episodes.csv"
    if not episodes_path.exists():
        return None, {}, None
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
    return df, config, log


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Episode Viewer", layout="wide")
st.title("RL Infrastructure Maintenance — Episode Viewer")

# Sidebar
with st.sidebar:
    st.header("Settings")
    results_dir = st.text_input("Results directory", value=args.results)

df, config, log = load_data(results_dir)

if df is None:
    st.error(f"No `eval_episodes.csv` found in `{results_dir}`. "
             "Run an experiment first, or pass `--results <path>` on launch.")
    st.stop()

n_assets = get_n_assets(df)
episodes = sorted(df["episode"].unique())
n_episodes = len(episodes)

with st.sidebar:
    ep_idx = st.slider("Episode", min_value=0, max_value=n_episodes - 1, value=0)
    selected_ep = episodes[ep_idx]
    asset_idx = st.slider("Asset (for detail panel)", min_value=0, max_value=n_assets - 1, value=0)
    default_gamma = float(config.get("gamma", 0.99))
    gamma = st.number_input("Gamma (discount)", min_value=0.0, max_value=1.0,
                            value=default_gamma, step=0.01, format="%.3f")
    st.markdown("---")
    st.markdown(f"**Episodes:** {n_episodes}  \n**Assets:** {n_assets}  \n"
                f"**Timesteps/ep:** {len(df[df['episode'] == selected_ep])}")

# Build tabs
tab_labels = ["Episode rollout", "Cross-episode summary"]
if log is not None:
    tab_labels.append("Training curve")
tab_labels.append("State diversity")
tabs = st.tabs(tab_labels)

# ---- Tab 1: Episode rollout ----
with tabs[0]:
    ep_df = get_episode(df, selected_ep)
    st.subheader(f"Episode {selected_ep}")

    log_y = st.checkbox("Log-scale y-axis (cost chart)", value=True)
    st.plotly_chart(make_cost_chart(ep_df, gamma, log_y), use_container_width=True)

    st.markdown("---")
    st.plotly_chart(make_heatmap(ep_df, n_assets), use_container_width=True)

    st.markdown("---")
    st.plotly_chart(make_asset_detail(ep_df, asset_idx), use_container_width=True)

# ---- Tab 2: Cross-episode summary ----
with tabs[1]:
    st.subheader("Cross-episode summary")
    with st.spinner("Computing cross-episode charts…"):
        hist_fig, act_fig, fail_fig, cond_fig = make_cross_episode_charts(df, n_assets, selected_ep)

    st.plotly_chart(hist_fig, use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(act_fig, use_container_width=True)
    with col2:
        st.plotly_chart(fail_fig, use_container_width=True)
    st.plotly_chart(cond_fig, use_container_width=True)

# ---- Tab 3: Training curve (optional) ----
if log is not None:
    with tabs[2]:
        st.subheader("Training curve")
        st.plotly_chart(make_training_curve(log), use_container_width=True)
        with st.expander("Raw training log"):
            st.dataframe(log, use_container_width=True)

# ---- Tab 4: State diversity ----
with tabs[-1]:
    st.subheader("State diversity")
    with st.spinner("Computing state diversity charts…"):
        div_charts = make_state_diversity_charts(df, n_assets)

    st.plotly_chart(div_charts['coverage'], use_container_width=True)
    st.markdown("---")
    st.plotly_chart(div_charts['violin'], use_container_width=True)
    st.markdown("---")
    pca_note = div_charts.get('pca_note', '')
    if pca_note:
        st.caption(f"PCA note: {pca_note} — color = training progress (dark = early, light = late)")
    st.plotly_chart(div_charts['pca'], use_container_width=True)
