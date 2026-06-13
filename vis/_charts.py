"""Shared pure chart functions for RL infrastructure maintenance visualisations.

All functions return Plotly figures with no Streamlit calls.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTION_NAMES = {0: "Do-nothing", 1: "Repair", 2: "Renovate", 3: "Restrict"}
ACTION_SYMBOLS = {0: None, 1: "diamond", 2: "square", 3: "triangle-up"}
ACTION_COLORS = {0: None, 1: "blue", 2: "orange", 3: "purple"}

BASE_YEAR = 2026
YEAR_AXIS = dict(title="Year", tick0=BASE_YEAR, dtick=10)


def _to_year(t, dt: float):
    return BASE_YEAR + np.asarray(t) * dt

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def get_n_assets(df: pd.DataFrame) -> int:
    return sum(1 for c in df.columns if c.startswith("d_"))


def get_episode(df: pd.DataFrame, ep: int) -> pd.DataFrame:
    return df[df["episode"] == ep].sort_values("t").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------


def make_cost_chart(ep_df: pd.DataFrame, gamma: float, log_y: bool,
                    dt: float = 0.5) -> go.Figure:
    t = _to_year(ep_df["t"].values, dt)
    cost = ep_df["cost"].values
    cum_disc = [sum(gamma ** i * cost[i] for i in range(k + 1)) for k in range(len(cost))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=cost, name="Step cost",
        line=dict(color="steelblue"),
        hovertemplate="year=%{x}<br>cost=%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=t, y=cum_disc, name="Cumulative discounted",
        line=dict(color="darkorange", dash="dash"),
        hovertemplate="year=%{x}<br>cum=%{y:.2f}<extra></extra>",
        yaxis="y2",
    ))

    # Failure markers (any asset at d >= 0.95)
    n_assets = get_n_assets(ep_df)
    d_cols = [f"d_{i}" for i in range(n_assets)]
    fail_mask = ep_df[d_cols].max(axis=1) >= 0.95
    fail_t = t[fail_mask.values]
    if len(fail_t):
        for ft in fail_t:
            fig.add_vline(x=ft, line=dict(color="red", dash="dot", width=1), opacity=0.4)

    fig.update_layout(
        title="Cost over time",
        xaxis=YEAR_AXIS,
        yaxis_title="Step cost",
        yaxis2=dict(title="Cumulative discounted cost", overlaying="y", side="right",
                    type="log" if log_y else "linear"),
        yaxis_type="log" if log_y else "linear",
        legend=dict(orientation="h", y=1.1),
        height=300,
        margin=dict(t=40, b=30),
    )
    return fig


def make_heatmap(ep_df: pd.DataFrame, n_assets: int, dt: float = 0.5) -> go.Figure:
    d_cols = [f"d_{i}" for i in range(n_assets)]
    pivot = ep_df[["t"] + d_cols].set_index("t")[d_cols].T
    pivot.index = list(range(n_assets))
    pivot.columns = _to_year(pivot.columns, dt)

    h_cols = [f"h_{i}" for i in range(n_assets)]
    if all(c in ep_df.columns for c in h_cols):
        h_pivot = ep_df[["t"] + h_cols].set_index("t")[h_cols].T
        h_pivot.index = list(range(n_assets))
        h_pivot.columns = _to_year(h_pivot.columns, dt)
        ren_z = np.where(h_pivot.values > 0, 1.0, np.nan)
    else:
        ren_z = None

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=list(range(n_assets)),
        colorscale=[[0, "green"], [0.5, "yellow"], [1, "red"]],
        zmin=0, zmax=1,
        colorbar=dict(title="d"),
        hovertemplate="asset=%{y}<br>t=%{x}<br>d=%{z:.3f}<extra></extra>",
    ))

    if ren_z is not None:
        fig.add_trace(go.Heatmap(
            z=ren_z,
            x=pivot.columns.tolist(),
            y=list(range(n_assets)),
            colorscale=[[0, "grey"], [1, "grey"]],
            zmin=0, zmax=1,
            showscale=False,
            hovertemplate="asset=%{y}<br>t=%{x}<br>renovating<extra></extra>",
            name="renovating",
        ))

    for action_code, symbol, color in [
        (1, "diamond", "blue"),
        (2, "square", "darkorange"),
        (3, "triangle-up", "purple"),
    ]:
        xs, ys, texts = [], [], []
        for i in range(n_assets):
            a_col = f"a_{i}"
            if a_col not in ep_df.columns:
                continue
            mask = ep_df[a_col] == action_code
            rows = ep_df[mask]
            xs.extend(_to_year(rows["t"], dt).tolist())
            ys.extend([i] * len(rows))
            for _, row in rows.iterrows():
                texts.append(
                    f"asset={i}, t={int(row['t'])}<br>"
                    f"action={ACTION_NAMES[action_code]}<br>"
                    f"d={row[f'd_{i}']:.3f}"
                )
        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers",
                marker=dict(symbol=symbol, color=color, size=8, line=dict(width=1, color="white")),
                name=ACTION_NAMES[action_code],
                text=texts,
                hovertemplate="%{text}<extra></extra>",
            ))

    fig.update_layout(
        title="Asset condition heatmap (d)",
        xaxis=YEAR_AXIS,
        yaxis_title="Asset",
        yaxis=dict(tickmode="linear", dtick=1),
        height=450,
        margin=dict(t=40, b=30),
        legend=dict(orientation="h", y=1.05),
    )
    return fig


def make_asset_detail(ep_df: pd.DataFrame, asset_idx: int, dt: float = 0.5) -> go.Figure:
    t = _to_year(ep_df["t"].values, dt)
    d = ep_df[f"d_{asset_idx}"].values
    h = ep_df[f"h_{asset_idx}"].values
    ell = ep_df[f"ell_{asset_idx}"].values
    r = ep_df[f"r_{asset_idx}"].values if f"r_{asset_idx}" in ep_df.columns else None

    fig = go.Figure()

    # Renovation bands (h > 0)
    in_band = False
    band_start = None
    for idx, (ti, hi) in enumerate(zip(t, h)):
        if hi > 0 and not in_band:
            in_band = True
            band_start = ti
        elif hi == 0 and in_band:
            fig.add_vrect(x0=band_start, x1=ti,
                          fillcolor="orange", opacity=0.15, line_width=0,
                          annotation_text="reno", annotation_position="top left")
            in_band = False
    if in_band:
        fig.add_vrect(x0=band_start, x1=t[-1],
                      fillcolor="orange", opacity=0.15, line_width=0)

    # Restriction bands (ell == 1)
    in_band = False
    for ti, ei in zip(t, ell):
        if ei == 1 and not in_band:
            in_band = True
            band_start = ti
        elif ei == 0 and in_band:
            fig.add_vrect(x0=band_start, x1=ti,
                          fillcolor="purple", opacity=0.12, line_width=0,
                          annotation_text="restr", annotation_position="top right")
            in_band = False
    if in_band:
        fig.add_vrect(x0=band_start, x1=t[-1],
                      fillcolor="purple", opacity=0.12, line_width=0)

    fig.add_trace(go.Scatter(x=t, y=d, name="d (condition)", line=dict(color="crimson", width=2),
                             hovertemplate="t=%{x}<br>d=%{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=t, y=h, name="h (reno counter)", line=dict(color="orange", dash="dash"),
                             hovertemplate="t=%{x}<br>h=%{y}<extra></extra>"))
    fig.add_trace(go.Scatter(x=t, y=ell, name="ell (restriction)", line=dict(color="purple", dash="dot"),
                             hovertemplate="t=%{x}<br>ell=%{y}<extra></extra>"))
    if r is not None:
        fig.add_trace(go.Scatter(x=t, y=r, name="r", line=dict(color="gray", dash="longdash"),
                                 hovertemplate="t=%{x}<br>r=%{y:.3f}<extra></extra>"))

    a_col = f"a_{asset_idx}"
    if a_col in ep_df.columns:
        for action_code in [1, 2, 3]:
            mask = ep_df[a_col] == action_code
            rows = ep_df[mask]
            if len(rows):
                fig.add_trace(go.Scatter(
                    x=_to_year(rows["t"].values, dt).tolist(),
                    y=rows[f"d_{asset_idx}"].tolist(),
                    mode="markers",
                    marker=dict(symbol=ACTION_SYMBOLS[action_code],
                                color=ACTION_COLORS[action_code], size=10,
                                line=dict(width=1, color="white")),
                    name=ACTION_NAMES[action_code],
                    hovertemplate=f"action={ACTION_NAMES[action_code]}<br>t=%{{x}}<br>d=%{{y:.3f}}<extra></extra>",
                ))

    fig.update_layout(
        title=f"Asset {asset_idx} — detail",
        xaxis=YEAR_AXIS,
        yaxis_title="Value",
        height=380,
        margin=dict(t=40, b=30),
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def make_cross_episode_charts(df: pd.DataFrame, n_assets: int, selected_ep: int,
                               dt: float = 0.5, gamma_per_epoch: float = 1.0):
    """Returns (hist_fig, action_freq_fig, fail_heatmap_fig, mean_cond_fig)."""

    # --- Cost distribution ---
    if gamma_per_epoch >= 1.0:
        ep_costs_s = df.groupby("episode")["cost"].sum()
    else:
        ep_costs_s = df.groupby("episode")[["t", "cost"]].apply(
            lambda g: float((gamma_per_epoch ** g["t"].values * g["cost"].values).sum())
        )
    ep_costs = ep_costs_s.reset_index(name="total_cost")
    colors = ["red" if e == selected_ep else "steelblue" for e in ep_costs["episode"]]
    hist_fig = go.Figure(go.Bar(
        x=ep_costs["episode"], y=ep_costs["total_cost"],
        marker_color=colors,
        hovertemplate="episode=%{x}<br>total cost=%{y:.2f}<extra></extra>",
    ))
    hist_fig.update_layout(
        title="Total cost per episode (selected episode in red)",
        xaxis_title="Episode", yaxis_title="Total cost",
        yaxis=dict(rangemode="tozero"),
        height=300, margin=dict(t=40, b=30),
    )

    # --- Action frequency heatmap ---
    action_records = []
    for i in range(n_assets):
        a_col = f"a_{i}"
        if a_col not in df.columns:
            continue
        for code, name in ACTION_NAMES.items():
            if code == 0:
                continue
            freq = (df[a_col] == code).groupby(df["episode"]).mean().mean()
            action_records.append({"asset": i, "action": name, "frequency": freq})
    act_df = pd.DataFrame(action_records)
    if not act_df.empty:
        act_pivot = act_df.pivot(index="asset", columns="action", values="frequency").fillna(0)
        act_fig = px.imshow(
            act_pivot,
            labels=dict(x="Action", y="Asset", color="Frequency"),
            color_continuous_scale="Blues",
            title="Action frequency (mean over episodes)",
            height=400,
        )
        act_fig.update_layout(margin=dict(t=40, b=30))
    else:
        act_fig = go.Figure()

    # --- Failure heatmap ---
    episodes = df["episode"].unique()
    fail_matrix = pd.DataFrame(index=range(n_assets), columns=episodes, dtype=float)
    for ep in episodes:
        ep_df = df[df["episode"] == ep]
        for i in range(n_assets):
            fail_matrix.loc[i, ep] = (ep_df[f"d_{i}"] >= 0.95).sum()
    fail_fig = px.imshow(
        fail_matrix.astype(float),
        labels=dict(x="Episode", y="Asset", color="Near-failure steps"),
        color_continuous_scale="OrRd",
        title="Near-failure timesteps per asset × episode (d ≥ 0.95)",
        height=400,
    )
    fail_fig.update_layout(margin=dict(t=40, b=30))

    # --- Mean condition per asset over time ---
    d_cols = [f"d_{i}" for i in range(n_assets)]
    mean_cond = df.groupby("t")[d_cols].mean().reset_index()
    cond_fig = go.Figure()
    for i in range(n_assets):
        cond_fig.add_trace(go.Scatter(
            x=_to_year(mean_cond["t"], dt), y=mean_cond[f"d_{i}"],
            name=f"asset {i}", mode="lines",
            line=dict(width=1),
            hovertemplate=f"asset {i}<br>year=%{{x}}<br>mean d=%{{y:.3f}}<extra></extra>",
        ))
    cond_fig.update_layout(
        title="Mean condition per asset over time (all episodes)",
        xaxis=YEAR_AXIS, yaxis_title="Mean d",
        height=400, margin=dict(t=40, b=30),
        legend=dict(orientation="h", y=-0.2),
    )
    return hist_fig, act_fig, fail_fig, cond_fig


def make_pred_residual_chart(df: pd.DataFrame, pred_col: str, target_col: str,
                              title: str, color_col: str | None = None) -> go.Figure:
    """Scatter: predicted value (x) vs residual = actual − predicted (y).
    Horizontal zero line indicates perfect predictions.
    """
    residuals = df[target_col] - df[pred_col]
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color='gray', dash='dash', width=1))
    scatter_kw: dict = dict(x=df[pred_col], y=residuals, mode='markers',
                            marker=dict(size=4, opacity=0.5))
    if color_col and color_col in df.columns:
        scatter_kw['marker']['color'] = df[color_col]
        scatter_kw['marker']['colorscale'] = 'Viridis'
        scatter_kw['marker']['showscale'] = True
    fig.add_trace(go.Scatter(**scatter_kw,
                             hovertemplate='v_pred=%{x:.2f}<br>residual=%{y:.2f}<extra></extra>'))
    fig.update_layout(title=title, xaxis_title='Predicted V(s_post)',
                      yaxis_title='Residual (actual − predicted)',
                      height=380, margin=dict(t=40, b=30))
    return fig


def make_state_diversity_charts(df: pd.DataFrame, n_assets: int) -> dict:
    """Returns dict of Plotly figures for the State Diversity tab.

    Keys: 'pca', 'violin', 'coverage'.
    """
    import numpy as np

    var_groups = ['d', 'h', 'ell', 'r', 'n_fail']
    state_cols = [f'{v}_{i}' for v in var_groups for i in range(n_assets)]
    # Keep only columns that exist (graceful: older CSVs may lack n_fail_*)
    state_cols = [c for c in state_cols if c in df.columns]

    episodes = sorted(df['episode'].unique())
    n_ep = len(episodes)

    # ------------------------------------------------------------------ #
    # Chart C — Coverage index over training (lightweight, computed first)
    # ------------------------------------------------------------------ #
    coverage_records = []
    for ep in episodes:
        ep_df = df[df['episode'] == ep]
        rec = {'episode': ep}
        for v in ['d', 'h', 'n_fail']:
            cols = [f'{v}_{i}' for i in range(n_assets) if f'{v}_{i}' in df.columns]
            if cols:
                rec[v] = ep_df[cols].std(axis=1).mean()
        coverage_records.append(rec)
    cov_df = pd.DataFrame(coverage_records)

    cov_fig = go.Figure()
    colors_cov = {'d': 'crimson', 'h': 'darkorange', 'n_fail': 'steelblue'}
    for v, color in colors_cov.items():
        if v in cov_df.columns:
            cov_fig.add_trace(go.Scatter(
                x=cov_df['episode'], y=cov_df[v], name=v,
                line=dict(color=color, width=2),
                hovertemplate=f'{v}: ep=%{{x}}<br>coverage=%{{y:.4f}}<extra></extra>',
            ))
    cov_fig.update_layout(
        title='Coverage index over training — mean std across assets per episode',
        xaxis_title='Episode', yaxis_title='Mean cross-asset std',
        height=320, margin=dict(t=40, b=30),
        legend=dict(orientation='h', y=1.1),
    )

    # ------------------------------------------------------------------ #
    # Chart B — Per-variable violin distributions (5 buckets × 5 groups)
    # ------------------------------------------------------------------ #
    K = 5
    bucket_size = max(1, n_ep // K)
    ep_arr = np.array(episodes)
    bucket_labels = []
    bucket_map = {}
    for k in range(K):
        lo = k * bucket_size
        hi = (k + 1) * bucket_size if k < K - 1 else n_ep
        label = f'ep {ep_arr[lo]}–{ep_arr[min(hi, n_ep) - 1]}'
        bucket_labels.append(label)
        for ep in ep_arr[lo:hi]:
            bucket_map[ep] = label

    violin_fig = go.Figure()
    # One trace per (group, bucket); use legendgroup to associate
    group_colors = {'d': 'crimson', 'h': 'darkorange', 'ell': 'purple',
                    'r': 'gray', 'n_fail': 'steelblue'}
    # subplot-style via xaxis positions (offsetgroups)
    # Use a simple single-axis multi-group violin approach
    for v in var_groups:
        cols = [f'{v}_{i}' for i in range(n_assets) if f'{v}_{i}' in df.columns]
        if not cols:
            continue
        for k, label in enumerate(bucket_labels):
            bucket_eps = [ep for ep, bl in bucket_map.items() if bl == label]
            vals = df[df['episode'].isin(bucket_eps)][cols].values.flatten()
            violin_fig.add_trace(go.Violin(
                y=vals,
                name=label,
                legendgroup=label,
                showlegend=(v == var_groups[0]),
                x0=v,
                offsetgroup=label,
                box_visible=True,
                meanline_visible=True,
                line_color=group_colors.get(v, 'black'),
                opacity=0.5 + 0.1 * k,
                hoverinfo='y',
            ))
    violin_fig.update_layout(
        title='Per-variable distribution evolution across training buckets',
        xaxis_title='State variable group', yaxis_title='Value',
        violinmode='group',
        height=420, margin=dict(t=40, b=30),
        legend=dict(orientation='h', y=1.08),
    )

    # ------------------------------------------------------------------ #
    # Chart A — PCA 2D projection (sklearn optional)
    # ------------------------------------------------------------------ #
    pca_note = ''
    try:
        from sklearn.decomposition import PCA
        X = df[state_cols].values.astype(float)
        # Replace NaN with 0 (e.g. missing n_fail columns in old files)
        X = np.nan_to_num(X)
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X)
        var_exp = pca.explained_variance_ratio_
        pca_note = f'PC1+PC2 explain {100*sum(var_exp):.1f}% of variance'

        # Color by episode (normalised 0→1 for colorscale)
        ep_vals = df['episode'].values
        ep_norm = (ep_vals - ep_vals.min()) / max(ep_vals.max() - ep_vals.min(), 1)

        pca_fig = go.Figure(go.Scatter(
            x=coords[:, 0], y=coords[:, 1],
            mode='markers',
            marker=dict(
                color=ep_norm, colorscale='Viridis', size=3, opacity=0.6,
                colorbar=dict(title='Training progress<br>(0=early, 1=late)'),
            ),
            hovertemplate='PC1=%{x:.2f}<br>PC2=%{y:.2f}<extra></extra>',
        ))
        pca_fig.update_layout(
            title=f'PCA 2D projection of visited states — {pca_note}',
            xaxis_title=f'PC1 ({100*var_exp[0]:.1f}% var)',
            yaxis_title=f'PC2 ({100*var_exp[1]:.1f}% var)',
            height=420, margin=dict(t=40, b=30),
        )
    except ImportError:
        pca_note = 'sklearn not available — PCA chart disabled.'
        pca_fig = go.Figure()
        pca_fig.add_annotation(text=pca_note, x=0.5, y=0.5, xref='paper', yref='paper',
                               showarrow=False, font=dict(size=14))
        pca_fig.update_layout(height=300, margin=dict(t=40, b=30))

    return {'pca': pca_fig, 'violin': violin_fig, 'coverage': cov_fig,
            'pca_note': pca_note}


def make_cost_breakdown_chart(df: pd.DataFrame, dt: float = 0.5) -> go.Figure:
    """Stacked bar: mean cost per timestep averaged over all episodes,
    broken down into c_travel, c_maint, c_risk.
    Returns an empty figure with annotation if breakdown columns are absent.
    """
    cols = ['c_travel', 'c_maint', 'c_risk']
    labels = {'c_travel': 'Travel', 'c_maint': 'Maintenance', 'c_risk': 'Risk'}
    colors = {'c_travel': '#4C78A8', 'c_maint': '#F58518', 'c_risk': '#E45756'}

    if not all(c in df.columns for c in cols):
        fig = go.Figure()
        fig.add_annotation(text="No cost breakdown data (re-run evaluation)",
                           showarrow=False, font=dict(size=14))
        return fig

    mean_by_t = df.groupby('t')[cols].mean().reset_index()

    fig = go.Figure()
    for col in cols:
        fig.add_trace(go.Bar(
            x=_to_year(mean_by_t['t'], dt), y=mean_by_t[col],
            name=labels[col], marker_color=colors[col],
        ))
    fig.update_layout(
        barmode='stack',
        title='Mean cost per timestep by category (avg over episodes)',
        xaxis=YEAR_AXIS, yaxis_title='Mean cost (€)',
        height=380, margin=dict(t=40, b=30),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )
    return fig


# ---------------------------------------------------------------------------
# Problem-structure helpers and fan charts
# ---------------------------------------------------------------------------


def _extract_renovation_durations(df: pd.DataFrame, n_assets: int, dt: float = 0.5) -> list[float]:
    """Return list of renovation event durations in years (steps × dt)."""
    durations: list[float] = []
    for i in range(n_assets):
        col = f"h_{i}"
        if col not in df.columns:
            continue
        for _, ep_df in df.groupby("episode"):
            h = ep_df.sort_values("t")[col].values
            in_run = False
            run_len = 0
            for val in h:
                if val > 0:
                    if not in_run:
                        in_run = True
                        run_len = 1
                    else:
                        run_len += 1
                else:
                    if in_run:
                        durations.append(run_len * dt)
                        in_run = False
                        run_len = 0
            if in_run and run_len > 0:
                durations.append(run_len * dt)
    return durations


def make_degradation_fan(df: pd.DataFrame, asset_idx: int, dt: float = 0.5,
                         show_lines: bool = True) -> go.Figure:
    """Fan chart of degradation paths across eval episodes for one asset."""
    col = f"d_{asset_idx}"
    if col not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text=f"Column {col} not found", x=0.5, y=0.5,
                           xref="paper", yref="paper", showarrow=False)
        return fig

    pivot = df.pivot_table(index="t", columns="episode", values=col, aggfunc="first")
    t_years = _to_year(pivot.index.values, dt)
    vals = pivot.values  # shape (T, n_episodes)
    episodes = pivot.columns.tolist()
    n_ep = len(episodes)

    p5  = np.nanpercentile(vals, 5,  axis=1)
    p25 = np.nanpercentile(vals, 25, axis=1)
    p50 = np.nanpercentile(vals, 50, axis=1)
    p75 = np.nanpercentile(vals, 75, axis=1)
    p95 = np.nanpercentile(vals, 95, axis=1)

    fig = go.Figure()

    # Spaghetti lines (subsample to ≤20 for large episode counts)
    if show_lines:
        rng = np.random.default_rng(0)
        spaghetti_eps = (episodes if n_ep <= 20
                         else rng.choice(episodes, size=20, replace=False).tolist())
        for ep in spaghetti_eps:
            fig.add_trace(go.Scatter(
                x=t_years, y=pivot[ep].values,
                mode="lines",
                line=dict(color="steelblue", width=1),
                opacity=0.12,
                showlegend=False,
                hoverinfo="skip",
            ))

    # 5–95th percentile band
    fig.add_trace(go.Scatter(
        x=np.concatenate([t_years, t_years[::-1]]),
        y=np.concatenate([p95, p5[::-1]]),
        fill="toself",
        fillcolor="rgba(70,130,180,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="5–95th pct",
        hoverinfo="skip",
    ))

    # IQR band
    fig.add_trace(go.Scatter(
        x=np.concatenate([t_years, t_years[::-1]]),
        y=np.concatenate([p75, p25[::-1]]),
        fill="toself",
        fillcolor="rgba(70,130,180,0.35)",
        line=dict(color="rgba(0,0,0,0)"),
        name="IQR (25–75th)",
        hoverinfo="skip",
    ))

    # Median
    fig.add_trace(go.Scatter(
        x=t_years, y=p50,
        mode="lines",
        line=dict(color="darkblue", width=2.5),
        name="Median",
        hovertemplate="year=%{x:.1f}<br>median d=%{y:.3f}<extra></extra>",
    ))

    # Failure threshold
    fig.add_hline(y=1.0, line=dict(color="red", dash="dash", width=1.5),
                  annotation_text="Failure threshold", annotation_position="top right")

    fig.update_layout(
        title=f"Degradation paths — Asset {asset_idx}  ({n_ep} eval episodes)",
        xaxis=dict(**YEAR_AXIS),
        yaxis=dict(title="Condition d (0=pristine, 1=failed)", range=[0, 1.05]),
        height=380,
        margin=dict(t=50, b=30),
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def make_renovation_fan(df: pd.DataFrame, asset_idx: int, dt: float = 0.5,
                        show_lines: bool = True) -> tuple[go.Figure, go.Figure]:
    """Return (h_fan_fig, duration_hist_fig) for one asset."""
    col = f"h_{asset_idx}"
    n_assets = get_n_assets(df)

    # --- h-trajectory fan (event-aligned: x = time since renovation start) ---
    if col not in df.columns:
        h_fig = go.Figure()
        h_fig.add_annotation(text=f"Column {col} not found", x=0.5, y=0.5,
                              xref="paper", yref="paper", showarrow=False)
    else:
        # Extract individual renovation events for this asset, aligned at step 0
        events: list[np.ndarray] = []
        for _, ep_df in df.groupby("episode"):
            h = ep_df.sort_values("t")[col].values
            in_run = False
            run: list[float] = []
            for val in h:
                if val > 0:
                    if not in_run:
                        in_run = True
                        run = [val]
                    else:
                        run.append(val)
                else:
                    if in_run:
                        events.append(np.array(run))
                        in_run = False
                        run = []
            if in_run and run:
                events.append(np.array(run))

        h_fig = go.Figure()

        if not events:
            h_fig.add_annotation(text="No renovation events found for this asset",
                                  x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        else:
            max_len = max(len(e) for e in events)
            # Pad shorter events with NaN so we can stack
            mat = np.full((max_len, len(events)), np.nan)
            for j, ev in enumerate(events):
                mat[:len(ev), j] = ev
            rel_t = np.arange(max_len) * dt  # relative years since start

            p25 = np.nanpercentile(mat, 25, axis=1)
            p50 = np.nanpercentile(mat, 50, axis=1)
            p75 = np.nanpercentile(mat, 75, axis=1)

            n_events = len(events)
            if show_lines:
                rng = np.random.default_rng(0)
                idx_sample = (list(range(n_events)) if n_events <= 20
                              else rng.choice(n_events, size=20, replace=False).tolist())
                for j in idx_sample:
                    ev_len = (~np.isnan(mat[:, j])).sum()
                    h_fig.add_trace(go.Scatter(
                        x=rel_t[:ev_len], y=mat[:ev_len, j],
                        mode="lines",
                        line=dict(color="darkorange", width=1),
                        opacity=0.15,
                        showlegend=False,
                        hoverinfo="skip",
                    ))

            # IQR band (only where we have data)
            valid = ~np.isnan(p50)
            h_fig.add_trace(go.Scatter(
                x=np.concatenate([rel_t[valid], rel_t[valid][::-1]]),
                y=np.concatenate([p75[valid], p25[valid][::-1]]),
                fill="toself",
                fillcolor="rgba(255,140,0,0.25)",
                line=dict(color="rgba(0,0,0,0)"),
                name="IQR (25–75th)",
                hoverinfo="skip",
            ))

            # Median
            h_fig.add_trace(go.Scatter(
                x=rel_t[valid], y=p50[valid],
                mode="lines",
                line=dict(color="darkorange", width=2.5),
                name="Median",
                hovertemplate="Δt=%{x:.1f}yr<br>median h=%{y:.3f}<extra></extra>",
            ))

            # Zero reference
            h_fig.add_hline(y=0, line=dict(color="gray", dash="dot", width=1))

        h_fig.update_layout(
            title=f"Renovation progress h — Asset {asset_idx} (aligned at event start)",
            xaxis_title="Time since renovation start (years)",
            yaxis_title="Renovation counter h",
            height=380,
            margin=dict(t=50, b=30),
            legend=dict(orientation="h", y=1.1),
        )

    # --- Renovation duration histogram ---
    durations = _extract_renovation_durations(df, n_assets, dt)
    if durations:
        mean_dur = float(np.mean(durations))
        hist_fig = go.Figure(go.Histogram(
            x=durations,
            xbins=dict(size=dt),
            marker_color="darkorange",
            opacity=0.8,
            hovertemplate="duration=%{x:.1f}yr<br>count=%{y}<extra></extra>",
        ))
        hist_fig.add_vline(x=mean_dur, line=dict(color="black", dash="dash", width=1.5),
                           annotation_text=f"mean={mean_dur:.1f}yr",
                           annotation_position="top right")
    else:
        hist_fig = go.Figure()
        hist_fig.add_annotation(text="No renovation events found", x=0.5, y=0.5,
                                xref="paper", yref="paper", showarrow=False)

    hist_fig.update_layout(
        title="Renovation event durations (all assets × all episodes)",
        xaxis_title="Renovation duration (years)",
        yaxis_title="Count",
        height=380,
        margin=dict(t=50, b=30),
    )
    return h_fig, hist_fig


def make_training_curve(log: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "mean_cost" in log.columns:
        y = log["mean_cost"]
        x = log["episode"] if "episode" in log.columns else log.index
        fig.add_trace(go.Scatter(x=x, y=y, name="Mean cost",
                                 line=dict(color="steelblue", width=2),
                                 hovertemplate="ep=%{x}<br>mean cost=%{y:.2f}<extra></extra>"))
        if "std_cost" in log.columns:
            std = log["std_cost"]
            fig.add_trace(go.Scatter(
                x=list(x) + list(x)[::-1],
                y=list(y + std) + list((y - std).clip(lower=0))[::-1],
                fill="toself", fillcolor="rgba(70,130,180,0.2)",
                line=dict(color="rgba(0,0,0,0)"),
                name="±1 std", showlegend=True,
                hoverinfo="skip",
            ))
    fig.update_layout(
        title="Training curve",
        xaxis_title="Episode", yaxis_title="Mean cost",
        height=350, margin=dict(t=40, b=30),
        legend=dict(orientation="h", y=1.1),
    )
    return fig
