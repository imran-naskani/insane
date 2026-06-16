"""
Sector Flow — Smart Money Tracker
Streamlit page: pages/sector_flow.py

Run via: streamlit run insane.py   (Streamlit auto-discovers pages/)
"""

import io
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sector_scanner import (
    DATA_DIR,
    OHLCV_FILE,
    SECTOR_CACHE_FILE,
    THEME_FILE,
    UNIVERSE_FILE,
    compute_metrics,
    compute_trend_series,
    get_all_sectors,
    get_constituents,
    get_highlights,
    get_industry_stocks,
    get_sector_industries,
    get_universe,
    update_daily,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _color(val: float | None, center: float = 0.0) -> str:
    if val is None or np.isnan(val):
        return "#555555"
    if val > center + 3:
        return "#1a7a1a"
    if val > center + 1:
        return "#2ecc71"
    if val > center:
        return "#7dbb7d"
    if val < center - 3:
        return "#8b0000"
    if val < center - 1:
        return "#e74c3c"
    return "#b35858"


def _fmt(val, suffix="", prefix="") -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{prefix}{val:+.2f}{suffix}" if isinstance(val, float) else f"{val}"


def _data_status() -> str:
    if not OHLCV_FILE.exists():
        return "No data — run: python sector_scanner.py --init"
    df = pd.read_parquet(OHLCV_FILE, columns=[])
    last = df.index[-1].date() if len(df) > 0 else "unknown"
    tags_ok = SECTOR_CACHE_FILE.exists()
    return f"Data through **{last}** | Tags: {'✅' if tags_ok else '⚠️ missing — run --tags'}"


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("## Sector Flow — Smart Money Tracker")
st.markdown(_data_status())

# Controls
ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1, 1, 1, 4])
with ctrl_col1:
    period = st.selectbox("Period", ["1D", "1W", "1M", "3M"], index=1)
with ctrl_col2:
    view = st.selectbox("View", ["All", "Sectors", "Industries", "Themes"], index=0)
with ctrl_col3:
    if st.button("Refresh Data"):
        with st.spinner("Fetching latest prices..."):
            update_daily()
        st.rerun()

# Load data
with st.spinner("Computing metrics..."):
    metrics = compute_metrics(period)

if metrics.empty:
    st.error("No data loaded. Run `python sector_scanner.py --init` to download historical data.")
    st.code("python sector_scanner.py --init")
    st.stop()

ret_col = f"ret_{period}"

# ---------------------------------------------------------------------------
# TRENDING — DRILL DOWN CHARTS
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("### Trending — Smart Money Flow")

# Init session state
for key, default in [
    ("sd_level", 0), ("sd_sector", None), ("sd_industry", None),
    ("theme_drill", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

trend_period = st.select_slider(
    "Lookback period",
    options=["1M", "3M", "6M", "1Y", "2Y", "3Y"],
    value="3M",
    key="trend_period",
)


def _trend_chart(series: dict, title: str, height: int = 420) -> go.Figure:
    """Build a normalized multi-line Plotly chart."""
    fig = go.Figure()
    # Assign a distinct color palette (up to ~35 themes)
    colors = [
        "#2ecc71","#e74c3c","#3498db","#f39c12","#9b59b6","#1abc9c","#e67e22",
        "#c0392b","#2980b9","#27ae60","#8e44ad","#d35400","#16a085","#f1c40f",
        "#2c3e50","#7f8c8d","#e91e63","#00bcd4","#ff5722","#607d8b","#795548",
        "#4caf50","#ff9800","#9c27b0","#03a9f4","#8bc34a","#ffc107","#673ab7",
        "#009688","#f44336","#3f51b5","#cddc39","#ff4081","#00e5ff","#76ff03",
    ]
    for i, (name, s) in enumerate(series.items()):
        last_val = s.iloc[-1]
        delta = last_val - 100
        label = f"{name} ({delta:+.1f}%)"
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values,
            name=label,
            mode="lines",
            line=dict(width=1.8, color=colors[i % len(colors)]),
            hovertemplate=f"<b>{name}</b><br>%{{x}}<br>%{{y:.1f}} (indexed)<extra></extra>",
        ))
    fig.add_hline(y=100, line_dash="dot", line_color="#444444", line_width=1)
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#cccccc")),
        height=height,
        margin=dict(l=0, r=0, t=35, b=0),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#cccccc", size=10),
        xaxis=dict(gridcolor="#1f1f1f", showgrid=True),
        yaxis=dict(gridcolor="#1f1f1f", showgrid=True, title="Indexed (100 = period start)"),
        legend=dict(
            orientation="v", font=dict(size=9),
            bgcolor="rgba(0,0,0,0.4)", bordercolor="#333",
            x=1.01, y=1,
        ),
        hovermode="x unified",
    )
    return fig


# --- LEFT: Sector → Industry → Stocks ---
tcol1, gap, tcol2 = st.columns([5, 0.3, 5])

with tcol1:
    # Breadcrumb
    crumb = ["Sectors"]
    if st.session_state.sd_sector:
        crumb.append(st.session_state.sd_sector)
    if st.session_state.sd_industry:
        crumb.append(st.session_state.sd_industry)

    bc_cols = st.columns(len(crumb))
    for i, label in enumerate(crumb):
        style = "**" if i == len(crumb) - 1 else ""
        if bc_cols[i].button(f"{'↩ ' if i < len(crumb)-1 else ''}{style}{label}{style}",
                             key=f"bc_{i}"):
            if i == 0:
                st.session_state.sd_level = 0
                st.session_state.sd_sector = None
                st.session_state.sd_industry = None
            elif i == 1:
                st.session_state.sd_level = 1
                st.session_state.sd_industry = None
            st.rerun()

    level = st.session_state.sd_level

    if level == 0:
        names = get_all_sectors()
        gtype = "Sector"
        chart_title = "All Sectors — Indexed Return"
        drill_label = "Drill into sector:"
        drill_options = names
    elif level == 1:
        names = get_sector_industries(st.session_state.sd_sector)
        gtype = "Industry"
        chart_title = f"{st.session_state.sd_sector} — Industries"
        drill_label = "Drill into industry:"
        drill_options = names
    else:
        names = get_industry_stocks(st.session_state.sd_industry)
        gtype = "Stock"
        chart_title = f"{st.session_state.sd_industry} — Stocks"
        drill_label = None
        drill_options = []

    with st.spinner("Loading..."):
        series = compute_trend_series(names, gtype, trend_period)

    if gtype == "Stock":
        found = list(series.keys())
        missing = [n for n in names if n not in series]
        if found:
            st.caption(f"**{len(found)} stocks:** {', '.join(found)}")
        if missing:
            st.caption(f"⚠️ Not in universe: {', '.join(missing)}")

    if series:
        st.plotly_chart(_trend_chart(series, chart_title), use_container_width=True)
    else:
        st.warning(f"No stocks from this industry are in the downloaded universe (S&P500 + Nasdaq-100). "
                   f"Industry may be dominated by non-US companies (e.g. Luxury Goods → LVMH, Hermès).")

    if drill_options:
        chosen = st.selectbox(drill_label, ["— select —"] + drill_options, key=f"sd_sel_{level}")
        if chosen != "— select —":
            if level == 0:
                st.session_state.sd_sector = chosen
                st.session_state.sd_level = 1
            elif level == 1:
                st.session_state.sd_industry = chosen
                st.session_state.sd_level = 2
            st.rerun()


# --- RIGHT: Theme → Stocks ---
with tcol2:
    themes_data = json.loads(THEME_FILE.read_text()) if THEME_FILE.exists() else {}
    themes_list = sorted(themes_data.keys())

    # Breadcrumb
    if st.session_state.theme_drill:
        tb1, tb2 = st.columns([1, 3])
        if tb1.button("↩ All Themes", key="theme_back"):
            st.session_state.theme_drill = None
            st.rerun()
        tb2.markdown(f"**{st.session_state.theme_drill}**")
    else:
        st.markdown("**All Themes**")

    if st.session_state.theme_drill is None:
        t_names = themes_list
        t_gtype = "Theme"
        t_title = "All Themes — Indexed Return"
    else:
        t_names = themes_data.get(st.session_state.theme_drill, [])
        t_gtype = "Stock"
        t_title = f"{st.session_state.theme_drill} — Stocks"

    with st.spinner("Loading..."):
        t_series = compute_trend_series(t_names, t_gtype, trend_period)

    if t_gtype == "Stock":
        found_t = list(t_series.keys())
        missing_t = [n for n in t_names if n not in t_series]
        if found_t:
            st.caption(f"**{len(found_t)} stocks:** {', '.join(found_t)}")
        if missing_t:
            st.caption(f"⚠️ Not in universe: {', '.join(missing_t)}")

    if t_series:
        st.plotly_chart(_trend_chart(t_series, t_title), use_container_width=True)
    else:
        st.warning("No stocks from this theme are in the downloaded universe.")

    if st.session_state.theme_drill is None and themes_list:
        chosen_theme = st.selectbox(
            "Drill into theme:", ["— select —"] + themes_list, key="theme_sel"
        )
        if chosen_theme != "— select —":
            st.session_state.theme_drill = chosen_theme
            st.rerun()


# ---------------------------------------------------------------------------
# SECTOR HEATMAP
# ---------------------------------------------------------------------------

if view in ("All", "Sectors"):
    st.markdown("---")
    st.markdown("### Sectors")

    sectors = metrics[metrics["type"] == "Sector"].copy()

    if not sectors.empty:
        # Build plotly heatmap as colored tiles
        cols_per_row = 4
        sector_list = sectors.sort_values(ret_col, ascending=False).to_dict("records")
        n_rows = -(-len(sector_list) // cols_per_row)

        fig = go.Figure()
        tile_w, tile_h = 1.0, 0.8
        for idx, row in enumerate(sector_list):
            r = idx // cols_per_row
            c = idx % cols_per_row
            x0, x1 = c * tile_w, c * tile_w + tile_w * 0.95
            y0, y1 = (n_rows - r - 1) * tile_h, (n_rows - r) * tile_h * 0.95

            ret_val = row.get(ret_col)
            bg = _color(ret_val)
            vr = row.get("vol_ratio")
            vol_str = f"Vol {vr:.2f}x" if vr else ""
            rs = row.get("rs_vs_spx")
            rs_str = f"RS {rs:+.1f}%" if rs is not None else ""

            fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                          fillcolor=bg, line_color="#111", line_width=1)
            fig.add_annotation(
                x=(x0 + x1) / 2, y=(y0 + y1) / 2 + 0.15,
                text=f"<b>{row['name']}</b>",
                showarrow=False, font=dict(size=11, color="white"),
                xanchor="center", yanchor="middle"
            )
            fig.add_annotation(
                x=(x0 + x1) / 2, y=(y0 + y1) / 2 - 0.05,
                text=f"{_fmt(ret_val, '%')}",
                showarrow=False, font=dict(size=14, color="white", family="monospace"),
                xanchor="center", yanchor="middle"
            )
            fig.add_annotation(
                x=(x0 + x1) / 2, y=(y0 + y1) / 2 - 0.22,
                text=f"{rs_str}  {vol_str}",
                showarrow=False, font=dict(size=9, color="#cccccc"),
                xanchor="center", yanchor="middle"
            )

        fig.update_layout(
            height=n_rows * 110,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(visible=False, range=[0, cols_per_row * tile_w]),
            yaxis=dict(visible=False, range=[0, n_rows * tile_h]),
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# INDUSTRY TABLE
# ---------------------------------------------------------------------------

if view in ("All", "Industries"):
    st.markdown("---")
    st.markdown("### Industries")

    industries = metrics[metrics["type"] == "Industry"].copy()
    if not industries.empty:
        display = industries[["name", "tickers", ret_col, "ret_1D", "rs_vs_spx", "vol_ratio", "breadth_pct", "score"]].copy()
        display.columns = ["Industry", "# Stocks", f"Ret {period}", "Ret 1D", "RS vs SPX", "Vol Ratio", "Breadth %", "Score"]
        display = display.sort_values("Score", ascending=False).reset_index(drop=True)

        def style_ret(v):
            try:
                c = "#2ecc71" if v > 0 else "#e74c3c"
                return f"color: {c}"
            except Exception:
                return ""

        styled = display.style.applymap(style_ret, subset=[f"Ret {period}", "Ret 1D", "RS vs SPX", "Score"])
        st.dataframe(styled, use_container_width=True, height=500)

# ---------------------------------------------------------------------------
# THEME TABLE
# ---------------------------------------------------------------------------

if view in ("All", "Themes"):
    st.markdown("---")
    st.markdown("### Themes")

    themes_df = metrics[metrics["type"] == "Theme"].copy()
    if not themes_df.empty:
        display = themes_df[["name", "tickers", ret_col, "ret_1D", "rs_vs_spx", "vol_ratio", "breadth_pct", "score"]].copy()
        display.columns = ["Theme", "# Stocks", f"Ret {period}", "Ret 1D", "RS vs SPX", "Vol Ratio", "Breadth %", "Score"]
        display = display.sort_values("Score", ascending=False).reset_index(drop=True)

        def style_ret(v):
            try:
                c = "#2ecc71" if v > 0 else "#e74c3c"
                return f"color: {c}"
            except Exception:
                return ""

        styled = display.style.applymap(style_ret, subset=[f"Ret {period}", "Ret 1D", "RS vs SPX", "Score"])
        st.dataframe(styled, use_container_width=True, height=600)

# ---------------------------------------------------------------------------
# DRILL-DOWN
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("### Drill Down — Constituents")

d_col1, d_col2 = st.columns([1, 1])
with d_col1:
    drill_type = st.selectbox("Group Type", ["Theme", "Sector", "Industry"])
with d_col2:
    if drill_type == "Theme":
        opts = sorted(json.loads(THEME_FILE.read_text()).keys()) if THEME_FILE.exists() else []
    elif drill_type == "Sector":
        opts = sorted(metrics[metrics["type"] == "Sector"]["name"].tolist())
    else:
        opts = sorted(metrics[metrics["type"] == "Industry"]["name"].tolist())
    drill_name = st.selectbox("Name", opts)

if drill_name:
    constituents = get_constituents(drill_type, drill_name, metrics, period)
    if not constituents.empty:
        def style_c(v):
            try:
                c = "#2ecc71" if v > 0 else "#e74c3c"
                return f"color: {c}"
            except Exception:
                return ""

        styled_c = constituents.style.applymap(
            style_c, subset=[f"ret_{period}", "ret_1D", "vs_20sma"]
        )
        st.dataframe(styled_c, use_container_width=True)

# ---------------------------------------------------------------------------
# HIGHLIGHTS
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("### Highlights")

highlights = get_highlights(metrics)

h1, h2, h3 = st.columns(3)

with h1:
    st.markdown("#### Top Flow (RS vs SPX)")
    top = highlights.get("top_themes", [])
    for item in top:
        rs = item.get("rs_vs_spx")
        vr = item.get("vol_ratio")
        rs_str = f"{rs:+.1f}%" if rs is not None else "—"
        vr_str = f"{vr:.2f}x vol" if vr else ""
        st.markdown(f"🟢 **{item['name']}** — RS {rs_str}  {vr_str}")

    st.markdown("#### Laggards")
    bottom = highlights.get("bottom_themes", [])
    for item in bottom:
        rs = item.get("rs_vs_spx")
        rs_str = f"{rs:+.1f}%" if rs is not None else "—"
        st.markdown(f"🔴 **{item['name']}** — RS {rs_str}")

with h2:
    st.markdown("#### Smart Money Signature")
    st.caption("Volume surge AND positive RS vs SPX")
    smart = highlights.get("volume_surge", [])
    if smart:
        for item in smart:
            vr = item.get("vol_ratio", 1)
            rs = item.get("rs_vs_spx")
            rs_str = f"{rs:+.1f}%" if rs is not None else "—"
            st.markdown(f"💰 **{item['name']}** — {vr:.2f}x vol, RS {rs_str}")
    else:
        st.caption("No unusual volume signals right now")

    st.markdown("#### Reversal Watch")
    st.caption("Down but volume picking up — potential accumulation")
    rev = highlights.get("reversal_watch", [])
    if rev:
        for item in rev:
            ret_val = item.get(ret_col) or item.get("ret_1D")
            vr = item.get("vol_ratio", 1)
            ret_str = f"{ret_val:+.1f}%" if ret_val is not None else "—"
            st.markdown(f"👀 **{item['name']}** — {ret_str}, {vr:.2f}x vol")
    else:
        st.caption("No reversal candidates")

with h3:
    st.markdown("#### Momentum Building")
    st.caption("Positive RS + broad breadth (>55% above 20MA)")
    mom = highlights.get("momentum_building", [])
    if mom:
        for item in mom:
            rs = item.get("rs_vs_spx")
            breadth = item.get("breadth_pct", 0)
            rs_str = f"{rs:+.1f}%" if rs is not None else "—"
            st.markdown(f"📈 **{item['name']}** — RS {rs_str}, {breadth:.0f}% above 20MA")
    else:
        st.caption("No strong momentum clusters")

    st.markdown("#### Top Sectors (RS vs SPX)")
    sector_top = highlights.get("top_sectors", [])
    for item in sector_top:
        rs = item.get("rs_vs_spx")
        rs_str = f"{rs:+.1f}%" if rs is not None else "—"
        st.markdown(f"🏆 **{item['name']}** — {rs_str}")

# ---------------------------------------------------------------------------
# EXPORT
# ---------------------------------------------------------------------------

st.markdown("---")
exp_col1, exp_col2 = st.columns([1, 4])
with exp_col1:
    export_type = st.selectbox("Export", ["Themes", "Industries", "Sectors", "All"])

if export_type == "All":
    export_df = metrics.copy()
elif export_type == "Themes":
    export_df = metrics[metrics["type"] == "Theme"].copy()
elif export_type == "Sectors":
    export_df = metrics[metrics["type"] == "Sector"].copy()
else:
    export_df = metrics[metrics["type"] == "Industry"].copy()

csv_buf = io.BytesIO()
export_df.to_csv(csv_buf, index=False)
csv_buf.seek(0)

with exp_col1:
    st.download_button(
        label="Download CSV",
        data=csv_buf,
        file_name=f"sector_flow_{export_type.lower()}_{period}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )
