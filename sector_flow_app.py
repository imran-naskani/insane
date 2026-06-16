"""
Sector Flow — standalone app.
Run: streamlit run sector_flow_app.py --server.port 8502
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import io, json
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sector_scanner import (
    OHLCV_FILE, SECTOR_CACHE_FILE, THEME_FILE,
    compute_metrics, compute_trend_series, get_all_sectors,
    get_constituents, get_highlights, get_industry_stocks,
    get_sector_industries, update_daily,
)

st.set_page_config(page_title="Sector Flow — INSANE", layout="wide")

# ---------------------------------------------------------------------------
def _color(val, center=0.0):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "#555555"
    if val > center + 3:  return "#1a7a1a"
    if val > center + 1:  return "#2ecc71"
    if val > center:      return "#7dbb7d"
    if val < center - 3:  return "#8b0000"
    if val < center - 1:  return "#e74c3c"
    return "#b35858"

def _fmt(val, suffix=""):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{val:+.2f}{suffix}"

def _data_status():
    if not OHLCV_FILE.exists():
        return "No data — run: python sector_scanner.py --init"
    df = pd.read_parquet(OHLCV_FILE, columns=[])
    last = df.index[-1].date() if len(df) else "unknown"
    return f"Data through **{last}** | Tags: {'✅' if SECTOR_CACHE_FILE.exists() else '⚠️ missing'}"

def _trend_chart(series: dict, title: str, height: int = 420):
    colors = [
        "#2ecc71","#e74c3c","#3498db","#f39c12","#9b59b6","#1abc9c","#e67e22",
        "#c0392b","#2980b9","#27ae60","#8e44ad","#d35400","#16a085","#f1c40f",
        "#2c3e50","#7f8c8d","#e91e63","#00bcd4","#ff5722","#607d8b","#795548",
        "#4caf50","#ff9800","#9c27b0","#03a9f4","#8bc34a","#ffc107","#673ab7",
        "#009688","#f44336","#3f51b5","#cddc39","#ff4081","#00e5ff","#76ff03",
    ]
    fig = go.Figure()
    for i, (name, s) in enumerate(series.items()):
        delta = s.iloc[-1] - 100
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values,
            name=f"{name} ({delta:+.1f}%)",
            mode="lines",
            line=dict(width=1.8, color=colors[i % len(colors)]),
            hovertemplate=f"<b>{name}</b><br>%{{x}}<br>%{{y:.1f}}<extra></extra>",
        ))
    fig.add_hline(y=100, line_dash="dot", line_color="#444", line_width=1)
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#ccc")),
        height=height, margin=dict(l=0, r=0, t=35, b=0),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#ccc", size=10),
        xaxis=dict(gridcolor="#1f1f1f"), yaxis=dict(gridcolor="#1f1f1f", title="Indexed (100 = start)"),
        legend=dict(orientation="v", font=dict(size=9), bgcolor="rgba(0,0,0,0.4)",
                    bordercolor="#333", x=1.01, y=1),
        hovermode="x unified",
    )
    return fig

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("## Sector Flow — Smart Money Tracker")
st.markdown(_data_status())

c1, c2, c3, _ = st.columns([1, 1, 1, 4])
with c1: period = st.selectbox("Period", ["1D", "1W", "1M", "3M"], index=1)
with c2: view   = st.selectbox("View", ["All", "Sectors", "Industries", "Themes"], index=0)
with c3:
    if st.button("Refresh Data"):
        with st.spinner("Fetching..."):
            update_daily()
        st.rerun()

with st.spinner("Computing metrics..."):
    metrics = compute_metrics(period)

if metrics.empty:
    st.error("No data. Run: python sector_scanner.py --init")
    st.stop()

ret_col = f"ret_{period}"

# ---------------------------------------------------------------------------
# TRENDING
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### Trending — Smart Money Flow")

for k, v in [("sd_level",0),("sd_sector",None),("sd_industry",None),("theme_drill",None)]:
    if k not in st.session_state:
        st.session_state[k] = v

trend_period = st.select_slider(
    "Lookback", options=["1M","3M","6M","1Y","2Y","3Y"], value="3M", key="trend_period"
)

tcol1, _, tcol2 = st.columns([5, 0.3, 5])

# LEFT: Sector → Industry → Stocks
with tcol1:
    crumb = ["Sectors"]
    if st.session_state.sd_sector:   crumb.append(st.session_state.sd_sector)
    if st.session_state.sd_industry: crumb.append(st.session_state.sd_industry)

    bc_cols = st.columns(len(crumb))
    for i, lbl in enumerate(crumb):
        if bc_cols[i].button(f"{'↩ ' if i < len(crumb)-1 else ''}{lbl}", key=f"bc_{i}"):
            if i == 0:
                st.session_state.update({"sd_level":0,"sd_sector":None,"sd_industry":None})
            elif i == 1:
                st.session_state.update({"sd_level":1,"sd_industry":None})
            st.rerun()

    lv = st.session_state.sd_level
    if lv == 0:
        names, gtype = get_all_sectors(), "Sector"
        chart_title, drill_label = "All Sectors — Indexed Return", "Drill into sector:"
    elif lv == 1:
        names, gtype = get_sector_industries(st.session_state.sd_sector), "Industry"
        chart_title, drill_label = f"{st.session_state.sd_sector} — Industries", "Drill into industry:"
    else:
        names, gtype = get_industry_stocks(st.session_state.sd_industry), "Stock"
        chart_title, drill_label = f"{st.session_state.sd_industry} — Stocks", None

    with st.spinner("Loading..."):
        series = compute_trend_series(names, gtype, trend_period)

    if gtype == "Stock":
        found = list(series.keys())
        missing = [n for n in names if n not in series]
        if found:   st.caption(f"**{len(found)} stocks:** {', '.join(found)}")
        if missing: st.caption(f"⚠️ Not in universe: {', '.join(missing)}")

    if series:
        st.plotly_chart(_trend_chart(series, chart_title), use_container_width=True)
    else:
        st.warning("No data — industry may be dominated by non-US companies.")

    if drill_label and names:
        chosen = st.selectbox(drill_label, ["— select —"] + names, key=f"sd_sel_{lv}")
        if chosen != "— select —":
            if lv == 0: st.session_state.update({"sd_sector": chosen, "sd_level": 1})
            else:       st.session_state.update({"sd_industry": chosen, "sd_level": 2})
            st.rerun()

# RIGHT: Theme → Stocks
with tcol2:
    themes_data = json.loads(THEME_FILE.read_text()) if THEME_FILE.exists() else {}
    themes_list = sorted(themes_data.keys())

    if st.session_state.theme_drill:
        tb1, tb2 = st.columns([1, 3])
        if tb1.button("↩ All Themes", key="theme_back"):
            st.session_state.theme_drill = None
            st.rerun()
        tb2.markdown(f"**{st.session_state.theme_drill}**")
    else:
        st.markdown("**All Themes**")

    if st.session_state.theme_drill is None:
        t_names, t_gtype, t_title = themes_list, "Theme", "All Themes — Indexed Return"
    else:
        t_names = themes_data.get(st.session_state.theme_drill, [])
        t_gtype, t_title = "Stock", f"{st.session_state.theme_drill} — Stocks"

    with st.spinner("Loading..."):
        t_series = compute_trend_series(t_names, t_gtype, trend_period)

    if t_gtype == "Stock":
        found_t = list(t_series.keys())
        missing_t = [n for n in t_names if n not in t_series]
        if found_t:   st.caption(f"**{len(found_t)} stocks:** {', '.join(found_t)}")
        if missing_t: st.caption(f"⚠️ Not in universe: {', '.join(missing_t)}")

    if t_series:
        st.plotly_chart(_trend_chart(t_series, t_title), use_container_width=True)
    else:
        st.warning("No data for this theme.")

    if st.session_state.theme_drill is None and themes_list:
        chosen_t = st.selectbox("Drill into theme:", ["— select —"] + themes_list, key="theme_sel")
        if chosen_t != "— select —":
            st.session_state.theme_drill = chosen_t
            st.rerun()

# ---------------------------------------------------------------------------
# SECTOR HEATMAP
# ---------------------------------------------------------------------------
if view in ("All", "Sectors"):
    st.markdown("---")
    st.markdown("### Sectors")
    sectors = metrics[metrics["type"] == "Sector"].copy()
    if not sectors.empty:
        cols_per_row, tile_w, tile_h = 4, 1.0, 0.8
        sector_list = sectors.sort_values(ret_col, ascending=False).to_dict("records")
        n_rows = -(-len(sector_list) // cols_per_row)
        fig = go.Figure()
        for idx, row in enumerate(sector_list):
            r, c = idx // cols_per_row, idx % cols_per_row
            x0, x1 = c*tile_w, c*tile_w + tile_w*0.95
            y0, y1 = (n_rows-r-1)*tile_h, (n_rows-r)*tile_h*0.95
            bg = _color(row.get(ret_col))
            vr = row.get("vol_ratio"); rs = row.get("rs_vs_spx")
            fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                          fillcolor=bg, line_color="#111", line_width=1)
            fig.add_annotation(x=(x0+x1)/2, y=(y0+y1)/2+0.15,
                text=f"<b>{row['name']}</b>", showarrow=False,
                font=dict(size=11, color="white"), xanchor="center", yanchor="middle")
            fig.add_annotation(x=(x0+x1)/2, y=(y0+y1)/2-0.05,
                text=_fmt(row.get(ret_col), "%"), showarrow=False,
                font=dict(size=14, color="white", family="monospace"), xanchor="center", yanchor="middle")
            fig.add_annotation(x=(x0+x1)/2, y=(y0+y1)/2-0.22,
                text=f"{'RS '+str(round(rs,1))+'%' if rs else ''}  {'Vol '+str(vr)+'x' if vr else ''}",
                showarrow=False, font=dict(size=9, color="#ccc"), xanchor="center", yanchor="middle")
        fig.update_layout(height=n_rows*110, margin=dict(l=0,r=0,t=10,b=0),
            xaxis=dict(visible=False, range=[0, cols_per_row*tile_w]),
            yaxis=dict(visible=False, range=[0, n_rows*tile_h]),
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# INDUSTRY TABLE
# ---------------------------------------------------------------------------
if view in ("All", "Industries"):
    st.markdown("---"); st.markdown("### Industries")
    ind_df = metrics[metrics["type"] == "Industry"].copy()
    if not ind_df.empty:
        d = ind_df[["name","tickers",ret_col,"ret_1D","rs_vs_spx","vol_ratio","breadth_pct","score"]].copy()
        d.columns = ["Industry","# Stocks",f"Ret {period}","Ret 1D","RS vs SPX","Vol Ratio","Breadth %","Score"]
        d = d.sort_values("Score", ascending=False).reset_index(drop=True)
        def _sc(v):
            try: return f"color: {'#2ecc71' if v > 0 else '#e74c3c'}"
            except: return ""
        st.dataframe(d.style.applymap(_sc, subset=[f"Ret {period}","Ret 1D","RS vs SPX","Score"]),
                     use_container_width=True, height=500)

# ---------------------------------------------------------------------------
# THEME TABLE
# ---------------------------------------------------------------------------
if view in ("All", "Themes"):
    st.markdown("---"); st.markdown("### Themes")
    th_df = metrics[metrics["type"] == "Theme"].copy()
    if not th_df.empty:
        d = th_df[["name","tickers",ret_col,"ret_1D","rs_vs_spx","vol_ratio","breadth_pct","score"]].copy()
        d.columns = ["Theme","# Stocks",f"Ret {period}","Ret 1D","RS vs SPX","Vol Ratio","Breadth %","Score"]
        d = d.sort_values("Score", ascending=False).reset_index(drop=True)
        def _sc(v):
            try: return f"color: {'#2ecc71' if v > 0 else '#e74c3c'}"
            except: return ""
        st.dataframe(d.style.applymap(_sc, subset=[f"Ret {period}","Ret 1D","RS vs SPX","Score"]),
                     use_container_width=True, height=600)

# ---------------------------------------------------------------------------
# HIGHLIGHTS
# ---------------------------------------------------------------------------
st.markdown("---"); st.markdown("### Highlights")
hl = get_highlights(metrics)
h1, h2, h3 = st.columns(3)

with h1:
    st.markdown("#### Top Flow (RS vs SPX)")
    for item in hl.get("top_themes", []):
        rs = item.get("rs_vs_spx"); vr = item.get("vol_ratio")
        st.markdown(f"🟢 **{item['name']}** — RS {rs:+.1f}%  {f'{vr:.2f}x vol' if vr else ''}")
    st.markdown("#### Laggards")
    for item in hl.get("bottom_themes", []):
        rs = item.get("rs_vs_spx")
        st.markdown(f"🔴 **{item['name']}** — RS {rs:+.1f}%")

with h2:
    st.markdown("#### Smart Money Signature")
    st.caption("Volume surge AND positive RS")
    for item in hl.get("volume_surge", []) or [st.caption("None right now")]:
        if isinstance(item, dict):
            st.markdown(f"💰 **{item['name']}** — {item.get('vol_ratio',1):.2f}x vol")

    st.markdown("#### Reversal Watch")
    for item in hl.get("reversal_watch", []) or [st.caption("None")]:
        if isinstance(item, dict):
            st.markdown(f"👀 **{item['name']}** — {item.get('vol_ratio',1):.2f}x vol")

with h3:
    st.markdown("#### Momentum Building")
    for item in hl.get("momentum_building", []) or [st.caption("None")]:
        if isinstance(item, dict):
            rs = item.get("rs_vs_spx"); b = item.get("breadth_pct", 0)
            st.markdown(f"📈 **{item['name']}** — RS {rs:+.1f}%, {b:.0f}% above 20MA")
    st.markdown("#### Top Sectors")
    for item in hl.get("top_sectors", []):
        rs = item.get("rs_vs_spx")
        st.markdown(f"🏆 **{item['name']}** — {rs:+.1f}%")

# ---------------------------------------------------------------------------
# EXPORT
# ---------------------------------------------------------------------------
st.markdown("---")
ec1, _ = st.columns([1, 4])
with ec1:
    export_type = st.selectbox("Export", ["Themes","Industries","Sectors","All"])
    export_df = metrics if export_type == "All" else metrics[metrics["type"] == export_type[:-1] if export_type != "All" else ""].copy()
    if export_type != "All":
        type_map = {"Themes": "Theme", "Industries": "Industry", "Sectors": "Sector"}
        export_df = metrics[metrics["type"] == type_map[export_type]].copy()
    buf = io.BytesIO()
    export_df.to_csv(buf, index=False); buf.seek(0)
    st.download_button("Download CSV", data=buf,
        file_name=f"sector_flow_{export_type.lower()}_{period}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv")
