import streamlit as st
import pandas as pd
import datetime as dt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")
import streamlit.components.v1 as components
import numpy as np
import re
import datetime
from build_dataset import build_feature_dataset, floor_5_or_int
from technical_features import add_all_labels
from backtest_same_day_close import generate_trade_log, compute_trade_stats
from backtest_next_day_open import generate_trade_log_next_open
from backtest_intraday_same_bar_close import backtest_intraday_close, backtest_intraday_close_sell_only
from backtest_intraday_next_bar_open import backtest_intraday_next_open, backtest_intraday_next_open_sell_only
from model import secret_sauce, spicy_sauce
from openai import OpenAI
import json
import glob
from streamlit_autorefresh import st_autorefresh
from daily_engine import load_ticker_history, snapshot_all_signals_first_time
import os



client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ----------------------------------------------------
# HELPER FUNCTIONS — AI Overlay
# ----------------------------------------------------
def snap_to_price(target_price, df, tolerance_pct=0.03, role=None):
    """Snap an AI-estimated price to the nearest actual High/Low within tolerance.
    If role='upper', only snap to Highs. If role='lower', only snap to Lows.
    """
    if role == "upper":
        candidates = df["High"].reset_index(drop=True)
    elif role == "lower":
        candidates = df["Low"].reset_index(drop=True)
    else:
        candidates = pd.concat([df["High"], df["Low"]]).reset_index(drop=True)
    diffs = (candidates - target_price).abs()
    nearest_price = float(candidates.iloc[diffs.argmin()])
    if abs(nearest_price - target_price) / target_price <= tolerance_pct:
        return nearest_price
    return target_price

def detect_swing_points(df, window=5, max_points=8):
    """Detect swing highs and swing lows from OHLC data.
    A swing high is a bar whose High is the highest in a 2*window+1 neighborhood.
    Returns (swing_highs, swing_lows) as lists of {'date': str, 'price': float}.
    """
    highs = df["High"]
    lows = df["Low"]
    swing_highs = []
    swing_lows = []
    for i in range(window, len(df) - window):
        local_high = highs.iloc[i - window:i + window + 1]
        if highs.iloc[i] == local_high.max():
            swing_highs.append({
                "date": df.index[i].strftime("%Y-%m-%d"),
                "price": round(float(highs.iloc[i]), 2)
            })
        local_low = lows.iloc[i - window:i + window + 1]
        if lows.iloc[i] == local_low.min():
            swing_lows.append({
                "date": df.index[i].strftime("%Y-%m-%d"),
                "price": round(float(lows.iloc[i]), 2)
            })
    # Deduplicate consecutive same-price swings, keep most recent
    def dedup(pts):
        if not pts:
            return pts
        result = [pts[0]]
        for p in pts[1:]:
            if p["price"] != result[-1]["price"]:
                result.append(p)
        return result
    swing_highs = dedup(swing_highs)[-max_points:]
    swing_lows = dedup(swing_lows)[-max_points:]
    return swing_highs, swing_lows

def parse_overlay_from_response(response_text):
    """Extract overlay JSON block and clean markdown from AI response.
    Returns (clean_markdown, overlay_dict) tuple.
    """
    pattern = r'```json\s*(\{.*?\})\s*```'
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        json_str = match.group(1)
        clean_markdown = response_text[:match.start()].rstrip()
        try:
            overlay = json.loads(json_str)
            return clean_markdown, overlay
        except json.JSONDecodeError:
            return response_text, None
    return response_text, None

def validate_overlay(overlay, current_close):
    """Validate and fix directional consistency of overlay data.
    For BULLISH: target > current_close, invalidation < current_close.
    For BEARISH: target < current_close, invalidation > current_close.
    Returns corrected overlay dict.
    """
    if not overlay:
        return overlay
    direction = overlay.get("direction", "BULLISH").upper()
    tp = overlay.get("target_price")
    inv = overlay.get("invalidation_price")
    if tp and inv:
        if direction == "BULLISH":
            # Target should be above close, invalidation below
            if tp < current_close and inv > current_close:
                # Swapped — fix
                overlay["target_price"], overlay["invalidation_price"] = inv, tp
            elif tp < current_close:
                overlay["target_price"] = None  # nonsensical, remove
        elif direction == "BEARISH":
            # Target should be below close, invalidation above
            if tp > current_close and inv < current_close:
                # Swapped — fix
                overlay["target_price"], overlay["invalidation_price"] = inv, tp
            elif tp > current_close:
                overlay["target_price"] = None  # nonsensical, remove
    # Cap pattern points to 6
    pat = overlay.get("pattern")
    if pat and pat.get("points") and len(pat["points"]) > 6:
        overlay["pattern"]["points"] = pat["points"][:6]
    return overlay

OVERLAY_JSON_INSTRUCTION = """

Additionally, at the END of your response, include a structured JSON block wrapped in ```json ... ``` code fences with this exact schema:
```json
{
  "support_levels": [price1, price2],
  "resistance_levels": [price1, price2],
  "direction": "BULLISH",
  "pattern": {
    "name": "Pattern Name",
    "points": [
      {"date": "YYYY-MM-DD", "price": 123.45, "role": "upper"},
      {"date": "YYYY-MM-DD", "price": 100.00, "role": "lower"}
    ]
  },
  "target_price": 123.45,
  "invalidation_price": 100.00
}
```
JSON rules:
- support_levels: 1-3 most important support prices visible on the chart
- resistance_levels: 1-3 most important resistance prices visible on the chart
- direction: "BULLISH" or "BEARISH" — your directional bias based on the analysis
- pattern.name: The chart pattern identified — ONLY linear-boundary patterns that can be drawn with straight trendlines:
  ALLOWED: Ascending Triangle, Descending Triangle, Symmetrical Triangle, Rising Wedge, Falling Wedge, Bull Flag, Bear Flag, Channel Up, Channel Down, Head & Shoulders, Inverse Head & Shoulders, Double Top, Double Bottom
  NOT ALLOWED (set pattern to null instead): Cup & Handle, Rounding Bottom, Rounding Top, Saucer, or any pattern requiring curved lines. Describe these in the text analysis but mention the pattern overlay cannot be drawn for this pattern type due to its curved nature.
- pattern.points: 3-6 key boundary vertices that outline the pattern shape (NOT individual price swings). Each point has:
  - "date": YYYY-MM-DD format
  - "price": numeric price at that vertex
  - "role": either "upper" (upper trendline/boundary) or "lower" (lower trendline/boundary)
  For channels/triangles/wedges: place 2-3 points on the upper trendline and 2-3 on the lower trendline. Upper points should form one straight line, lower points another.
  For bull flag / bear flag: The flag is the CONSOLIDATION zone only (NOT the flagpole). Place 2 upper points along the top edge of the consolidation (recent swing highs within the flag) and 2 lower points along the bottom edge (recent swing lows within the flag). These should form two roughly parallel, slightly sloping lines. Do NOT use the flagpole base or tip as flag boundary points.
  For head & shoulders: mark the left shoulder peak, head peak, right shoulder peak (upper), and the two neckline points (lower).
  For double top/bottom: mark the two peaks (upper) and the valley/neckline (lower), or vice versa.
  IMPORTANT: Each boundary (upper and lower) needs at least 2 points to draw a trendline. Never leave a boundary with only 1 point.
  Points should be in chronological order.
- target_price: Your projected target price (numeric). MUST be above current close for BULLISH, below for BEARISH.
- invalidation_price: Price level that invalidates the thesis (numeric). MUST be below current close for BULLISH, above for BEARISH.
- All prices MUST be numeric (no strings). All dates MUST be YYYY-MM-DD format.
- CRITICAL: For pattern points, you MUST use dates and prices ONLY from the "Swing Highs" and "Swing Lows" lists provided in the context data. Use swing highs for "upper" role points and swing lows for "lower" role points. Do NOT estimate prices from the chart image.
- For support_levels, use prices from the Swing Lows list. For resistance_levels, use prices from the Swing Highs list.
- If no clear pattern exists, or the pattern requires curved lines (cup & handle, rounding bottom, etc.), set pattern to null.
- IMPORTANT: In your text response, NEVER mention "JSON", "null", "set to null", or any implementation details. If a pattern cannot be overlaid, simply say the pattern overlay cannot be drawn for this pattern type due to its curved nature."""

# ----------------------------------------------------
# Application state
# ----------------------------------------------------
if "run_model" not in st.session_state:
    st.session_state.run_model = False
if "ai_analysis" not in st.session_state:
    st.session_state.ai_analysis = None
if "ai_ticker" not in st.session_state:
    st.session_state.ai_ticker = None
if "ai_overlay_visible" not in st.session_state:
    st.session_state.ai_overlay_visible = True
if "ai_toast" not in st.session_state:
    st.session_state.ai_toast = None
# # Chat state
# if "chat_open" not in st.session_state:
#     st.session_state.chat_open = False

# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []


st.set_page_config(page_title="INSANE Trading Model", layout="wide")

# --- Show deferred toast from AI analysis (after st.rerun) ---
if st.session_state.ai_toast:
    st.toast(st.session_state.ai_toast)
    st.session_state.ai_toast = None

# ----------------------------------------------------
# GLOBAL CSS (DARK THEME SAFE)
# ----------------------------------------------------
st.markdown("""
<style>
/* Index cards – dark theme safe */
.index-card {
    background: #1e1e1e;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 8px;
}

.index-title {
    font-size: 14px;
    font-weight: 700;
    color: #e0e0e0;
}

.index-signal {
    font-size: 15px;
    font-weight: 800;
    margin-top: 2px;
}

.index-date {
    font-size: 11px;
    color: #9e9e9e;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# TITLE
# ----------------------------------------------------
import base64

with open("assets/INSANE_Logo.png", "rb") as _logo_file:
    _logo_b64 = base64.b64encode(_logo_file.read()).decode()

st.markdown(
    f"""
    <div style="display:flex; align-items:center; gap:16px; margin-bottom:8px;">
        <img src="data:image/png;base64,{_logo_b64}" style="height:60px;" />
        <h1 id='MainTitle' style="margin:0;">Intelligent Statistical Algorithm for Navigating Equities</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------------------------------
# PRELOAD SIGNAL DATA (used by index bar + right column)
# ----------------------------------------------------
INDEX_TICKERS = {
    "SPY": "SPY",
    "QQQ": "QQQ",
    "^GSPC": "S&P 500 (SPX)",
    "^IXIC": "NASDAQ Composite",
    "^RUT": "Russell 2000",
    "^VIX": "VIX"
}

_long_rows = []
_short_rows = []
_index_signals = {}
_today_date = None

try:
    with open("daily_closes.json", "r") as f:
        _daily_data = json.load(f)
    _today_date = _daily_data["date"]
    _today_closes = _daily_data["closes"]

    _signal_files = glob.glob("signal_history/*.json")

    for _file in _signal_files:
        _ticker = os.path.basename(_file).replace(".json", "")

        with open(_file, "r") as f:
            _history = json.load(f)

        if not isinstance(_history, list) or len(_history) == 0:
            continue

        _last = _history[-1]
        _signal = _last["signal"]
        _signal_date = _last["date"]
        _signal_close = _last["close"]
        _today_close = _today_closes.get(_ticker)

        # INDEX HANDLING
        if _ticker in INDEX_TICKERS:
            _index_signals[_ticker] = {
                "signal": "LONG" if _signal == "UP" else "SHORT",
                "date": _signal_date
            }
            continue

        # STOCK SIGNALS
        _delta = None
        _delta_pct = None

        if _today_close is not None and _signal_close != 0:
            if _signal == "UP":
                _delta = _today_close - _signal_close
            elif _signal == "DOWN":
                _delta = _signal_close - _today_close

            _delta = round(_delta, 2)
            _delta_pct = round((_delta / _signal_close) * 100, 2)

        _row = {
            "Ticker": _ticker,
            "Signal Date": _signal_date,
            "Signal Close": round(_signal_close, 2),
            "Today Close": round(_today_close, 2) if _today_close else None,
            "Delta": _delta,
            "Delta %": _delta_pct
        }

        if _signal == "UP":
            _long_rows.append(_row)
        elif _signal == "DOWN":
            _short_rows.append(_row)

    _data_loaded = True
except Exception:
    _data_loaded = False

# ----------------------------------------------------
# INDEX TREND SIGNALS — HORIZONTAL BAR (FULL WIDTH)
# ----------------------------------------------------
if _data_loaded and _index_signals:
    st.markdown(
        f"<div style='display:flex; align-items:baseline; gap:12px; margin-bottom:8px;'>"
        f"<span style='font-size:18px; font-weight:700;'>📅 Daily Market Signals (@ 3PM CST)</span>"
        f"<span style='font-size:13px; color:#9e9e9e;'>Last Update: {_today_date}</span>"
        f"</div>",
        unsafe_allow_html=True
    )
    st.markdown("""
    <style>
    .index-bar {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin-bottom: 12px;
    }
    .index-pill {
        background: #1e1e1e;
        border: 1px solid #2a2a2a;
        border-radius: 8px;
        padding: 8px 14px;
        min-width: 120px;
        text-align: center;
    }
    .index-pill-name {
        font-size: 12px;
        font-weight: 700;
        color: #e0e0e0;
    }
    .index-pill-signal {
        font-size: 14px;
        font-weight: 800;
        margin-top: 2px;
    }
    .index-pill-date {
        font-size: 10px;
        color: #9e9e9e;
    }
    </style>
    """, unsafe_allow_html=True)

    pills_html = '<div class="index-bar">'
    for tkr, name in INDEX_TICKERS.items():
        if tkr in _index_signals:
            sig = _index_signals[tkr]
            color = "#4CAF50" if sig["signal"] == "LONG" else "#F44336" if sig["signal"] == "SHORT" else "#B0BEC5"
            pills_html += f'''
            <div class="index-pill">
                <div class="index-pill-name">{name}</div>
                <div class="index-pill-signal" style="color:{color};">{sig["signal"]}</div>
                <div class="index-pill-date">{sig["date"]}</div>
            </div>'''
    pills_html += '</div>'
    st.markdown(pills_html, unsafe_allow_html=True)

# ----------------------------------------------------
# LAYOUT COLUMNS
# ----------------------------------------------------
filters_col, main_col, chat_col = st.columns([1, 4, 1])

# ----------------------------------------------------
# CHAT COLUMN (RIGHT SIDE)
# ----------------------------------------------------
with chat_col:

    # -----------------------------------------------------------
    # EARNINGS WATCHLIST (enriched live from signal_strength_history)
    # -----------------------------------------------------------
    try:
        with open("earnings_watchlist.json", "r") as f:
            _earnings = json.load(f)

        _wl = _earnings.get("watchlist", [])
        if _wl:
            # Enrich each row with live signal + current price
            _enriched = []
            for _e in _wl:
                _sym = _e["symbol"]
                _sig_file = os.path.join("signal_history", f"{_sym}.json")
                _signal = "N/A"
                _strength = "N/A"
                _sig_date = ""
                if os.path.exists(_sig_file):
                    try:
                        with open(_sig_file, "r") as _sf:
                            _sig_hist = json.load(_sf)
                        if _sig_hist:
                            _last = _sig_hist[-1]
                            _signal = "Long" if _last["signal"] == "UP" else ("Short" if _last["signal"] == "DOWN" else "N/A")
                            _strength = _last.get("strength", "Strong")
                            _sig_date = _last.get("date", "")
                    except Exception:
                        pass
                # Current close from daily_closes.json (already loaded as _today_closes)
                _cur_price = _today_closes.get(_sym) if _data_loaded else None
                _enriched.append({
                    "Ticker": _sym,
                    "Earnings": _e.get("earnings_date", ""),
                    "Hour": _e.get("hour", ""),
                    "EPS Est": _e.get("eps_estimate"),
                    "Rev Est": _e.get("revenue_estimate"),
                    "Signal": _signal,
                    "Strength": _strength,
                    "Sig Date": _sig_date,
                    "Price": round(_cur_price, 2) if _cur_price is not None else None,
                })

            with st.expander(f"📅 Earnings This Week ({len(_wl)})", expanded=True):
                st.caption(f"Scanned: {_earnings.get('scan_from', '')} → {_earnings.get('scan_to', '')}  |  Generated: {_earnings.get('generated', '')}")
                _edf = pd.DataFrame(_enriched)
                _edf = _edf.sort_values("Earnings")
                if "EPS Est" in _edf.columns:
                    _edf["EPS Est"] = _edf["EPS Est"].round(2)
                st.dataframe(_edf, use_container_width=True, height=400, hide_index=True)
    except FileNotFoundError:
        pass
    except Exception as e:
        st.warning(f"Earnings data error: {e}")

    # -----------------------------------------------------------
    # DAILY SIGNALS DISPLAY (COMPACT VERSION)
    # -----------------------------------------------------------

    st.markdown("""
    <style>
        .signal-header {
            font-size: 18px !important;
            font-weight: 700;
            margin-bottom: 6px;
        }
        .signal-subheader {
            font-size: 15px !important;
            font-weight: 600;
            margin-top: 10px;
            margin-bottom: 4px;
        }
        .signal-text {
            font-size: 13px !important;
            margin-bottom: 4px;
        }
        .signal-box {
            padding: 6px;
            border-radius: 6px;
            background: #f5f5f5;
            margin-bottom: 6px;
            font-size: 13px !important;
        }
    </style>
    """, unsafe_allow_html=True)

    if _data_loaded:

        # --------------------------------------------------
        # DISPLAY LONG SIGNALS
        # --------------------------------------------------
        with st.expander(f"🟢 Long Signals ({len(_long_rows)})", expanded=True):
            if _long_rows:
                st.dataframe(
                    pd.DataFrame(_long_rows).sort_values("Signal Date", ascending=False),
                    use_container_width=True,
                    height=300,
                    hide_index=True
                )
            else:
                st.write("No Long signals.")

        # --------------------------------------------------
        # DISPLAY SHORT SIGNALS
        # --------------------------------------------------
        with st.expander(f"🔴 Short Signals ({len(_short_rows)})", expanded=True):
            if _short_rows:
                st.dataframe(
                    pd.DataFrame(_short_rows).sort_values("Signal Date", ascending=False),
                    use_container_width=True,
                    height=300,
                    hide_index=True
                )
            else:
                st.write("No Short signals.")

    else:
        st.warning("Signal data not available. Run daily_engine.py first.")

    #     # Today's Stock Signals
    #     st.markdown("#### 🟢 Long Signals")
    #     if signals["long"]:
    #         # st.table(pd.DataFrame(signals["long"], columns=["Ticker"]))
    #         st.dataframe(
    #             pd.DataFrame(signals["long"], columns=["Ticker"]),
    #             height=200,
    #             use_container_width=True
    #         )
    #     else:
    #         st.write("No Long signals today.")

    #     st.markdown("#### 🔴 Short Signals")
    #     if signals["short"]:
    #         # st.table(pd.DataFrame(signals["short"], columns=["Ticker"]))
    #         st.dataframe(
    #             pd.DataFrame(signals["short"], columns=["Ticker"]),
    #             height=200,
    #             use_container_width=True
    #         )
    #     else:
    #         st.write("No Short signals today.")

    #     # INDEX SIGNALS
    #     st.markdown("### 📈 Index Trend Signals")

    #     def show_index(name, obj):
    #         signal = obj["signal"]
    #         date = obj["date"]

    #         color = (
    #             "#4CAF50" if signal == "LONG"
    #             else "#F44336" if signal == "SHORT"
    #             else "#B0BEC5"
    #         )

    #         st.markdown(
    #             f"""
    #             <div class="index-card">
    #                 <div class="index-title">{name}</div>
    #                 <div class="index-signal" style="color:{color};">{signal}</div>
    #                 <div class="index-date">Last signal: {date}</div>
    #             </div>
    #             """,
    #             unsafe_allow_html=True
    #         )

    #     show_index("SPY", signals["spy"])
    #     show_index("QQQ", signals["qqq"])
    #     show_index("S&P 500 (SPX)", signals["spx"])
    #     show_index("NASDAQ Composite", signals["nasdaq"])
    #     show_index("Russell 2000", signals["russell"])


    # except Exception:
    #     st.warning("Daily signals not found. Run daily_engine.py first.")

    # -----------------------------------------------------------
    # CHATBOT (BELOW DAILY SIGNALS)
    # -----------------------------------------------------------
    # st.markdown("### 🤖 INSANEgpt")

    # # Show chat history
    # for msg in st.session_state.chat_history:
    #     st.markdown(f"<div style='margin-bottom:8px;'>{msg}</div>", unsafe_allow_html=True)

    # # User input
    # user_msg = st.text_input("Message:", key="chat_input_field_right")

    # if st.button("Send Message", key="chat_send_right"):
    #     if user_msg.strip():
    #         st.session_state.chat_history.append(f"Question: {user_msg}")

    #         response = client.chat.completions.create(
    #             model="gpt-4o-mini",
    #             messages=[
    #                 {"role": "system", "content": "You are INSANE — Imran's trading assistant. Expert Financial Analyst specializing in stock trading strategies. Explain indicators, provide concise, accurate answers based on statistical analysis and trading principles. Do not answer anything outside of stock trading or finance."},
    #                 {"role": "user", "content": user_msg}
    #             ]
    #         )
    #         reply = response.choices[0].message.content

    #         st.session_state.chat_history.append(f"INSANEgpt: {reply}")

    #         # Clear input field
    #         st.session_state.pop("chat_input_field_right", None)
    #         st.rerun()


# ----------------------------------------------------
# FILTER PANEL (LEFT SIDE)
# ----------------------------------------------------
with filters_col:
    st.markdown("""
        <p style='background-color:#9BD36A;
                  padding:10px;
                  border-radius:6px;
                  font-size:20px;
                  font-weight:bold;
                  text-align:center;
                  margin-bottom:15px;'>
            FILTERS
        </p>
    """, unsafe_allow_html=True)

    ticker = st.text_input("Ticker", value="NVDA")
    ticker = ticker.strip().upper()
    if ticker == "SPX":
        ticker = "^GSPC"
    elif ticker == "NASDAQ" or ticker == "NDX":
        ticker = "^IXIC"
    elif ticker == "RUSSELL":
        ticker = "^RUT"
    elif ticker == "VIX":
        ticker = "^VIX"
    timeframe_options = ["5m", "15m", "30m", "1h", "4h", "1d"]
    timeframe = st.selectbox("Timeframe", timeframe_options, index=5)   # default = "1d"
    
    # ## ----- Auto Refresh-------
    # # Convert timeframe (like '5m', '15m', '1h') to minutes
    # def tf_to_minutes(tf):
    #     tf = tf.lower()
    #     if tf.endswith("m"):
    #         return int(tf.replace("m", ""))
    #     if tf.endswith("h"):
    #         return int(tf.replace("h", "")) * 60
    #     return None

    # interval_minutes = tf_to_minutes(timeframe)

    # # Auto-refresh ONLY for intraday timeframes
    # if interval_minutes is not None:
    #     now = dt.datetime.now()

    #     # Find the minute when the MOST RECENT bar closed
    #     last_close_minute = (now.minute // interval_minutes) * interval_minutes

    #     # Create timestamp for the last closed bar
    #     last_close_time = now.replace(
    #         minute=last_close_minute,
    #         second=0,
    #         microsecond=0
    #     )

    #     # Refresh 1 minute AFTER bar close
    #     refresh_trigger_time = last_close_time + dt.timedelta(seconds=30)

    #     # If current time >= trigger time → refresh page
    #     if now >= refresh_trigger_time:
    #         st_autorefresh(interval=1000, key="refresh_after_bar_close")

    if timeframe in ["5m", "15m", "30m", "1h", "4h"]:
        default_start = dt.date.today() - dt.timedelta(days=7)
    else:
        default_start = dt.date.today() - dt.timedelta(days=365*5)    
    start_date = st.date_input("Start Date", value=default_start)
    start_date_user = start_date

    if timeframe in ["5m", "15m", "30m", "1h", "4h"]:
        extended_start = start_date_user - dt.timedelta(days=23)
        print("Extended Start for Intraday:", extended_start)
    elif timeframe == '1d': 
        extended_start = start_date_user - dt.timedelta(days=290)
    
    # if timeframe  == "1d":
    #     end_date = st.date_input("End Date", value=dt.date.today())
    # else:
    #     end_date = st.date_input("End Date", value=dt.date.today() + dt.timedelta(days=1))
    end_date = st.date_input("End Date", value=dt.date.today() + dt.timedelta(days=1))
    capital = st.number_input("Capital ($)", 1000, 1_000_000, 10000, 500)
    # Clear AI analysis if ticker or timeframe changed (runs on every rerun, not just button click)
    current_key = f"{ticker}_{timeframe}"
    if st.session_state.ai_ticker != current_key:
        st.session_state.ai_analysis = None
        st.session_state.ai_overlay_visible = False
        st.session_state.ai_ticker = current_key

    if st.button("Run Model", key="run_button"):
        st.session_state.run_model = True


# ----------------------------------------------------
# MAIN CONTENT SECTION
# ----------------------------------------------------
if st.session_state.run_model:

    with main_col:
        with st.spinner("Fetching data and running model..."):

            # ------------------------------------------------------------
            # 1) BUILD DATASET
            # ------------------------------------------------------------
            data = build_feature_dataset(
                ticker,
                start_date=extended_start.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"), 
                timeframe=timeframe
            )
            
            if timeframe in ["5m", "15m", "30m", "1h", "4h"]:
                data = data.fillna(0).copy()
            else:
                data = data.dropna().copy()

            # ------------------------------------------------------------
            # REMOVE PARTIAL LAST CANDLE (PERFECT SAFE LOGIC)
            # ------------------------------------------------------------
            tf = timeframe.lower()

            # Determine interval length in minutes
            if tf.endswith("m"):
                interval_minutes = int(tf.replace("m", ""))
            elif tf.endswith("h"):
                interval_minutes = int(tf.replace("h", "")) * 60
            else:
                interval_minutes = None  # daily timeframe → no partial candles

            if interval_minutes is not None:
                now = pd.Timestamp.now(tz=data.index.tz)     # current time in same timezone
                last = data.index[-1]                         # last candle timestamp

                # Last fully completed candle time
                last_full_close = now.floor(f"{interval_minutes}min")

                # If last returned candle ends AFTER the last full interval → it's partial
                if last > last_full_close:
                    df = data.iloc[:-1]


            # Add Labels
            df = add_all_labels(data)
            # df = df[df.index >= pd.to_datetime(start_date_user).tz_localize(df.index.tz)]

            # ------------------------------------------------------------
            # 2) COMPUTE KALMAN SIGNALS
            # ------------------------------------------------------------
            price = df['Close']
            if ticker == "^GSPC":
                p_win = 84
            else:
                p_win = 84  
            if timeframe in ["5m", "15m", "30m", "1h", "4h"]:
                # price = (df['High'] + df['Low']) / 2
                df["Smooth"], df["Slope"] = spicy_sauce(df["Close"])
                df['price_delta'] = df["Close"] - df["Smooth"]
                price_delta = df['price_delta']
                df["q05"] = df["price_delta"].rolling(p_win).quantile(0.05) # 0.05 quantile
                df["q35"] = df["price_delta"].rolling(p_win).quantile(0.25)
                df["q50"] = df["price_delta"].rolling(p_win).quantile(0.50)
                df["q65"] = df["price_delta"].rolling(p_win).quantile(0.75)
                df["q95"] = df["price_delta"].rolling(p_win).quantile(0.95) # 0.95 quantile
                df["date"] = df.index.date
                day_high = df.groupby("date")["High"].cummax()
                day_low  = df.groupby("date")["Low"].cummin()
                df["today_range"] = day_high - day_low
                df['price_delta_shift'] = df['price_delta'] - df['price_delta'].shift(1)
                df['price_delta_shift'] = df['price_delta_shift'].fillna(0)
                df["q01"] = df["price_delta_shift"].rolling(p_win).quantile(0.25) #0.05
                df["q99"] = df["price_delta_shift"].rolling(p_win).quantile(0.75) #0.95
                df['vwap_range'] = round(df["VWAP_Upper"] - df["VWAP_Lower"])
                daily_thr = floor_5_or_int(df['today_range'].median())
                vwap_thr  = floor_5_or_int(df['vwap_range'].median())
                
                # df["Slope_Neg"] = ((df["price_delta"] < df["q05"]) | (df['price_delta_shift'] <  df["q01"] )) & (df["Close"] < df["TOS_Trail"]) & ((df['vwap_range'] >= vwap_thr) | (df["today_range"]  >= daily_thr )) #  .shift(1)
                # df["Slope_Pos"] = ((df["price_delta"] > df["q95"]) | (df['price_delta_shift'] >  df["q99"] )) & (df["Close"] > df["TOS_Trail"]) & ((df['vwap_range'] >= vwap_thr) | (df["today_range"]  >= daily_thr ))  # .shift(1)

                df["Slope_Neg"] = ((df['price_delta_shift'] <  df["q01"] ) ) & (df["Close"] < df["TOS_Trail"]) & ((df['vwap_range'] >= vwap_thr) | (df["today_range"]  >= daily_thr )) & (df['TOS_RSI'] < 50)
                df["Slope_Pos"] = ((df['price_delta_shift'] >  df["q99"] ) ) & (df["Close"] > df["TOS_Trail"]) & ((df['vwap_range'] >= vwap_thr) | (df["today_range"]  >= daily_thr )) & (df['TOS_RSI'] > 50)
                
                
                df["Turn_Up"]   = df["Slope_Pos"] & (~df["Slope_Pos"].shift(1).fillna(False))
                df["Turn_Down"] = df["Slope_Neg"] & (~df["Slope_Neg"].shift(1).fillna(False))
                            
            
            else:                                    
                df["Smooth"], df["Slope"] = secret_sauce(price)
                df["price_delta"] = df["Close"] - df["Smooth"]

                slope_q = df["Slope"].quantile([0.05, 0.35, 0.5, 0.65, 0.95]).tolist()
                slope_vals = [round(x / 0.25) * 0.25 for x in slope_q]
                print(slope_vals)

                df["Slope_Neg"] = (
                    (df["Slope"] < (slope_vals[0] + slope_vals[1])/2) & # ) / 2
                    (df["Close"] < df["Smooth"]) &
                    (df["Slope"] < df["Slope"].shift(1))
                )

                df["Slope_Pos"] = (
                    (df["Slope"] > (slope_vals[3] + slope_vals[4]) / 2) &
                    (df["Close"] > df["Smooth"]) &
                    (df["Slope"] > df["Slope"].shift(1))
                )
            
            # # if timeframe in ["5m", "15m", "30m", "1h", "4h"]:
            # #     q_roll = 120
            # # else: 
            # #     q_roll = 252

            # # df["q05"] = df["Slope"].expanding().quantile(0.05)
            # # df["q35"] = df["Slope"].expanding().quantile(0.35)
            # # df["q50"] = df["Slope"].expanding().quantile(0.50)
            # # df["q65"] = df["Slope"].expanding().quantile(0.65)
            # # df["q95"] = df["Slope"].expanding().quantile(0.95)
            
            # # df["Slope_Neg"] = (
            # #     (df["Slope"] < (df["q05"] + df["q35"]) / 2) &
            # #     (df["Close"] < df["Smooth"]) &
            # #     (df["Slope"] < df["Slope"].shift(1))
            # # )

            # # df["Slope_Pos"] = (
            # #     (df["Slope"] > (df["q65"] + df["q95"]) / 2) &
            # #     (df["Close"] > df["Smooth"]) &
            # #     (df["Slope"] > df["Slope"].shift(1))
            # # )

            # # df["Slope_Neg"] = (
            # #     ((df["Slope"] < (df["q95"]) ) &
            # #     (df["Slope"].shift(1) > (df["q95"].shift(1)))) | 
            # #     ((df["Slope"].shift(1) < (df["Slope"])))

            # # )

            # # df["Slope_Pos"] = (
            # #     ((df["Slope"] > (df["q05"])) &
            # #     (df["Slope"].shift(1) < (df["q05"].shift(1)))) |
            # #     ((df["Slope"].shift(1) > (df["Slope"])))
            # # )
            
            # # df["Turn_Up"]   = df["Slope_Pos"] & (~df["Slope_Pos"].shift(1).fillna(False))
            # # df["Turn_Down"] = df["Slope_Neg"] & (~df["Slope_Neg"].shift(1).fillna(False))
            if timeframe == "1d":
                frozen = load_ticker_history(ticker)
                if len(frozen) == 0:
                    # New ticker — compute signals from scratch
                    df["Turn_Up"]   = df["Slope_Pos"] & (~df["Slope_Pos"].shift(1).fillna(False))
                    df["Turn_Down"] = df["Slope_Neg"] & (~df["Slope_Neg"].shift(1).fillna(False))
                    # Attempt to persist (time-gated inside snapshot_all_signals_first_time)
                    snapshot_all_signals_first_time(ticker, df)
                    frozen = load_ticker_history(ticker)
                
                # Only overwrite from frozen history if it actually exists on disk;
                # otherwise keep the computed signals for display
                if len(frozen) > 0:
                    df["Turn_Up"] = False
                    df["Turn_Down"] = False
                    for record in frozen:
                        date = record["date"]
                        mask = df.index.strftime("%Y-%m-%d") == date
                        if mask.any():
                            idx = df.index[mask][0]
                            if record["signal"] == "UP":
                                df.at[idx, "Turn_Up"] = True
                            if record["signal"] == "DOWN":
                                df.at[idx, "Turn_Down"] = True

            df["Turn_Up"] = df["Turn_Up"].fillna(False)
            df["Turn_Down"] = df["Turn_Down"].fillna(False)


            # ------------------------------------------------------------
            # 3) INITIALIZE POSITION
            # ------------------------------------------------------------
            df["Position"] = 0


            # ------------------------------------------------------------
            # 4) POSITION STATE MACHINE (ENTRY ONLY)
            # ------------------------------------------------------------
            for i in range(1, len(df)):
                if df["Turn_Up"].iloc[i]:
                    df.at[df.index[i], "Position"] = 1

                elif df["Turn_Down"].iloc[i]:
                    df.at[df.index[i], "Position"] = -1

                else:
                    df.at[df.index[i], "Position"] = df["Position"].iloc[i-1]


            # ------------------------------------------------------------
            # 5) EXIT LOGIC (RAW EXIT SIGNALS)
            # ------------------------------------------------------------
            
            if timeframe in ["5m", "15m", "30m", "1h", "4h"]:
                
                df["Sell_Long"] = (
                    (df["Position"] == 1) &
                    (
                        ((df["Close"].shift(1) >= df["VWAP_Upper"].shift(1)) & 
                        (df["Close"] < df["VWAP_Upper"]) & (df['vwap_range'] >= vwap_thr)) | 
                        # ((df["Close"].shift(1) >= df["VWAP"].shift(1)) &
                        # (df["Close"] < df["VWAP"]) & (df['vwap_range'] >= vwap_thr)) |
                        ((df["Close"].shift(1) >= df["TOS_Trail"].shift(1)) &
                        (df["Close"] < df["TOS_Trail"])) |
                        ((df["TOS_RSI"].shift(1) > 70) &
                        (df["TOS_RSI"] < 70)) #| 
                        # ((df['Close'] < df['Open'].shift(1))) #|
                        # ((df["Low"].shift(1) >= df["Low"]) &
                        # (df["Close"].shift(1) >= df["Close"])) |
                        # ((df["High"].shift(1) >= df["High"]) &
                        # (df["Close"].shift(1) >= df["Close"])) |
                        # ((df["High"].shift(1) >= df["High"]) &
                        # (df["Low"].shift(1) >= df["Low"])) 

                    )
                )

                df["Sell_Short"] = (
                    (df["Position"] == -1) &
                    (
                        ((df["Close"].shift(1) <= df["VWAP_Lower"].shift(1)) &
                        (df["Close"] > df["VWAP_Lower"]) & (df['vwap_range'] >= vwap_thr)) |
                        # ((df["Close"].shift(1) <= df["VWAP"].shift(1)) &
                        # (df["Close"] > df["VWAP"]) & (df['vwap_range'] >= vwap_thr)) | 
                        ((df["Close"].shift(1) <= df["TOS_Trail"].shift(1)) &
                        (df["Close"] > df["TOS_Trail"])) |
                        ((df["TOS_RSI"].shift(1) < 30) &
                        (df["TOS_RSI"] > 30)) #|
                        # ((df['Close'] > df['Open'].shift(1))) #|
                        # ((df["Low"].shift(1) <= df["Low"]) &
                        # (df["Close"].shift(1) <= df["Close"])) |
                        # ((df["High"].shift(1) <= df["High"]) &
                        # (df["Close"].shift(1) <= df["Close"])) |
                        # ((df["High"].shift(1) <= df["High"]) &
                        # (df["Low"].shift(1) <= df["Low"])) 
                    )
                )

                # df[['Close', 'Low', 'VWAP_Lower', 'VWAP', 'Position', 'Sell_Short', 'Sell_Long']].to_csv('test.csv')
            else:
                df["Sell_Long"] = (
                    (df["Position"] == 1) &
                    (df["High"].shift(1) >= df["BB_Upper"].shift(1)) &
                    (df["Close"].shift(1) >= df["BB_Upper"].shift(1)) & 
                    ((df["High"] <= df["BB_Upper"]) | 
                    (df["Close"] <= df["BB_Upper"]))
                ) 
                # | ((df["Position"] == 1) & 
                #     (df["Slope"] >= df['q65'])  
                #     )

                df["Sell_Short"] = (
                    (df["Position"] == -1) &
                    (df["Low"].shift(1) <= df["BB_Lower"].shift(1)) &
                    (df["Close"].shift(1) <= df["BB_Lower"].shift(1)) &
                    ((df["Low"] >= df["BB_Lower"]) |
                    (df["Close"] >= df["BB_Lower"]))
                ) 
                # | ((df["Position"] == -1) & 
                #     (df["Slope"] <= df['q35'])  
                #     )
            
            df["Sell_Long"]  = df["Sell_Long"].fillna(False)
            df["Sell_Short"] = df["Sell_Short"].fillna(False)


            # ------------------------------------------------------------
            # 6) FINAL POSITION STATE MACHINE (EXIT TRADES)
            # ------------------------------------------------------------
            for i in range(1, len(df)):
                prev = df["Position"].iloc[i-1]

                if prev == 1 and df["Sell_Long"].iloc[i]:
                    df.at[df.index[i], "Position"] = 0

                elif prev == -1 and df["Sell_Short"].iloc[i]:
                    df.at[df.index[i], "Position"] = 0

                else:
                    df.at[df.index[i], "Position"] = df["Position"].iloc[i]


            # # ------------------------------------------------------------
            # # 7) FIRST EXIT ONLY (NO DUPLICATE EXIT MARKERS)
            # # ------------------------------------------------------------
            # df["Sell_Long_Plot"] = False
            # df["Sell_Short_Plot"] = False

            # in_long_trade = False
            # in_short_trade = False

            # for i in range(1, len(df)):
            #     # New long entry resets long exit plot control
            #     if df["Turn_Up"].iloc[i]:
            #         in_long_trade = True
            #         in_short_trade = False

            #     # New short entry resets short exit plot control
            #     if df["Turn_Down"].iloc[i]:
            #         in_short_trade = True
            #         in_long_trade = False

            #     # First long exit after entry
            #     if in_long_trade and df["Sell_Long"].iloc[i]:
            #         df.at[df.index[i], "Sell_Long_Plot"] = True
            #         in_long_trade = False

            #     # First short exit after entry
            #     if in_short_trade and df["Sell_Short"].iloc[i]:
            #         df.at[df.index[i], "Sell_Short_Plot"] = True
            #         in_short_trade = False

            # df.loc[df["Turn_Up"], "Sell_Long_Plot"] = False
            # df.loc[df["Turn_Down"], "Sell_Short_Plot"] = False

            # ------------------------------------------------------------
            # 7) ONE EXIT PER TURN SIGNAL (NO SAME-BAR EXIT)
            # ------------------------------------------------------------
            df["Sell_Long_Plot"] = False
            df["Sell_Short_Plot"] = False

            exit_armed_long = False
            exit_armed_short = False

            last_turn_long_idx = None
            last_turn_short_idx = None

            for i in range(1, len(df)):

                # Arm LONG exit on Turn Up
                if df["Turn_Up"].iloc[i]:
                    exit_armed_long = True
                    last_turn_long_idx = i

                # Arm SHORT exit on Turn Down
                if df["Turn_Down"].iloc[i]:
                    exit_armed_short = True
                    last_turn_short_idx = i

                # First LONG exit AFTER Turn Up (not same bar)
                if (
                    exit_armed_long and
                    df["Sell_Long"].iloc[i] and
                    last_turn_long_idx is not None and
                    i > last_turn_long_idx
                ):
                    df.at[df.index[i], "Sell_Long_Plot"] = True
                    exit_armed_long = False

                # First SHORT exit AFTER Turn Down (not same bar)
                if (
                    exit_armed_short and
                    df["Sell_Short"].iloc[i] and
                    last_turn_short_idx is not None and
                    i > last_turn_short_idx
                ):
                    df.at[df.index[i], "Sell_Short_Plot"] = True
                    exit_armed_short = False


            df = df[df.index >= pd.to_datetime(start_date_user).tz_localize(df.index.tz)]
            print(df[['Close','Turn_Up','Turn_Down','Sell_Long','Sell_Short','Sell_Long_Plot','Sell_Short_Plot','Position']].head(10))
            # print(default_start, start_date_user, extended_start)
            # ------------------------------------------------------------
            # 3) PRICE vs MOMENTUM — INTERACTIVE PLOT
            # ------------------------------------------------------------
            # --- AI Button (top-right of chart) ---
            date_range_days = (end_date - start_date_user).days
            is_daily = timeframe == "1d"
            ai_eligible = is_daily and date_range_days <= 365

            # --- Pre-load AI analysis cache (needed for overlay before chart renders) ---
            if ai_eligible:
                os.makedirs("ai_analysis", exist_ok=True)
                analysis_date = datetime.date.today().strftime("%Y-%m-%d")
                analysis_file = f"ai_analysis/{ticker}_{timeframe}_{analysis_date}.json"
                if st.session_state.ai_analysis is None and os.path.exists(analysis_file):
                    with open(analysis_file, "r") as f:
                        st.session_state.ai_analysis = json.load(f)
                    # Auto-enable overlay if cached analysis has overlay data
                    if st.session_state.ai_analysis.get("overlay"):
                        st.session_state.ai_overlay_visible = True

            has_overlay = (st.session_state.ai_analysis is not None and
                           st.session_state.ai_analysis.get("overlay") is not None)

            chart_title_col, overlay_col, ai_btn_col = st.columns([5, 1, 2])
            with chart_title_col:
                st.subheader("📌 Price vs Momentum Trend")
            with overlay_col:
                if has_overlay:
                    st.checkbox("Overlay Pattern", key="ai_overlay_visible")
            with ai_btn_col:
                if not ai_eligible:
                    st.button(
                        "🤖 Generate AI Analysis",
                        disabled=True,
                        help="AI analysis is available for daily charts with a date range ≤ 1 year"
                    )
                else:
                    ai_btn_clicked = st.button("🤖 Generate AI Analysis", key="ai_analysis_btn")

            # if timeframe in ["5m", "15m", "30m", "1h", "4h"]: 
            #     df = df.iloc[:-1]
            # # x = np.arange(len(df))
            # Columns we do NOT consider as indicators
            exclude_cols = ["Open", "High", "Low", "Close", "Volume", 
                            "Smooth",  "price_delta", 
                            "Slope_Pos", "Slope_Neg", "Turn_Up", "Turn_Down"] #"Slope",

            if timeframe in ["5m", "15m", "30m", "1h", "4h"]:
                indicator_cols = ['BB_Upper', 'BB_Lower', 'FRAMA', 'SMA_Short', 'SMA_Medium', 'SMA_Long', 'VWAP_Upper', 'VWAP', 'VWAP_Lower', 'VIX']
            else:
                indicator_cols = ['BB_Upper', 'BB_Lower', 'FRAMA', 'SMA_Short', 'SMA_Medium', 'SMA_Long', 'VIX']
            print(indicator_cols)

            selected_indicators = st.multiselect(
                "Select indicators to overlay:",
                indicator_cols,
                default=[]
            )

            fig = make_subplots(
                rows=3,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.60, 0.20, 0.20],
                specs=[
                    [{"secondary_y": True}],   # Row 1 → Candles + ATR
                    [{"secondary_y": False}],  # Row 2 → Volume
                    [{"secondary_y": False}]   # Row 3 → RSI
                ],
            )

            fig.update_yaxes(
                title_text="VIX",
                fixedrange=True,
                secondary_y=True,
                showgrid=False,
                zeroline=False
            )

            # -------------------------------
            # 1) CANDLESTICK (using df OHLCV)
            # -------------------------------
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    high=df["High"],
                    open=df["Open"],
                    close=df["Close"],
                    low=df["Low"],
                    name="Candles",
                    increasing_line_color="green",
                    decreasing_line_color="red",
                    increasing_fillcolor="rgba(0, 180, 0, 0.7)",
                    decreasing_fillcolor="rgba(220, 0, 0, 0.7)"
                ),
                row=1, col=1, secondary_y=False
            )

            # -------------------------------
            # 2) SMOOTH LINE (Kalman)
            # -------------------------------
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["Smooth"],
                    mode="lines",
                    name="Predicted Momentum",
                    line=dict(color="orange", width=2)
                ),
                row=1, col=1, secondary_y=False
            )
            
            # -------------------------------
            # ATR Trailing STOP LEVELS
            # -------------------------------
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["TOS_Trail"],
                    mode="markers",
                    name="Support/Resistance",
                    # line=dict(width=2) #color="orange", 
                    marker=dict(
                        size=6,
                        color="magenta",     # optional
                        symbol="circle"   # or "square", "diamond", "triangle-up"...
                    )
                ),
                row=1, col=1, secondary_y=False
            )   

            # 
            # -------------------------------
            # 3) RSI - Overbought / Oversold reference lines
            # -------------------------------
            # RSI line
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["TOS_RSI"],
                    mode="lines",
                    name="RSI",
                    line=dict(width=2, color="lightgreen")
                ),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=[70] * len(df),
                    mode="lines",
                    line=dict(width=1, dash="dash", color="orange"),
                    name="Overbought"
                ),
                row=3, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=[30] * len(df),
                    mode="lines",
                    line=dict(width=1, dash="dash", color="cyan"),
                    name="Oversold"
                ),
                row=3, col=1
            )

            # -------------------------------
            # 3) BUY / SELL MARKERS
            # -------------------------------
            # Turn Up (BUY)
            fig.add_trace(
                go.Scatter(
                    x=df.index[df["Turn_Up"]],
                    y=df["Low"][df["Turn_Up"]] - 0.5,
                    mode="markers",
                    marker=dict(
                        color="lime",
                        symbol="triangle-up",
                        size=14,
                        line=dict(color="black", width=1.4)
                    ),
                    name="Turn Up"
                ),
                row=1, col=1
            )
            
            # ------------------------------------------------------------
            # PLOT SELL SIGNALS (Long Exit)
            # ------------------------------------------------------------
            # if timeframe in ["5m", "15m", "30m", "1h", "4h"]:
            sell_longs = df[df["Sell_Long_Plot"]]
            
            fig.add_scatter(
                x=sell_longs.index,
                y=sell_longs["High"],
                mode="markers",
                name="Exit Warning",
                marker=dict(
                    symbol="triangle-down",
                    size=14,
                    color="white",
                    line=dict(width=1, color="black")
                )
            )

            # Turn Down (SELL)
            fig.add_trace(
                go.Scatter(
                    x=df.index[df["Turn_Down"]],
                    y=df["High"][df["Turn_Down"]] + 0.5,
                    mode="markers",
                    marker=dict(
                        color="yellow",
                        symbol="triangle-down",
                        size=14,
                        line=dict(color="black", width=1.4)
                    ),
                    name="Turn Down"
                ),
                row=1, col=1
            )

            # ------------------------------------------------------------
            # PLOT SELL SIGNALS (Short Exit)
            # ------------------------------------------------------------
            # if timeframe in ["5m", "15m", "30m", "1h", "4h"]:
            sell_shorts = df[df["Sell_Short_Plot"]]
            
            fig.add_scatter(
                x=sell_shorts.index,
                y=sell_shorts["Low"],
                mode="markers",
                name="Exit Warning",
                marker=dict(
                    symbol="triangle-up",
                    size=14,
                    color="white",
                    line=dict(width=1, color="black")
                )
            )

            # ---------------------------------------
            # Add Selected Indicators to the Chart
            # ---------------------------------------
            for ind in selected_indicators:
                if ind == 'VIX':
                    tick = 'VIX_Close' if 'VIX_Close' in df else 'VIX'
                    df[tick] = df[tick].replace(0, np.nan)
                    print(df[tick])
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df[tick],
                            mode="lines",
                            name='VIX',
                            line=dict(width=2)
                        ),
                        row=1, col=1, 
                        secondary_y=True
                    )
                else:
                    print(ind)
                    df[ind] = df[ind].replace(0, np.nan)
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df[ind],
                            mode="lines",
                            name=ind,
                            line=dict(width=2)
                        ),
                        row=1, col=1#, 
                        # secondary_y=True
                    )   

            # -------------------------------
            # 4) VOLUME BARS
            # -------------------------------
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df["Volume"],
                    name="Volume",
                    marker=dict(
                        color=["green" if df["Close"][i] >= df["Open"][i] else "red" for i in df.index]
                    )
                ),
                row=2, col=1
            )

            # -------------------------------
            # 5) Layout Settings
            # -------------------------------
            fig.update_layout(
                hovermode="x unified",    # shared vertical line
                hoverlabel=dict(
                    bgcolor="rgba(30,30,30,0.7)",
                    bordercolor="rgba(255,255,255,0.3)",
                    font_size=12,
                ),
                hoverdistance=5,
                spikedistance=-1,
                height=950,
                showlegend=True,
                dragmode="pan",
                xaxis=dict(
                    # rangeslider=dict(visible=True),
                    type="date"
                ),
                yaxis=dict(fixedrange=False),
                template="plotly_white",
                # margin=dict(l=20, r=20, t=20, b=20),
                margin=dict(t=60, b=60),
                legend=dict(
                    orientation="h",      # horizontal legend
                    yanchor="bottom",
                    y=-0.15,               # move it below the chart
                    xanchor="center",
                    x=0.5
                    ),
                )
            # Moves unified tooltip to the top outside the figure
            fig.update_layout(
                hoverlabel=dict(
                    align="left",
                ),
                margin=dict(t=120)  # give space above
            )

            if timeframe in ["5m", "15m", "30m", "1h", "4h"]:
                # 🚀 ADD RANGE BREAKS HERE (right after layout)
                if ticker in ["^GSPC", "^IXIC", "^RUT", "^VIX", "^DJI"]:
                    # Market Indices have different trading hours (9:30am - 4:00pm EST)
                    fig.update_xaxes(
                        rangebreaks=[
                            dict(bounds=["sat", "mon"]),          # skip weekends
                            dict(bounds=[15.1, 24], pattern="hour"),  # skip overnight hours
                            dict(bounds=[0, 8.5], pattern="hour")
                        ]
                    )
                else:
                    fig.update_xaxes(
                        rangebreaks=[
                            dict(bounds=["sat", "mon"]),          # skip weekends
                            dict(bounds=[19.05, 24], pattern="hour"),  # skip overnight hours
                            dict(bounds=[0, 3], pattern="hour")
                        ]
                    )

                # (Optional) set on the second row as well:
                fig.update_xaxes(
                    rangebreaks=[
                        dict(bounds=["sat", "mon"]),
                        dict(bounds=[19, 24], pattern="hour"),
                        dict(bounds=[0, 3], pattern="hour")
                    ],
                    row=2, col=1
                )

        
                # Count x-axes in the figure (price, volume, RSI = usually 3)
                xaxes = [ax for ax in fig.layout if ax.startswith("xaxis")]
                fig.update_traces(
                    hoverinfo="text",
                    hovertemplate="%{y:.2f}",
                    row=3, col=1
                )
                
                for ax in xaxes:
                    fig.layout[ax].update(
                        showspikes=True,
                        spikemode="across",
                        spikesnap="cursor",
                        spikethickness=1,
                        spikedash="dot",
                        spikecolor="rgba(180,180,180,0.8)"
                    )

                fig.update_traces(
                    hoverinfo="text",
                    hovertemplate="%{y:.2f}",
                    row=3, col=1
                )

            fig.update_xaxes(rangeslider_visible=False)

            # -----------------------------------------------
            # AI PATTERN OVERLAY (drawn before chart render)
            # -----------------------------------------------
            if (ai_eligible and
                st.session_state.get("ai_overlay_visible", True) and
                st.session_state.ai_analysis and
                st.session_state.ai_analysis.get("overlay")):

                _overlay = st.session_state.ai_analysis["overlay"]

                # Support levels (green dashed lines)
                for _lvl in _overlay.get("support_levels", []):
                    _snapped = snap_to_price(_lvl, df, role="lower")
                    fig.add_hline(
                        y=_snapped, line_dash="dash", line_color="#00ff88", line_width=1,
                        annotation_text=f"S: {_snapped:.2f}",
                        annotation_position="bottom right",
                        annotation_font=dict(color="#00ff88", size=10),
                        row=1, col=1
                    )

                # Resistance levels (red dashed lines)
                for _lvl in _overlay.get("resistance_levels", []):
                    _snapped = snap_to_price(_lvl, df, role="upper")
                    fig.add_hline(
                        y=_snapped, line_dash="dash", line_color="#ff4444", line_width=1,
                        annotation_text=f"R: {_snapped:.2f}",
                        annotation_position="top right",
                        annotation_font=dict(color="#ff4444", size=10),
                        row=1, col=1
                    )

                # Target price (dotted green)
                if _overlay.get("target_price"):
                    _tp = _overlay["target_price"]
                    fig.add_hline(
                        y=_tp, line_dash="dot", line_color="#00ff88", line_width=1.5,
                        annotation_text=f"\U0001f3af Target: {_tp:.2f}",
                        annotation_position="top left",
                        annotation_font=dict(color="#00ff88", size=11),
                        row=1, col=1
                    )

                # Invalidation price (dotted red)
                if _overlay.get("invalidation_price"):
                    _inv = _overlay["invalidation_price"]
                    fig.add_hline(
                        y=_inv, line_dash="dot", line_color="#ff4444", line_width=1.5,
                        annotation_text=f"\u26d4 Invalidation: {_inv:.2f}",
                        annotation_position="bottom left",
                        annotation_font=dict(color="#ff4444", size=11),
                        row=1, col=1
                    )

                # Pattern outline — draw upper and lower boundaries separately
                _pattern = _overlay.get("pattern")
                if _pattern and _pattern.get("points") and len(_pattern.get("points", [])) >= 2:
                    _pts = _pattern["points"]
                    _upper = [p for p in _pts if p.get("role") == "upper"]
                    _lower = [p for p in _pts if p.get("role") == "lower"]

                    # If roles not specified, fall back to connecting all points
                    if not _upper and not _lower:
                        _upper = _pts

                    # Find the full date range across all pattern points
                    _all_dates = [pd.Timestamp(p["date"]) for p in _pts]
                    _rightmost_date = max(_all_dates).strftime("%Y-%m-%d") if _all_dates else None

                    def _extrapolate_line(points, target_date, role_snap):
                        """Extend a trendline to target_date using linear extrapolation."""
                        if len(points) < 2 or not target_date:
                            return points
                        last_date = pd.Timestamp(points[-1]["date"])
                        if last_date >= pd.Timestamp(target_date):
                            return points  # already extends far enough
                        # Linear extrapolation from last two points
                        d0 = pd.Timestamp(points[-2]["date"]).timestamp()
                        d1 = pd.Timestamp(points[-1]["date"]).timestamp()
                        dt = pd.Timestamp(target_date).timestamp()
                        p0 = snap_to_price(points[-2]["price"], df, role=role_snap)
                        p1 = snap_to_price(points[-1]["price"], df, role=role_snap)
                        if d1 != d0:
                            slope = (p1 - p0) / (d1 - d0)
                            projected_price = p1 + slope * (dt - d1)
                        else:
                            projected_price = p1
                        extended = list(points) + [{"date": target_date, "price": projected_price, "role": points[-1].get("role"), "_ext": True}]
                        return extended

                    # Extend shorter line to match the rightmost date
                    if _upper and _lower and _rightmost_date:
                        _upper = _extrapolate_line(_upper, _rightmost_date, "upper")
                        _lower = _extrapolate_line(_lower, _rightmost_date, "lower")

                    def _resolve_prices(pts, role_snap):
                        """Snap original points to real prices; keep extrapolated prices as-is."""
                        return [p["price"] if p.get("_ext") else snap_to_price(p["price"], df, role=role_snap) for p in pts]

                    # Draw upper boundary
                    if len(_upper) >= 2:
                        fig.add_trace(
                            go.Scatter(
                                x=[p["date"] for p in _upper],
                                y=_resolve_prices(_upper, "upper"),
                                mode="lines+markers+text",
                                name=f"\U0001f4d0 {_pattern.get('name', 'Pattern')} (upper)",
                                line=dict(color="cyan", width=2, dash="dashdot"),
                                marker=dict(size=8, color="cyan", symbol="diamond"),
                                text=[_pattern.get("name", "")] + [""] * (len(_upper) - 1),
                                textposition="top center",
                                textfont=dict(color="cyan", size=11),
                                showlegend=bool(not _lower or len(_lower) < 2),
                            ),
                            row=1, col=1, secondary_y=False
                        )

                    # Draw lower boundary
                    if len(_lower) >= 2:
                        fig.add_trace(
                            go.Scatter(
                                x=[p["date"] for p in _lower],
                                y=_resolve_prices(_lower, "lower"),
                                mode="lines+markers",
                                name=f"\U0001f4d0 {_pattern.get('name', 'Pattern')} (lower)",
                                line=dict(color="cyan", width=2, dash="dashdot"),
                                marker=dict(size=8, color="cyan", symbol="diamond"),
                                showlegend=True,
                            ),
                            row=1, col=1, secondary_y=False
                        )

            st.plotly_chart(fig, use_container_width=True)

            # ------------------------------------------------------------
            # BACKTEST TABLES SIDE BY SIDE
            # ------------------------------------------------------------
            is_intraday = timeframe in ["5m", "15m", "30m", "1h", "4h"]
            bt_left, bt_right = st.columns([1, 1])

            with bt_left:
                # ---- Net Profit for Same-Day Close ----
                if is_intraday:
                    # close_df, close_equity_end = backtest_intraday_close(df, capital)
                    close_df, close_equity_end = backtest_intraday_close_sell_only(df, capital)
                else:
                    close_df, close_equity_end = generate_trade_log(df, capital)
                    # close_df, close_equity_end = backtest_intraday_close(df, capital)
                close_net_profit = close_equity_end - capital
                close_roi = (close_net_profit / capital) * 100


                st.markdown(
                    f"""
                    <div style='font-size:20px; font-weight:700; margin-bottom:5px;'>
                        💰 Net Profit (Same-Bar Close): 
                        <span style='color:{"green" if close_net_profit > 0 else "red"};'>
                            {close_net_profit:,.2f} ({close_roi:,.2f}%)
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.subheader("📊 Same-Bar Close Backtest Table")
                # Format dates
                if timeframe in ["5m", "15m", "30m", "1h", "4h"]:
                    close_df["Entry_Date"] = pd.to_datetime(close_df["Entry_Date"])
                    close_df["Exit_Date"]  = pd.to_datetime(close_df["Exit_Date"])
                else:    
                    close_df["Entry_Date"] = pd.to_datetime(close_df["Entry_Date"]).dt.strftime("%Y-%m-%d")
                    close_df["Exit_Date"]  = pd.to_datetime(close_df["Exit_Date"]).dt.strftime("%Y-%m-%d")
                
                # Remove unwanted column
                if "Trade" in close_df.columns:
                    close_df = close_df.drop(columns=["Trade"])
                st.dataframe(close_df, height=250)
                close_stats, _ = compute_trade_stats(close_df, capital)

            with bt_right:
                # ---- Net Profit for Next-Day Open ----
                if is_intraday:
                    # open_df, open_equity_end = backtest_intraday_next_open(df, capital)
                    open_df, open_equity_end = backtest_intraday_next_open_sell_only(df, capital)
                else:
                    open_df, open_equity_end = generate_trade_log_next_open(df, capital)
                    # open_df, open_equity_end = backtest_intraday_next_open(df, capital)
                open_net_profit = open_equity_end - capital
                open_roi = (open_net_profit / capital) * 100

                st.markdown(
                    f"""
                    <div style='font-size:20px; font-weight:700; margin-bottom:5px;'>
                        💰 Net Profit (Next-Day Open): 
                        <span style='color:{"green" if open_net_profit > 0 else "red"};'>
                            {open_net_profit:,.2f} ({open_roi:,.2f}%)
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.subheader("📊 Next-Bar Open Backtest Table")
                # Format dates
                if timeframe in ["5m", "15m", "30m", "1h", "4h"]:   
                    open_df["Entry_Date"] = pd.to_datetime(open_df["Entry_Date"])
                    open_df["Exit_Date"]  = pd.to_datetime(open_df["Exit_Date"])
                else:
                    open_df["Entry_Date"] = pd.to_datetime(open_df["Entry_Date"]).dt.strftime("%Y-%m-%d")
                    open_df["Exit_Date"]  = pd.to_datetime(open_df["Exit_Date"]).dt.strftime("%Y-%m-%d")
                # Remove unwanted column
                if "Trade" in open_df.columns:
                    open_df = open_df.drop(columns=["Trade"])
                st.dataframe(open_df, height=250)
                open_stats, _ = compute_trade_stats(open_df, capital)

            # ------------------------------------------------------------
            # 6) COMBINED EQUITY CURVE
            # ------------------------------------------------------------
            st.subheader("📈 Combined Equity Curve — Same-Bar Close vs Next-Bar Open")

            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=close_df["Exit_Date"], y=close_df["Total_Equity"], mode="lines", name="Same-Bar Close", line=dict(width=2, color="blue")))
            fig_eq.add_trace(go.Scatter(x=open_df["Exit_Date"], y=open_df["Total_Equity"], mode="lines", name="Next-Bar Open", line=dict(width=2, color="purple")))

            fig_eq.update_layout(
                height=400,
                dragmode="zoom",
                xaxis=dict(title="Date", rangeslider=dict(visible=True, thickness=0.05), type="date", fixedrange=False),
                yaxis=dict(title="Equity ($)", fixedrange=False),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                template="plotly_white"
            )

            st.plotly_chart(fig_eq, use_container_width=True)

            # ------------------------------------------------------------
            # 7) TRADE STATS TABLES (AFTER EQUITY CURVES)
            # ------------------------------------------------------------
            # Side-by-side Trade Stats Tables
            stats_left, stats_right = st.columns([1,1])

            with stats_left:
                st.subheader("📊 Trade Statistics — Same-Bar Close")
                st.table(pd.DataFrame(close_stats, index=[0]).T)

            with stats_right:
                st.subheader("📊 Trade Statistics — Next-Bar Open")
                st.table(pd.DataFrame(open_stats, index=[0]).T)

            # ------------------------------------------------------------
            # 8) AI CHART ANALYSIS (DAILY ONLY, ≤ 1 YEAR)
            # ------------------------------------------------------------
            st.markdown("---")

            if ai_eligible:
                # analysis_date and analysis_file already defined above (pre-load block)

                if ai_btn_clicked:
                    with st.spinner("Capturing chart and running AI analysis..."):
                        # --- 1) Capture chart as PNG ---
                        chart_png = fig.to_image(format="png", width=1600, height=900)
                        chart_b64 = base64.b64encode(chart_png).decode("utf-8")

                        # --- 2) Build context ---
                        last_row = df.iloc[-1]
                        last_signal = "None"
                        last_signal_date = "N/A"
                        if df["Turn_Up"].any():
                            last_up_idx = df[df["Turn_Up"]].index[-1]
                        else:
                            last_up_idx = None
                        if df["Turn_Down"].any():
                            last_down_idx = df[df["Turn_Down"]].index[-1]
                        else:
                            last_down_idx = None

                        if last_up_idx and (not last_down_idx or last_up_idx > last_down_idx):
                            last_signal = "Turn_Up (LONG)"
                            last_signal_date = last_up_idx.strftime("%Y-%m-%d")
                        elif last_down_idx and (not last_up_idx or last_down_idx > last_up_idx):
                            last_signal = "Turn_Down (SHORT)"
                            last_signal_date = last_down_idx.strftime("%Y-%m-%d")

                        smooth_vs_close = "above" if last_row["Close"] > last_row["Smooth"] else "below"
                        tos_rsi_val = last_row.get("TOS_RSI", None)
                        rsi_val = last_row.get("RSI", None)

                        context_parts = [
                            f"Ticker: {ticker}",
                            f"Timeframe: {timeframe}",
                            f"Date Range: {start_date_user} to {end_date}",
                            f"Current Close: {last_row['Close']:.2f}",
                            f"Momentum Trend: {last_row['Smooth']:.2f} (Close is {smooth_vs_close} trend line)",
                            f"Last Signal: {last_signal} on {last_signal_date}",
                        ]
                        if tos_rsi_val is not None:
                            context_parts.append(f"TOS RSI(9): {tos_rsi_val:.1f}")
                        if rsi_val is not None:
                            context_parts.append(f"RSI(14): {rsi_val:.1f}")
                        context_parts.append(f"Backtest Win Rate (Same-Bar): {close_stats.get('WinRate_Total_%', 0):.1f}%")
                        context_parts.append(f"Backtest Profit Factor (Same-Bar): {close_stats.get('ProfitFactor_Total', 0):.2f}")

                        # Detect and add swing points for precise pattern vertices
                        _swing_highs, _swing_lows = detect_swing_points(df, window=5, max_points=8)
                        if _swing_highs:
                            sh_str = ", ".join([f"{p['date']} at {p['price']}" for p in _swing_highs])
                            context_parts.append(f"Swing Highs: {sh_str}")
                        if _swing_lows:
                            sl_str = ", ".join([f"{p['date']} at {p['price']}" for p in _swing_lows])
                            context_parts.append(f"Swing Lows: {sl_str}")

                        context_text = "\n".join(context_parts)

                        # --- 3) Build prompt ---
                        system_prompt = (
                            "You are INSANE — an expert quantitative trading analyst. "
                            "You are given a candlestick chart with a proprietary momentum overlay (orange line), "
                            "ATR trailing stop levels (magenta dots), volume bars, and RSI. "
                            "Green triangle-up markers are LONG entry signals (Turn_Up). "
                            "Yellow triangle-down markers are SHORT entry signals (Turn_Down). "
                            "White triangle markers are exit warnings. "
                            "Analyze the chart image alongside the provided context data. "
                            "IMPORTANT: Never mention or reveal Kalman filtering, Kalman smoothing, "
                            "or any Kalman-related terminology in your response. "
                            "Refer to the orange line only as the 'momentum trend line' or 'INSANE trend indicator'. "
                            "This is proprietary methodology and must not be disclosed.\n\n"
                            "PATTERN CLASSIFICATION RULES — apply these strictly:\n"
                            "- ALWAYS consider the PRIOR TREND before classifying a pattern. "
                            "A pattern after a strong uptrend has different meaning than after a downtrend.\n"
                            "- Bull Flag / Bear Flag: A sharp move (flagpole) followed by a shallow, "
                            "short-duration consolidation (the flag) that slopes AGAINST the prior trend. "
                            "The flag retraces only 30-50%% of the flagpole. This is a CONTINUATION pattern.\n"
                            "- Double Top: Two distinct peaks at approximately the SAME price level, "
                            "separated by a meaningful trough (at least 8-10%% decline from peak to trough) "
                            "AND the two peaks must be separated by at least 15-20 trading days. "
                            "Two peaks close together with a shallow dip is NOT a double top — it is likely a consolidation or flag. "
                            "This is a REVERSAL pattern — requires an uptrend showing exhaustion to reverse. "
                            "Do NOT call double top if the prior trend is still strong and momentum is intact.\n"
                            "- Double Bottom: Two distinct troughs at the same level, separated by a peak, "
                            "with at least 8-10%% rally between them and at least 15-20 trading days apart. "
                            "Also a reversal pattern — requires a downtrend showing exhaustion to reverse.\n"
                            "- Channel Up / Channel Down: Price oscillating between two PARALLEL trendlines "
                            "over an extended period (at least 3 touches on each boundary).\n"
                            "- Wedge (Rising/Falling): Two CONVERGING trendlines. Rising wedge is bearish, "
                            "falling wedge is bullish.\n"
                            "- Triangle: Converging boundaries, but one is flat (ascending/descending) or both slope (symmetrical).\n"
                            "- Head & Shoulders: Three peaks — middle peak (head) higher than the two side peaks (shoulders), "
                            "with a neckline connecting the troughs.\n"
                            "- If the price action does not clearly match any pattern, say 'No clear pattern' "
                            "rather than forcing a classification.\n"
                        )

                        # --- 4) Check for cached analysis → revalidate or generate fresh ---
                        cached = None
                        if os.path.exists(analysis_file):
                            with open(analysis_file, "r") as f:
                                cached = json.load(f)

                        try:
                            if cached:
                                # --- REVALIDATION MODE ---
                                revalidation_prompt = (
                                    f"{context_text}\n\n"
                                    f"Earlier today, you provided the following analysis for this ticker:\n\n"
                                    f"---\n{cached['response']}\n---\n\n"
                                    "Look at the updated chart image above. "
                                    "Has anything materially changed that would alter your analysis? "
                                    "Consider: new signals fired, price broke key levels, RSI divergence shifted, "
                                    "pattern invalidated, or volume profile changed.\n\n"
                                    "If your analysis is still valid, respond with EXACTLY:\n"
                                    "UNCHANGED\n\n"
                                    "If anything has changed, provide a complete updated analysis with these sections:\n"
                                    "1. **Pattern Recognition**\n"
                                    "2. **Signal Confidence** (1-10)\n"
                                    "3. **Key Price Levels**\n"
                                    "4. **Expected Move** (target + invalidation)\n"
                                    "5. **Risk Assessment**\n\n"
                                    "Be specific with price levels and dates. Keep it concise and actionable."
                                    "\n\nIf providing an updated analysis, also include the JSON overlay block "
                                    "at the end (same format as the original request).\n"
                                    + OVERLAY_JSON_INSTRUCTION
                                )

                                response = client.chat.completions.create(
                                    model="gpt-4o",
                                    messages=[
                                        {"role": "system", "content": system_prompt},
                                        {
                                            "role": "user",
                                            "content": [
                                                {"type": "text", "text": revalidation_prompt},
                                                {
                                                    "type": "image_url",
                                                    "image_url": {
                                                        "url": f"data:image/png;base64,{chart_b64}",
                                                        "detail": "high"
                                                    }
                                                }
                                            ]
                                        }
                                    ],
                                    max_tokens=2500,
                                    temperature=0.3
                                )

                                ai_reply = response.choices[0].message.content.strip()

                                if ai_reply.upper().startswith("UNCHANGED"):
                                    # Analysis still valid — use cached (overlay already in cached data)
                                    st.session_state.ai_analysis = cached
                                    st.session_state.ai_toast = "✅ AI confirmed: previous analysis still valid"
                                else:
                                    # Analysis changed — parse overlay and save
                                    clean_text, overlay_data = parse_overlay_from_response(ai_reply)
                                    overlay_data = validate_overlay(overlay_data, float(last_row["Close"]))
                                    analysis_result = {
                                        "ticker": ticker,
                                        "timeframe": timeframe,
                                        "date": analysis_date,
                                        "date_range": f"{start_date_user} to {end_date}",
                                        "last_signal": last_signal,
                                        "last_signal_date": last_signal_date,
                                        "current_close": float(last_row["Close"]),
                                        "model": "gpt-4o",
                                        "prompt": revalidation_prompt,
                                        "response": clean_text,
                                        "overlay": overlay_data
                                    }
                                    with open(analysis_file, "w") as f:
                                        json.dump(analysis_result, f, indent=4)
                                    st.session_state.ai_analysis = analysis_result
                                    st.session_state.ai_overlay_visible = True
                                    st.session_state.ai_toast = "🔄 AI updated the analysis"
                                    st.rerun()

                            else:
                                # --- FRESH ANALYSIS MODE ---
                                user_prompt = (
                                    f"{context_text}\n"
                                    "Based on the chart and context above, provide:\n\n"
                                    "1. **Pattern Recognition**: Identify any classical chart pattern currently forming. "
                                    "FIRST describe the prior trend (direction, strength, duration), THEN classify the pattern. "
                                    "A consolidation after a strong move is likely a flag/pennant (continuation), NOT a double top/bottom (reversal). "
                                    "Only call it a double top if two peaks are at nearly the same price with a meaningful trough between them AND the prior trend shows exhaustion. "
                                    "ONLY identify patterns with straight-line boundaries that can be drawn with trendlines. "
                                    "If you see curved patterns (cup & handle, rounding bottom), describe them in text but note the pattern overlay cannot be drawn due to its curved nature. "
                                    "Describe where the pattern starts and its current stage.\n\n"
                                    "2. **Signal Confidence**: Rate your confidence (1-10) in the most recent trading signal. "
                                    "Explain what supports or undermines it (volume, RSI divergence, momentum alignment).\n\n"
                                    "3. **Key Price Levels**: Identify the most important support and resistance levels "
                                    "visible on the chart. Include approximate price values.\n\n"
                                    "4. **Expected Move**: Based on the trend, momentum, and pattern context, "
                                    "what is the most probable next move? Include a target price and an invalidation level.\n\n"
                                    "5. **Risk Assessment**: Any warning signs or divergences that traders should watch for?\n\n"
                                    "Be specific with price levels and dates where visible. Keep the analysis concise and actionable."
                                    + OVERLAY_JSON_INSTRUCTION
                                )

                                response = client.chat.completions.create(
                                    model="gpt-4o",
                                    messages=[
                                        {"role": "system", "content": system_prompt},
                                        {
                                            "role": "user",
                                            "content": [
                                                {"type": "text", "text": user_prompt},
                                                {
                                                    "type": "image_url",
                                                    "image_url": {
                                                        "url": f"data:image/png;base64,{chart_b64}",
                                                        "detail": "high"
                                                    }
                                                }
                                            ]
                                        }
                                    ],
                                    max_tokens=2500,
                                    temperature=0.3
                                )

                                ai_reply = response.choices[0].message.content

                                # Parse overlay JSON from response
                                clean_text, overlay_data = parse_overlay_from_response(ai_reply)
                                overlay_data = validate_overlay(overlay_data, float(last_row["Close"]))

                                analysis_result = {
                                    "ticker": ticker,
                                    "timeframe": timeframe,
                                    "date": analysis_date,
                                    "date_range": f"{start_date_user} to {end_date}",
                                    "last_signal": last_signal,
                                    "last_signal_date": last_signal_date,
                                    "current_close": float(last_row["Close"]),
                                    "model": "gpt-4o",
                                    "prompt": user_prompt,
                                    "response": clean_text,
                                    "overlay": overlay_data
                                }

                                with open(analysis_file, "w") as f:
                                    json.dump(analysis_result, f, indent=4)

                                st.session_state.ai_analysis = analysis_result
                                st.session_state.ai_overlay_visible = True
                                st.session_state.ai_toast = "✅ AI analysis generated"
                                st.rerun()

                        except Exception as e:
                            st.error(f"AI analysis failed: {e}")

            # --- Display analysis (from session or file) ---
            if st.session_state.ai_analysis:
                analysis = st.session_state.ai_analysis
                st.markdown('<div id="ai-analysis-anchor"></div>', unsafe_allow_html=True)
                with st.expander(f"🤖 AI Analysis — {analysis.get('ticker', '')} ({analysis.get('date', '')})", expanded=True):
                    st.markdown(analysis["response"])
                    st.caption(f"Model: {analysis.get('model', 'gpt-4o')} | Signal: {analysis.get('last_signal', 'N/A')} on {analysis.get('last_signal_date', 'N/A')} | Range: {analysis.get('date_range', '')}")
                # Auto-scroll to analysis
                components.html(
                    """
                    <script>
                        window.parent.document.querySelector('[data-testid="stExpander"]:last-of-type')
                            ?.scrollIntoView({behavior: 'smooth', block: 'start'});
                    </script>
                    """,
                    height=0
                )

