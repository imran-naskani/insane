import streamlit as st
import pandas as pd
import datetime as dt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")
import streamlit.components.v1 as components
import numpy as np
import datetime
from build_dataset import build_feature_dataset
from technical_features import add_all_labels
from backtest_same_day_close import generate_trade_log, compute_trade_stats
from backtest_next_day_open import generate_trade_log_next_open
from backtest_intraday_same_bar_close import backtest_intraday_close
from backtest_intraday_next_bar_open import backtest_intraday_next_open
from model import secret_sauce, spicy_sauce
from openai import OpenAI
import json
import glob
from streamlit_autorefresh import st_autorefresh
from daily_engine import load_ticker_history, snapshot_all_signals_first_time
import os



client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
# ----------------------------------------------------
# Application state
# ----------------------------------------------------
if "run_model" not in st.session_state:
    st.session_state.run_model = False
# # Chat state
# if "chat_open" not in st.session_state:
#     st.session_state.chat_open = False

# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []


st.set_page_config(page_title="INSANE Trading Model", layout="wide")

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
st.markdown("<h1 id='MainTitle'>📈 INSANE — Intelligent Statistical Algorithm for Navigating Equities</h1>", unsafe_allow_html=True)

# ----------------------------------------------------
# LAYOUT COLUMNS
# ----------------------------------------------------
filters_col, main_col, chat_col = st.columns([1, 4, 1])

# ----------------------------------------------------
# CHAT COLUMN (RIGHT SIDE)
# ----------------------------------------------------
with chat_col:


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


    st.markdown("<div class='signal-header'>📅 Daily Market Signals (@ 3PM CST)</div>", unsafe_allow_html=True)

    try:
        # with open("daily_signals.json", "r") as f:
        #     signals = json.load(f)
        
        # --------------------------------------------------
        # LOAD DAILY CLOSES (LATEST DAY)
        # --------------------------------------------------
        with open("daily_closes.json", "r") as f:
            daily_data = json.load(f)
        today_date = daily_data["date"]
        today_closes = daily_data["closes"]
        st.write(f"**Last Update:** {today_date}")

        # --------------------------------------------------
        # LOAD SIGNAL HISTORY FILES
        # --------------------------------------------------
        signal_files = glob.glob("signal_history/*.json")

        long_rows = []
        short_rows = []

        index_signals = {}

        INDEX_TICKERS = {
            "SPY": "SPY",
            "QQQ": "QQQ",
            "^GSPC": "S&P 500 (SPX)",
            "^IXIC": "NASDAQ Composite",
            "^RUT": "Russell 2000",
            "^VIX": "VIX"
        }

        for file in signal_files:
            ticker = os.path.basename(file).replace(".json", "")

            with open(file, "r") as f:
                history = json.load(f)

            # Ensure history is a non-empty list
            if not isinstance(history, list) or len(history) == 0:
                continue

            last = history[-1]   # last flip only

            signal = last["signal"]
            signal_date = last["date"]
            signal_close = last["close"]
            today_close = today_closes.get(ticker)

            # INDEX HANDLING
            if ticker in INDEX_TICKERS:
                index_signals[ticker] = {
                    "signal": "LONG" if signal == "UP" else "SHORT",
                    "date": signal_date
                }
                continue

            # STOCK SIGNALS
            delta = None
            delta_pct = None

            if today_close is not None and signal_close != 0:
                if signal == "UP":        # LONG
                    delta = today_close - signal_close
                elif signal == "DOWN":    # SHORT
                    delta = signal_close - today_close

                delta = round(delta, 2)
                delta_pct = round((delta / signal_close) * 100, 2)

            row = {
                "Ticker": ticker,
                "Signal Date": signal_date,
                "Signal Close": round(signal_close, 2),
                "Today Close": round(today_close, 2) if today_close else None,
                "Delta": delta,
                "Delta %": delta_pct
            }

            if signal == "UP":
                long_rows.append(row)
            elif signal == "DOWN":
                short_rows.append(row)

        # --------------------------------------------------
        # DISPLAY LONG SIGNALS
        # --------------------------------------------------
        st.markdown("#### 🟢 Long Signals")

        if long_rows:
            st.dataframe(
                pd.DataFrame(long_rows).sort_values("Signal Date", ascending=False),
                use_container_width=True,
                height=300,
                hide_index=True
            )
        else:
            st.write("No Long signals.")

        # --------------------------------------------------
        # DISPLAY SHORT SIGNALS
        # --------------------------------------------------
        st.markdown("#### 🔴 Short Signals")

        if short_rows:
            st.dataframe(
                pd.DataFrame(short_rows).sort_values("Signal Date", ascending=False),
                use_container_width=True,
                height=300,
                hide_index=True
            )
        else:
            st.write("No Short signals.")

        # --------------------------------------------------
        # INDEX SIGNALS
        # --------------------------------------------------
        st.markdown("### 📈 Index Trend Signals")

        def show_index(name, obj):
            signal = obj["signal"]
            date = obj["date"]

            color = (
                "#4CAF50" if signal == "LONG"
                else "#F44336" if signal == "SHORT"
                else "#B0BEC5"
            )

            st.markdown(
                f"""
                <div class="index-card">
                    <div class="index-title">{name}</div>
                    <div class="index-signal" style="color:{color};">{signal}</div>
                    <div class="index-date">Last signal: {date}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        for tkr, name in INDEX_TICKERS.items():
            if tkr in index_signals:
                show_index(name, index_signals[tkr])

    except Exception as e:
        st.warning(f"Signal data not available: {e}")

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
        extended_start = start_date_user - dt.timedelta(days=30)
    elif timeframe == '1d': 
        extended_start = start_date_user - dt.timedelta(days=290)
    
    # if timeframe  == "1d":
    #     end_date = st.date_input("End Date", value=dt.date.today())
    # else:
    #     end_date = st.date_input("End Date", value=dt.date.today() + dt.timedelta(days=1))
    end_date = st.date_input("End Date", value=dt.date.today() + dt.timedelta(days=1))
    capital = st.number_input("Capital ($)", 1000, 1_000_000, 10000, 500)
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
            df = df[df.index >= pd.to_datetime(start_date_user).tz_localize(df.index.tz)]

            # ------------------------------------------------------------
            # 2) COMPUTE KALMAN SIGNALS
            # ------------------------------------------------------------
            price = df['Close']
            
            if timeframe in ["5m", "15m", "30m", "1h", "4h"]:
                # price = (df['High'] + df['Low']) / 2
                df["Smooth"], df["Slope"] = spicy_sauce(df["Close"])
                df['price_delta'] = df["Close"] - df["Smooth"]
                price_delta = df['price_delta']
                df["q05"] = df["price_delta"].rolling(84).quantile(0.05) # 0.05 quantile
                df["q35"] = df["price_delta"].rolling(84).quantile(0.25)
                df["q50"] = df["price_delta"].rolling(84).quantile(0.50)
                df["q65"] = df["price_delta"].rolling(84).quantile(0.75)
                df["q95"] = df["price_delta"].rolling(84).quantile(0.95) # 0.95 quantile
                df["Slope_Neg"] = (df["price_delta"] < df["q05"]) & (df["Close"] < df["TOS_Trail"])
                df["Slope_Pos"] = (df["price_delta"] > df["q95"]) & (df["Close"] > df["TOS_Trail"])
                df["Turn_Up"]   = df["Slope_Pos"] & (~df["Slope_Pos"].shift(1).fillna(False))
                df["Turn_Down"] = df["Slope_Neg"] & (~df["Slope_Neg"].shift(1).fillna(False))
                            
            
            else:                                    
                df["Smooth"], df["Slope"] = secret_sauce(price)
                df["price_delta"] = df["Close"] - df["Smooth"]

                # slope_q = df["Slope"].quantile([0.05, 0.35, 0.5, 0.65, 0.95]).tolist()
                # slope_vals = [round(x / 0.25) * 0.25 for x in slope_q]
                # print(slope_vals)

                # df["Slope_Neg"] = (
                #     (df["Slope"] < (slope_vals[0] + slope_vals[1])/2) & # ) / 2
                #     (df["Close"] < df["Smooth"]) &
                #     (df["Slope"] < df["Slope"].shift(1))
                # )

                # df["Slope_Pos"] = (
                #     (df["Slope"] > (slope_vals[3] + slope_vals[4]) / 2) &
                #     (df["Close"] > df["Smooth"]) &
                #     (df["Slope"] > df["Slope"].shift(1))
                # )
            
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
                    df["Turn_Up"]   = df["Slope_Pos"] & (~df["Slope_Pos"].shift(1).fillna(False))
                    df["Turn_Down"] = df["Slope_Neg"] & (~df["Slope_Neg"].shift(1).fillna(False))
                    snapshot_all_signals_first_time(ticker, df)
                    frozen = load_ticker_history(ticker)
                
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
                        ((df["High"].shift(1) >= df["VWAP_Upper"].shift(1)) &
                        (df["Close"] < df["VWAP_Upper"])) | 
                        ((df["Close"].shift(1) >= df["VWAP"].shift(1)) &
                        (df["Close"] < df["VWAP"])) |
                        ((df["Low"].shift(1) >= df["TOS_Trail"].shift(1)) &
                        (df["Low"] < df["TOS_Trail"])) 
                    )
                )

                df["Sell_Short"] = (
                    (df["Position"] == -1) &
                    (
                        ((df["Low"].shift(1) <= df["VWAP_Lower"].shift(1)) &
                        (df["Close"] > df["VWAP_Lower"])) |
                        ((df["Close"].shift(1) <= df["VWAP"].shift(1)) &
                        (df["Close"] > df["VWAP"])) | 
                        ((df["High"].shift(1) <= df["TOS_Trail"].shift(1)) &
                        (df["High"] > df["TOS_Trail"]))
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


            
            print(df[['Close','Turn_Up','Turn_Down','Sell_Long','Sell_Short','Sell_Long_Plot','Sell_Short_Plot','Position']].head(10))
            # print(default_start, start_date_user, extended_start)
            # ------------------------------------------------------------
            # 3) PRICE vs MOMENTUM — INTERACTIVE PLOT
            # ------------------------------------------------------------
            st.subheader("📌 Price vs Momentum Trend")

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

            st.plotly_chart(fig, use_container_width=True)

            # ------------------------------------------------------------
            # BACKTEST TABLES SIDE BY SIDE
            # ------------------------------------------------------------
            is_intraday = timeframe in ["5m", "15m", "30m", "1h", "4h"]
            bt_left, bt_right = st.columns([1, 1])

            with bt_left:
                # ---- Net Profit for Same-Day Close ----
                if is_intraday:
                    close_df, close_equity_end = backtest_intraday_close(df, capital)
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
                    open_df, open_equity_end = backtest_intraday_next_open(df, capital)
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
                
            
