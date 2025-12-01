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
from model import kalman_basic
from openai import OpenAI
import json
    
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
# ----------------------------------------------------
# Application state
# ----------------------------------------------------
if "run_model" not in st.session_state:
    st.session_state.run_model = False
# Chat state
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


st.set_page_config(page_title="INSANE Trading Model", layout="wide")

# ----------------------------------------------------
# TITLE
# ----------------------------------------------------
st.markdown("<h1 id='MainTitle'>📈 INSANE — Imran Naskani's Statistical Algorithm for Navigating Equities</h1>", unsafe_allow_html=True)

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


    import json
    try:
        with open("daily_signals.json", "r") as f:
            signals = json.load(f)

        st.write(f"**Last Update:** {signals['date']}")

        # Today's Stock Signals
        st.markdown("#### 🟢 Long Signals")
        if signals["long"]:
            # st.table(pd.DataFrame(signals["long"], columns=["Ticker"]))
            st.dataframe(
                pd.DataFrame(signals["long"], columns=["Ticker"]),
                height=200,
                use_container_width=True
            )
        else:
            st.write("No Long signals today.")

        st.markdown("#### 🔴 Short Signals")
        if signals["short"]:
            # st.table(pd.DataFrame(signals["short"], columns=["Ticker"]))
            st.dataframe(
                pd.DataFrame(signals["short"], columns=["Ticker"]),
                height=200,
                use_container_width=True
            )
        else:
            st.write("No Short signals today.")

        # INDEX SIGNALS
        st.markdown("### 📈 Index Trend Signals")

        def show_index(name, obj):
            signal = obj["signal"]
            date = obj["date"]
            color = (
                "green" if signal == "LONG"
                else "red" if signal == "SHORT"
                else "gray"
            )
            st.markdown(
                f"""
                <div style="padding:8px; border-radius:6px; background:#f5f5f5; margin-bottom:6px;">
                    <b>{name}:</b> 
                    <span style="color:{color}; font-weight:bold">{signal}</span>
                    <br><small>Last signal date: {date}</small>
                </div>
                """,
                unsafe_allow_html=True
            )

        show_index("SPY", signals["spy"])
        show_index("QQQ", signals["qqq"])
        show_index("S&P 500 (SPX)", signals["spx"])
        show_index("NASDAQ Composite", signals["nasdaq"])
        show_index("Russell 2000", signals["russell"])

    except Exception:
        st.warning("Daily signals not found. Run daily_engine.py first.")

    # -----------------------------------------------------------
    # CHATBOT (BELOW DAILY SIGNALS)
    # -----------------------------------------------------------
    st.markdown("### 🤖 INSANEgpt")

    # Show chat history
    for msg in st.session_state.chat_history:
        st.markdown(f"<div style='margin-bottom:8px;'>{msg}</div>", unsafe_allow_html=True)

    # User input
    user_msg = st.text_input("Message:", key="chat_input_field_right")

    if st.button("Send Message", key="chat_send_right"):
        if user_msg.strip():
            st.session_state.chat_history.append(f"Question: {user_msg}")

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are INSANE — Imran's trading assistant. Expert Financial Analyst specializing in stock trading strategies. Explain indicators, provide concise, accurate answers based on statistical analysis and trading principles. Do not answer anything outside of stock trading or finance."},
                    {"role": "user", "content": user_msg}
                ]
            )
            reply = response.choices[0].message.content

            st.session_state.chat_history.append(f"INSANEgpt: {reply}")

            # Clear input field
            st.session_state.pop("chat_input_field_right", None)
            st.rerun()


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
    timeframe_options = ["5m", "15m", "30m", "1h", "4h", "1d"]
    timeframe = st.selectbox("Timeframe", timeframe_options, index=5)   # default = "1d"
    
    if timeframe in ["5m", "15m", "30m", "1h", "4h"]:
        default_start = dt.date.today() - dt.timedelta(days=5)
    else:
        default_start = dt.date.today() - dt.timedelta(days=365*5)    
    start_date = st.date_input("Start Date", value=default_start)
    start_date_user = start_date

    if timeframe in ["5m", "15m", "30m", "1h", "4h"]:
        extended_start = start_date_user - dt.timedelta(days=49)
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

            # Add Labels
            df = add_all_labels(data)
            df = df[df.index >= pd.to_datetime(start_date_user).tz_localize(df.index.tz)]

            # Columns we do NOT consider as indicators
            exclude_cols = ["Open", "High", "Low", "Close", "Volume", 
                            "Smooth", "Slope", "price_delta", 
                            "Slope_Pos", "Slope_Neg", "Turn_Up", "Turn_Down"]

            indicator_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype != 'O']
            # print(indicator_cols)
            # print(df.tail(5))

            # ------------------------------------------------------------
            # 2) COMPUTE KALMAN SIGNALS
            # ------------------------------------------------------------
            df["Smooth"], df["Slope"] = kalman_basic(df["Close"])
            df["price_delta"] = df["Close"] - df["Smooth"]

            slope_q = df["Slope"].quantile([0.05, 0.35, 0.5, 0.65, 0.95]).tolist()
            slope_vals = [round(x / 0.25) * 0.25 for x in slope_q]

            # STRONGER SHORT LOGIC
            df["Slope_Neg"] = (
                (df["Slope"] < (slope_vals[0]+slope_vals[1])/2) &          # very negative slope
                (df["Close"] < df["Smooth"]) &             # breakdown
                (df["Slope"] < df["Slope"].shift(1))       # momentum still worsening
            )

            df["Slope_Pos"] = (
                (df["Slope"] > (slope_vals[3]+slope_vals[4])/2) &          # strong slope
                (df["Close"] > df["Smooth"]) &             # breakout
                (df["Slope"] > df["Slope"].shift(1))       # momentum improving
            )

            df["Turn_Up"] = df["Slope_Pos"] & (~df["Slope_Pos"].shift(1).fillna(False))
            df["Turn_Down"] = df["Slope_Neg"] & (~df["Slope_Neg"].shift(1).fillna(False))

            # Safety
            df["Turn_Up"] = df["Turn_Up"].fillna(False)
            df["Turn_Down"] = df["Turn_Down"].fillna(False)
            
            print(df[['Close', 'Smooth', 'Slope', 'price_delta', 'Turn_Up', 'Turn_Down']].tail(10))

            # ------------------------------------------------------------
            # 3) PRICE vs MOMENTUM — INTERACTIVE PLOT
            # ------------------------------------------------------------
            st.subheader("📌 Price vs Momentum Trend")

            if timeframe in ["5m", "15m", "30m", "1h", "4h"]: 
                df = df.iloc[:-1]
            # x = np.arange(len(df))

            selected_indicators = st.multiselect(
                "Select indicators to overlay:",
                indicator_cols,
                default=[]
            )

            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3],
                # specs=[
                #     [{"secondary_y": True}],   # Row 1: price + predictions + indicators
                #     [{"secondary_y": False}]   # Row 2: volume only
                # ],
            )

            # -------------------------------
            # 1) CANDLESTICK (using df OHLCV)
            # -------------------------------
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df["Open"],
                    high=df["High"],
                    low=df["Low"],
                    close=df["Close"],
                    name="Candles",
                    increasing_line_color="green",
                    decreasing_line_color="red",
                    increasing_fillcolor="rgba(0, 180, 0, 0.7)",
                    decreasing_fillcolor="rgba(220, 0, 0, 0.7)"
                ),
                row=1, col=1#, secondary_y=False
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
                row=1, col=1#, secondary_y=False
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

            # ---------------------------------------
            # Add Selected Indicators to the Chart
            # ---------------------------------------
            for ind in selected_indicators:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[ind],
                        mode="lines",
                        name=ind,
                        line=dict(width=2)
                    ),
                    row=1, col=1#, secondary_y=False
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
                height=650,
                showlegend=True,
                dragmode="zoom",
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    type="date"
                ),
                yaxis=dict(fixedrange=False),
                template="plotly_white",
                margin=dict(l=20, r=20, t=20, b=20),
            )

            if timeframe in ["5m", "15m", "30m", "1h", "4h"]:
                # 🚀 ADD RANGE BREAKS HERE (right after layout)
                fig.update_xaxes(
                    rangebreaks=[
                        dict(bounds=["sat", "mon"]),          # skip weekends
                        dict(bounds=[19, 24], pattern="hour"),  # skip overnight hours
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

            st.plotly_chart(fig, use_container_width=True)

            # ------------------------------------------------------------
            # BACKTEST TABLES SIDE BY SIDE
            # ------------------------------------------------------------
            bt_left, bt_right = st.columns([1, 1])

            with bt_left:
                # ---- Net Profit for Same-Day Close ----
                close_df, close_equity_end = generate_trade_log(df, capital)
                close_net_profit = close_equity_end - capital
                close_roi = (close_net_profit / capital) * 100


                st.markdown(
                    f"""
                    <div style='font-size:20px; font-weight:700; margin-bottom:5px;'>
                        💰 Net Profit (Same-Day Close): 
                        <span style='color:{"green" if close_net_profit > 0 else "red"};'>
                            {close_net_profit:,.2f} ({close_roi:,.2f}%)
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.subheader("📊 Same-Day Close Backtest Table")
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
                open_df, open_equity_end = generate_trade_log_next_open(df, capital)
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
                st.subheader("📊 Next-Day Open Backtest Table")
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
            st.subheader("📈 Combined Equity Curve — Same-Day Close vs Next-Day Open")

            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=close_df["Exit_Date"], y=close_df["Total_Equity"], mode="lines", name="Same-Day Close", line=dict(width=2, color="blue")))
            fig_eq.add_trace(go.Scatter(x=open_df["Exit_Date"], y=open_df["Total_Equity"], mode="lines", name="Next-Day Open", line=dict(width=2, color="purple")))

            fig_eq.update_layout(
                height=400,
                dragmode="zoom",
                xaxis=dict(title="Date", rangeslider=dict(visible=True), type="date", fixedrange=False),
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
                st.subheader("📊 Trade Statistics — Same-Day Close")
                st.table(pd.DataFrame(close_stats, index=[0]).T)

            with stats_right:
                st.subheader("📊 Trade Statistics — Next-Day Open")
                st.table(pd.DataFrame(open_stats, index=[0]).T)
                
            
