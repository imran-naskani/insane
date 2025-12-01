import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import plotly.graph_objects as go

from build_dataset import build_feature_dataset
from technical_features import add_all_labels
from backtest_same_day_close import generate_trade_log, compute_trade_stats
from backtest_next_day_open import generate_trade_log_next_open
from model import kalman_basic   # signal logic comes from model.py

import numpy as np
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="INSANE Trading Model", layout="wide")

st.title("📈 INSANE — Statistical Analysis for Navigating Equities")

ticker = st.text_input("Enter Ticker Symbol", value="NVDA")
# Default Start = 5 years ago
default_start = dt.date.today() - dt.timedelta(days=365*5)

# User inputs
start_date = st.date_input("Start Date", value=default_start)
end_date = st.date_input("End Date (optional)", value=dt.date.today())

capital = st.number_input(
    "Initial Trading Capital ($)", 
    min_value=1000, max_value=1000000, value=10000, step=500
)


run = st.button("Run Model")

if run:

    with st.spinner("Fetching data and running model..."):

        # ------------------------------------------------------------------
        # 1. BUILD FEATURE SET
        # ------------------------------------------------------------------
        data = build_feature_dataset(
            ticker,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
        data = data.dropna().copy()

        # Add labels (not used in backtest but good for display)
        main_df = add_all_labels(data)
        df = main_df.copy()

        # ------------------------------------------------------------------
        # 2. KALMAN SIGNAL COMPUTATION (your original logic)
        # ------------------------------------------------------------------
        df["Smooth"], df["Slope"] = kalman_basic(df["Close"])
        df["price_delta"] = df["Close"] - df["Smooth"]

        slope_quantiles = df["Slope"].quantile([0.05, 0.35, 0.5, 0.65, 0.95])
        slope_quantiles = slope_quantiles.values.tolist()
        slope_values = [round(x / 0.25) * 0.25 for x in slope_quantiles]

        price_quantiles = df["price_delta"].quantile([0.05, 0.35, 0.5, 0.65, 0.95])
        price_quantiles = price_quantiles.values.tolist()
        price_delta_values = [round(x / 0.25) * 0.25 for x in price_quantiles]

        # STRONG SHORT
        df["Slope_Neg"] = (
            (df["Slope"] < slope_values[0]) &
            (df["Close"] < df["Smooth"]) &
            (df["Slope"] < df["Slope"].shift(1))
        )

        # STRONG LONG
        df["Slope_Pos"] = (
            (df["Slope"] > slope_values[3]) &
            (df["Close"] > df["Smooth"]) &
            (df["Slope"] > df["Slope"].shift(1))
        )

        # FINAL SIGNALS
        df["Turn_Up"] = df["Slope_Pos"] & (~df["Slope_Pos"].shift(1).fillna(False))
        df["Turn_Down"] = df["Slope_Neg"] & (~df["Slope_Neg"].shift(1).fillna(False))

        # ------------------------------------------------------------------
        # 3. ENSURE SIGNAL COLUMNS EXIST (safety)
        # ------------------------------------------------------------------
        if "Turn_Up" not in df.columns:
            df["Turn_Up"] = False
        if "Turn_Down" not in df.columns:
            df["Turn_Down"] = False

        # ------------------------------------------------------------------
        # 4. PRICE / MOMENTUM PLOT
        # ------------------------------------------------------------------
        st.subheader("📌 Price vs Momentum Trend (Interactive)")

        fig = go.Figure()

        # Price
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name="Price",
            line=dict(width=1.2)
        ))

        # Momentum (Smooth)
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["Smooth"],
            mode="lines",
            name="Momentum (Smooth)",
            line=dict(width=2, color="orange")
        ))

        # Turn Up signals
        fig.add_trace(go.Scatter(
            x=df.index[df["Turn_Up"]],
            y=df["Smooth"][df["Turn_Up"]],
            mode="markers",
            name="Turn Up",
            marker=dict(color="green", size=10, symbol="triangle-up")
        ))

        # Turn Down signals
        fig.add_trace(go.Scatter(
            x=df.index[df["Turn_Down"]],
            y=df["Smooth"][df["Turn_Down"]],
            mode="markers",
            name="Turn Down",
            marker=dict(color="red", size=10, symbol="triangle-down")
        ))

        fig.update_layout(
            height=500,
            dragmode="zoom",            # <-- Enables drag-zoom anywhere
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="date",
                fixedrange=False        # <-- Allow X-axis zoom/pan
            ),
            yaxis=dict(
                fixedrange=False        # <-- FULLY ENABLE Y-AXIS SCALING
            ),
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

        # ------------------------------------------------------------------
        # 5. SAME-DAY CLOSE BACKTEST
        # ------------------------------------------------------------------
        st.subheader("📊 Same-Day Close Backtest")
        close_trade_df, close_final_value = generate_trade_log(df, capital)

        st.write(f"**Final portfolio value:** ${close_final_value:,.2f}")
        st.dataframe(close_trade_df)

        close_stats, _ = compute_trade_stats(close_trade_df, capital)
        st.table(pd.DataFrame(close_stats, index=[0]).T)

        # ------------------------------------------------------------------
        # 6. NEXT-DAY OPEN BACKTEST
        # ------------------------------------------------------------------
        st.subheader("📊 Next-Day Open Backtest")
        open_trade_df, open_final_value = generate_trade_log_next_open(df, capital)

        st.write(f"**Final portfolio value:** ${open_final_value:,.2f}")
        st.dataframe(open_trade_df)

        open_stats, _ = compute_trade_stats(open_trade_df, capital)
        st.table(pd.DataFrame(open_stats, index=[0]).T)

        # ------------------------------------------------------------------
        # 7. EQUITY CURVES
        # ------------------------------------------------------------------
        st.subheader("📈 Combined Equity Curve — Same-Day Close vs Next-Day Open (Interactive)")
        fig_eq = go.Figure()

        # Same-Day Close curve
        fig_eq.add_trace(go.Scatter(
            x=close_trade_df["Exit_Date"],
            y=close_trade_df["Total_Equity"],
            mode="lines",
            name="Same-Day Close",
            line=dict(width=2, color="blue")
        ))

        # Next-Day Open curve
        fig_eq.add_trace(go.Scatter(
            x=open_trade_df["Exit_Date"],
            y=open_trade_df["Total_Equity"],
            mode="lines",
            name="Next-Day Open",
            line=dict(width=2, color="purple")
        ))

        # Layout — make it fully zoomable in both axes
        fig_eq.update_layout(
            height=500,
            dragmode="zoom",
            xaxis=dict(
                title="Date",
                rangeslider=dict(visible=True),
                type="date",
                fixedrange=False
            ),
            yaxis=dict(
                title="Equity ($)",
                fixedrange=False   # <---- THIS ENABLES Y-AXIS STRETCH
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            template="plotly_white"
        )

        st.plotly_chart(fig_eq, use_container_width=True)

