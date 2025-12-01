import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import ta

def encode_label(val):
    if val == 1:
        return "LONG"
    elif val == -1:
        return "SHORT"
    return "EXIT"

def label_simple_return(df, forward=5, thresh=0.02):
    future_ret = df["Close"].shift(-forward) / df["Close"] - 1
    
    cond_long  = future_ret >  thresh
    cond_short = future_ret < -thresh

    labels = np.where(cond_long, 1,
               np.where(cond_short, -1, 0))

    return pd.Series(labels, index=df.index)


def label_atr(df, forward=5, atr_mult=1.5):
    # ATR already exists in your dataset
    atr_pct = df["ATR"] / df["Close"]

    future_ret = df["Close"].shift(-forward) / df["Close"] - 1

    cond_long  = future_ret >  atr_mult * atr_pct
    cond_short = future_ret < -atr_mult * atr_pct

    labels = np.where(cond_long, 1,
               np.where(cond_short, -1, 0))

    return pd.Series(labels, index=df.index)


def label_trend(df):
    sma_fast = df["SMA_Short"]
    sma_slow = df["SMA_Mid"]

    frama_slope = df["FRAMA"].diff()

    cond_long  = (sma_fast > sma_slow) & (frama_slope > 0)
    cond_short = (sma_fast < sma_slow) & (frama_slope < 0)

    labels = np.where(cond_long, 1,
               np.where(cond_short, -1, 0))

    return pd.Series(labels, index=df.index)


def label_regime(df):
    # Trend regime
    uptrend   = df["SMA_Short"] > df["SMA_Mid"]
    downtrend = df["SMA_Short"] < df["SMA_Mid"]

    # Volatility regime
    vol_expanding = df["BB_Width"] > df["BB_Width"].rolling(20).mean()

    # Momentum confirmation
    mom_pos = df["RSI"] > 55
    mom_neg = df["RSI"] < 45

    # Define
    long_regime  = uptrend & mom_pos & (~vol_expanding)
    short_regime = downtrend & mom_neg & (~vol_expanding)

    labels = np.where(long_regime, 1,
               np.where(short_regime, -1, 0))

    return pd.Series(labels, index=df.index)


def add_all_labels(df):
    out = df.copy()

    out["Label_Return"] = label_simple_return(df)
    out["Label_ATR"]    = label_atr(df)
    out["Label_Trend"]  = label_trend(df)
    out["Label_Regime"] = label_regime(df)

    # Convert 1/-1/0 → LONG/SHORT/EXIT
    for col in ["Label_Return", "Label_ATR", "Label_Trend", "Label_Regime"]:
        out[col] = out[col].apply(
            lambda x: "LONG" if x == 1 else ("SHORT" if x == -1 else "EXIT")
        )

    return out


