import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import ta


def floor_5_or_int(x):
    return int((x // 5) * 5) if x >= 5 else int(x // 1)

# ----------------------------------------------------
#  COMPUTE FRAMA
# ----------------------------------------------------
def compute_frama(series, window=16, fc=1, sc=200):
    """
    Compute FRAMA (Fractal Adaptive Moving Average)
    John Ehlers original algorithm.
    series : pandas Series of prices
    window : lookback length for fractal dimension (default 16)
    fc : fast constant (default 1)
    sc : slow constant (default 200)
    """
    close = series.values
    n = window

    frama = np.zeros(len(close))
    frama[:] = np.nan

    for i in range(n * 2, len(close)):
        # First half
        hl1 = max(close[i-n:i]) - min(close[i-n:i])
        # Second half
        hl2 = max(close[i-2*n:i-n]) - min(close[i-2*n:i-n])
        # Whole window
        hl = max(close[i-2*n:i]) - min(close[i-2*n:i])

        # Avoid division errors
        if hl1 == 0 or hl2 == 0 or hl == 0:
            continue

        # Fractal dimension
        dim = (np.log(hl1 + hl2) - np.log(hl)) / np.log(2)

        # Smoothing factor
        alpha = np.exp(-4.6 * (dim - 1))
        alpha = max(min(alpha, 1), 0.01)  # Bound between 0.01 and 1

        # FRAMA recursive formula
        if np.isnan(frama[i-1]):
            frama[i] = close[i]
        else:
            frama[i] = alpha * close[i] + (1 - alpha) * frama[i-1]

    return pd.Series(frama, index=series.index)


# ---------------------------------------------------------
# 1) TOS RSI With Divergence
# ---------------------------------------------------------
def compute_tos_rsi(df, n=14, over_bought=70, over_sold=30):
    """
    Exact ThinkOrSwim-style RSI:
    NetChgAvg = ExpAverage(c - c[1], n)
    TotChgAvg = ExpAverage(|c - c[1]|, n)
    RSI       = 50 * (NetChgAvg / TotChgAvg + 1)
    """
    c = df["Close"].astype(float)

    # price change
    delta = c.diff()

    # TOS ExpAverage = EMA with alpha = 2/(n+1)
    net_chg_avg = delta.ewm(span=n, adjust=False).mean()
    tot_chg_avg = delta.abs().ewm(span=n, adjust=False).mean()

    chg_ratio = np.where(tot_chg_avg != 0, net_chg_avg / tot_chg_avg, 0.0)
    rsi = 50 * (chg_ratio + 1)

    out = pd.DataFrame(index=df.index)
    out["TOS_RSI"] = rsi
    out["OverBought"] = over_bought
    out["OverSold"] = over_sold

    return out

# ---------------------------------------------------------
# 1) EXACT TOS MODIFIED ATR
# ---------------------------------------------------------
def tos_atr_modified(df, atr_period=10):
    high = df["High"].values.astype(float)
    low  = df["Low"].values.astype(float)
    close = df["Close"].values.astype(float)
    n = len(df)

    # ----- HiLo -----
    hl = high - low
    sma_hl = pd.Series(hl).rolling(atr_period).mean().values

    # TOS behavior: if SMA not ready, just use hl (no NaNs)
    hilo = np.where(np.isnan(sma_hl), hl, np.minimum(hl, 1.5 * sma_hl))

    # ----- HRef / LRef -----
    href = np.zeros(n)
    lref = np.zeros(n)

    for i in range(1, n):
        # HRef
        if low[i] <= high[i-1]:
            href[i] = high[i] - close[i-1]
        else:
            href[i] = (high[i] - close[i-1]) - 0.5 * (low[i] - high[i-1])

        # LRef
        if high[i] >= low[i-1]:
            lref[i] = close[i-1] - low[i]
        else:
            lref[i] = (close[i-1] - low[i]) - 0.5 * (low[i-1] - high[i])

    # ----- True Range -----
    # After the HiLo fix, none of these should be NaN
    tr = np.maximum(hilo, np.maximum(href, lref))

    # ----- Wilder-style ATR, TOS-style init -----
    atr = np.full(n, np.nan)
    if n > 1:
        atr[1] = tr[1]  # seed with first TR

    alpha = 1.0 / atr_period
    for i in range(2, n):
        atr[i] = atr[i-1] + alpha * (tr[i] - atr[i-1])

    return pd.Series(atr, index=df.index, name="TOS_ATR")

# ---------------------------------------------------------
# 2) FULL TOS TRAILING STOP (STATE MACHINE)
# ---------------------------------------------------------
def tos_trailing_stop(df, atr_period=10, atr_factor=1.5, first_trade="long"):
    atr = tos_atr_modified(df, atr_period).values
    close = df["Close"].values.astype(float)
    n = len(df)

    loss  = atr_factor * atr
    state = np.full(n, "init", dtype=object)
    trail = np.full(n, np.nan)

    for i in range(1, n):
        # INIT: first bar with non-NaN loss starts the system
        if state[i-1] == "init":
            if not np.isnan(loss[i]):
                if first_trade == "long":
                    state[i] = "long"
                    trail[i] = close[i] - loss[i]
                else:
                    state[i] = "short"
                    trail[i] = close[i] + loss[i]
            else:
                state[i] = "init"
                trail[i] = np.nan
            continue

        prev_state = state[i-1]
        prev_trail = trail[i-1]

        if prev_state == "long":
            if close[i] > prev_trail:
                state[i] = "long"
                trail[i] = max(prev_trail, close[i] - loss[i])
            else:
                state[i] = "short"
                trail[i] = close[i] + loss[i]
        else:  # prev_state == "short"
            if close[i] < prev_trail:
                state[i] = "short"
                trail[i] = min(prev_trail, close[i] + loss[i])
            else:
                state[i] = "long"
                trail[i] = close[i] - loss[i]

    # ---- Buy / Sell when state flips ----
    buy  = np.zeros(n, dtype=bool)
    sell = np.zeros(n, dtype=bool)

    for i in range(1, n):
        buy[i]  = (state[i] == "long"  and state[i-1] != "long")
        sell[i] = (state[i] == "short" and state[i-1] != "short")

    return pd.DataFrame(
        {
            "TOS_ATR": atr,
            "TOS_Trail": trail,
            "TOS_State": state,
            "TOS_Buy": buy,
            "TOS_Sell": sell,
        },
        index=df.index,
    )



# def compute_vwap_with_bands(df, num_dev_up=2.0, num_dev_dn=-2.0):
#     """
#     Exact ThinkOrSwim VWAP + Standard Deviation Bands
#     VWAP resets daily.
#     """
#     # Typical price (same as TOS 'vwap')
#     tp = (df["High"] + df["Low"] + df["Close"]) / 3

#     # Daily grouping
#     g = df.index.date

#     # Cumulative sums per day
#     vol_cum = df["Volume"].groupby(g).cumsum()
#     tp_vol_cum = (tp * df["Volume"]).groupby(g).cumsum()
#     tp2_vol_cum = ((tp ** 2) * df["Volume"]).groupby(g).cumsum()

#     # VWAP
#     vwap = tp_vol_cum / vol_cum

#     # Volume-weighted variance
#     variance = (tp2_vol_cum / vol_cum) - (vwap ** 2)
#     variance = variance.clip(lower=0)  # prevent negative due to FP error

#     deviation = np.sqrt(variance)

#     upper = vwap + num_dev_up * deviation
#     lower = vwap + num_dev_dn * deviation

#     return upper, vwap, lower

def compute_vwap_with_bands(df, num_dev_up=2.0, num_dev_dn=-2.0, anchor="DAY"):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3

    # --- FIX ZERO VOLUME ---
    vol = df["Volume"].replace(0, 1)   # essential for indices like SPX

    # --- anchor periods ---
    if anchor == "DAY":
        period = df.index.date
    elif anchor == "WEEK":
        period = df.index.to_period("W").astype(str)
    elif anchor == "MONTH":
        period = df.index.to_period("M").astype(str)

    # --- cumulative values ---
    vol_sum = vol.groupby(period).cumsum()
    vol_tp_sum = (vol * tp).groupby(period).cumsum()
    vol_tp2_sum = (vol * (tp ** 2)).groupby(period).cumsum()

    # --- VWAP ---
    vwap = vol_tp_sum / vol_sum

    # --- STD DEV ---
    variance = (vol_tp2_sum / vol_sum) - (vwap ** 2)
    variance = variance.clip(lower=0)
    deviation = np.sqrt(variance)

    upper = vwap + num_dev_up * deviation
    lower = vwap + num_dev_dn * deviation

    # --- Remove first bar of each day/month/week ---
    first_idx = df.index.to_series().groupby(period).head(1).index
    vwap.loc[first_idx] = np.nan
    upper.loc[first_idx] = np.nan
    lower.loc[first_idx] = np.nan

    return upper, vwap, lower


# ----------------------------------------------------
#  SAFE PRICE DOWNLOADER (YFINANCE WITHOUT MULTIINDEX)
# ----------------------------------------------------
def download_price(ticker, start, end, timeframe=None):
    """
    ALWAYS returns a clean DataFrame:
    Date | Open | High | Low | Close | Volume | Adj_Close
    """
    if timeframe == '1d' or timeframe is None:
        df = yf.download(ticker, start=start, end=end, auto_adjust=False)
    else:
        df = yf.download(ticker, start=start, end=end, interval=timeframe,  prepost=True, auto_adjust=False)
        df = df.tz_convert("America/Chicago")
    # --------------- FIX MULTIINDEX ---------------
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # --------------- STANDARDIZE NAMES ---------------
    df = df.rename(columns={
        "Adj Close": "Adj_Close",
        "Close": "Close",
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Volume": "Volume"
    })

    # Ensure correct dtype
    df = df[["Open","High","Low","Close","Volume","Adj_Close"]]
    # print(df.tail(5))
    return df


# ----------------------------------------------------
#  Feature Configuration
# ----------------------------------------------------
feature_params = {
    "sma_short": 20,
    "sma_mid": 50,
    "sma_long": 200,
    "trend_slope_short": 5,
    "trend_slope_long": 20,
    "rsi_length": 14,
    "stoch_length": 14,
    "stoch_signal": 3,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "adx_length": 14,
    "atr_length": 14,
    "bollinger_length": 20,
    "volatility_length": 20,
    "volume_zscore_length": 20,
    "volume_change_length": 1,
}


# ----------------------------------------------------
#  Helper: Linear Regression Slope
# ----------------------------------------------------
def trend_slope(series):
    y = np.asarray(series).astype(float)
    x = np.arange(len(y))
    return np.polyfit(x, y, 1)[0]


# ----------------------------------------------------
#  Market Context (VIX + SPY)
# ----------------------------------------------------
def get_market_context(start, end, timeframe):

    vix = download_price("^VIX", start, end, timeframe)
    vix = vix.rename(columns={"Close": "VIX_Close"})[["VIX_Close"]]

    spy = download_price("SPY", start, end, timeframe)
    spy["SPY_Return"] = spy["Close"].pct_change()
    spy["SPY_Volatility_20"] = spy["Close"].pct_change().rolling(20).std()

    return pd.concat([vix, spy[["SPY_Return","SPY_Volatility_20"]]], axis=1)


# ----------------------------------------------------
#  Core Feature Engineering
# ----------------------------------------------------
def add_features(df, p, timeframe=None):

    # -------- Trend --------
    df["SMA_Short"] = df["Close"].rolling(p["sma_short"]).mean()
    df["SMA_Mid"]   = df["Close"].rolling(p["sma_mid"]).mean()
    df["SMA_Long"]  = df["Close"].rolling(p["sma_long"]).mean()

    df["Slope_Short"] = df["Close"].rolling(p["trend_slope_short"]) \
        .apply(trend_slope, raw=False)
    df["Slope_Long"]  = df["Close"].rolling(p["trend_slope_long"]) \
        .apply(trend_slope, raw=False)

    # -------- FRAMA --------
    df["FRAMA"] = compute_frama(df["Close"], window=16)
    
    # -------- Momentum --------
    df["RSI"] = ta.momentum.rsi(df["Close"], window=p["rsi_length"])

    st = ta.momentum.StochasticOscillator(
        df["High"], df["Low"], df["Close"],
        window=p["stoch_length"],
        smooth_window=p["stoch_signal"]
    )
    df["Stoch_K"] = st.stoch()
    df["Stoch_D"] = st.stoch_signal()

    macd = ta.trend.MACD(
        close=df["Close"],
        window_slow=p["macd_slow"],
        window_fast=p["macd_fast"],
        window_sign=p["macd_signal"]
    )
    df["MACD"]        = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"]   = macd.macd_diff()

    df["ADX"] = ta.trend.adx(df["High"], df["Low"], df["Close"], window=p["adx_length"])

    # -------- Volatility --------
    df["ATR"] = ta.volatility.average_true_range(
        df["High"], df["Low"], df["Close"], window=p["atr_length"]
    )
    df["Daily_Return"] = df["Close"].pct_change()
    df["Volatility_20"] = df["Daily_Return"].rolling(
        p["volatility_length"]
    ).std()

    bb = ta.volatility.BollingerBands(df["Close"], window=p["bollinger_length"], window_dev=2)
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Mid"]   = bb.bollinger_mavg()
    df["BB_Lower"] = bb.bollinger_lband()
    df["BB_Width"] = bb.bollinger_wband()

    # -------- Volume --------
    df["Volume_Zscore"] = (
        df["Volume"] - df["Volume"].rolling(p["volume_zscore_length"]).mean()
    ) / df["Volume"].rolling(p["volume_zscore_length"]).std()

    df["Volume_Change"] = df["Volume"].pct_change()
    df["OBV"] = ta.volume.on_balance_volume(df["Close"], df["Volume"])

    df["CMF"] = ta.volume.chaikin_money_flow(
        df["High"], df["Low"], df["Close"], df["Volume"], window=20
    )

    # -------- Candle Structure --------
    df["Gap"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
    df["Candle_Body"] = (df["Close"] - df["Open"]) / (df["High"] - df["Low"] + 1e-9)
    
    # -------- VWAP --------
    if timeframe in ["5m", "15m", "30m", "1h", "4h"]:
        df["VWAP_Upper"], df["VWAP"], df["VWAP_Lower"] = compute_vwap_with_bands(df)

    # -------- TOS ATR --------
    tos = tos_trailing_stop(df, atr_period=5, atr_factor=1.5)
    df["TOS_Trail"] = tos["TOS_Trail"]


    # # -------- TOS RSI With Divergence  --------
    tos_rsi = compute_tos_rsi(df, 9, 70, 30)
    df["TOS_RSI"]      = tos_rsi["TOS_RSI"]


    return df


# ----------------------------------------------------
#  FINAL DATASET BUILDER
# ----------------------------------------------------
def build_feature_dataset(ticker, start_date="2010-01-01", end_date=None, timeframe=None):

    ohlcv = download_price(ticker, start_date, end_date, timeframe=timeframe)

    feat  = add_features(ohlcv.copy(), feature_params, timeframe=timeframe)

    market = get_market_context(start_date, end_date, timeframe=timeframe)
    market = market.reindex(feat.index)

    out = feat.join(market)

    # DO NOT DROPNA HERE — we drop AFTER labeling
    return out
