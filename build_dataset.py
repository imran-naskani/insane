import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import ta


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


def compute_vwap_with_bands(df, num_dev_up=2.0, num_dev_dn=-2.0):
    """
    Exact ThinkOrSwim VWAP + Standard Deviation Bands
    VWAP resets daily.
    """
    # Typical price (same as TOS 'vwap')
    tp = (df["High"] + df["Low"] + df["Close"]) / 3

    # Daily grouping
    g = df.index.date

    # Cumulative sums per day
    vol_cum = df["Volume"].groupby(g).cumsum()
    tp_vol_cum = (tp * df["Volume"]).groupby(g).cumsum()
    tp2_vol_cum = ((tp ** 2) * df["Volume"]).groupby(g).cumsum()

    # VWAP
    vwap = tp_vol_cum / vol_cum

    # Volume-weighted variance
    variance = (tp2_vol_cum / vol_cum) - (vwap ** 2)
    variance = variance.clip(lower=0)  # prevent negative due to FP error

    deviation = np.sqrt(variance)

    upper = vwap + num_dev_up * deviation
    lower = vwap + num_dev_dn * deviation

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
def get_market_context(start, end):

    vix = download_price("^VIX", start, end)
    vix = vix.rename(columns={"Close": "VIX_Close"})[["VIX_Close"]]

    spy = download_price("SPY", start, end)
    spy["SPY_Return"] = spy["Close"].pct_change()
    spy["SPY_Volatility_20"] = spy["Close"].pct_change().rolling(20).std()

    return pd.concat([vix, spy[["SPY_Return","SPY_Volatility_20"]]], axis=1)


# ----------------------------------------------------
#  Core Feature Engineering
# ----------------------------------------------------
def add_features(df, p):

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

    df["VWAP_Upper"], df["VWAP"], df["VWAP_Lower"] = compute_vwap_with_bands(df)

    return df


# ----------------------------------------------------
#  FINAL DATASET BUILDER
# ----------------------------------------------------
def build_feature_dataset(ticker, start_date="2010-01-01", end_date=None, timeframe=None):

    ohlcv = download_price(ticker, start_date, end_date, timeframe=timeframe)

    feat  = add_features(ohlcv.copy(), feature_params)

    market = get_market_context(start_date, end_date)
    market = market.reindex(feat.index)

    out = feat.join(market)

    # DO NOT DROPNA HERE — we drop AFTER labeling
    return out
