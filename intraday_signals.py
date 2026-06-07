"""
Shared intraday signal logic: MR (Mean Reversion) + ORB-Slope strategies.
Full specification, formulas, and backtest results in SIGNAL_LOGIC.md.

Quick reference
---------------
add_intraday_features(df)         -> adds lr_slope, lr_r2 to a df that has ATR + TOS_Trail
run_new(dv, s_thr, atr_floor)     -> (long_sig, short_sig)  MR zero-crossings
run_orb_slope(dv)                 -> (long_sig, short_sig)  ORB at bar 6 of each session
_orb_reversal_confirmed(...)      -> bool  gate: allow MR to flip an ORB position
compute_chart_signals(df, ticker) -> df with Turn_Up, Turn_Down, Signal_Source added

Used by
-------
  alert_spicy_engine.py  - live 5m loop, Telegram alerts, reversal gate enforced
  insane.py              - Streamlit chart display, gate not applied (historical view)
"""

import numpy as np
import pandas as pd

# ── Per-ticker parameters (SIGNAL_LOGIC.md Section 8) ────────────────────────
S_THR = {
    "TSLA": 0.65,
    "SPY":  0.55,
    "NVDA": 0.65,
    "QQQ":  0.55,
    "AAPL": 0.60,
}
ATR_FLOOR = {
    "TSLA": 1.67,   # STRIKE_GAP / 1.5  (STRIKE_GAP = $2.50)
    "SPY":  0.33,   # STRIKE_GAP / 1.5  (STRIKE_GAP = $0.50)
    "NVDA": 0.67,   # STRIKE_GAP / 1.5  (STRIKE_GAP = $1.00)
    "QQQ":  0.67,
    "AAPL": 1.67,
}

ORB_SLOPE_BARS   = 6     # first 6 bars = 30-min opening window (08:30-08:55 CT)
ORB_SLOPE_DEG    = 30.0  # angle threshold — tan(30 deg) ~= 0.58 ATR/bar
ORB_REVERSAL_DEG = 10.0  # min re-computed angle for MR to flip an ORB position


# ── Feature engineering ───────────────────────────────────────────────────────

def rolling_linreg_fast(series, window=14):
    """
    Vectorized rolling OLS regression on Close.

    Returns
    -------
    lr_slope : pd.Series  $/bar — positive = uptrend, negative = downtrend
    lr_r2    : pd.Series  [0, 1] — near 1 = clean linear trend, near 0 = choppy
    """
    values = series.to_numpy(dtype=float)
    x      = np.arange(window, dtype=float)
    xd     = x - x.mean()
    denom  = np.dot(xd, xd)

    slopes = np.full(len(values), np.nan)
    r2s    = np.full(len(values), np.nan)

    for i in range(window - 1, len(values)):
        y = values[i - window + 1 : i + 1]
        if np.isnan(y).any():
            continue
        yd        = y - y.mean()
        slopes[i] = np.dot(xd, yd) / denom
        c         = np.corrcoef(x, y)[0, 1]
        r2s[i]    = c ** 2 if np.isfinite(c) else np.nan

    return pd.Series(slopes, index=series.index), pd.Series(r2s, index=series.index)


def add_intraday_features(df):
    """
    Add lr_slope and lr_r2 to a df that already has ATR and TOS_Trail
    (both provided by build_feature_dataset for 5m timeframe).

    IMPORTANT: call on the FULL multi-day df before any session filtering.
    TOS_Trail is stateful and must carry across day boundaries.
    NaN warmup bars are filled with 0 (first 13 bars have no signal by design).
    """
    df = df.copy()
    slopes, r2s    = rolling_linreg_fast(df["Close"], window=14)
    df["lr_slope"] = slopes
    df["lr_r2"]    = r2s
    df[["lr_slope", "lr_r2"]] = df[["lr_slope", "lr_r2"]].fillna(0)
    return df


# ── Signal generation ─────────────────────────────────────────────────────────

def _cooldown(mask, cd):
    """
    Suppress the cd bars immediately after each True signal.
    Prevents rapid same-direction re-triggers within the cooldown window.
    """
    arr = mask.to_numpy(dtype=bool).copy()
    i = 0
    while i < len(arr):
        if arr[i]:
            arr[i + 1 : i + 1 + cd] = False
            i += cd + 1
        else:
            i += 1
    return pd.Series(arr, index=mask.index)


def run_new(dv, s_thr, atr_floor=0.0):
    """
    Strategy A — Mean Reversion (MR).

    Fires when ATR-normalized OLS slope crosses zero, provided prior-trend
    quality gates all pass.

    Gates (all required)
    --------------------
    wneg/wpos : strong prior trend in any of the 8 bars before the crossing
    wr2       : R^2 > 0.50 in those same 8 bars (linear, not choppy)
    prox      : price within 2x ATR of TOS trailing stop (not overextended)
    atr_ok    : ATR above per-ticker floor (quiet-day filter)
    cooldown  : 8-bar suppression after each signal

    Parameters
    ----------
    dv        : session DataFrame with lr_slope, lr_r2, ATR, TOS_Trail, Close
    s_thr     : slope_sq threshold for "strong trend" (per-ticker from S_THR)
    atr_floor : minimum ATR to generate any signal (per-ticker from ATR_FLOOR)

    Returns
    -------
    (long_sig, short_sig) : boolean pd.Series indexed like dv
    """
    snorm    = dv["lr_slope"] / dv["ATR"].replace(0, np.nan)
    s        = np.tanh(5 * snorm).fillna(0)   # bounded (-1, +1); saturates at ~0.2 ATR/bar
    s_1      = s.shift(1).fillna(0)

    r2_gate  = (1 / (1 + np.exp(-14 * (dv["lr_r2"] - 0.50)))) > 0.50  # sigmoid at R^2=0.50
    prox     = abs(dv["Close"] - dv["TOS_Trail"]) <= dv["ATR"] * 2.0
    atr_ok   = dv["ATR"] >= atr_floor

    wneg = pd.Series(False, index=s.index)
    wpos = pd.Series(False, index=s.index)
    wr2  = pd.Series(False, index=s.index)
    for k in range(1, 9):  # look back 1..8 bars
        wneg |= s.shift(k) <= -s_thr
        wpos |= s.shift(k) >=  s_thr
        wr2  |= r2_gate.shift(k).fillna(False)

    zcu = (s > 0) & (s_1 <= 0)   # upward zero-crossing   -> LONG candidate
    zcd = (s < 0) & (s_1 >= 0)   # downward zero-crossing -> SHORT candidate

    return (
        _cooldown(zcu & wneg & prox & wr2 & atr_ok, 8),
        _cooldown(zcd & wpos & prox & wr2 & atr_ok, 8),
    )


def run_orb_slope(dv, slope_bars=ORB_SLOPE_BARS, angle_deg=ORB_SLOPE_DEG):
    """
    Strategy B — Opening Range Breakout via slope angle.

    Fits OLS to the first slope_bars (6) bars of the session, normalizes the
    slope by ATR, converts to degrees via arctan.  Fires at most once per
    session at the last bar of the opening window (~08:55 CT).

    Entry is at the Open of the NEXT bar (bar 7, ~09:00 CT).

    Parameters
    ----------
    dv        : today's session DataFrame (08:30-15:00 CT), features pre-computed
    slope_bars: number of opening bars (default 6 = 30 min)
    angle_deg : threshold in degrees (default 30 — tan(30 deg) ~= 0.58 ATR/bar)

    Returns
    -------
    (long_sig, short_sig) : boolean pd.Series, True on bar 6 when angle qualifies
    """
    long_sig  = pd.Series(False, index=dv.index)
    short_sig = pd.Series(False, index=dv.index)

    if len(dv) < slope_bars:
        return long_sig, short_sig

    window = dv.iloc[:slope_bars]
    closes = window["Close"].values.astype(float)
    atr    = float(window["ATR"].iloc[-1])

    if atr <= 0 or np.isnan(atr):
        return long_sig, short_sig

    x      = np.arange(slope_bars, dtype=float)
    xd     = x - x.mean()
    slope  = np.dot(xd, closes - closes.mean()) / np.dot(xd, xd)
    angle  = np.degrees(np.arctan(slope / atr))

    last_bar = window.index[-1]   # timestamp of bar 6 — signal bar
    if angle >= angle_deg:
        long_sig.loc[last_bar]  = True
    elif angle <= -angle_deg:
        short_sig.loc[last_bar] = True

    return long_sig, short_sig


def _orb_reversal_confirmed(dv, orb_bar_idx, orb_dir, current_idx,
                            threshold=ORB_REVERSAL_DEG):
    """
    Reversal gate: returns True if MR is allowed to flip the active ORB position.

    Re-computes the OLS angle from the ORB signal bar to the current bar.
    The flip is allowed only if the trend has genuinely reversed by at least
    `threshold` degrees (default 10, tuned via sweep in SIGNAL_LOGIC.md).

    Parameters
    ----------
    dv          : today's session DataFrame
    orb_bar_idx : integer iloc of the ORB signal bar (always ORB_SLOPE_BARS-1 = 5)
    orb_dir     : 'long' or 'short' — direction the ORB position is holding
    current_idx : integer iloc of the bar where MR wants to flip
    threshold   : degrees required to confirm reversal

    Returns
    -------
    bool : True = allow flip; False = suppress MR signal (gate blocks it)
    """
    n_bars = current_idx - orb_bar_idx + 1
    if n_bars < 2:
        return False

    closes = dv["Close"].iloc[orb_bar_idx : current_idx + 1].values.astype(float)
    atr    = float(dv["ATR"].iloc[current_idx])
    if atr <= 0 or np.isnan(atr):
        return False

    x      = np.arange(n_bars, dtype=float)
    xd     = x - x.mean()
    slope  = np.dot(xd, closes - closes.mean()) / np.dot(xd, xd)
    angle  = np.degrees(np.arctan(slope / atr))

    if orb_dir == "short":
        return angle >= threshold    # upward angle needed to flip a short ORB
    else:
        return angle <= -threshold   # downward angle needed to flip a long ORB


# ── Chart display helper (used by insane.py) ──────────────────────────────────

def compute_chart_signals(df, ticker):
    """
    Compute MR + ORB signals across ALL sessions in df for historical chart display.
    Adds Turn_Up, Turn_Down, Signal_Source columns to df.

    Note: the ORB reversal gate is NOT applied here — it is a live-trading
    concept that requires session state.  For historical display, all signals
    are shown as-is.

    Prerequisites
    -------------
    Call add_intraday_features(df) before this function.
    df must cover the full multi-day range (not pre-filtered to one session).
    """
    s_thr     = S_THR.get(ticker, 0.60)
    atr_floor = ATR_FLOOR.get(ticker, 0.50)

    df = df.copy()
    df["Turn_Up"]       = False
    df["Turn_Down"]     = False
    df["Signal_Source"] = ""

    # MR: run on full multi-day df — features (lr_slope, ATR, TOS_Trail) carry across sessions
    lmr, smr = run_new(df, s_thr, atr_floor)
    df.loc[lmr, "Turn_Up"]       = True
    df.loc[lmr, "Signal_Source"] = "MR"
    df.loc[smr, "Turn_Down"]     = True
    df.loc[smr, "Signal_Source"] = "MR"

    # ORB: must be computed per-session (first 6 bars of each trading day)
    session_start = pd.Timestamp("08:30").time()
    session_end   = pd.Timestamp("15:00").time()
    for d in sorted(set(df.index.date)):
        mask    = df.index.date == d
        session = df[mask]
        session = session[
            (session.index.time >= session_start) &
            (session.index.time <= session_end)
        ]
        if len(session) < ORB_SLOPE_BARS:
            continue
        lorb, sorb = run_orb_slope(session)
        orb_long_idx  = lorb[lorb].index
        orb_short_idx = sorb[sorb].index
        df.loc[orb_long_idx,  "Turn_Up"]       = True
        df.loc[orb_long_idx,  "Signal_Source"] = "ORB"
        df.loc[orb_short_idx, "Turn_Down"]     = True
        df.loc[orb_short_idx, "Signal_Source"] = "ORB"

    return df
