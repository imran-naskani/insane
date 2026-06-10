"""
Shared intraday signal logic: MR (Mean Reversion) + ORB-Slope strategies.
Full specification, formulas, and backtest results in SIGNAL_LOGIC.md.

Quick reference
---------------
add_intraday_features(df)              -> adds lr_slope, lr_r2 to df
run_new(dv, s_thr, atr_floor)          -> (long_sig, short_sig)  MR zero-crossings
run_orb_slope(dv, ...)                 -> (long_sig, short_sig)  ORB accumulation window
run_orb_sliding(dv, ...)               -> (long_sig, short_sig)  sliding 6-bar ORB
sliding_qualifies_direction(dv, ...)   -> bool  checks if current 6-bar window qualifies
                                          (used for MR->ORB upgrade in alert engine)
_orb_reversal_confirmed(...)           -> bool  gate: allow MR to flip an ORB position
compute_chart_signals(df, ticker)      -> df with Turn_Up, Turn_Down, Signal_Source added

Used by
-------
  alert_spicy_engine.py  - live 5m loop, Telegram alerts, reversal gate enforced
  insane.py              - Streamlit chart display, gate not applied (historical view)
"""

import numpy as np
import pandas as pd

# ── Per-ticker MR parameters ──────────────────────────────────────────────────
S_THR = {
    "TSLA": 0.65,
    "SPY":  0.55,
    "QQQ":  0.55,
    "TQQQ": 0.55,
}
ATR_FLOOR = {
    "TSLA": 1.67,   # STRIKE_GAP / 1.5  (STRIKE_GAP = $2.50)
    "SPY":  0.33,   # STRIKE_GAP / 1.5  (STRIKE_GAP = $0.50)
    "QQQ":  0.67,   # STRIKE_GAP / 1.5  (STRIKE_GAP = $1.00)
    "TQQQ": 0.67,   # STRIKE_GAP / 1.5  (STRIKE_GAP = $1.00)
}

# ── Per-ticker ORB parameters ─────────────────────────────────────────────────
ORB_SLOPE_BARS        = 6       # opening window width (6 x 5min = 30 min, 08:30-08:55 CT)
ORB_SLOPE_DEG_DEFAULT = 20.0    # angle threshold fallback (most common across tuned tickers)
ORB_SLOPE_DEG = {               # per-ticker angle threshold
    "TSLA":  30.0,
    "SPY":   20.0,
    "QQQ":   20.0,
    "TQQQ":  20.0,
    "^GSPC": 10.0,
}

ORB_R2_THR_DEFAULT = 0.60       # R2 gate fallback (most common across tuned tickers)
ORB_R2_THR = {                  # per-ticker R2 quality gate on ORB/sliding windows
    "TSLA":  0.80,
    "SPY":   0.60,
    "QQQ":   0.60,
    "TQQQ":  0.60,
    "^GSPC": 0.60,
}

ORB_ACCUM_START_DEFAULT = 5     # accumulation 5→6 bars (most common across tuned tickers)
ORB_ACCUM_START = {             # accumulation: check growing windows from this bar
    "TSLA":  5,                 # checks bars 1-5, then 1-6 (fires at first qualifying)
    "SPY":   5,
    "^GSPC": 5,
    "QQQ":   None,              # fixed bar 6 only
    "TQQQ":  None,
}

ORB_SLIDE_LIM = 18              # sliding window: scan up to bar 18 (~09:55 CT, 0-indexed=17)
ORB_USE_SLIDING_DEFAULT = True  # sliding enabled by default (most common across tuned tickers)
ORB_USE_SLIDING = {             # whether to run sliding window when fixed/accum ORB misses
    "TSLA":  True,
    "SPY":   True,
    "QQQ":   True,
    "TQQQ":  True,
    "^GSPC": False,             # no sliding for SPX — accum 5-6 alone is best
}

ORB_REVERSAL_DEG    = 10.0      # legacy default (kept for backtest scripts)
REVERSAL_ANGLE      = 20.0      # uniform reversal gate angle for all tickers (ext_i+1 anchor)


# ── Internal OLS helpers ──────────────────────────────────────────────────────

def _ols_angle_r2(closes, atr):
    """OLS slope angle (degrees) and R² for a price array."""
    closes = np.asarray(closes, dtype=float)
    n  = len(closes)
    x  = np.arange(n, dtype=float); xd = x - x.mean()
    s  = np.dot(xd, closes - closes.mean()) / np.dot(xd, xd)
    ang = np.degrees(np.arctan(s / atr)) if atr > 0 else 0.0
    c   = np.corrcoef(x, closes)[0, 1]
    r2  = float(c ** 2) if np.isfinite(c) else 0.0
    return ang, r2


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
    Add lr_slope and lr_r2 to a df that already has ATR and TOS_Trail.
    Call on the FULL multi-day df before any session filtering.
    """
    df = df.copy()
    slopes, r2s    = rolling_linreg_fast(df["Close"], window=14)
    df["lr_slope"] = slopes
    df["lr_r2"]    = r2s
    df[["lr_slope", "lr_r2"]] = df[["lr_slope", "lr_r2"]].fillna(0)
    return df


# ── Signal generation ─────────────────────────────────────────────────────────

def _cooldown(mask, cd):
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
    Fires when ATR-normalized OLS slope crosses zero with prior-trend quality gates.
    """
    snorm   = dv["lr_slope"] / dv["ATR"].replace(0, np.nan)
    s       = np.tanh(5 * snorm).fillna(0)
    s_1     = s.shift(1).fillna(0)
    r2_gate = (1 / (1 + np.exp(-14 * (dv["lr_r2"] - 0.50)))) > 0.50
    prox    = abs(dv["Close"] - dv["TOS_Trail"]) <= dv["ATR"] * 2.0
    atr_ok  = dv["ATR"] >= atr_floor

    wneg = pd.Series(False, index=s.index)
    wpos = pd.Series(False, index=s.index)
    wr2  = pd.Series(False, index=s.index)
    for k in range(1, 9):
        wneg |= s.shift(k) <= -s_thr
        wpos |= s.shift(k) >=  s_thr
        wr2  |= r2_gate.shift(k).fillna(False)

    zcu = (s > 0) & (s_1 <= 0)
    zcd = (s < 0) & (s_1 >= 0)
    return (
        _cooldown(zcu & wneg & prox & wr2 & atr_ok, 8),
        _cooldown(zcd & wpos & prox & wr2 & atr_ok, 8),
    )


def run_orb_slope(dv, slope_bars=ORB_SLOPE_BARS, angle_deg=ORB_SLOPE_DEG_DEFAULT,
                  r2_thr=0.0, accum_start=None):
    """
    Strategy B — Opening Range Breakout via slope angle.

    If accum_start is not None, checks growing windows from accum_start through
    slope_bars (e.g. accum_start=5 checks bars 1-5 then 1-6). Fires on the first
    qualifying window. R2 gate skipped on non-qualifying windows (waits for next bar).

    Parameters
    ----------
    dv          : today's session DataFrame (08:30-15:00 CT)
    slope_bars  : opening window size in bars (default 6)
    angle_deg   : angle threshold in degrees
    r2_thr      : minimum R2 for the window (0.0 = no gate)
    accum_start : first bar to check (None = fixed slope_bars only)

    Returns
    -------
    (long_sig, short_sig) : boolean pd.Series, True at the signal bar
    """
    long_sig  = pd.Series(False, index=dv.index)
    short_sig = pd.Series(False, index=dv.index)

    checks = list(range(accum_start, slope_bars + 1)) if accum_start else [slope_bars]

    for nb in checks:
        if len(dv) < nb:
            break
        window = dv.iloc[:nb]
        atr    = float(window["ATR"].iloc[-1])
        if atr <= 0 or np.isnan(atr):
            continue
        closes = window["Close"].values.astype(float)
        ang, r2 = _ols_angle_r2(closes, atr)

        if r2_thr > 0 and r2 < r2_thr:
            continue   # quality not yet there — try next (larger) window

        last_bar = window.index[-1]
        if ang >= angle_deg:
            long_sig.loc[last_bar]  = True
            return long_sig, short_sig
        elif ang <= -angle_deg:
            short_sig.loc[last_bar] = True
            return long_sig, short_sig

    return long_sig, short_sig


def run_orb_sliding(dv, angle_deg, r2_thr=0.0, slide_lim=ORB_SLIDE_LIM):
    """
    Sliding 6-bar ORB window. Scans bars 7 through slide_lim looking for the
    first 6-bar window that meets the angle + R2 threshold.

    Caller is responsible for only calling this when fixed/accum ORB has NOT fired.

    Parameters
    ----------
    dv        : today's session DataFrame
    angle_deg : angle threshold
    r2_thr    : minimum R2 (0.0 = no gate)
    slide_lim : last bar index to scan (0-based, default 17 = bar 18)

    Returns
    -------
    (long_sig, short_sig) : boolean pd.Series, True at the last bar of the
                            qualifying window (entry on next bar open)
    """
    long_sig  = pd.Series(False, index=dv.index)
    short_sig = pd.Series(False, index=dv.index)

    for i in range(ORB_SLOPE_BARS - 1, min(len(dv) - 1, slide_lim)):
        w     = dv.iloc[i - ORB_SLOPE_BARS + 1: i + 1]
        atr_w = float(w["ATR"].iloc[-1])
        if atr_w <= 0:
            continue
        ang_w, r2_w = _ols_angle_r2(w["Close"].values.astype(float), atr_w)
        if r2_thr > 0 and r2_w < r2_thr:
            continue
        if ang_w >= angle_deg:
            long_sig.loc[w.index[-1]]  = True
            return long_sig, short_sig
        elif ang_w <= -angle_deg:
            short_sig.loc[w.index[-1]] = True
            return long_sig, short_sig

    return long_sig, short_sig


def sliding_qualifies_direction(dv, bar_i, angle_deg, r2_thr, direction):
    """
    Check if the 6-bar window ending at bar_i qualifies in the given direction.
    Used by the alert engine for MR->ORB upgrade: when in an MR position, call
    this each bar — if it returns True, upgrade pos_src to 'orb'.

    No time restriction (runs any time during the session).

    Parameters
    ----------
    dv        : today's session DataFrame
    bar_i     : integer iloc of the current bar (last bar of the 6-bar window)
    angle_deg : angle threshold
    r2_thr    : minimum R2 (0.0 = no gate)
    direction : 'long' or 'short'

    Returns
    -------
    bool : True if the window qualifies in the given direction
    """
    if bar_i < ORB_SLOPE_BARS - 1:
        return False
    w = dv.iloc[bar_i - ORB_SLOPE_BARS + 1: bar_i + 1]
    if len(w) < ORB_SLOPE_BARS:
        return False
    atr = float(w["ATR"].iloc[-1])
    if atr <= 0:
        return False
    ang, r2 = _ols_angle_r2(w["Close"].values.astype(float), atr)
    if r2_thr > 0 and r2 < r2_thr:
        return False
    if direction == "long":  return ang >= angle_deg
    if direction == "short": return ang <= -angle_deg
    return False


def _orb_reversal_confirmed(dv, orb_bar_idx, orb_dir, current_idx,
                            threshold=ORB_REVERSAL_DEG, r2_thr=0.0,
                            accum_start=5, slope_bars=6):
    """
    Reversal gate: returns True if MR is allowed to flip the active ORB position.
    Anchors at the day's extreme (lowest for short ORB, highest for long ORB) since
    ORB entry, then checks accum 5-6 bar OLS windows from that extreme.
    Both angle AND R2 must pass (AND logic).
    """
    if current_idx <= orb_bar_idx:
        return False
    closes = dv["Close"].iloc[orb_bar_idx: current_idx + 1].values.astype(float)
    if len(closes) == 0:
        return False
    atr = float(dv["ATR"].iloc[current_idx])
    if atr <= 0 or np.isnan(atr):
        return False

    # Find extreme since ORB entry
    rel   = int(np.argmin(closes)) if orb_dir == "short" else int(np.argmax(closes))
    ext_i = orb_bar_idx + rel      # absolute session iloc of extreme

    # Check accum windows starting from bar AFTER the extreme
    anchor     = ext_i + 1
    all_closes = dv["Close"].values.astype(float)
    for nb in range(accum_start, slope_bars + 1):
        end = anchor + nb
        if end - 1 > current_idx:
            break
        w = all_closes[anchor: end]
        ang, r2 = _ols_angle_r2(w, atr)
        if r2_thr > 0 and r2 < r2_thr:
            continue                      # R2 fails — try next window
        if orb_dir == "short" and ang >= threshold:
            return True
        if orb_dir == "long"  and ang <= -threshold:
            return True
    return False


# ── Chart display helper (used by insane.py) ──────────────────────────────────

def compute_chart_signals(df, ticker):
    """
    Compute MR + ORB (accumulation + sliding) signals across ALL sessions in df.
    Adds Turn_Up, Turn_Down, Signal_Source columns to df.
    ORB reversal gate IS applied: MR signals that cannot flip an active ORB
    position are suppressed so chart matches backtest and live engine.
    """
    s_thr     = S_THR.get(ticker, 0.60)
    atr_floor = ATR_FLOOR.get(ticker, 0.50)
    ang_deg   = ORB_SLOPE_DEG.get(ticker, ORB_SLOPE_DEG_DEFAULT)
    r2_thr    = ORB_R2_THR.get(ticker, ORB_R2_THR_DEFAULT)
    accum     = ORB_ACCUM_START.get(ticker, ORB_ACCUM_START_DEFAULT)
    use_slide = ORB_USE_SLIDING.get(ticker, ORB_USE_SLIDING_DEFAULT)

    df = df.copy()
    df["Turn_Up"]       = False
    df["Turn_Down"]     = False
    df["Signal_Source"] = ""

    # Compute MR signals on full df (TOS_Trail warmup requires full history)
    lmr, smr = run_new(df, s_thr, atr_floor)

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

        # ── ORB / sliding ─────────────────────────────────────────────────
        lorb, sorb = run_orb_slope(session, angle_deg=ang_deg,
                                   r2_thr=r2_thr, accum_start=accum)
        orb_fired = lorb.any() or sorb.any()
        if not orb_fired and use_slide:
            lsl, ssl = run_orb_sliding(session, ang_deg, r2_thr)
            lorb = lorb | lsl
            sorb = sorb | ssl

        # Mark ORB signals and capture gate anchor
        orb_bar_i = None
        orb_dir   = None
        if sorb.any():
            orb_ts    = sorb[sorb].index[0]
            orb_bar_i = session.index.get_loc(orb_ts)
            orb_dir   = "short"
            df.at[orb_ts, "Turn_Down"]     = True
            df.at[orb_ts, "Signal_Source"] = "ORB"
        elif lorb.any():
            orb_ts    = lorb[lorb].index[0]
            orb_bar_i = session.index.get_loc(orb_ts)
            orb_dir   = "long"
            df.at[orb_ts, "Turn_Up"]       = True
            df.at[orb_ts, "Signal_Source"] = "ORB"

        # ── MR signals with reversal gate ─────────────────────────────────
        # in_orb: True while ORB position is active (gate applies)
        in_orb = orb_bar_i is not None

        for ts in session.index:
            is_ml = bool(lmr.get(ts, False)) if ts in lmr.index else False
            is_ms = bool(smr.get(ts, False)) if ts in smr.index else False
            if not (is_ml or is_ms):
                continue

            ts_i = session.index.get_loc(ts)

            if in_orb:
                # Gate: MR must confirm reversal to flip ORB position
                if is_ml and orb_dir == "short":
                    if _orb_reversal_confirmed(session, orb_bar_i, "short", ts_i,
                                               threshold=REVERSAL_ANGLE, r2_thr=r2_thr):
                        df.at[ts, "Turn_Up"]       = True
                        df.at[ts, "Signal_Source"] = "MR"
                        in_orb = False   # ORB flipped — now MR position
                    # else: gate blocks, don't mark
                elif is_ms and orb_dir == "long":
                    if _orb_reversal_confirmed(session, orb_bar_i, "long", ts_i,
                                               threshold=REVERSAL_ANGLE, r2_thr=r2_thr):
                        df.at[ts, "Turn_Down"]     = True
                        df.at[ts, "Signal_Source"] = "MR"
                        in_orb = False
                    # else: gate blocks, don't mark
                # MR same direction as ORB: not a flip, skip
            else:
                # No active ORB — mark MR freely
                if is_ml:
                    df.at[ts, "Turn_Up"]       = True
                    df.at[ts, "Signal_Source"] = "MR"
                elif is_ms:
                    df.at[ts, "Turn_Down"]     = True
                    df.at[ts, "Signal_Source"] = "MR"

    return df
