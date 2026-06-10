# Signal Logic — Intraday MR + ORB-Slope Combined Strategy

This document is a complete specification of every formula, parameter, and implementation
detail needed to replicate the strategy from scratch. A fresh Claude instance with this
file plus the repo's Python files should be able to reproduce all results exactly.

---

## Table of Contents

1. [Strategy Overview](#1-strategy-overview)
2. [Data Requirements and Caching](#2-data-requirements-and-caching)
3. [Shared Indicators](#3-shared-indicators)
4. [Strategy A — Mean Reversion (MR)](#4-strategy-a--mean-reversion-mr)
5. [Strategy B — ORB-Slope](#5-strategy-b--orb-slope)
6. [Combined Strategy — MR + ORB with Reversal Gate](#6-combined-strategy--mr--orb-with-reversal-gate)
7. [Trade Execution Rules](#7-trade-execution-rules)
8. [Per-Ticker Parameters](#8-per-ticker-parameters)
9. [Backtest Results (42 days, Apr 9 – Jun 5 2026)](#9-backtest-results)
10. [Files and Entry Points](#10-files-and-entry-points)
11. [Design Decisions and Known Limitations](#11-design-decisions-and-known-limitations)
12. [Dependencies](#12-dependencies)

---

## 1. Strategy Overview

Two independent signal sources are combined into a single daily position with a
reversal gate that prevents them from destructively interfering on trend days.

### Strategy A — Mean Reversion (MR)
- Fires **throughout the day** based on ATR-normalized OLS slope zero-crossings
- Long when a confirmed downtrend reverses upward; short when confirmed uptrend reverses down
- Requires: strong prior trend in lookback window, high R², price near TOS trailing stop
- Multiple signals possible per session; 8-bar cooldown between same-direction signals

### Strategy B — ORB-Slope (Opening Range Breakout via Slope Angle)
- Fires **exactly once per session**, at bar 6 (the 30-minute mark, ~09:00 CT)
- Fits OLS to the first 6 bars of the session, normalizes slope by ATR, converts to degrees
- Long if opening angle >= +30 deg (strong upward thrust); short if <= -30 deg
- Holds the full day to EOD with no intraday re-entries

### Combined Execution with Reversal Gate
- Both signal streams are merged: `long_combined = long_MR | long_ORB`
- **On days where ORB fires**, MR signals that oppose the ORB direction are gated:
  they may only flip the ORB position if the OLS slope re-computed from the ORB
  signal bar to the current bar has reversed by at least **10 degrees**
- This prevents mean-reversion logic from prematurely exiting a strong trending open
- The gate applies only to MR-vs-ORB conflicts; ORB-vs-ORB never conflicts (one per day)

### Intended Use — 0DTE Options
The strategy is designed for buying ATM 0DTE options:
- Buy ATM call (ORB long or MR long) or put (ORB short or MR short)
- Hold all day; close before EOD
- Maximum loss = premium paid (defined risk)
- ORB trades are the highest-conviction: strong trend at open, expected to continue

---

## 2. Data Requirements and Caching

### Interval and Timezone
| Parameter | Value |
|-----------|-------|
| Bar interval | 5 minutes |
| Timezone | America/Chicago (US Central) |
| Regular session | 08:30 CT – 15:00 CT |
| Required columns | Open, High, Low, Close, Volume |
| NaN handling | `fillna(0)` after all feature engineering |

### yfinance Limitation
Yahoo Finance provides only ~60 days of 5-minute history. Download in 25-day chunks
and save locally as parquet to avoid re-downloading. See `download_history.py`.

```python
# download_history.py — key config
TICKERS    = ["TSLA", "SPY", "NVDA", "AAPL", "QQQ"]
DATA_DIR   = Path(__file__).parent / "data"
CHUNK_DAYS = 25        # safe chunk size for yfinance 5m requests
MAX_DAYS   = 58        # Yahoo's practical 5m lookback limit
INTERVAL   = "5m"
TZ_TARGET  = "America/Chicago"
```

Run: `python download_history.py` (incremental) or `python download_history.py --rebuild`

### Loading in Backtest
```python
DATA_DIR   = Path(__file__).parent / "data"
start_date = date(2026, 4, 9)
end_date   = date(2026, 6, 5)

df = pd.read_parquet(DATA_DIR / f"{ticker}_5m.parquet")
if df.index.tz is None:
    df = df.tz_localize("America/Chicago")
df = add_features(df)    # compute ATR, TOS_Trail, lr_slope, lr_r2 on full history

# Filter to backtest date range per-day inside the main loop
trading_days = sorted(d for d in set(df.index.date) if start_date <= d <= end_date)
```

### Session Filtering
```python
def session_bars(df, d):
    tz = df.index.tz
    dv = df[(df.index >= pd.Timestamp(d, tz=tz)) &
            (df.index <  pd.Timestamp(d + timedelta(days=1), tz=tz))]
    return dv[(dv.index.time >= pd.Timestamp("08:30").time()) &
              (dv.index.time <= pd.Timestamp("15:00").time())]
```

**Critical:** Features are computed on the **full multi-day DataFrame** before session
filtering. The TOS trailing stop is stateful and must carry over day boundaries.

---

## 3. Shared Indicators

### 3.1 Standard ATR (ta library, window=14)

Used for: slope normalization, proximity filter, ATR floor gate.

```python
import ta
df["ATR"] = ta.volatility.average_true_range(
    df["High"], df["Low"], df["Close"], window=14
)
```

### 3.2 TOS ATR (ThinkorSwim-style, used only inside TOS trailing stop)

Distinct from the standard ATR above. Called with `atr_period=5` by `tos_trailing_stop`.
Features HiLo range capping and Wilder EMA smoothing.

```python
def tos_atr_modified(df, atr_period=10):
    high  = df["High"].values.astype(float)
    low   = df["Low"].values.astype(float)
    close = df["Close"].values.astype(float)
    n     = len(df)

    hl     = high - low
    sma_hl = pd.Series(hl).rolling(atr_period).mean().values
    hilo   = np.where(np.isnan(sma_hl), hl, np.minimum(hl, 1.5 * sma_hl))  # cap at 1.5x mean

    href = np.zeros(n)
    lref = np.zeros(n)
    for i in range(1, n):
        if low[i] <= high[i-1]:
            href[i] = high[i] - close[i-1]
        else:
            href[i] = (high[i] - close[i-1]) - 0.5 * (low[i] - high[i-1])
        if high[i] >= low[i-1]:
            lref[i] = close[i-1] - low[i]
        else:
            lref[i] = (close[i-1] - low[i]) - 0.5 * (low[i-1] - high[i])

    tr  = np.maximum(hilo, np.maximum(href, lref))
    atr = np.full(n, np.nan)
    if n > 1:
        atr[1] = tr[1]
    alpha = 1.0 / atr_period    # Wilder smoothing
    for i in range(2, n):
        atr[i] = atr[i-1] + alpha * (tr[i] - atr[i-1])

    return pd.Series(atr, index=df.index)
```

### 3.3 TOS Trailing Stop

Stateful bar-by-bar trailing stop. Starts in "long" state. Flips when close crosses
the trail level. Used as the proximity anchor for MR signals.

**Parameters: `atr_period=5`, `atr_factor=1.5`. Do not change — these control signal sensitivity.**

```python
def tos_trailing_stop(df, atr_period=5, atr_factor=1.5):
    atr   = tos_atr_modified(df, atr_period).values
    close = df["Close"].values.astype(float)
    n     = len(df)
    loss  = atr_factor * atr    # trail distance = 1.5 * TOS_ATR(5)

    state = np.full(n, "init", dtype=object)
    trail = np.full(n, np.nan)

    for i in range(1, n):
        if state[i-1] == "init":
            if not np.isnan(loss[i]):
                state[i] = "long"
                trail[i] = close[i] - loss[i]
            else:
                state[i] = "init"
            continue
        ps, pt = state[i-1], trail[i-1]
        if ps == "long":
            if close[i] > pt:
                state[i] = "long";  trail[i] = max(pt, close[i] - loss[i])  # ratchet up
            else:
                state[i] = "short"; trail[i] = close[i] + loss[i]            # flip
        else:
            if close[i] < pt:
                state[i] = "short"; trail[i] = min(pt, close[i] + loss[i])  # ratchet down
            else:
                state[i] = "long";  trail[i] = close[i] - loss[i]            # flip

    return pd.Series(trail, index=df.index, name="TOS_Trail")
```

**Note:** TOS_Trail is computed across all days together; do NOT reset per session.
The trail level from the previous day's close carries into the next day's open.

### 3.4 Rolling Linear Regression (Slope + R²)

14-bar OLS regression on Close. Vectorized using centered x values.

```python
def rolling_linreg_fast(series, window=14):
    values = series.to_numpy(dtype=float)
    x      = np.arange(window, dtype=float)
    xd     = x - x.mean()        # centered x
    denom  = np.sum(xd ** 2)

    slopes = np.full(len(values), np.nan)
    r2s    = np.full(len(values), np.nan)

    for i in range(window - 1, len(values)):
        y = values[i - window + 1 : i + 1]
        if np.isnan(y).any():
            continue
        yd        = y - y.mean()
        slopes[i] = np.sum(xd * yd) / denom
        c         = np.corrcoef(x, y)[0, 1]
        r2s[i]    = c ** 2 if np.isfinite(c) else np.nan

    return pd.Series(slopes, index=series.index), pd.Series(r2s, index=series.index)
```

Output:
- `lr_slope`: $/bar. Positive = uptrend, negative = downtrend
- `lr_r2`: [0, 1]. Near 1 = linear trend, near 0 = choppy

### 3.5 Feature Assembly

Call on the full multi-day DataFrame before any session filtering:

```python
def add_features(df):
    df = df.copy()
    df["ATR"]       = ta.volatility.average_true_range(
                          df["High"], df["Low"], df["Close"], window=14)
    df["TOS_Trail"] = tos_trailing_stop(df, atr_period=5, atr_factor=1.5)
    slopes, r2s     = rolling_linreg_fast(df["Close"], window=14)
    df["lr_slope"]  = slopes
    df["lr_r2"]     = r2s
    return df.fillna(0)
```

`fillna(0)` is intentional. Rolling warmup NaNs become 0, not dropped. The first 13
bars of each feature are 0 and will not generate signals (slope_sq = 0, no strong zone).

---

## 4. Strategy A — Mean Reversion (MR)

### Parameters
| Name | Value | Description |
|------|-------|-------------|
| `WINDOW` | 14 | OLS regression window (bars) |
| `SLOPE_TANH_K` | 5 | tanh compression factor |
| `R2_SIG_K` | 14 | sigmoid steepness for R² gate |
| `R2_SIG_THR` | 0.50 | sigmoid midpoint |
| `R2_GATE_THR` | 0.50 | threshold on transformed R² |
| `NEW_PROX_MULT` | 2.0 | ATR multiplier for proximity to TOS trail |
| `NEW_MIN_LB` | 8 | bars to look back for strong-zone prior |
| `NEW_COOLDOWN` | 8 | bars to suppress same-direction re-trigger |

### Step-by-Step Algorithm

**Step 1 — ATR-normalized slope → `slope_sq`**
```python
snorm    = dv["lr_slope"] / dv["ATR"].replace(0, np.nan)
slope_sq = np.tanh(5 * snorm).fillna(0)   # bounded to (-1, +1)
slope_sq_prev = slope_sq.shift(1).fillna(0)
```
`tanh(5x)` saturates: at `snorm = ±0.2`, `slope_sq ≈ ±0.84`. The factor 5 means
the indicator is near ±1 for any moderately trending bar.

**Step 2 — R² gate**
```python
r2_gate = (1 / (1 + np.exp(-14 * (dv["lr_r2"] - 0.50)))) > 0.50
```
Sigmoid with k=14, midpoint=0.50. `lr_r2 > 0.50` → `r2_gate = True` (trending).
Checked in the lookback window, NOT at the zero-crossing bar (which always has low R²).

**Step 3 — Proximity filter**
```python
prox = abs(dv["Close"] - dv["TOS_Trail"]) <= dv["ATR"] * 2.0
```
Signal only fires when price is within 2×ATR of the TOS trail. Filters signals when
price is already extended far from its mean-reversion anchor.

**Step 4 — Lookback windows (strong-zone + R²)**
```python
wneg = pd.Series(False, index=slope_sq.index)
wpos = pd.Series(False, index=slope_sq.index)
wr2  = pd.Series(False, index=slope_sq.index)
for k in range(1, 9):   # k = 1 .. 8 bars back
    wneg |= (slope_sq.shift(k) <= -s_thr)
    wpos |= (slope_sq.shift(k) >=  s_thr)
    wr2  |= (r2_gate.shift(k).fillna(False))
```
`wneg[i]` is True if any of the 8 bars before bar i had slope_sq below -s_thr.
This confirms a real downtrend preceded the upward zero-crossing.
8 bars = 40 minutes: captures slow-forming trends that pause before reversing.

**Step 5 — ATR floor gate**
```python
atr_ok = dv["ATR"] >= ATR_FLOOR    # per-ticker, see Section 8
```

**Step 6 — Zero-crossing detection**
```python
zcu = (slope_sq > 0) & (slope_sq_prev <= 0)   # upward crossing
zcd = (slope_sq < 0) & (slope_sq_prev >= 0)   # downward crossing
```

**Step 7 — Combine all conditions**
```python
raw_long  = zcu & wneg & prox & wr2 & atr_ok
raw_short = zcd & wpos & prox & wr2 & atr_ok
```

**Step 8 — Cooldown (8-bar suppression)**
```python
def _cooldown(mask, cd):
    arr = mask.to_numpy(dtype=bool).copy()
    i = 0
    while i < len(arr):
        if arr[i]: arr[i+1:i+1+cd] = False; i += cd + 1
        else: i += 1
    return pd.Series(arr, index=mask.index)

long_sig  = _cooldown(raw_long,  8)
short_sig = _cooldown(raw_short, 8)
```

### Complete `run_new()` Function
```python
def run_new(dv, s_thr, atr_floor=0.0):
    snorm = dv["lr_slope"] / dv["ATR"].replace(0, np.nan)
    s     = np.tanh(5 * snorm).fillna(0)
    s_1   = s.shift(1).fillna(0)
    r2_gate  = (1 / (1 + np.exp(-14 * (dv["lr_r2"] - 0.50)))) > 0.50
    prox     = abs(dv["Close"] - dv["TOS_Trail"]) <= dv["ATR"] * 2.0
    atr_ok   = dv["ATR"] >= atr_floor
    wneg = pd.Series(False, index=s.index)
    wpos = pd.Series(False, index=s.index)
    wr2  = pd.Series(False, index=s.index)
    for k in range(1, 9):
        wneg |= s.shift(k) <= -s_thr
        wpos |= s.shift(k) >=  s_thr
        wr2  |= r2_gate.shift(k).fillna(False)
    zcu = (s > 0) & (s_1 <= 0)
    zcd = (s < 0) & (s_1 >= 0)
    return (_cooldown(zcu & wneg & prox & wr2 & atr_ok, 8),
            _cooldown(zcd & wpos & prox & wr2 & atr_ok, 8))
```

### Signal Interpretation
| slope_sq value | Meaning |
|----------------|---------|
| <= -s_thr | Strong downtrend confirmed |
| -s_thr to 0 | Weakening downtrend |
| crosses 0 upward | **LONG SIGNAL** (if prior downtrend + R² confirmed) |
| 0 to +s_thr | Weakening uptrend |
| >= +s_thr | Strong uptrend confirmed |
| crosses 0 downward | **SHORT SIGNAL** (if prior uptrend + R² confirmed) |

---

## 5. Strategy B — ORB-Slope

### Concept
Fit a linear regression to the first 6 bars (30 minutes) of the session. Normalize
the slope by ATR and convert to degrees via arctan. If the opening angle is steep enough,
the market is trending strongly at open and is likely to continue.

### Parameters
| Name | Value | Description |
|------|-------|-------------|
| `ORB_SLOPE_BARS` | 6 | Number of bars for opening window (6 × 5min = 30 min) |
| `ORB_SLOPE_DEG` | 30.0 | Angle threshold in degrees. tan(30°) ≈ 0.58 ATR/bar |

**Threshold reasoning:** tan(30°) ≈ 0.58 means slope = 0.58 ATR per bar. At 5-min bars
this requires a sustained, significant directional move at the open — not a drift.
tan(45°) = 1.0 ATR/bar is too rare (almost never fires). Sweep across [15, 20, 25, 30, 35]
confirmed 30° as the best balance of trade frequency vs quality.

### Algorithm

```python
ORB_SLOPE_BARS = 6
ORB_SLOPE_DEG  = 30.0

def run_orb_slope(dv, slope_bars=ORB_SLOPE_BARS, angle_deg=ORB_SLOPE_DEG):
    long_sig  = pd.Series(False, index=dv.index)
    short_sig = pd.Series(False, index=dv.index)

    if len(dv) < slope_bars:
        return long_sig, short_sig

    window = dv.iloc[:slope_bars]          # first 6 bars of the session
    closes = window["Close"].values.astype(float)
    atr    = float(window["ATR"].iloc[-1]) # ATR at bar 6

    if atr <= 0 or np.isnan(atr):
        return long_sig, short_sig

    x     = np.arange(slope_bars, dtype=float)
    xd    = x - x.mean()
    slope = np.dot(xd, closes - closes.mean()) / np.dot(xd, xd)   # $/bar
    angle = np.degrees(np.arctan(slope / atr))                      # ATR-normalized degrees

    last_bar = window.index[-1]    # timestamp of bar 6 (the signal bar)
    if angle >= angle_deg:
        long_sig.loc[last_bar]  = True
    elif angle <= -angle_deg:
        short_sig.loc[last_bar] = True

    return long_sig, short_sig
```

### Key Properties
- Fires at the **last bar of the opening window** (bar 6, ~08:55 CT timestamp)
- Entry is at bar 7 open (~09:00 CT)
- At most **one signal per session** (long, short, or none)
- Standalone: holds to 14:30 CT EOD force-close — no intraday flip exits
- No cooldown needed (single signal per day)
- No R² gate, no proximity filter — the angle itself is the quality filter

### ATR Normalization Detail
```
slope_norm = slope / ATR       # dimensionless: ATR-units per bar
angle      = degrees(arctan(slope_norm))

tan(15°) = 0.27 → slope = 0.27 ATR/bar  (gentle)
tan(20°) = 0.36 → slope = 0.36 ATR/bar
tan(25°) = 0.47 → slope = 0.47 ATR/bar
tan(30°) = 0.58 → slope = 0.58 ATR/bar  (threshold used)
tan(35°) = 0.70 → slope = 0.70 ATR/bar  (steep)
tan(45°) = 1.00 → slope = 1.00 ATR/bar  (very rare)
```

### ORB Standalone Simulation
Standalone ORB holds its position to EOD — no MR signals can flip it:
```python
def simulate_orb_standalone(dv, long_sig, short_sig):
    # open position on signal, hold to 14:30 CT EOD close, no flips
    pos = None; ep = None
    times = list(dv.index); opens = dv["Open"].values; n = len(dv)
    for i in range(n):
        t = times[i].time()
        if pos and t >= dtime(14, 30):
            xp = float(opens[i+1]) if i+1<n else float(dv["Close"].iloc[i])
            pnl = (xp - ep) if pos=="long" else (ep - xp)
            return [{"direction": pos, "pnl": round(pnl, 4), "exit": "eod"}]
        if pos is None and i+1 < n:
            if bool(long_sig.iloc[i]):   pos="long";  ep=float(opens[i+1])
            elif bool(short_sig.iloc[i]): pos="short"; ep=float(opens[i+1])
    if pos is not None:
        xp = float(dv["Close"].iloc[-1])
        return [{"direction": pos, "pnl": round((xp-ep) if pos=="long" else (ep-xp), 4)}]
    return []
```

---

## 6. Combined Strategy — MR + ORB with Reversal Gate

### The Problem Without the Gate
On strong trend days, ORB fires correctly (e.g., SPY -39° on Jun 5 2026 = strong short).
But MR, being a mean-reversion system, will see the early price drop and fire a LONG
signal, flipping the ORB short. Result: lose the profitable ORB trade AND lose money
on the wrong-direction MR trade. Example:

```
Jun 5 SPY:  ORB SHORT at 09:00 -> holds all day -> +$10.84 (standalone)
Combined without gate: ORB SHORT flipped to LONG at 09:50 -> -$0.62 + -$11.46 = -$12.08
```

### The Gate: Extreme-Anchored Accumulation Check (updated 2026-06-09)

Before allowing an MR signal to flip an ORB position:

1. Find the **day's extreme** since ORB entry: lowest close for a short ORB, highest for a long ORB
2. Check growing OLS windows **anchored at that extreme** (5-bar first, then 6-bar)
3. A window passes only when **both** angle AND R2 exceed per-ticker thresholds (AND logic)

This asks "is there a clean structured bounce off the day's extreme?" rather than "has price
recovered from the ORB entry?" The original full-span approach was too strict for afternoon
signals: on V-reversal days the long historical drag kept the slope negative even after a
genuine bounce.

```python
def _orb_reversal_confirmed(dv, orb_bar_idx, orb_dir, current_idx,
                             threshold=ORB_REVERSAL_DEG, r2_thr=0.0,
                             accum_start=5, slope_bars=6):
    """
    Returns True if MR is allowed to flip the ORB position.

    orb_bar_idx : session iloc of the ORB signal bar
    orb_dir     : 'long' or 'short'
    current_idx : session iloc of the bar where MR wants to flip
    threshold   : angle threshold in degrees (per-ticker: SPY 20, TSLA 30, QQQ/TQQQ 20, GSPC 10)
    r2_thr      : R2 quality gate (per-ticker: SPY/QQQ/TQQQ/GSPC 0.60, TSLA 0.80)
    """
    if current_idx <= orb_bar_idx:
        return False
    closes_seg = dv["Close"].iloc[orb_bar_idx: current_idx + 1].values.astype(float)
    if len(closes_seg) == 0:
        return False
    atr = float(dv["ATR"].iloc[current_idx])
    if atr <= 0 or np.isnan(atr):
        return False

    # Step 1: find extreme since ORB entry
    rel   = int(np.argmin(closes_seg)) if orb_dir == "short" else int(np.argmax(closes_seg))
    ext_i = orb_bar_idx + rel   # absolute session iloc of the extreme bar

    # Step 2: check accum windows anchored at extreme
    all_closes = dv["Close"].values.astype(float)
    for nb in range(accum_start, slope_bars + 1):
        end = ext_i + nb
        if end - 1 > current_idx:
            break                         # window extends past current bar
        w = all_closes[ext_i: end]
        ang, r2 = _ols_angle_r2(w, atr)
        if r2_thr > 0 and r2 < r2_thr:
            continue                      # R2 fails -> try next window
        if orb_dir == "short" and ang >= threshold:
            return True
        if orb_dir == "long"  and ang <= -threshold:
            return True
    return False
```

### How compute_chart_signals calls the gate

```python
# Per-ticker thresholds are passed — same values used for ORB entry
_orb_reversal_confirmed(session, orb_bar_i, "short", ts_i,
                        threshold=ang_deg,   # e.g. 20 for SPY
                        r2_thr=r2_thr)       # e.g. 0.60 for SPY
```

### Example: SPY Jun 9 2026

```
ORB SHORT fires at 09:05, price = 743.98
Day's low reached at 11:35, price = 723.34
Gate check when MR LONG fires at 12:15:
  extreme at bar 37 (11:35)
  5-bar anchored [11:35-11:55]: ang=13.46, R2=0.469 -> R2 fails
  6-bar anchored [11:35-12:00]: ang=20.18, R2=0.696 -> PASS (ang>=20 and R2>=0.6)
  -> MR LONG at 12:15 is allowed through
```

### Example: SPY Jun 5 2026 (false signal correctly blocked)

```
ORB SHORT fires at 08:50, price = 750.01
Day's low reached at 11:40
Gate check when MR LONG fires at 12:35:
  No 5 or 6 bar window from 11:40 reaches ang>=20 with R2>=0.6
  -> MR LONG at 12:35 is BLOCKED
```

### Per-Ticker Gate Parameters

| Ticker | Angle threshold | R2 threshold | Notes |
|--------|----------------|--------------|-------|
| SPY | 20 deg | 0.60 | |
| TSLA | 30 deg | 0.80 | Higher bar — TSLA reversals are clean when they happen |
| QQQ | 20 deg | 0.60 | |
| TQQQ | 20 deg | 0.60 | |
| ^GSPC | 10 deg | 0.60 | Lower angle — index is smoother |

### Backtest Comparison (45 days, Jun 2026)

| Ticker | Dir | OLD PF (full-span) | NEW PF (extreme-anchored) |
|--------|-----|--------------------|---------------------------|
| SPY | LONG | 1.44 | 1.12 |
| SPY | SHORT | 2.85 | 2.02 |
| TSLA | SHORT | 4.37 | 4.63 (+0.26) |
| QQQ | LONG | 2.36 | 1.48 |
| TQQQ | LONG | 2.95 | 3.33 (+0.38) |
| ^GSPC | TOTAL | 1.89 | 1.38 |

The new gate trades some PF on SPY/QQQ/^GSPC in exchange for catching genuine
V-reversals that the full-span gate blocked. TSLA and TQQQ improve. Further
per-ticker threshold tuning (especially R2) is the next optimization lever.

### Complete `simulate_combined()` Function

```python
def simulate_combined(dv, long_sig, short_sig, lorb, sorb,
                      reversal_deg=ORB_REVERSAL_DEG):
    """
    Combined MR+ORB simulation with ORB reversal gate.

    dv         : session DataFrame with features
    long_sig   : combined long signals  (lmr | lorb)
    short_sig  : combined short signals (smr | sorb)
    lorb       : ORB-only long signals  (for gate detection)
    sorb       : ORB-only short signals (for gate detection)
    reversal_deg: minimum re-computed angle (degrees) to allow MR to flip ORB position
    """
    trades = []
    pos = None        # 'long' | 'short' | None
    ep  = None        # entry price
    et  = None        # entry timestamp
    ei  = None        # entry bar index
    pos_src  = None   # 'orb' | 'mr' — source of the current position
    orb_bar_i = None  # bar index where ORB signal fired (for reversal check)

    times  = list(dv.index)
    opens  = dv["Open"].values
    n      = len(dv)
    CUTOFF = dtime(14, 30)

    # Pre-build set of ORB signal bar indices
    orb_set = set(np.where(lorb.values)[0]) | set(np.where(sorb.values)[0])

    def close_trade(i, reason):
        nonlocal pos, ep, et, ei, pos_src, orb_bar_i
        xp  = float(opens[i+1]) if i+1 < n else float(dv["Close"].iloc[i])
        xt  = times[i+1]        if i+1 < n else times[i]
        pnl = (xp - ep) if pos == "long" else (ep - xp)
        trades.append({
            "entry_time":  et,   "entry_price": round(ep, 4),
            "exit_time":   xt,   "exit_price":  round(xp, 4),
            "direction":   pos,  "pnl":         round(pnl, 4),
            "exit_reason": reason, "bars_held":  i - ei,
            "source":      pos_src,
        })
        pos = ep = et = ei = pos_src = orb_bar_i = None
        return xp, xt, i + 1

    for i in range(n):
        t  = times[i].time()
        sl = bool(long_sig.iloc[i])
        ss = bool(short_sig.iloc[i])

        # ── Attempt to flip a LONG position ───────────────────────────────
        if pos == "long" and ss:
            # Gate: if current position is ORB-sourced and flip signal is MR
            if pos_src == "orb" and i not in orb_set:
                if not _orb_reversal_confirmed(dv, orb_bar_i, "long", i, reversal_deg):
                    continue    # trend not reversed — suppress MR short
            xp, xt, ni = close_trade(i, "flip")
            if t < CUTOFF and ni < n:
                src = "orb" if i in orb_set else "mr"
                pos = "short"; ep = xp; et = xt; ei = ni
                pos_src = src; orb_bar_i = i if src == "orb" else None
            continue

        # ── Attempt to flip a SHORT position ──────────────────────────────
        if pos == "short" and sl:
            if pos_src == "orb" and i not in orb_set:
                if not _orb_reversal_confirmed(dv, orb_bar_i, "short", i, reversal_deg):
                    continue    # trend not reversed — suppress MR long
            xp, xt, ni = close_trade(i, "flip")
            if t < CUTOFF and ni < n:
                src = "orb" if i in orb_set else "mr"
                pos = "long"; ep = xp; et = xt; ei = ni
                pos_src = src; orb_bar_i = i if src == "orb" else None
            continue

        # ── EOD force-close ───────────────────────────────────────────────
        if pos and t >= CUTOFF:
            close_trade(i, "eod")
            break

        if t >= CUTOFF:
            continue

        # ── Open new position ─────────────────────────────────────────────
        if pos is None and i + 1 < n:
            if sl:
                src = "orb" if i in orb_set else "mr"
                pos = "long";  ep = float(opens[i+1]); et = times[i+1]; ei = i+1
                pos_src = src; orb_bar_i = i if src == "orb" else None
            elif ss:
                src = "orb" if i in orb_set else "mr"
                pos = "short"; ep = float(opens[i+1]); et = times[i+1]; ei = i+1
                pos_src = src; orb_bar_i = i if src == "orb" else None

    # Safety close if session ends with open position
    if pos is not None:
        xp  = float(dv["Close"].iloc[-1]); xt = times[-1]
        pnl = (xp - ep) if pos == "long" else (ep - xp)
        trades.append({
            "entry_time": et, "entry_price": round(ep, 4),
            "exit_time":  xt, "exit_price":  round(xp, 4),
            "direction":  pos, "pnl":        round(pnl, 4),
            "exit_reason": "eod", "bars_held": n - 1 - ei,
            "source": pos_src,
        })

    return trades
```

### Main Loop Integration

```python
for d in trading_days:
    dv = session_bars(df, d)
    if len(dv) < WINDOW + 5:
        continue

    # Compute signals
    lsig_mr,  ssig_mr  = run_new(dv, s_thr=NEW_S_THR[ticker], atr_floor=ATR_FLOOR[ticker])
    lsig_orb, ssig_orb = run_orb_slope(dv)

    # Combined signals (merged boolean Series)
    lsig_comb = lsig_mr | lsig_orb
    ssig_comb = ssig_mr | ssig_orb

    # Run combined with reversal gate
    trades = simulate_combined(dv, lsig_comb, ssig_comb, lsig_orb, ssig_orb,
                               reversal_deg=10.0)
```

---

## 7. Trade Execution Rules

These rules apply to all strategies (MR standalone, ORB standalone, combined).

### Entry
- Signal fires at bar `i` → enter at **Open of bar `i+1`**
- If `i` is at or after 14:30 CT: do not enter

### Flip Exit
- Opposite signal fires at bar `j` while position is open → close at Open[j+1]
- If `j` is before 14:30 CT: immediately open opposite at same price (Open[j+1])
- If `j` is at or after 14:30 CT: close only, do not reopen

### EOD Force-Close
- If position open when loop reaches bar at or after 14:30 CT: close at Open[next bar]
- Typically exits at 14:35 CT open (next bar after 14:30)
- Most common exit for ORB trades and TSLA MR trades (trending names)

### P&L
```
Long:  P&L = exit_price - entry_price
Short: P&L = entry_price - exit_price
```

### Session Timeline (5-min bars, Chicago time)
```
08:30 CT — session starts, signals can fire
08:55 CT — ORB signal bar (bar 6, last bar of 30-min opening window)
09:00 CT — ORB entry (open of bar 7)
14:25 CT — last bar that can trigger a new entry (enters at 14:30 open)
14:30 CT — cutoff: force-close trigger, no new entries
14:35 CT — typical EOD exit price (open of bar after 14:30)
15:00 CT — session end
```

---

## 8. Per-Ticker Parameters

| Ticker | `s_thr` | `STRIKE_GAP` | `ATR_FLOOR` | Notes |
|--------|---------|--------------|-------------|-------|
| TSLA | 0.65 | $2.50 | $1.67 | High volatility; s_thr=0.65 keeps quality high |
| SPY | 0.55 | $0.50 | $0.33 | Index ETF; lower threshold captures more reversals |
| NVDA | 0.65 | $1.00 | $0.67 | High vol like TSLA; ORB threshold may need tuning to 35-40° |
| AAPL | 0.60 | $2.50 | $1.67 | Low intraday vol; MR fires rarely; may need lower s_thr |
| QQQ | 0.55 | $1.00 | $0.67 | Index ETF like SPY; consider long-only for MR |

**`STRIKE_GAP`** = minimum P&L that would clear an ATM 0DTE options strike width.
This is used for gap-clear rate reporting, not for signal logic itself.

**`ATR_FLOOR`** = `STRIKE_GAP / 1.5`. Days where ATR is below this value are too
quiet for 0DTE options to clear even one strike — no signals generated.

**Per-ticker threshold rationale:**
- Volatile names (TSLA, NVDA): higher `s_thr` (0.65) ensures only strong prior trends count
- ETFs (SPY, QQQ): lower `s_thr` (0.55) captures smaller but still tradeable reversals
- AAPL: rarely meets MR conditions at 0.60; s_thr=0.50-0.55 would increase frequency

**ORB threshold per ticker:** 30° is validated for TSLA, SPY, QQQ. NVDA is more volatile;
consider 35° to reduce false ORBs (two -$7 to -$9 losers on Jun 2 and Jun 4 in this period).

---

## 9. Backtest Results

**Period:** 42 trading days, April 9 – June 5 2026, 5-minute bars, session hours only.
**Reversal gate:** 10 degrees. **ORB threshold:** 30 degrees.

### MR+ORB Combined (primary strategy)

| Ticker | Trades | Win% | Gap% | Long W% | Short W% | Total P&L | Avg/trade |
|--------|--------|------|------|---------|----------|-----------|-----------|
| TSLA | 28 | 64% | 43% | 62% | 67% | +$83.43 | +$2.98 |
| SPY | 78 | 45% | 33% | 51% | 39% | +$26.50 | +$0.34 |
| NVDA | 41 | 59% | 41% | 65% | 50% | +$19.92 | +$0.49 |
| AAPL | 2 | 100% | 50% | 100% | — | +$4.92 | +$2.46 |
| QQQ | 61 | 48% | 31% | 59% | 38% | +$11.34 | +$0.19 |
| **ALL** | **210** | **52%** | **37%** | | | **+$146.10** | **+$0.70** |

### ORB-Slope Standalone

| Ticker | Trades | Win% | Gap% | Total P&L | Avg/trade |
|--------|--------|------|------|-----------|-----------|
| TSLA | 5 | 60% | 60% | +$42.17 | +$8.43 |
| SPY | 6 | 50% | 50% | +$10.67 | +$1.78 |
| NVDA | 9 | 56% | 44% | -$10.41 | -$1.16 |
| AAPL | 2 | 100% | 50% | +$4.92 | +$2.46 |
| QQQ | 5 | 60% | 60% | +$17.09 | +$3.42 |
| **ALL** | **27** | **59%** | **56%** | **+$64.43** | **+$2.39** |

### MR Standalone

| Ticker | Trades | Win% | Total P&L |
|--------|--------|------|-----------|
| TSLA | 27 | 63% | +$41.95 |
| SPY | 81 | 43% | +$4.25 |
| NVDA | 35 | 57% | +$20.56 |
| AAPL | 0 | — | $0.00 |
| QQQ | 63 | 46% | -$20.75 |
| **ALL** | **206** | **47%** | **+$46.00** |

### Key Findings

1. **TSLA is the strongest ticker** for both strategies. ORB mean +$8.43/trade, MR mean
   +$1.55/trade, combined +$2.98/trade. Short signals slightly outperform longs (67% vs 62%).

2. **ORB is the quality signal** — 27 trades, 59% win, +$2.39/trade vs MR's 206 trades at
   $0.22/trade. ORB trades are high-conviction, hold all day, big winners (TSLA Apr 15 +$22.38,
   QQQ Jun 5 +$16.02, TSLA Jun 5 +$18.92).

3. **NVDA ORB needs higher threshold** — two large losers ($-8.97, $-7.87) suggest NVDA's
   volatility requires 35-40° ORB threshold to filter out weaker openings.

4. **QQQ and SPY MR shorts are problematic** — win rates 38-39%. Index ETFs have upward drift
   bias even intraday. Consider long-only MR mode for ETFs, or raise s_thr for shorts.

5. **Combined beats naive sum** — at 10° gate, combined $+146.10 vs MR+ORB independent sum
   $+110.43. The flip exits on confirmed reversals add $35.67 of synergy value.

6. **AAPL is too quiet** — only 2 ORB signals in 42 days, 0 MR signals. ATR floor and s_thr
   are filtering everything. Lower thresholds or a different ticker choice for AAPL.

---

## 10. Files and Entry Points

```
ibkr_mean_reversion/
├── backtest.py               # Main combined backtest — run this for full results
│   ├── tos_atr_modified()    # TOS-style ATR (used inside TOS trailing stop)
│   ├── tos_trailing_stop()   # Stateful trailing stop
│   ├── rolling_linreg_fast() # 14-bar OLS slope + R²
│   ├── add_features()        # Assembles ATR, TOS_Trail, lr_slope, lr_r2
│   ├── run_new()             # MR signal generation
│   ├── run_orb_slope()       # ORB-Slope signal generation
│   ├── _orb_reversal_confirmed()  # Gate: checks if trend reversed since ORB
│   ├── simulate_trades()     # Standard flip simulation (no gate)
│   ├── simulate_combined()   # MR+ORB with reversal gate (primary)
│   └── session_bars()        # Filter df to one session's trading hours
│
├── download_history.py       # Download + cache 5m OHLCV to data/*.parquet
│                             # Run: python download_history.py [--rebuild]
│
├── sweep_slope_deg.py        # Sweep ORB angle threshold [15,20,25,30,35]
│                             # Validates ORB standalone across 42 days
│
├── sweep_reversal_deg.py     # Sweep reversal gate [0,5,10,15,20,25,30] deg
│                             # Confirmed 10° as optimal for TSLA+SPY
│
├── analyze_mfe.py            # MFE/MAE analysis for ORB trades
│                             # Validates that big winners have tiny MAE
│
├── plot_conflict_day.py      # Visualize MR vs ORB conflict on a specific day
│                             # Default: SPY Jun 5 2026 (clearest example)
│
├── data/
│   ├── TSLA_5m.parquet       # 3,198 bars, Apr 9 – Jun 5 2026
│   ├── SPY_5m.parquet
│   ├── NVDA_5m.parquet
│   ├── AAPL_5m.parquet
│   └── QQQ_5m.parquet
│
└── plots_30d_full/           # Generated charts (one per ticker)
    ├── TSLA_30day.png
    ├── SPY_30day.png
    ...
```

### Running the Full Backtest

```bash
# 1. Download data (if not already cached)
python download_history.py

# 2. Run combined backtest with all 5 tickers
python backtest.py

# 3. (Optional) Re-run sweep to validate gate threshold
python sweep_reversal_deg.py

# 4. (Optional) View a conflict-day plot
python plot_conflict_day.py
```

### Python Environment

The project uses `C:\Program Files\Python311\python.exe` on Windows. The default
`python` in PATH may point to a virtualenv without the required packages. Use the
full path if `python backtest.py` fails with `ModuleNotFoundError`.

---

## 11. Design Decisions and Known Limitations

1. **No intrabar stop-loss.** Only exits are flip signal and EOD force-close. The 0DTE
   options approach provides natural defined risk (premium paid = max loss). Adding
   a stop risks cutting large winners like TSLA Apr 15 (+$22.38).

2. **EOD exits create trend-continuation dependence.** Most P&L comes from trades held
   4+ hours to the 14:35 force-close. The strategy captures intraday trend persistence,
   not just short-term mean reversion.

3. **ORB is trend-following; MR is mean-reverting.** They conflict on trend days. The
   reversal gate resolves this by protecting ORB positions until the trend actually breaks.

4. **Short trades underperform on ETFs.** SPY and QQQ have an upward intraday drift bias.
   Consider long-only for index products, or raise s_thr for short signals specifically.

5. **NVDA ORB at 30° is too loose.** Two large ORB losses ($-8.97, $-7.87) in June 2026
   suggest NVDA's high ATR means 30° is still a relatively gentle opening slope for that
   ticker. Test 35-40° for NVDA specifically.

6. **42-day sample is limited.** Results span one market regime (post-Liberation-Day
   recovery + bull run). The strategy should be validated on at least 6 months of data
   from a paid data provider before live trading.

7. **`fillna(0)` warmup.** The first 13 bars of each feature are 0. This is intentional
   but means the first 13 bars of a session (if the DataFrame starts that day) will have
   no signals. Always compute features on the full multi-day history.

8. **TOS_Trail crosses day boundaries.** It is stateful and carries the trail level from
   one session into the next. Never reset per-day. This is intentional — it provides a
   stable proximity anchor even at the open.

9. **Source tagging in combined.** Each trade has a `source` field: `"orb"` if the
   position was opened by an ORB signal, `"mr"` if opened by MR. When an ORB position
   is flipped by a confirmed MR reversal, the MR leg is tagged `"mr"`. This allows
   per-source P&L analysis.

10. **Windows encoding.** All print statements must use ASCII characters only — no `°`, `±`,
    `→`, `—`, or `★`. Use `deg`, `+-`, `->`, `-`, `*` instead. Python's default print
    on Windows (cp1252) raises `UnicodeEncodeError` otherwise.

---

## 12. Dependencies

```
numpy
pandas
ta           # pip install ta
pyarrow      # pip install pyarrow  (for parquet read/write)
yfinance     # pip install yfinance (data download only)
matplotlib   # pip install matplotlib (plots)
scikit-learn # not used in backtest.py but required by notebook
```

The signal logic itself (`run_new`, `run_orb_slope`, `simulate_combined`) depends only
on `numpy` and `pandas`. If your data pipeline provides the four feature columns
(`ATR`, `TOS_Trail`, `lr_slope`, `lr_r2`), you can call these functions directly
without `ta` or `yfinance`.
