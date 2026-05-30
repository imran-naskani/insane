# INSANE Daily Signal Logic — Critique & Improvement Plan

## Overview

The daily signal pipeline runs a 1-D Kalman filter (`secret_sauce`) on the closing price to produce a smoothed trend and slope, then fires LONG/SHORT signals when the slope crosses dynamic quantile thresholds while price is on the correct side of the Kalman smooth. Signals are edge-detected (first bar of the condition flip) and persisted to `signal_history/` JSON files.

---

## Problems Found

### 1. Lookahead Bias — Global Quantile Thresholds (Critical)

**Files:** `insane.py:877`, `daily_engine.py:201`

```python
slope_q = df["Slope"].quantile([0.05, 0.35, 0.5, 0.65, 0.95]).tolist()
```

Quantiles are computed over the **entire dataframe** — past and future. When evaluating bar 100 (e.g. mid-2022), the threshold already knows slope values from 2026. In live trading this information does not exist. Every historical signal on the chart is computed with future knowledge baked into the threshold. All backtests using this logic are contaminated.

---

### 2. Lookahead Bias — Kalman Slope Uses Central Differences (Critical)

**File:** `model.py:93`

```python
slope = np.gradient(smooth)
```

`np.gradient` uses central differences for interior points:
```
slope[i] = (smooth[i+1] - smooth[i-1]) / 2
```

Bar `i`'s slope is computed using bar `i+1`. Every daily signal is therefore one bar ahead of where it appears on the chart.

---

### 3. Rounding Thresholds to Nearest $0.25 Is Fragile

**Files:** `insane.py:878`, `daily_engine.py:202`

```python
slope_vals = [round(x / 0.25) * 0.25 for x in slope_q]
```

Kalman slopes for a $200 stock range roughly $0.05–$1.50/day. Rounding to the nearest $0.25 creates coarse, unstable thresholds — a small data shift can snap the threshold from $0.25 to $0.50, doubling or halving signal frequency overnight. It is also not scale-invariant: thresholds for a $10 stock behave completely differently than a $400 stock, making cross-ticker comparisons meaningless.

---

### 4. Volume Confirmation Ignored

**File:** `build_dataset.py`

`OBV`, `CMF`, and `Volume_Zscore` are all computed but none are used in the daily signal condition. A slope flip on below-average volume is far less reliable than one accompanied by a volume surge. This is a foundational best practice the model skips entirely.

---

### 5. ADX Computed but Not Used — No Regime Filter

**File:** `build_dataset.py:389`

`ADX` is computed but the daily signal fires regardless of whether the market is trending or consolidating. In low-ADX (choppy) markets, Kalman slope signals produce significantly more noise. A simple `ADX > 20` gate would eliminate a large share of false signals at low cost.

---

### 6. `daily_engine.py` Bypasses the Full Feature Pipeline

**File:** `daily_engine.py:236`

`run_daily_engine()` performs a raw `yf.download()` directly instead of calling `build_feature_dataset()`. The batch engine and the interactive app use different data construction paths. If any feature from `build_dataset.py` is ever added to the daily signal condition, the engine silently won't have it and will error or compute wrong signals.

---

### 7. Weak→Strong Signal Promotion Is Brittle

**File:** `daily_engine.py:111–118`

The promotion check looks for `Turn_Up`/`Turn_Down` at the exact same date string. Because quantile thresholds shift every day as new data arrives (global quantiles recalculate), a signal can silently shift one bar earlier or later. It no longer matches the exact date, stays `"Weak"` forever, and stale entries accumulate indefinitely with no cleanup.

---

### 8. "Strength" Label Is Misleading

**File:** `daily_engine.py:79–80`

All historical signals are labelled `"Strong"` except the last one which is `"Weak"`. This label only means "this signal appeared in historical data" — it says nothing about slope magnitude, volume confirmation, or whether the trade was profitable. Calling these "Strong" creates false confidence in marginal signals.

---

## Improvement Suggestions

| Priority | Action | Rationale |
|---|---|---|
| **Critical** | Replace global `.quantile()` with `.expanding().quantile()` | Eliminates the primary lookahead bias. Thresholds at each bar only see data up to that bar, matching live trading conditions. |
| **Critical** | Replace `np.gradient(smooth)` with `np.diff` | Removes the 1-bar-ahead lookahead. Use: `slope = np.concatenate([[0], np.diff(smooth)])` |
| **High** | Unify batch engine to call `build_feature_dataset()` | Single source of truth. Eliminates silent divergence between the app and the daily cron job. |
| **High** | Normalize slope as `Slope / Close` (percentage) | Makes thresholds scale-invariant across tickers. A $0.50/day slope means something very different for NVDA vs. a $10 stock. |
| **High** | Remove `slope_vals` rounding | Use raw quantile values directly. The $0.25 snap is artificial precision that adds instability. |
| **Medium** | Add ADX regime filter (`ADX > 20`) | Suppress signals in choppy markets. Low implementation cost, high false-signal reduction. |
| **Medium** | Add volume confirmation (`Volume_Zscore > 0.5` or `CMF > 0`) | Only take signals where volume supports the move. Filters low-conviction slope flips. |
| **Low** | Replace binary Weak/Strong with a scored signal strength (1–5) | Score based on slope magnitude, volume rank, ADX level. Makes the label actually meaningful. |
| **Low** | Add weekly trend alignment check | Only take longs when the weekly slope is also positive. Reduces counter-trend trades. |
| **Low** | Run a walk-forward validation after fixing lookahead | Fixes #1 and #2 will degrade apparent historical performance — that degradation is the true cost of the bias and should be measured explicitly. |

---

## Code Fix Sketch

```python
# Fix 1: expanding quantiles (no lookahead)
df["q05"] = df["Slope"].expanding().quantile(0.05)
df["q35"] = df["Slope"].expanding().quantile(0.35)
df["q65"] = df["Slope"].expanding().quantile(0.65)
df["q95"] = df["Slope"].expanding().quantile(0.95)

df["Slope_Pos"] = (
    (df["Slope"] > (df["q65"] + df["q95"]) / 2) &
    (df["Close"] > df["Smooth"]) &
    (df["Slope"] > df["Slope"].shift(1))
)
df["Slope_Neg"] = (
    (df["Slope"] < (df["q05"] + df["q35"]) / 2) &
    (df["Close"] < df["Smooth"]) &
    (df["Slope"] < df["Slope"].shift(1))
)

# Fix 2: backward difference (no lookahead on slope)
# In secret_sauce(), replace:
#   slope = np.gradient(smooth)
# with:
slope = np.concatenate([[0], np.diff(smooth)])

# Fix 3: normalized slope
df["Slope_Pct"] = df["Slope"] / df["Close"]
# use Slope_Pct for quantile computation instead of raw Slope

# Fix 4: ADX + volume gate
df["Slope_Pos"] = (
    df["Slope_Pos"] &
    (df["ADX"] > 20) &
    (df["Volume_Zscore"] > 0)
)
```
