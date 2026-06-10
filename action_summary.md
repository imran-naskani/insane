# ORB Reversal Gate Tuning (2026-06-10)

## Changes

Two targeted improvements to `_orb_reversal_confirmed`:

### 1. Post-extreme anchor (`ext_i + 1`)

Previously the accumulation window started **at** the day's extreme bar (included). Now it starts from the **bar after** the extreme. This removes the bias introduced by the extreme bar itself — the reversal structure should be measured from where price begins moving away from the extreme, not from the turning point.

Effect: ~2-3° steeper angles on 5-bar windows, and cleaner R2 since the extreme bar is typically a low-momentum bar that dilutes the regression.

### 2. Uniform `REVERSAL_ANGLE = 20.0` for all tickers

Previously the reversal gate reused per-ticker ORB entry angles (TSLA: 30°, ^GSPC: 10°, others: 20°). These were too strict for TSLA (blocked valid flips like TSLA Jun 3 10:30 SHORT) and too loose for ^GSPC.

A 45-day sweep across 5 angle thresholds (10/15/20/25/30°) with next-bar accumulation showed 20° as the optimal uniform threshold:
- MR_REV SHORTs: SPY PF=1.61, QQQ PF=1.57, ^GSPC PF=6.60, TSLA 100% win
- MR_REV LONGs: QQQ PF=5.72 (best at 20°), others thin but positive

ORB entry angles (`ORB_SLOPE_DEG`) are **unchanged** — only the reversal gate uses `REVERSAL_ANGLE`.

### Fail safe experiments (not implemented)

Tested two fail safe variants: flip on ORB open breach, and exit-only on ORB bar LOW breach. Both degraded overall stats vs gate only — the breach condition fires too frequently, clearing ORB state and flooding the system with ungated MR signals of lower quality. TSLA was the only ticker where exit-only marginally helped (ALL LONG PF 3.52 → 4.32). Decision: keep gate only.

## Files Changed

- `intraday_signals.py` — `ext_i + 1` anchor; `REVERSAL_ANGLE = 20.0` constant; `compute_chart_signals` passes `REVERSAL_ANGLE`
- `alert_spicy_engine.py` — imports `REVERSAL_ANGLE`; both gate calls pass `threshold=REVERSAL_ANGLE, r2_thr=r2_thr`
- `INSANE-PRO/backend/engine/intraday.py` — same two changes synced (not committed separately)

## Helper Scripts Added

- `_sweep_reversal_gate.py` — updated to accept `--ticker` argument (was SPY-only)
- `_sweep_reversal_angles.py` — sweeps 5 angle thresholds across all tickers, shows MR_REV stats per direction with MFE
- `_test_orb_failsafe.py` — three-way comparison: gate only vs fail safe flip vs fail safe exit, full stats per category

## Gate Summary (production)

| Parameter | Value |
|-----------|-------|
| Anchor | bar AFTER day's extreme (`ext_i + 1`) |
| Window sizes | 5-bar then 6-bar (accum) |
| Angle threshold | 20° uniform (all tickers) |
| R2 gate | per-ticker (TSLA=0.80, others=0.60) |
| Logic | angle AND R2 both must pass |

---

# ORB Reversal Gate Redesign (2026-06-09)

## Problem

The original ORB reversal gate (`_orb_reversal_confirmed`) anchored at the ORB entry bar
and computed a full-span OLS angle from ORB entry to the current bar. On strong trend days
this worked well. But on V-reversal days (e.g. SPY Jun 9 2026: ORB short at 09:05, price
fell from 744 to 724 then recovered to 730 by 12:15), the full-span slope stayed negative
(-13.15 deg) even after a clear bounce — blocking a genuine MR LONG that the user expected
to see. The gate's anchor point was too far back in time.

## New Gate Design

**Extreme-anchored accumulation gate.**

Instead of measuring from the ORB entry bar, the gate:
1. Finds the **day's extreme** since ORB entry: lowest close for a short ORB, highest for a long ORB
2. Checks **growing windows** anchored at that extreme (5-bar, then 6-bar)
3. Requires both **angle >= threshold AND R2 >= r2_thr** to pass (AND logic, not OR)

This asks "is there a clean structured bounce off the day's low?" rather than "has price
fully recovered from the ORB entry?" — a more meaningful reversal detection question.

### Verification on Key Days

| Date | ORB | Extreme | Gate Check | Result |
|------|-----|---------|------------|--------|
| Jun 9 SPY | SHORT 09:05 at 744 | LOW 11:35 at 723.34 | 6-bar from 11:35: ang=20.18, R2=0.696 | PASS — MR LONG at 12:15 allowed |
| Jun 5 SPY | SHORT 08:50 at 750 | LOW 11:40 | No window from low reached 20 deg + R2>=0.6 | BLOCK — false 12:35 long correctly suppressed |

## Backtest Comparison (45 days, all 5 tickers)

Gate variants tested: OLD (full-span, 10 deg, no R2) vs NEW (extreme-anchored, per-ticker ang/R2) vs TUNED (extreme-anchored, 15 deg, R2=0.80 uniform).

Key findings:
- **TSLA**: TUNED gate is best (SHORT PF 4.37 -> 5.39, MFE $5.01 -> $5.53)
- **SPY, QQQ, ^GSPC**: OLD gate outperforms on PF (new gate adds more trades but lower quality)
- **TQQQ**: TUNED returns same stats as OLD (R2=0.80 filters everything back)
- **MFE confirms**: SPY TUNED MR LONG MFE=$0.93 (noise), TSLA TUNED MR SHORT MFE=$5.53 (genuine moves)

Decision: implement the NEW gate (per-ticker angle + R2) as a baseline for user spot-checking
and further optimization. The extreme-anchor logic is directionally correct for all tickers;
threshold tuning (e.g. per-ticker R2 for reversal vs ORB entry) is a future iteration.

## Files Changed

- `intraday_signals.py` — `_orb_reversal_confirmed()` redesigned; `compute_chart_signals()` passes per-ticker ang/R2
- `INSANE-PRO/backend/engine/intraday.py` — same changes synced
- `backtest_intraday_same_bar_close.py` and `backtest_intraday_next_bar_open.py` — unchanged (gate is redundant; signals are pre-filtered by compute_chart_signals)

## Helper Scripts Added (not production)

- `_sweep_reversal_gate.py` — diagnostic: shows gate decision per MR signal for a given date
- `_stats_new_gate.py` — three-way backtest comparison (old/new/tuned) with MFE by trade type

---

# Ranking Workflow Summary (Updated 2026-05-31)

## Objective

Build next-week long/short ranking exports from the full signal history universe, then iterate filters and scoring so outputs are practical for trading.

## What Was Implemented

1. Started with AI analysis ranking and moved to full signal_history universe ranking.
2. Added volume spike features to ranking:
    - VolSpikeCount1w
    - VolSpikeIntensity1w
    - VolumeSpikeAccum (composite)
3. Maintained separate LONG and SHORT rankings with overlap check.
4. Added practical filtering iterations:
    - Price band filter (Close between 10 and 500)
    - Market cap filters (Large cap, then Mid+Large comparisons)
    - Freshness gate experiments (SignalAgeDays <= 14), then removed on request
5. Exported CSV outputs for both base and recent approaches.

## Key Scripts Used

- _rank_chunks_verbose.py: chunked full-universe ranking with progress logging and robust yfinance handling
- _marketcap_topslice.py: market-cap filtered top-slice review
- _midpluslarge_compare.py: Mid+Large vs Large-only comparison
- _midlarge_rescore_check.py: score re-normalization inside filtered universe
- _fresh_largecap_2w.py: freshness plus large-cap analysis snapshot
- _export_recent_csvs.py: export recent-approach CSV files

## Current Export Definition (as of 2026-05-31)

Recent exports currently apply:

- Price filter only: Close in [10, 500]
- Market cap filter: MarketCap >= 10B
- No freshness gate (removed)

## Output Files Generated

- ranking_nextweek_longs.csv
- ranking_nextweek_shorts.csv
- ranking_nextweek_longs_recent.csv
- ranking_nextweek_shorts_recent.csv

## Latest Confirmed Row Counts

- ranking_nextweek_longs_recent.csv: 254 rows
- ranking_nextweek_shorts_recent.csv: 228 rows

## Notes on Volume Fields

- VolSpikeCount1w: count of last-5-session spikes above baseline mean + 1 standard deviation
- VolSpikeIntensity1w: summed positive normalized excess volume above baseline
- VolumeSpikeAccum: weighted composite of count, intensity, and big spikes

## Operational Notes

- Some yfinance symbols intermittently fail or appear delisted; pipeline continues with partial data.
- Pandas bottleneck warning is non-blocking in current runs.

---

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
