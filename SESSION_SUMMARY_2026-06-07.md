# Session Summary — 2026-06-07

## Overview

Replaced the intraday signal engine (alert engine + Streamlit dashboard) with a fully backtested MR + ORB-Slope strategy, verified to reproduce the exact results documented in `SIGNAL_LOGIC.md`. Also added CLAUDE.md, fixed backtest flip logic, and improved chart visualization.

---

## New Files

### `intraday_signals.py`
Shared signal module used by both the alert engine and Streamlit. Contains:
- `rolling_linreg_fast()` — 14-bar rolling OLS regression (slope + R²)
- `add_intraday_features(df)` — adds `lr_slope`, `lr_r2` to any df that has ATR + TOS_Trail
- `run_new(dv, s_thr, atr_floor)` — Strategy A: MR zero-crossing signals with 5 quality gates
- `run_orb_slope(dv)` — Strategy B: Opening Range Breakout via slope angle at bar 6 (~08:55 CT)
- `_orb_reversal_confirmed()` — reversal gate (10°) preventing MR from prematurely exiting ORB positions
- `compute_chart_signals(df, ticker)` — historical chart display helper (no gate, all sessions)
- Per-ticker config: `S_THR`, `ATR_FLOOR` (TSLA, SPY, NVDA, QQQ, AAPL)

### `CLAUDE.md`
Codebase documentation for Claude Code — commands, architecture, key hyperparameters, known issues.

### `SIGNAL_LOGIC.md`
Complete specification of the MR + ORB-Slope strategy including all formulas, parameters, backtest results, and design decisions.

---

## Modified Files

### `alert_spicy_engine.py` — Full replacement
| Before | After |
|---|---|
| 2D Kalman (`spicy_sauce`) + `price_delta` quantile bands | OLS slope zero-crossing (MR) + ORB angle at bar 6 |
| `last_alert = {}` — simple dedup dict | `session_state` per ticker: tracks position, ORB fired, reversal gate |
| Tickers: `["^GSPC", "TSLA", "AAPL"]` | Tickers: `["TSLA", "SPY", "NVDA", "QQQ", "AAPL"]` |
| No EOD exit concept | EOD force-exit alert at 14:30 CT |
| Signal types: TURN_UP / TURN_DOWN / EXIT | Signal types: ORB_LONG / ORB_SHORT / MR_LONG / MR_SHORT + EOD Exit |
| No reversal gate | Gate: MR suppressed if it opposes active ORB and angle reversal < 10° |

Old signal logic kept verbatim as commented-out code with a plain-English summary header.

### `insane.py` — Four targeted changes

**1. Intraday signal block** (5m, 15m, 30m, 1h, 4h)
- Replaced spicy_sauce + quantile band logic with `add_intraday_features()` + `compute_chart_signals()`
- Old code kept as comments under `── OLD: 2D Kalman + quantile band signal logic ──`

**2. Intraday exit block**
- Replaced VWAP/TOS_Trail/RSI exits with position-flip detection + 14:30 CT EOD mask
- Old code kept as comments under `── OLD: VWAP + TOS_Trail + TOS_RSI exit logic ──`

**3. Chart — 4th subplot row**
- Chart expanded from 3 rows to 4 rows
- Row 4: OLS Slope (aqua) + R² (gold dotted) on shared y-axis
- Zero reference line for slope; 0.5 dashed line for R² gate threshold
- Renders for **both intraday and daily** timeframes
- Daily: `add_intraday_features()` called in the daily branch to compute lr_slope/lr_r2

**4. MR vs ORB signal distinction on chart**
- No new legend entries, no new traces
- `Signal_Source` column ('MR' or 'ORB') drives variable marker sizes on existing traces:
  - **ORB signals** → size 22 (larger = high-conviction, holds all day)
  - **MR signals** → size 14 (standard)
- Daily timeframe: no distinction (all size 14)

**5. Smooth line guard**
- Changed `if timeframe != "1m"` → `if timeframe == "1d"` to prevent KeyError on `df["Smooth"]` for intraday

**6. Empty backtest guard**
- Added `if close_df.empty` / `if open_df.empty` guards before date formatting, equity curve traces, and stats tables

### `backtest_intraday_same_bar_close.py`
Added flip re-entry after each exit block. When `Sell_Long` fires on the same bar as `Turn_Down`, the function now exits the long AND enters the short in the same iteration (previously went flat and missed the short entry).

### `backtest_intraday_next_bar_open.py`
Same flip re-entry fix — after exiting at next bar's open, immediately enters opposite direction at that same next-bar open if a flip signal is present.

### `.gitignore`
Added `_*.py` (scratch/helper scripts) and `ranking_*.csv` (output CSVs) to keep them local only.

---

## Backtest Verification

Implementation verified against `SIGNAL_LOGIC.md` expected results for TSLA, Apr 9 – Jun 5 2026 (41 trading days):

| Strategy | Expected | Actual | Match |
|---|---|---|---|
| MR+ORB Combined | 28 trades, 64% win, +$83.43 | 28 trades, 64% win, +$83.43 | ✓ exact |
| ORB Standalone | 5 trades, 60% win, +$42.17 | 5 trades, 60% win, +$42.17 | ✓ exact |
| MR Standalone | 27 trades, 63% win, +$41.95 | 27 trades, 63% win, +$41.95 | ✓ exact |

Long/short win rates also match exactly (62%/67% combined, 69%/57% MR standalone).

---

## Architecture Notes

- **Feature pipeline**: `build_feature_dataset()` provides ATR + TOS_Trail. `add_intraday_features()` adds `lr_slope` + `lr_r2` on top. Called on full multi-day df before session filtering — TOS_Trail is stateful and must carry across day boundaries.
- **Alert engine vs dashboard**: both import from `intraday_signals.py`. Alert engine enforces the reversal gate (live trading). Dashboard uses `compute_chart_signals()` which skips the gate (historical display).
- **Dead code policy**: all replaced code is commented out in-place (not deleted), with a plain-English summary at the top of each commented block for easy reversion.

---

## Commits

| Hash | Description |
|---|---|
| `9cda347` | Replace intraday signal logic with MR + ORB-Slope strategy |
| `a8707dc` | Auto update 2026-06-07 — signal history, AI analysis, daily data |

---

# Session Update — 2026-06-08

## Overview

Three changes to `insane.py`: Weak/Strong signal color coding on both the dashboard signal tables and the chart markers; DeepSeek V4 Pro as a selectable AI model alongside GPT-4o; and an intraday outside-hours toggle checkbox.

---

## Change 1 — Weak/Strong Signal Color Coding

### Problem
`daily_engine.py` correctly saves `"strength": "Weak"` or `"strength": "Strong"` on every signal record in `signal_history/`, but `insane.py` never read the field — dashboard showed all signals identically with no visual distinction.

### Signal Tables
- Added `"Strength"` to the `_row` dict (was silently dropped before)
- Replaced `st.dataframe()` with `pd.Styler` using `apply(..., axis=1)` per-row coloring
- Expander titles now show split counts: `🟢 Long Signals — 45 Strong · 17 Weak`

| Signal | Background | Text |
|---|---|---|
| Strong Long | `#1B5E20` dark green | white |
| Weak Long | `#C8E6C9` light green | dark green |
| Strong Short | `#F9A825` dark amber | dark |
| Weak Short | `#FFF9C4` light yellow | brown |

### Chart Markers (daily 1d only)
- Populated `df["Strength"]` from the frozen signal history loop
- Passed per-point color lists to Plotly `go.Scatter` instead of a single color string

| Marker | Color |
|---|---|
| Strong Long ▲ | `#1B5E20` dark green |
| Weak Long ▲ | `#A5D6A7` light green |
| Strong Short ▼ | `#F9A825` dark amber |
| Weak Short ▼ | `#FFF9C4` light yellow |

Intraday markers unchanged — Weak/Strong has no meaning for intraday signals.

---

## Change 2 — DeepSeek V4 Pro AI Model Selector

### UI
New **"Model"** dropdown added next to the "Generate AI Analysis" button — choices: `GPT-4o` | `DeepSeek`

### Cache — Separate Folders Per Model
| Model | Cache path |
|---|---|
| GPT-4o | `ai_analysis/openai/{ticker}_{timeframe}.json` |
| DeepSeek | `ai_analysis/deepseek/{ticker}_{timeframe}.json` |

Results never overwrite each other. Cache caption shows both date and model: `Cached: 2026-06-05 · gpt-4o`

### API Routing
- **GPT-4o**: vision + text (chart PNG captured and sent, unchanged from before)
- **DeepSeek V4 Pro**: text context only — no vision API on `deepseek-v4-pro`; text context includes close, RSI, swing highs/lows, backtest stats, and all signal history

```python
deepseek_client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),  # from .env
    base_url="https://api.deepseek.com",
)
```

Available models confirmed via live API: `deepseek-v4-pro`, `deepseek-v4-flash`

### Model Isolation
`current_key` now includes the model name → switching models immediately clears `st.session_state.ai_analysis` so GPT-4o's overlay never bleeds into DeepSeek's view and vice versa.

### Bug Fixed — Empty DeepSeek Response
**Root cause:** `deepseek-v4-pro` is a reasoning model. It runs an internal `<think>` chain-of-thought before writing any response, and those reasoning tokens count against `max_tokens`. With the original `max_tokens=2500` and a ~1550-token prompt, reasoning exhausted the entire budget → empty response string saved to cache.

**Diagnosed by:** Live API test showed `reasoning_tokens=353` on a simple 500-token call; confirmed with the full NVDA prompt at `max_tokens=8000` → 3180 chars of content returned correctly.

**Fix:** `max_tokens=8000` for DeepSeek calls only. GPT-4o stays at 2500 (no reasoning overhead).

---

## Change 3 — Intraday Outside Market Hours Checkbox

Checkbox `"Show outside market hours"` added to the left filter panel, visible only when an intraday timeframe is selected.

| State | Effect |
|---|---|
| Unchecked (default) | `df.between_time("08:30", "15:00")` — regular CT session only |
| Checked | All bars shown including pre-market and after-hours |

Filter applied **after** all signal computation so TOS Trail warmup and MR/ORB signal logic still process the full dataset; only the chart view is clipped.

---

## Files Changed This Session

| File | Changes |
|---|---|
| `insane.py` | All 3 changes above — color coding, DeepSeek selector, outside-hours checkbox |

## Folders Created

| Path | Purpose |
|---|---|
| `ai_analysis/openai/` | GPT-4o analysis cache (model-specific) |
| `ai_analysis/deepseek/` | DeepSeek V4 Pro analysis cache |
