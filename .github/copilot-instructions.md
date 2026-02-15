# INSANE Trading Model — AI Agent Instructions

## Project Overview
**INSANE** (*Imran Naskani's Statistical Analysis for Navigating Equities*) is a quantitative trading system that combines Kalman filtering, technical analysis, and Streamlit visualization to generate trading signals across multiple timeframes and tickers.

### Core Architecture

| Module | Purpose |
|--------|---------|
| `build_dataset.py` | Feature pipeline — downloads OHLCV via yfinance, computes indicators (FRAMA, TOS RSI, TOS ATR, TOS Trailing Stop, VWAP, Bollinger Bands, SMAs, MACD, ADX, Stochastic, OBV, CMF) |
| `model.py` | Signal generation — two Kalman filter variants (`secret_sauce`, `spicy_sauce`) that smooth price and produce `Smooth`/`Slope` columns |
| `technical_features.py` | Supervised labeling — four label functions (`label_simple_return`, `label_atr`, `label_trend`, `label_regime`) producing LONG/SHORT/EXIT strings |
| `daily_engine.py` | Daily batch engine — scans all S&P 500 + custom tickers, persists signal flips to `signal_history/` JSON files, writes `daily_signals.json` and `daily_closes.json` |
| `insane.py` | Streamlit live dashboard — 3-column layout (filters | chart+backtest | signals/earnings), interactive Plotly charts, side-by-side backtest comparison, GPT-4o chart analysis |
| `alert_spicy_engine.py` | Intraday alert loop — polls yfinance every 5 minutes, runs `spicy_sauce` on 5m bars, sends Telegram on `Turn_Up`/`Turn_Down`/exit signals |
| `alert_spicy_engine_options.py` | Options alert engine — connects to Interactive Brokers via `ib_insync`, scans 5m bars, auto-fetches next OTM option price, tracks active trades with trailing exit logic |
| `backtest_same_day_close.py` | Daily backtest — enter/exit at Close on signal bar; includes `compute_trade_stats()` |
| `backtest_next_day_open.py` | Daily backtest — enter/exit at next bar's Open |
| `backtest_intraday_same_bar_close.py` | Intraday backtest — enter/exit at Close on signal bar; supports `Sell_Long`/`Sell_Short` exit signals + position flipping + sell-only variant with forced EOD exit at 15:55 |
| `backtest_intraday_next_bar_open.py` | Intraday backtest — enter/exit at next bar's Open; same exit/flip logic + sell-only variant |
| `earnings_scanner.py` | Fetches upcoming NASDAQ earnings from Finnhub API → writes `earnings_watchlist.json` |
| `refac_lstm.py` | Experimental — GAN + Bidirectional GRU + CatBoost ensemble for next-day S&P 500 close prediction; uses Finnhub sentiment |
| `frama.py` | Standalone FRAMA implementation (duplicate of the one in `build_dataset.py`) |
| `auto_git_push.ps1` | PowerShell script — auto `git add . && commit && push` with timestamped message |

### Data Flow
1. `build_feature_dataset(ticker, start_date, end_date, timeframe)` downloads OHLCV via `download_price()` → `add_features()` computes 30+ indicators → joins VIX/SPY market context
2. **Daily path**: `daily_engine.py` calls `secret_sauce(close)` → slope quantile thresholds → `Turn_Up`/`Turn_Down` → `snapshot_new_signals_only()` persists flips with Weak→Strong strength promotion
3. **Intraday path**: `alert_spicy_engine.py` / `insane.py` call `spicy_sauce(close)` → `price_delta_shift` crosses rolling 25th/75th quantile + `TOS_Trail` + `TOS_RSI` filter → entry/exit signals
4. Signals feed into backtests (4 modules) or Streamlit UI for visualization
5. **AI analysis path**: Daily chart → PNG capture via `fig.to_image()` → base64 encode → GPT-4o vision API → pattern recognition, signal confidence, key levels, expected move, risk assessment → cached to `ai_analysis/` JSON
6. Alert engines send Telegram messages; options engine additionally fetches IB option chains and tracks live P&L

## Module Deep Dive

### `model.py` — Kalman Filter Variants

#### `secret_sauce(close)` — 1D Online Kalman Filter
- **State**: scalar price (1D)
- **Method**: Streaming `filter_update()` loop (online, not batch) — processes one observation at a time
- **Parameters**: `observation_covariance=5`, `transition_covariance=1`, `initial_state_covariance=1`
- **Returns**: `(smooth, slope)` where `slope = np.gradient(smooth)`
- **Used by**: `daily_engine.py` for daily signals

#### `spicy_sauce(close)` — 2D Batch Kalman Filter
- **State**: `[price, slope]` (2D — jointly estimates level and momentum)
- **Method**: Batch `kf.filter(close)` — processes entire series at once
- **Transition matrix**: `[[1, dt], [0, 1]]` with `dt=0.7` (controls slope contribution to price update)
- **Parameters**: `observation_covariance=8`, `transition_covariance=diag(1, 0.01)`
- **Returns**: `(smooth, slope)` directly from state matrix columns
- **Used by**: `alert_spicy_engine.py`, `alert_spicy_engine_options.py`, `insane.py` for intraday signals

### `build_dataset.py` — Feature Engineering

#### Key Functions
| Function | Description |
|----------|-------------|
| `download_price(ticker, start, end, timeframe)` | Safe yfinance wrapper — fixes MultiIndex columns, renames to standard schema, converts intraday to CST |
| `compute_frama(series, window=16, fc=1, sc=200)` | Fractal Adaptive Moving Average — John Ehlers algorithm. `fc`/`sc` params accepted but unused in implementation |
| `compute_tos_rsi(df, n=14, over_bought=70, over_sold=30)` | ThinkOrSwim-style RSI using `ExpAverage(delta, n)` and `ExpAverage(|delta|, n)` |
| `tos_atr_modified(df, atr_period=10)` | TOS Modified ATR — uses HiLo/HRef/LRef true range + Wilder smoothing |
| `tos_trailing_stop(df, atr_period=10, atr_factor=1.5)` | ATR trailing stop state machine — tracks long/short state, emits `TOS_Buy`/`TOS_Sell` on flips |
| `compute_vwap_with_bands(df, num_dev_up=2, num_dev_dn=-2, anchor="DAY")` | VWAP + standard deviation bands — supports DAY/WEEK/MONTH anchoring, handles zero-volume (index tickers) |
| `add_features(df, p, timeframe)` | Computes all indicators: 3 SMAs, 2 slope regressions, FRAMA, RSI, Stochastic, MACD, ADX, ATR, Bollinger, Volume Z-score, OBV, CMF, Gap, Candle Body, VWAP (intraday only), TOS Trail, TOS RSI(9) |
| `get_market_context(start, end, timeframe)` | Fetches VIX close + SPY return + SPY 20-day volatility as context features |
| `build_feature_dataset(ticker, start_date, end_date, timeframe)` | Master builder — downloads OHLCV → `add_features()` → joins market context. Does NOT dropna (deferred to caller) |
| `floor_5_or_int(x)` | Rounds down to nearest 5 (or nearest 1 if <5) — used for VWAP/range thresholds |

#### Feature Parameter Defaults (`feature_params` dict)
```python
sma_short=20, sma_mid=50, sma_long=200
rsi_length=14, stoch_length=14, stoch_signal=3
macd_fast=12, macd_slow=26, macd_signal=9
adx_length=14, atr_length=14, bollinger_length=20
volatility_length=20, volume_zscore_length=20
trend_slope_short=5, trend_slope_long=20
```

### `daily_engine.py` — Daily Signal Engine

#### Signal Logic (uses `secret_sauce`)
1. Compute `Slope` quantiles: `[0.05, 0.35, 0.50, 0.65, 0.95]` → rounded to 0.25 increments
2. `Slope_Neg`: Slope < midpoint(q05, q35) AND Close < Smooth AND Slope decreasing
3. `Slope_Pos`: Slope > midpoint(q65, q95) AND Close > Smooth AND Slope increasing
4. `Turn_Up` / `Turn_Down`: First bar where `Slope_Pos` / `Slope_Neg` becomes True

#### Signal Strength (Weak → Strong Promotion)
- New flips at edge-of-data start as **Weak** (only backward Slope available)
- On next run, if the flip still exists after recalculation (central difference), it's promoted to **Strong**
- Vanished Weak signals are never deleted (immutable audit trail)

#### Ticker Universe
Union of: S&P 500 constituents (fetched from GitHub CSV) + `{SPY, QQQ, ^GSPC, ^IXIC, ^RUT, ^VIX}` + existing `signal_history/` tickers + `earnings_watchlist.json` tickers

#### Outputs
- `signal_history/{ticker}.json` — chronological ledger of every UP/DOWN flip with `{date, signal, strength, close, smooth, slope}`
- `daily_signals.json` — today's LONG/SHORT lists + last signal for each index (SPY/QQQ/SPX/NASDAQ/Russell/VIX)
- `daily_closes.json` — `{date, closes: {ticker: price}}` for all processed tickers

#### Persistence Guard
Signal history is only saved to disk **after 3 PM CST** (prevents partial-day overwrites via `pytz.timezone('America/Chicago')`)

### `alert_spicy_engine.py` — Intraday Telegram Alerts

#### Signal Logic (uses `spicy_sauce`, 5m bars)
1. `price_delta = Close - Smooth`
2. `price_delta_shift = price_delta - price_delta.shift(1)` (rate of change of deviation)
3. Entry conditions (all must be true):
   - `price_delta_shift` crosses rolling 25th/75th percentile (`p_win=84` bars)
   - Price above/below `TOS_Trail`
   - Sufficient range: `vwap_range >= threshold` OR `today_range >= threshold`
   - `TOS_RSI` above 50 (for longs) or below 50 (for shorts)
4. `Turn_Up`/`Turn_Down`: First bar where entry conditions become True

#### Exit Logic
- **Sell_Long**: VWAP Upper cross-under, OR TOS Trail cross-under, OR RSI 70 cross-under
- **Sell_Short**: VWAP Lower cross-over, OR TOS Trail cross-over, OR RSI 30 cross-over

#### Alert Deduplication
Tracks `last_alert = {ticker: (bar_timestamp, signal_type)}` — blocks repeated identical signals and back-to-back EXIT alerts

#### Infrastructure
- Sleeps until next 5m boundary + 2s offset
- Up to 10 retries with 2s sleep on yfinance failures; skips ticker on total failure
- Sends combined Telegram message if multiple tickers trigger in same bar
- Default tickers: `["^GSPC", "TSLA", "AAPL"]`

### `alert_spicy_engine_options.py` — IB Options Alert Engine

#### Architecture
- Connects to Interactive Brokers TWS/Gateway via `ib_insync` (port 4002, clientId 111)
- **Dual-mode loop**: SCAN mode (every 5m) + TRACK mode (every 1 minute for active trades)

#### Instruments (configured via `INSTRUMENTS` dict)
Each instrument specifies: IB contract object, option symbol, multiplier, and `p_win` (quantile window)

#### Signal Logic
Same as `alert_spicy_engine.py` (spicy_sauce + price_delta_shift + TOS_Trail + TOS_RSI + range filters)

#### Option Execution Flow
1. On `Turn_Up`/`Turn_Down` → `fetch_next_otm_option_price()` gets nearest expiry, next OTM strike
2. Stores contract in `active_trades[sym]` with entry price + deque(maxlen=10) for price tracking
3. TRACK mode: polls option snapshot every minute, appends to deque
4. `evaluate_option_exit()`: exits when drawdown ≤ -5% AND stall ≥ 3 bars AND slope ≤ 0

#### Risk Parameters
- `ACCOUNT_CAPITAL = 100_000`, `RISK_PCT = 0.01` (1% risk per trade)
- `MAX_CONTRACTS = 1` (hard cap)
- Forced EOD exit at 14:55

### `insane.py` — Streamlit Dashboard

#### Layout
```
┌──────────────────────────────────────────────────────────────┐
│  INSANE Logo + Title                                         │
│  Index Trend Bar: SPY | QQQ | SPX | NASDAQ | Russell | VIX  │
├──────────┬────────────────────────────────┬──────────────────┤
│ Filters  │  Main Chart + Backtests       │ Signals/Earnings │
│ (col 1)  │  (col 4)                      │ (col 1)          │
├──────────┼────────────────────────────────┼──────────────────┤
│ Ticker   │  Plotly 3-row subplot:        │ Earnings Watchlist│
│ Timeframe│    Row 1: Candlestick+Smooth  │ Long Signals     │
│ Dates    │    Row 2: Volume              │ Short Signals    │
│ Capital  │    Row 3: RSI                 │ (Chat - disabled)│
│ Run Btn  │  Backtest tables (2 cols)     │                  │
│          │  Equity curve comparison      │                  │
│          │  Trade stats tables           │                  │
└──────────┴────────────────────────────────┴──────────────────┘
```

#### Key Behaviors
- Ticker aliases: `SPX` → `^GSPC`, `NASDAQ`/`NDX` → `^IXIC`, `RUSSELL` → `^RUT`, `VIX` → `^VIX`
- Intraday default: last 7 days with 23-day lookback extension for warmup
- Daily default: 5 years back with 290-day lookback extension
- Indicator overlay selector: BB bands, FRAMA, SMAs, VWAP bands, VIX Close
- Two backtest modes run side-by-side: Same-Bar Close vs Next-Bar Open
- Index trend bar reads from `daily_signals.json` + `daily_closes.json` at startup
- Earnings watchlist reads from `earnings_watchlist.json`, enriched with signal history
- `p_win = 84` for all tickers (quantile lookback window)
- OpenAI client instantiated for AI chart analysis (chatbot UI is commented out)
- Active `st.session_state` variables: `run_model` (boolean gate), `ai_analysis` (cached AI response dict), `ai_ticker` (tracks `{ticker}_{timeframe}` to clear stale analysis on change), `ai_overlay_visible` (toggle for chart overlay), `ai_toast` (deferred toast message shown after `st.rerun()`)

#### AI Chart Analysis (Section 8)
GPT-4o vision-based chart analysis available for **daily charts with ≤ 1 year date range**.

**UI Placement**: "🤖 Generate AI Analysis" button appears top-right of chart area via `st.columns([5, 1, 2])`. A "📊 Overlay" checkbox toggle appears in the middle column when overlay data exists. Disabled with tooltip when not eligible (intraday or >365 day range).

**Eligibility**: `ai_eligible = is_daily and date_range_days <= 365`

**Chart Capture**: `fig.to_image(format="png", width=1600, height=900)` via Kaleido → base64 encode → sent as `image_url` to GPT-4o.

**Context Sent to AI**: Ticker, timeframe, date range, current close, momentum trend value (Smooth — referred to as "Momentum Trend" to protect Kalman IP), close vs trend relationship, last signal + date, TOS RSI(9), RSI(14), backtest win rate, profit factor.

**System Prompt**: Instructs AI to act as "INSANE — an expert quantitative trading analyst". **Critical**: AI is explicitly told to NEVER reveal Kalman filtering/smoothing terminology — must refer to the orange line only as "momentum trend line" or "INSANE trend indicator".

**Analysis Output**: 5 sections — (1) Pattern Recognition (classical chart patterns), (2) Signal Confidence (1-10 with justification), (3) Key Price Levels (support/resistance), (4) Expected Move (target + invalidation), (5) Risk Assessment. Additionally, a structured JSON overlay block at the end of the response.

**Overlay JSON Schema** (appended to AI response in ```json fences):
```json
{
  "support_levels": [price1, price2],
  "resistance_levels": [price1, price2],
  "direction": "BULLISH",
  "pattern": {
    "name": "Pattern Name",
    "points": [
      {"date": "YYYY-MM-DD", "price": 123.45, "role": "upper"},
      {"date": "YYYY-MM-DD", "price": 100.00, "role": "lower"}
    ]
  },
  "target_price": 123.45,
  "invalidation_price": 100.00
}
```
- `direction`: `"BULLISH"` or `"BEARISH"` — directional bias; enforces target/invalidation consistency
- `pattern.points[].role`: `"upper"` or `"lower"` — determines which trendline boundary the vertex belongs to; upper and lower boundaries are drawn as separate lines
- Points capped at 3-6 key boundary vertices (not individual price swings)
- Parsed by `parse_overlay_from_response()` which splits markdown analysis from JSON block
- Validated by `validate_overlay()` which fixes swapped target/invalidation, removes nonsensical prices, caps points to 6
- Stored in the `"overlay"` key of the analysis JSON file

**Chart Overlay Drawing** (Phase 2):
Overlays are drawn on the Plotly chart BEFORE `st.plotly_chart()` renders, reading from `st.session_state.ai_analysis["overlay"]`:
- **Support levels**: Green dashed horizontal lines (`#00ff88`) with "S: {price}" labels
- **Resistance levels**: Red dashed horizontal lines (`#ff4444`) with "R: {price}" labels
- **Target price**: Dotted green line with 🎯 emoji label
- **Invalidation price**: Dotted red line with ⛔ emoji label
- **Pattern outline**: Upper and lower boundaries drawn as separate cyan dash-dot lines with diamond markers; pattern name annotated on upper boundary. Falls back to single line if `role` field is missing.

**Price Snapping**: `snap_to_price(target_price, df, tolerance_pct=0.03)` snaps AI-estimated prices to the nearest actual High/Low in the dataframe within 3% tolerance. Improves accuracy since AI reads a rasterized PNG.

**Direction Validation**: `validate_overlay(overlay, current_close)` ensures directional consistency:
- BULLISH: target must be above close, invalidation below (auto-swaps if inverted, nullifies if nonsensical)
- BEARISH: target must be below close, invalidation above

**Overlay Toggle**: "📊 Overlay" checkbox (bound to `st.session_state.ai_overlay_visible`) appears only when overlay data exists. Toggling re-renders the chart with/without overlays.

**Helper Functions** (module-level):
| Function | Description |
|----------|-------------|
| `snap_to_price(target_price, df, tolerance_pct, role)` | Snaps AI price to nearest actual High/Low within tolerance; role-aware (`"upper"` → snap to Highs only, `"lower"` → Lows only) |
| `detect_swing_points(df, window=5, max_points=8)` | Detects swing highs/lows (local max/min in 2×window+1 neighborhood); returns deduplicated lists of `{date, price}` dicts |
| `parse_overlay_from_response(response_text)` | Extracts JSON overlay block from AI response, returns `(clean_markdown, overlay_dict)` |
| `validate_overlay(overlay, current_close)` | Fixes directional inconsistencies in target/invalidation, caps pattern points to 6 |
| `OVERLAY_JSON_INSTRUCTION` | Constant string appended to both fresh and revalidation prompts requesting the structured JSON block |

**Swing Point Context**: `detect_swing_points()` computes recent swing highs and swing lows from the OHLC data. These are appended to the AI context text as `"Swing Highs: date at price, ..."` and `"Swing Lows: date at price, ..."`. The AI prompt instructs: *"For pattern points, you MUST use dates and prices ONLY from the Swing Highs / Swing Lows lists."* This ensures pattern vertices land precisely on real chart points instead of AI pixel-reading approximations.

### Backtesting Modules

#### Daily Backtests
| Module | Entry Fill | Exit Fill | Key Function |
|--------|-----------|-----------|--------------|
| `backtest_same_day_close.py` | Signal bar Close | Signal bar Close | `generate_trade_log(df, capital=10000)` |
| `backtest_next_day_open.py` | Next bar Open | Next bar Open | `generate_trade_log_next_open(df, capital=10000)` |

Both: `Turn_Up` → LONG, `Turn_Down` → SHORT, opposite signal closes + opens new position. Final open trade closed at last bar's Close.

#### Intraday Backtests
| Module | Entry Fill | Exit Fill | Key Functions |
|--------|-----------|-----------|---------------|
| `backtest_intraday_same_bar_close.py` | Signal bar Close | Signal bar Close | `backtest_intraday_close()`, `backtest_intraday_close_sell_only()` |
| `backtest_intraday_next_bar_open.py` | Next bar Open | Next bar Open | `backtest_intraday_next_open()`, `backtest_intraday_next_open_sell_only()` |

Intraday-specific features:
- Use `Sell_Long_Plot` / `Sell_Short_Plot` as independent exit signals (go flat, not flip)
- `Turn_Down` while LONG → close + flip to SHORT (and vice versa)
- **Sell-only variants**: skip extended hours (before 09:30, after 15:55), forced EOD exit at 15:55, only enter on Turn signals, only exit on Sell signals (no flip)

#### `compute_trade_stats()` (in `backtest_same_day_close.py`)
Calculates: win rate, profit factor, max drawdown, Sharpe ratio (annualized √252), average/max hold days — all broken down by total/long/short.

### `technical_features.py` — Supervised Labels

| Label | Logic | Parameters |
|-------|-------|------------|
| `label_simple_return` | Forward 5-day return > 2% → LONG, < -2% → SHORT | `forward=5, thresh=0.02` |
| `label_atr` | Forward 5-day return > 1.5× ATR% → LONG, < -1.5× ATR% → SHORT | `forward=5, atr_mult=1.5` |
| `label_trend` | SMA_Short > SMA_Mid AND FRAMA slope > 0 → LONG (inverse for SHORT) | Uses existing SMA/FRAMA columns |
| `label_regime` | Uptrend + RSI > 55 + low volatility → LONG (inverse for SHORT) | BB_Width vs 20-bar rolling mean |

`add_all_labels(df)` applies all four and encodes 1/-1/0 → LONG/SHORT/EXIT strings.

### `earnings_scanner.py`
- Calls Finnhub `/calendar/earnings` for next 7 days
- Filters to NASDAQ-listed symbols only (via `/stock/symbol` MIC=XNAS)
- Writes `earnings_watchlist.json`: `{generated, from_date, to_date, count, watchlist: [{symbol, earnings_date, hour, eps_estimate, revenue_estimate}]}`
- Designed for weekly scheduling (e.g., Sunday via Windows Task Scheduler)

### `refac_lstm.py` — Experimental ML Predictor
- **Title**: "S&P 500 Daily Predictor with GAN + GRU + CatBoost Ensemble"
- **Target**: `Slope_Long` (50-period linear regression slope of close)
- **Architecture**: LSTM-based GAN for data augmentation (25% synthetic ratio) → 3-layer Bidirectional GRU (256→128→64) with Huber loss → CatBoostRegressor → averaged ensemble with rolling bias correction
- **Additional data**: Finnhub social sentiment (Reddit/Twitter)
- **Stored models**: `lstm/*.h5` files for ^GSPC, TSLA, QQQ, TMUS, COIN

## Environment Variables & Secrets

| Variable | Source | Used By |
|----------|--------|---------|
| `TELEGRAM_BOT_TOKEN` | `.env` | `alert_spicy_engine.py` |
| `TELEGRAM_CHAT_ID` | `.env` | `alert_spicy_engine.py` |
| `TELEGRAM_OPTION_BOT_TOKEN` | `.env` | `alert_spicy_engine_options.py` |
| `TELEGRAM_OPTION_CHAT_ID` | `.env` | `alert_spicy_engine_options.py` |
| `OPENAI_API_KEY` | `st.secrets` | `insane.py` (AI chart analysis via GPT-4o vision) |
| Finnhub API key | embedded/env | `earnings_scanner.py`, `refac_lstm.py` |

## Data Persistence

| File | Format | Written By | Contents |
|------|--------|-----------|----------|
| `signal_history/{ticker}.json` | JSON array | `daily_engine.py` | Chronological ledger of every UP/DOWN flip: `{date, signal, strength, turn_up, turn_down, close, smooth, slope}` |
| `daily_signals.json` | JSON object | `daily_engine.py` | `{date, long: [...], short: [...], nasdaq: {signal, date}, spy: ..., spx: ..., qqq: ..., russell: ..., vix: ...}` |
| `daily_closes.json` | JSON object | `daily_engine.py` | `{date, closes: {ticker: price}}` for all processed tickers |
| `earnings_watchlist.json` | JSON object | `earnings_scanner.py` | Upcoming NASDAQ earnings with EPS/revenue estimates |
| `ai_analysis/{ticker}_{tf}_{date}.json` | JSON object | `insane.py` | AI chart analysis cache: `{ticker, timeframe, date, date_range, last_signal, last_signal_date, current_close, model, prompt, response}` |

## Key Developer Workflows

### Adding a New Ticker
1. Add ticker symbol to `TICKERS` list in `alert_spicy_engine.py` or type it in the Streamlit UI
2. `build_feature_dataset(ticker)` auto-fetches from yfinance; no manual registration needed
3. For daily engine: tickers in `signal_history/` are auto-included in the next run
4. Signal history auto-initialized when snapshot functions run (lazy creation in `signal_history/`)

### Deploying Live Alerts
- **Equity alerts**: Run `alert_spicy_engine.py` as persistent process — retries yfinance failures up to 10x
- **Options alerts**: Run `alert_spicy_engine_options.py` — requires IB TWS/Gateway on port 4002
- Both require `.env` with Telegram credentials
- Both sleep until next 5m boundary + offset for data availability

### Modifying Signal Logic

#### Kalman Parameters (`model.py`)
- `observation_covariance`: Higher = assumes price is noisier = smoother output = fewer signals
- `transition_covariance`: Higher = more reactive trend = more signals
- `spicy_sauce` `dt` (transition matrix): Lower = slope contributes less to price update

#### Entry Signal Tuning
- **Daily** (`daily_engine.py`): Adjust slope quantile thresholds (currently 5th/35th and 65th/95th percentiles)
- **Intraday** (`alert_spicy_engine.py`): Adjust `p_win` (quantile window, default 84), `q01`/`q99` quantile levels (currently 0.25/0.75), TOS RSI threshold (50), TOS Trailing Stop params (period=5, factor=1.5)

#### Exit Signal Tuning (Intraday only)
- VWAP band cross: requires `vwap_range >= threshold`
- TOS Trail cross: independent of range
- RSI reversal: 70 cross-under (exit long), 30 cross-over (exit short)

### Adding a New Technical Indicator
1. Add computation in `add_features()` in `build_dataset.py`
2. Reference in signal logic (`daily_engine.py` or `alert_spicy_engine.py`)
3. Optionally add to Streamlit indicator overlay list in `insane.py`

### Adding a New Label/Classifier
1. Create function in `technical_features.py` following `label_*()` pattern
2. Return `pd.Series` of 1/-1/0
3. Add to `add_all_labels()` for auto-inclusion in supervised datasets

## Common Debugging

### If signals are too noisy
- Increase `observation_covariance` in Kalman filter (trusts price less)
- Raise quantile thresholds (e.g., 0.25/0.75 → 0.10/0.90 for intraday `q01`/`q99`)
- Increase `p_win` for wider lookback window
- Add regime filter via `label_regime` check

### If signals lag price action
- Decrease `observation_covariance` (trusts price more)
- Increase `spicy_sauce` `dt` parameter (slope has more influence)
- Lower `p_win` for more responsive quantile bands
- Compare same-bar vs next-bar backtest results to isolate timing bias

### yfinance Data Issues
- `download_price()` fixes MultiIndex columns automatically
- Intraday data converted to `America/Chicago` timezone
- Zero-volume bars (common for indices like SPX) handled by VWAP replacing 0 with 1
- Alert engine retries 10x with 2s sleep; skips ticker on total failure

### IB Connection Issues (`alert_spicy_engine_options.py`)
- Disconnects existing connection before reconnecting (safe reconnect pattern)
- Port 4002 = IB Gateway paper; port 7497 = TWS paper
- `clientId=111` — change if running multiple instances
