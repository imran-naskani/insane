# INSANE Trading Model — AI Agent Instructions

## Project Overview
**INSANE** (*Intelligent Statistical Algorithm for Navigating Equities*) is a quantitative trading system that combines Kalman filtering, technical analysis, and Streamlit visualization to generate trading signals across multiple timeframes and tickers.

### Core Architecture
- **Feature Pipeline** (`build_dataset.py`): Downloads OHLCV data from yfinance, computes technical indicators (FRAMA, TOS RSI, ATR, Bollinger Bands, SMAs)
- **Signal Generation** (`model.py`): Kalman filter variants (`secret_sauce`, `spicy_sauce`) smooth price data and detect trend reversals via slope analysis
- **Labeling System** (`technical_features.py`): Multi-label approach (Return-based, ATR, Trend, Regime) used for backtesting and supervised learning
- **Backtesting** (`backtest_*.py`): Four distinct modules for daily & intraday execution (same-bar close, next-bar open) with drawdown/Sharpe metrics
- **Live Dashboard** (`insane.py`): Streamlit app with signal history snapshots (`signal_history/*.json`), trade logs, and OpenAI integration
- **Alerting Engines** (`alert_spicy_engine.py`, `alert_spicy_engine_options.py`): Telegram notification loops for 5m intraday signals; retry logic for data failures

### Data Flow
1. `build_feature_dataset()` fetches historical data + computes indicators (stored in-memory or cached)
2. `secret_sauce(close)` / `spicy_sauce(close)` apply Kalman smoothing → slope detection → `Turn_Up`/`Turn_Down` boolean signals
3. Signals fed to backtests (`generate_trade_log()`) or Streamlit UI for visualization
4. Daily snapshot logic in `daily_engine.py` persists signal flips to JSON ledger (only after 3 PM CST)
5. Alert engines poll every 5 minutes → send Telegram messages on signal changes

## Key Developer Workflows

### Adding a New Ticker
1. Add ticker symbol to `TICKERS` list in `alert_spicy_engine.py` or Streamlit UI
2. `build_feature_dataset(ticker)` auto-fetches from yfinance; no manual registration needed
3. Signal history auto-initialized when snapshot functions run (lazy creation in `signal_history/`)

### Backtesting a Strategy Change
```bash
# Single ticker, same-day close execution:
python backtest_same_day_close.py
# Then modify TICKERS or logic in backtest file

# Multi-timeframe comparison:
# Edit backtest_intraday_same_bar_close.py for 5m signals
# Compare results side-by-side with daily closes
```

### Deploying Live Alerts
- `alert_spicy_engine.py` runs as persistent loop (retry logic handles yfinance failures)
- Requires `.env`: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` (for Telegram)
- Sleeps until next 5m boundary + 2s offset for data availability
- Sends combined messages if multiple tickers trigger in same bar

### Modifying Signal Logic
- **Kalman parameters** in `model.py`: `transition_covariance` (lower = smoother), `observation_covariance` (higher = trusts price less)
- **Technical indicator inputs**: Adjust `window`, `fc`, `sc` in `compute_frama()`, RSI periods in `compute_tos_rsi()`
- **Turn_Up/Turn_Down detection**: Based on `Slope` sign change and smoothed price (`Smooth`) crossover logic in backtests

## Project-Specific Conventions

### Naming & Signals
- **Turn_Up / Turn_Down**: Boolean columns indicating trend reversal; primary entry signals
- **Smooth / Slope**: Kalman-filtered price & gradient; used for reversal confirmation
- **Labels (Return/ATR/Trend/Regime)**: Four classification schemes; combine for ensemble-like approach
- **LONG/SHORT/EXIT**: Encoded labels for supervised models

### Data Persistence
- **Signal History**: `signal_history/{ticker}.json` = chronological ledger of every flip (LONG/SHORT with price, slope, date)
- **Daily Closes**: `daily_closes.json` = cached snapshot of current day's OHLCV
- **Daily Signals**: `daily_signals.json` = list of all tickers w/ most recent signal + timestamp
- **Backups**: `backups/` folder = versioned Python files (auto-maintained via Git)

### Kalman Filter Variants
- `secret_sauce()`: 1D filter (price only) + gradient → baseline trend detection
- `spicy_sauce()`: 2D filter (price + slope) → more responsive to momentum shifts; default for intraday

### Backtesting Logic
All backtests assume **fixed capital per trade** (e.g., $10k) + **no leverage**:
- Entry: `Turn_Up` → LONG, `Turn_Down` → SHORT
- Exit: Opposite signal closes position + logs PnL
- Metrics: Win/loss count, avg return, Sharpe ratio, max drawdown
- **Time boundaries**: Same-bar vs next-bar affects realized entry/exit prices

## Critical Integration Points

### yfinance Failure Handling
- `alert_spicy_engine.py` retries up to 10x with 2s sleep between attempts
- If all retries fail, ticker is **skipped for that cycle** (loop continues)
- Exception messages logged to terminal; Telegram only sent on successful signal

### Streamlit Session State
- `st.session_state.run_model`: Boolean flag for manual model trigger
- Chat/LLM integration via OpenAI API (requires `st.secrets["OPENAI_API_KEY"]`)
- Auto-refresh via `streamlit-autorefresh` plugin every 60s

### CST Time Zone Handling
- Signal snapshots only persist **after 3 PM CST** (prevents intraday overwrites)
- All timestamps in `daily_engine.py` convert to America/Chicago timezone
- Critical for coordinating with market close times

## Common Debugging & Extension Points

### If signals are too noisy:
- Increase `transition_covariance` in Kalman filter (e.g., 0.1 → 1.0)
- Widen FRAMA window or adjust `fc`/`sc` parameters
- Add regime filter (check volatility or RSI overbought/oversold before taking signal)

### If signals lag price action:
- Decrease `observation_covariance` (trust price more)
- Switch to `spicy_sauce()` for 2D momentum tracking
- Check next-bar vs same-bar backtest to identify timing bias

### To add a new label/classifier:
- Create function in `technical_features.py` following `label_*()` pattern
- Return pd.Series of 1/-1/0 (or LONG/SHORT/EXIT strings)
- Add to `add_all_labels()` for auto-inclusion in supervised datasets

### To integrate with new broker API:
- Replace yfinance calls in `build_feature_dataset()` with broker's SDK
- Ensure output DataFrame matches schema: Index=dates, Columns=[Open, High, Low, Close, Volume, +indicators]
- Update alert engines to handle broker-specific rate limits/errors
