# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

INSANE is a quantitative trading system that generates daily and intraday signals using Kalman filtering, technical indicators, and backtesting. It targets 0DTE options and intraday derivatives, with a Streamlit dashboard, Telegram alert engines, and optional Interactive Brokers integration.

## Running the System

```powershell
# Daily signal generation (scans ~500 tickers, run after market close)
python daily_engine.py

# Interactive dashboard
streamlit run insane.py

# 5-minute intraday alert engine (runs forever, sends Telegram alerts)
python alert_spicy_engine.py

# Options alert engine (requires IB TWS/Gateway on port 4002)
python alert_spicy_engine_options.py

# Earnings calendar fetch (weekly)
python earnings_scanner.py

# Signal ranking for next-week longs/shorts
python _rank_all_signalhistory_fast.py
```

## Architecture

### Signal Pipeline

Two Kalman filter implementations in [model.py](model.py):
- `secret_sauce(close)` — 1D online Kalman filter; used by `daily_engine.py`
- `spicy_sauce(close)` — 2D batch Kalman filter (joint price + slope state); used by alert engines

Feature engineering lives in [build_dataset.py](build_dataset.py): downloads OHLCV via yfinance, computes 30+ indicators (FRAMA, TOS RSI, TOS ATR, Bollinger Bands, MACD, ADX, Stochastic, OBV, CMF, VWAP, volume z-score, market context).

**Important**: `daily_engine.py` does NOT call `build_feature_dataset()` — it runs its own lighter pipeline directly. The two pipelines are intentionally separate.

### Signal History & Persistence

`daily_engine.py` writes to these files:
- `signal_history/{ticker}.json` — chronological flip ledger per ticker: `{date, signal, strength, close, smooth, slope}`
- `daily_signals.json` — today's LONG/SHORT lists + index trends
- `daily_closes.json` — processed closes for all tickers

Guard: `daily_engine.py` only saves after 3 PM CST to prevent partial-day overwrites.

Signal strength:
- `"Strong"` — flip confirmed by backward recalculation on subsequent run
- `"Weak"` — edge-of-data signal, not yet confirmed

### Backtesting

Four modules covering fill/exit combinations:
- [backtest_same_day_close.py](backtest_same_day_close.py) — signal bar close → same bar close
- [backtest_next_day_open.py](backtest_next_day_open.py) — next bar open → next bar open
- [backtest_intraday_same_bar_close.py](backtest_intraday_same_bar_close.py) — intraday, same-bar fills
- [backtest_intraday_next_bar_open.py](backtest_intraday_next_bar_open.py) — intraday, next-bar fills

All share `compute_trade_stats()` which returns: win rate, profit factor, max drawdown, Sharpe, average hold.

### Streamlit Dashboard ([insane.py](insane.py))

3-column layout: Filters | Plotly chart + backtest results | Signals/Earnings panel.

Chart: 3-row Plotly subplots — Candlestick + Kalman overlay | Volume | RSI.

AI analysis calls GPT-4o vision (charts ≤365 days only) and caches responses to `ai_analysis/{ticker}_{tf}_{date}.json`. The overlay JSON includes support/resistance/target/invalidation levels snapped to actual High/Low candle prices within 3% tolerance.

Index trend bar reads from `daily_signals.json` at startup (not live).

### Intraday Alert Engine ([alert_spicy_engine.py](alert_spicy_engine.py))

- Runs on 5-minute bars; sleeps to next bar boundary between cycles
- Signal gates: `price_delta_shift` crosses rolling 25th/75th percentile (`p_win=84` lookback), confirmed by TOS Trail + TOS RSI (>50 long, <50 short)
- Retries yfinance up to 10× on failure; deduplicates alerts before sending

### IB Options Engine ([alert_spicy_engine_options.py](alert_spicy_engine_options.py))

- Connects to IB TWS/Gateway on port 4002 via `ib_insync`
- Auto-fetches next OTM option prices on entry signal
- Exit: -5% drawdown OR 3-bar stall; forced EOD exit at 14:55

## Key Hyperparameters

**Kalman (both filters):**
- `observation_covariance=5` (higher = smoother)
- `transition_covariance=1`
- `spicy_sauce` uses `dt=0.7` transition scaling for slope dimension

**TOS Trailing Stop** (stateful, carries across days intentionally):
- Period=5, factor=1.5 (ATR-based)

**Daily quantile thresholds:** `[0.05, 0.35, 0.50, 0.65, 0.95]`, rounded to nearest $0.25.

## Known Design Issues

Documented in [action_summary.md](action_summary.md) and [SIGNAL_LOGIC.md](SIGNAL_LOGIC.md):

1. **Lookahead bias** — `daily_engine.py` uses global `.quantile()` over the full price history; fix is `.expanding().quantile()`
2. **Kalman slope lookahead** — `np.gradient` uses central difference (future bars); fix is `np.diff`
3. **$0.25 rounding** of slope thresholds is scale-dependent and fragile
4. OBV, CMF, and ADX are computed but unused in live signal logic

## Environment & Secrets

`.env` file (loaded via `python-dotenv`):
```
TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
TELEGRAM_OPTION_BOT_TOKEN, TELEGRAM_OPTION_CHAT_ID
TWILIO_SID, TWILIO_TOKEN, TWILIO_FROM, ALERT_TO
FINNHUB_API_KEY
```

`.streamlit/secrets.toml`:
```
OPENAI_API_KEY = "sk-proj-..."
```

## Key Dependencies

```
numpy, pandas, pykalman
ta                  # technical analysis indicators
yfinance            # market data
streamlit, plotly   # dashboard
openai              # GPT-4o chart analysis
ib_insync           # Interactive Brokers TWS/Gateway
requests, python-dotenv
```

## Helper Scripts

Prefixed `_*.py` are one-off or experimental utilities:
- `_rank_all_signalhistory_fast.py` — scores all tickers by next-week signal strength
- `_export_recent_csvs.py` — exports ranking CSVs with market-cap + price filters
- `refac_lstm.py` — experimental GAN + BiGRU + CatBoost for next-day SPX prediction (not used in production)

`auto_git_push.ps1` — PowerShell script for scheduled automated commits.
