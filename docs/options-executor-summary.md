# Options Executor — Implementation Summary

**Date:** 2026-06-10  
**Branch:** main  
**Status:** Complete — 26 tests passing, paper trading ready

---

## What Was Built

Automated options execution (entry + exit) on IBKR paper account, triggered by ORB and MR signals from `alert_spicy_engine.py` for **SPY and TSLA only**.

---

## New Files

### `telegram.py`
Centralised Telegram sending for the entire project. Replaces the inline `send_telegram()` functions that previously existed in both alert engines.

- `send_alert(msg, channel="main")` — single public function
- `channel="main"` → `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID`
- `channel="options"` → `TELEGRAM_OPTION_BOT_TOKEN` / `TELEGRAM_OPTION_CHAT_ID`
- All credentials read from `.env` at call time (not at import)
- Swallows all network exceptions — never raises

### `options_executor.py`
Full IBKR options execution engine.

**Pure helpers (fully unit-tested):**
- `_find_atm_strike(strikes, price)` — closest strike to current price
- `_select_expiry(expirations, today, time)` — 0DTE if time ≤ 13:30 and available, else 1DTE, else None
- `_check_exit(entry, current, hwm, trailing_active)` — returns `(should_exit, reason)`; stop loss at 50% of entry takes priority over trailing

**`OptionsExecutor` class:**

| Method | Description |
|--------|-------------|
| `connect()` | Connects to IB TWS/Gateway at `127.0.0.1:4002`, loads persisted positions, starts background thread |
| `on_signal(ticker, direction)` | Called from alert engine — never raises; delegates to `_handle_signal` |
| `_handle_signal()` | Guards: unknown ticker, past 14:00 cutoff, same direction, signal reversal → exit + re-enter |
| `_select_option()` | Queries IB option chain; picks ATM strike; 0DTE/1DTE expiry |
| `_place_order()` | Limit order at bid/ask midpoint; widens 1¢ per retry; 3 retries × 30s |
| `_open_position()` | Full entry flow: price → option → order → streaming subscription → persist → Telegram |
| `_tracking_loop()` | Background daemon; every 5s reads live streaming midpoint; updates HWM; arms/triggers exits |
| `_exit_position()` | Market order SELL; Telegram exit alert with P&L and hold time; logs to CSV |
| `_reconnect()` | Retries IB connect every 30s; re-subscribes streaming for all open positions on success |
| `_save_positions()` | Writes `option_positions.json` (excluding non-serialisable IB objects) |
| `_load_positions()` | On startup: restores open positions; skips expired 0DTE from prior days |
| `_append_trade()` | Appends row to `option_trades.csv` with P&L%, hold time, exit reason |

**Exit priority order:**
1. Hard stop — premium ≤ 50% of entry price
2. EOD — time ≥ 14:55 CT (hard forced exit)
3. Trailing stop — armed at 150% of entry; exits at 75% of high-water mark
4. Signal reversal — opposite `on_signal()` call from alert engine

**Persistence files created at runtime:**
- `option_positions.json` — open positions; reloaded on restart
- `option_trades.csv` — append-only trade log with: ticker, direction, symbol, entry/exit price, P&L%, hold minutes, exit reason, timestamps

### `tests/test_telegram.py`
4 tests: main channel routing, options channel routing, default channel, exception swallowing.

### `tests/test_options_executor.py`
22 tests covering: ATM strike selection, expiry selection (0DTE/1DTE cutoff), exit logic (stop loss priority, trailing stop activation), persistence (JSON serialisation excludes IB objects), trade CSV logging, guard checks (unknown ticker, past cutoff, same direction, signal reversal, exception swallowing).

---

## Modified Files

### `alert_spicy_engine.py`
Five changes:
1. Removed `import os`, `import requests`, `BOT_TOKEN`, `CHAT_ID`, and inline `send_telegram()` function
2. Added `import telegram` and `from options_executor import OptionsExecutor`
3. Executor initialised before main loop — graceful fallback to `_executor = None` if IB is not running (alert engine continues unaffected)
4. `send_telegram(final_msg)` → `telegram.send_alert(final_msg)`
5. Added `_executor.on_signal(ticker, exec_dir)` inside the dedup block, gated on:
   - `_executor is not None`
   - `ticker in ("SPY", "TSLA")`
   - `signal_type in ("ORB_LONG", "MR_LONG", "ORB_SHORT", "MR_SHORT")`

Signal types `MR_ORB_UP_LONG` / `MR_ORB_UP_SHORT` (upgrades, no direction change) do **not** trigger `on_signal`.

---

## Retired Files

`alert_spicy_engine_options.py` → renamed to `_retired_alert_spicy_engine_options.py`

The old engine had its own signal detection (Slope_Pos/Neg Kalman logic, separate from the current ORB/MR strategy), fetched next-OTM option prices without placing real orders, and used 60-second polling. It is fully replaced by `options_executor.py`.

---

## How to Run

```powershell
# Standard operation — IB TWS/Gateway must be running on port 4002 (paper account)
python alert_spicy_engine.py
```

On startup the executor logs:
```
[executor] Connected to IB TWS/Gateway
```

If IB is not running:
```
[executor] Could not connect to IB (...). Options execution disabled.
```
The alert engine continues sending Telegram alerts normally in either case.

---

## Architecture Diagram

```
alert_spicy_engine.py
    ├── every 5m: scan TSLA, SPY, ^GSPC, QQQ, TQQQ
    ├── telegram.send_alert(msg)          ← all tickers, "main" channel
    └── if ticker in {SPY, TSLA} and ORB/MR signal:
            _executor.on_signal(ticker, "long"|"short")

options_executor.py
    ├── on_signal()
    │     └── _handle_signal() → guards → _open_position()
    ├── _open_position()
    │     ├── get underlying price (IB snapshot)
    │     ├── _select_option() → ATM, 0DTE/1DTE
    │     ├── _place_order() → limit order, 3 retries
    │     ├── reqMktData(snapshot=False) → live stream
    │     ├── persist to option_positions.json
    │     └── telegram.send_alert(entry msg, channel="options")
    └── background thread (every 5s)
          ├── read ticker_obj.midpoint() from live stream
          ├── update high-water mark
          ├── arm trailing stop at 150% of entry
          ├── check EOD (14:55) → exit
          ├── check stop loss (50%) → exit
          ├── check trailing stop (75% of HWM) → exit
          └── on exit: market order SELL + Telegram + CSV log

telegram.py
    └── send_alert(msg, channel)
          ├── "main"    → TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID
          └── "options" → TELEGRAM_OPTION_BOT_TOKEN / TELEGRAM_OPTION_CHAT_ID
```

---

## Phased Roadmap

| Phase | Scope | Status |
|-------|-------|--------|
| Phase 1 | Single-leg ATM calls/puts, 0DTE/1DTE, 1 contract, paper account | **Done** |
| Phase 2 | Delta-targeted strikes (0.40Δ), ATR-based strike distance | Pending |
| Phase 3 | Debit vertical spreads, risk-based sizing | Pending |
| Phase 4 | Live account | Pending |

---

## Commits

```
f7e0b9b chore: retire alert_spicy_engine_options.py — replaced by options_executor.py
d9be75b feat: wire OptionsExecutor into alert engine for SPY/TSLA signals
ec5e8a5 feat: implement tracking loop, exit logic, and IB reconnect
9506a3d feat: implement option selection, order placement, and position entry
5264fe8 feat: implement on_signal guard checks with full test coverage
90c7f4e feat: add OptionsExecutor skeleton with persistence and trade log
455501c feat: add options_executor pure helpers with full test coverage
bf62b50 fix: read telegram credentials at call-time; isolate test env vars with patch.dict
35387af fix: reject unknown telegram channels instead of silently falling back
b6634af feat: add standalone telegram module with dual-channel support
```
