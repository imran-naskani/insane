# Options Executor Design

**Date:** 2026-06-10  
**Scope:** Automate options entry/exit on IBKR paper account for SPY and TSLA, driven by ORB/MR signals from `alert_spicy_engine.py`.

---

## Goals

- Place real options orders on IBKR (paper account) when ORB or MR signals fire for SPY or TSLA
- Exit positions automatically based on premium-based rules
- Keep the alert engine unaffected if IB goes down
- Avoid code duplication (Telegram, signal logic)

## Out of Scope (Phase 1)

- Vertical spreads (debit or credit)
- Delta-targeted strikes
- Risk-based position sizing
- Live account

---

## Components

| File | Status | Responsibility |
|------|--------|----------------|
| `telegram.py` | New | Standalone `send_alert(msg, channel)` — owns both bot tokens/chat IDs |
| `options_executor.py` | New | `OptionsExecutor` class — all IB interaction, option selection, order placement, position tracking, exits |
| `alert_spicy_engine.py` | Modified | Import executor + telegram. Call `executor.on_signal(ticker, direction)` for SPY/TSLA only. Replace inline `send_telegram()` with `telegram.send_alert()` |
| `alert_spicy_engine_options.py` | Retired | Replaced entirely by `options_executor.py` |

---

## Signal Flow

```
alert_spicy_engine.py
    ├── ORB/MR signal fires for any ticker
    ├── telegram.send_alert(...)               ← Telegram alert (all tickers)
    └── if ticker in {"SPY", "TSLA"}:
            executor.on_signal(ticker, "long"|"short")

options_executor.py
    ├── on_signal() → IB order placement       ← called from alert engine thread
    ├── background thread → 5s tracking loop  ← live streaming data
    └── telegram.send_alert(..., channel="options")
```

---

## telegram.py

```python
send_alert(msg, channel="main")
```

- `channel="main"` → `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID`
- `channel="options"` → `TELEGRAM_OPTION_BOT_TOKEN` / `TELEGRAM_OPTION_CHAT_ID`
- Loaded from `.env` via `python-dotenv`
- All callers use this; no inline `send_telegram()` anywhere

---

## OptionsExecutor

### Initialization

```python
executor = OptionsExecutor()   # called once before alert engine main loop
```

- Connects to IB TWS/Gateway at `127.0.0.1:4002`, clientId=111
- Loads `option_positions.json` if present → re-subscribes streaming data for any recovered open positions
- Cancels any pending unconfirmed orders from before restart
- Starts background tracking thread (daemon)

### Ticker Config (internal)

```python
EXECUTOR_TICKERS = {
    "SPY":  {"contract": Stock("SPY",  "SMART", "USD")},
    "TSLA": {"contract": Stock("TSLA", "SMART", "USD")},
}
```

`on_signal()` silently ignores any ticker not in this dict.

---

## on_signal(ticker, direction)

Called from the alert engine's main thread. Must never raise — all exceptions caught internally.

### Guard Checks (no IB calls)

1. `ticker` not in `{"SPY", "TSLA"}` → return
2. Open position exists for ticker:
   - Same direction → return (already in trade)
   - Opposite direction → exit existing position first, then continue to entry
3. Current time > 14:00 → return (too late for 0DTE entries)
4. Market closed → return

### Option Selection (IB calls)

1. Request option chain via `reqSecDefOptParams`
2. **Expiry**: prefer today's 0DTE; fallback to tomorrow's 1DTE if 0DTE unavailable or time > 13:30
3. **Strike**: `min(strikes, key=lambda s: abs(s - current_price))` — ATM
4. **Right**: `CALL` for long, `PUT` for short

### Order Placement

1. Get bid/ask snapshot → place limit order at midpoint
2. Poll for fill every 30s; widen by 1 tick each retry
3. Cancel after 3 retries (90s) if unfilled
   - Send Telegram: `SPY | CALL entry missed — unfilled after 90s`
4. On fill:
   - Record position in `_positions` dict
   - Subscribe streaming data: `reqMktData(contract, "", False, False)`
   - Persist to `option_positions.json`
   - Send Telegram entry notification

**Entry Telegram (channel=options):**
```
SPY | CALL ATM | Strike: 595 | 0DTE | Fill: $1.50 | Signal: ORB Long
```

---

## Position State (per ticker)

```python
{
    "contract":        Option(...),      # IB contract object
    "symbol":          "SPY...",         # localSymbol string
    "direction":       "long",
    "entry_price":     1.50,
    "high_water_mark": 1.50,
    "trailing_active": False,            # True once up 50%
    "fill_time":       datetime,
    "contracts":       1,
    "ticker_obj":      Ticker(...),      # live streaming subscription
}
```

---

## Background Tracking Thread

Runs every **5 seconds**. Uses live streaming data (no new IB snapshot requests).

For each open position:
1. Read `ticker_obj.midpoint()` (bid/ask midpoint from live stream)
2. Update `high_water_mark` if current > previous high
3. Activate trailing if `current >= entry * 1.50` → set `trailing_active = True`
4. Evaluate exits in priority order:

| Priority | Rule | Condition |
|----------|------|-----------|
| 1 | Hard stop | `current <= entry * 0.50` |
| 2 | EOD | Time >= 14:55 |
| 3 | Trailing stop | `trailing_active AND current <= high_water_mark * 0.75` |
| 4 | Signal reversal | `on_signal()` called with opposite direction |

### Exit Execution

1. Place **market order** to close
2. Wait for fill confirmation
3. Cancel streaming subscription (`cancelMktData`)
4. Remove from `_positions`
5. Append to `option_trades.csv`
6. Persist updated `option_positions.json`
7. Send Telegram exit notification

**Exit Telegram (channel=options):**
```
SPY | CALL EXIT | Reason: Trailing Stop
Entry: $1.50 → Exit: $2.30 | P&L: +53.3% | Hold: 1h 45m
```

---

## Persistence

### option_positions.json

Written after every state change. Format:

```json
{
  "SPY": {
    "symbol": "SPY240610C00595000",
    "direction": "long",
    "entry_price": 1.50,
    "high_water_mark": 1.80,
    "trailing_active": false,
    "fill_time": "2026-06-10T09:45:00",
    "contracts": 1
  }
}
```

On startup: if fill_time date < today AND position is 0DTE → skip (expired), remove entry.

### option_trades.csv

Append-only. One row per completed trade:

| ticker | direction | option_symbol | entry_price | exit_price | pnl_pct | hold_minutes | exit_reason | entry_time | exit_time |
|--------|-----------|---------------|-------------|------------|---------|--------------|-------------|------------|-----------|

---

## IB Reconnect

- If IB disconnects: background thread detects `not ib.isConnected()`, retries `ib.connect()` every 30s
- On reconnect: re-subscribes streaming data for all open positions
- `on_signal()` during disconnect: catches exception, logs, sends Telegram warning, returns without placing order

---

## Changes to alert_spicy_engine.py

1. Add at top: `import telegram` and `from options_executor import OptionsExecutor`
2. Before main loop: `executor = OptionsExecutor()`
3. Replace all `send_telegram(msg)` → `telegram.send_alert(msg)`
4. After each ORB/MR signal block where Telegram alert is sent, add:
   ```python
   if ticker in ("SPY", "TSLA"):
       executor.on_signal(ticker, state["position"])  # "long" or "short"
   ```
5. No other changes — signal logic, state machine, all other tickers unchanged

---

## Phased Rollout

| Phase | Scope |
|-------|-------|
| Phase 1 (now) | Single leg ATM calls/puts, 0DTE/1DTE, 1 contract fixed, paper account |
| Phase 2 | Delta-targeted strikes (0.40Δ), ATR-based strike distance |
| Phase 3 | Debit vertical spreads, risk-based sizing |
| Phase 4 | Live account |
