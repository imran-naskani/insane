# Options Executor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire IBKR paper account options execution (entry, live tracking, exit) to ORB/MR signals from `alert_spicy_engine.py` for SPY and TSLA only.

**Architecture:** `telegram.py` centralises Telegram sending for both engines; `options_executor.py` owns all IB interaction via an `OptionsExecutor` class with a 5-second background tracking thread using live streaming data; `alert_spicy_engine.py` is minimally modified to import both and call `_executor.on_signal(ticker, direction)` for SPY/TSLA after a signal passes dedup.

**Tech Stack:** `ib_insync`, Python `threading`, `json`, `csv`, `datetime`, `pytest`, `requests`, `python-dotenv`

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `telegram.py` | Create | `send_alert(msg, channel)` — owns both bot tokens and chat IDs |
| `options_executor.py` | Create | `OptionsExecutor` class + pure helper functions (`_find_atm_strike`, `_select_expiry`, `_check_exit`) |
| `alert_spicy_engine.py` | Modify | Import telegram + executor; replace `send_telegram`; add `on_signal` calls for SPY/TSLA |
| `alert_spicy_engine_options.py` | Retire | Rename to `_retired_alert_spicy_engine_options.py` |
| `tests/__init__.py` | Create | Empty — makes tests/ a package |
| `tests/test_telegram.py` | Create | Unit tests for `telegram.py` |
| `tests/test_options_executor.py` | Create | Unit tests for pure helpers + persistence + guard checks |

---

## Task 1: telegram.py

**Files:**
- Create: `telegram.py`
- Create: `tests/__init__.py`
- Create: `tests/test_telegram.py`

- [ ] **Step 1: Create tests directory and write failing tests**

Create `tests/__init__.py` as an empty file.

Create `tests/test_telegram.py`:

```python
import os
import pytest
from unittest.mock import patch, MagicMock

os.environ.setdefault("TELEGRAM_BOT_TOKEN",         "test-main-token")
os.environ.setdefault("TELEGRAM_CHAT_ID",            "test-main-chat")
os.environ.setdefault("TELEGRAM_OPTION_BOT_TOKEN",  "test-opts-token")
os.environ.setdefault("TELEGRAM_OPTION_CHAT_ID",     "test-opts-chat")

import telegram as tg


def test_send_alert_main_channel_uses_main_token():
    with patch("telegram.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()
        tg.send_alert("hello", channel="main")
        args, kwargs = mock_post.call_args
        assert "test-main-token" in args[0]
        assert kwargs["json"]["chat_id"] == "test-main-chat"
        assert kwargs["json"]["text"] == "hello"


def test_send_alert_options_channel_uses_options_token():
    with patch("telegram.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()
        tg.send_alert("trade alert", channel="options")
        args, kwargs = mock_post.call_args
        assert "test-opts-token" in args[0]
        assert kwargs["json"]["chat_id"] == "test-opts-chat"


def test_send_alert_default_channel_is_main():
    with patch("telegram.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()
        tg.send_alert("hello")
        args, _ = mock_post.call_args
        assert "test-main-token" in args[0]


def test_send_alert_swallows_network_exception():
    with patch("telegram.requests.post", side_effect=Exception("timeout")):
        tg.send_alert("test", channel="main")  # must not raise
```

- [ ] **Step 2: Run to confirm FAIL**

```
pytest tests/test_telegram.py -v
```
Expected: `ModuleNotFoundError: No module named 'telegram'` or `ImportError`.

- [ ] **Step 3: Create `telegram.py`**

```python
import os
import requests
from dotenv import load_dotenv

load_dotenv()

_CHANNELS = {
    "main":    (os.environ.get("TELEGRAM_BOT_TOKEN"),        os.environ.get("TELEGRAM_CHAT_ID")),
    "options": (os.environ.get("TELEGRAM_OPTION_BOT_TOKEN"), os.environ.get("TELEGRAM_OPTION_CHAT_ID")),
}


def send_alert(msg: str, channel: str = "main") -> None:
    token, chat_id = _CHANNELS.get(channel, _CHANNELS["main"])
    if not token or not chat_id:
        print(f"[telegram] channel '{channel}' not configured")
        return
    url     = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"[telegram] send failed ({channel}): {e}")
```

- [ ] **Step 4: Run tests to confirm PASS**

```
pytest tests/test_telegram.py -v
```
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add telegram.py tests/__init__.py tests/test_telegram.py
git commit -m "feat: add standalone telegram module with dual-channel support"
```

---

## Task 2: options_executor.py — pure helpers

**Files:**
- Create: `options_executor.py` (constants + pure helpers only — class added in Task 3)
- Create: `tests/test_options_executor.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_options_executor.py`:

```python
import sys
import os
import datetime as dt
from unittest.mock import MagicMock

# Mock ib_insync before importing options_executor so the import doesn't fail
# without IB installed
sys.modules["ib_insync"] = MagicMock()

os.environ.setdefault("TELEGRAM_BOT_TOKEN",         "t")
os.environ.setdefault("TELEGRAM_CHAT_ID",            "c")
os.environ.setdefault("TELEGRAM_OPTION_BOT_TOKEN",  "t2")
os.environ.setdefault("TELEGRAM_OPTION_CHAT_ID",     "c2")

from options_executor import (
    _find_atm_strike,
    _select_expiry,
    _check_exit,
)


# ── _find_atm_strike ──────────────────────────────────────────────────────────

def test_atm_exact_match():
    assert _find_atm_strike([590.0, 595.0, 600.0], 595.0) == 595.0


def test_atm_rounds_to_nearest_above():
    assert _find_atm_strike([590.0, 595.0, 600.0], 596.0) == 595.0


def test_atm_rounds_to_nearest_below():
    assert _find_atm_strike([590.0, 595.0, 600.0], 594.0) == 595.0


def test_atm_single_strike():
    assert _find_atm_strike([500.0], 450.0) == 500.0


# ── _select_expiry ────────────────────────────────────────────────────────────

def test_select_expiry_prefers_0dte_before_cutoff():
    today = dt.date(2026, 6, 10)
    exps  = ["20260610", "20260611", "20260617"]
    assert _select_expiry(exps, today, dt.time(10, 0)) == "20260610"


def test_select_expiry_skips_0dte_after_cutoff():
    today = dt.date(2026, 6, 10)
    exps  = ["20260610", "20260611", "20260617"]
    # After 13:30 cutoff, even if 0DTE exists, use 1DTE
    assert _select_expiry(exps, today, dt.time(13, 45)) == "20260611"


def test_select_expiry_fallback_to_1dte():
    today = dt.date(2026, 6, 10)
    exps  = ["20260611", "20260617"]
    assert _select_expiry(exps, today, dt.time(10, 0)) == "20260611"


def test_select_expiry_returns_none_when_neither_available():
    today = dt.date(2026, 6, 10)
    exps  = ["20260617", "20260624"]
    assert _select_expiry(exps, today, dt.time(10, 0)) is None


# ── _check_exit ───────────────────────────────────────────────────────────────

def test_stop_loss_triggers_at_50pct():
    should, reason = _check_exit(2.00, 1.00, 2.00, False)
    assert should is True
    assert "Stop Loss" in reason


def test_no_exit_just_above_stop():
    should, reason = _check_exit(2.00, 1.01, 2.00, False)
    assert should is False
    assert reason is None


def test_trailing_stop_triggers_below_75pct_hwm():
    # entry=1.00, hwm=2.00 → trail at 75% of 2.00 = 1.50; current=1.40 < 1.50
    should, reason = _check_exit(1.00, 1.40, 2.00, True)
    assert should is True
    assert "Trailing" in reason


def test_trailing_stop_not_triggered_when_inactive():
    # trailing_active=False → no trailing stop regardless of price
    should, reason = _check_exit(1.00, 1.40, 2.00, False)
    assert should is False


def test_trailing_stop_not_triggered_above_threshold():
    # 75% of 2.00 = 1.50; current=1.60 > 1.50 → no exit
    should, reason = _check_exit(1.00, 1.60, 2.00, True)
    assert should is False


def test_stop_loss_takes_priority_over_trailing():
    # current=0.40 triggers stop loss; trailing also active — stop loss wins
    should, reason = _check_exit(1.00, 0.40, 2.00, True)
    assert should is True
    assert "Stop Loss" in reason
```

- [ ] **Step 2: Run to confirm FAIL**

```
pytest tests/test_options_executor.py -v
```
Expected: `ModuleNotFoundError: No module named 'options_executor'`.

- [ ] **Step 3: Create `options_executor.py` with constants and pure helpers**

```python
from ib_insync import IB, Stock, Option, LimitOrder, MarketOrder, util
import threading
import json
import csv
import datetime as dt
import os
import time

import telegram

# ── Start ib_insync event loop once at module level ───────────────────────────
util.startLoop()

# ── CONFIG ────────────────────────────────────────────────────────────────────

EXECUTOR_TICKERS = {
    "SPY":  Stock("SPY",  "SMART", "USD"),
    "TSLA": Stock("TSLA", "SMART", "USD"),
}

ENTRY_CUTOFF      = dt.time(14,  0)   # no new entries after 14:00
EOD_CUTOFF        = dt.time(14, 55)   # hard exit all positions
OTM_CUTOFF        = dt.time(13, 30)   # after this, use 1DTE instead of 0DTE
STOP_LOSS_PCT     = 0.50              # exit if premium drops to 50% of entry
TRAIL_ACTIVATE    = 1.50              # arm trailing once premium reaches 150% of entry
TRAIL_FROM_HWM    = 0.75             # trail at 75% of high-water mark
TRACK_INTERVAL    = 5                 # seconds between position checks
ORDER_RETRY_SECS  = 30               # seconds to wait per fill attempt
ORDER_MAX_RETRIES = 3                # give up after this many retries
POSITIONS_FILE    = "option_positions.json"
TRADES_FILE       = "option_trades.csv"


# ── PURE HELPERS (fully testable without IB) ──────────────────────────────────

def _find_atm_strike(strikes: list, current_price: float) -> float:
    """Return the strike closest to current_price."""
    return min(strikes, key=lambda s: abs(s - current_price))


def _select_expiry(expirations: list, today: dt.date, current_time: dt.time) -> str | None:
    """
    Return 0DTE if available AND current_time <= OTM_CUTOFF (13:30).
    Otherwise return 1DTE. Returns None if neither is in expirations.
    """
    today_str    = today.strftime("%Y%m%d")
    tomorrow_str = (today + dt.timedelta(days=1)).strftime("%Y%m%d")
    exps         = set(expirations)

    if today_str in exps and current_time <= OTM_CUTOFF:
        return today_str
    if tomorrow_str in exps:
        return tomorrow_str
    return None


def _check_exit(
    entry_price: float,
    current_price: float,
    high_water_mark: float,
    trailing_active: bool,
) -> tuple:
    """
    Returns (should_exit: bool, reason: str | None).
    EOD check is NOT here — the caller handles it (needs the real clock).
    Priority: stop loss first, then trailing.
    """
    if current_price <= entry_price * STOP_LOSS_PCT:
        return True, "Stop Loss (50%)"
    if trailing_active and current_price <= high_water_mark * TRAIL_FROM_HWM:
        return True, "Trailing Stop"
    return False, None
```

- [ ] **Step 4: Run tests to confirm PASS**

```
pytest tests/test_options_executor.py -v
```
Expected: 14 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add options_executor.py tests/__init__.py tests/test_options_executor.py
git commit -m "feat: add options_executor pure helpers with full test coverage"
```

---

## Task 3: OptionsExecutor — skeleton + persistence

**Files:**
- Modify: `options_executor.py` (append class with `__init__`, `connect`, `disconnect`, persistence methods, stubs)
- Modify: `tests/test_options_executor.py` (append persistence tests)

- [ ] **Step 1: Append persistence tests to `tests/test_options_executor.py`**

```python
import json
import csv
import threading
import pytest
from options_executor import OptionsExecutor, POSITIONS_FILE, TRADES_FILE


def _bare_executor():
    """Create OptionsExecutor without calling __init__ (no IB connection)."""
    ex = OptionsExecutor.__new__(OptionsExecutor)
    ex._ib        = MagicMock()
    ex._positions = {}
    ex._lock      = threading.Lock()
    ex._running   = False
    ex._thread    = None
    return ex


def test_save_positions_excludes_non_serialisable_fields(tmp_path, monkeypatch):
    monkeypatch.setattr("options_executor.POSITIONS_FILE", str(tmp_path / "pos.json"))
    ex = _bare_executor()
    ex._positions["SPY"] = {
        "contract":        MagicMock(),   # must NOT appear in JSON
        "ticker_obj":      MagicMock(),   # must NOT appear in JSON
        "contract_params": {"symbol": "SPY", "lastTradeDateOrContractMonth": "20260610",
                            "strike": 595.0, "right": "C", "exchange": "SMART",
                            "currency": "USD", "multiplier": "100"},
        "symbol":          "SPY260610C00595000",
        "direction":       "long",
        "entry_price":     1.50,
        "high_water_mark": 1.80,
        "trailing_active": False,
        "fill_time":       "2026-06-10T09:45:00",
        "contracts":       1,
    }
    ex._save_positions()
    with open(str(tmp_path / "pos.json")) as f:
        data = json.load(f)
    assert "SPY" in data
    assert data["SPY"]["entry_price"] == 1.50
    assert "contract" not in data["SPY"]
    assert "ticker_obj" not in data["SPY"]


def test_append_trade_writes_csv_row(tmp_path, monkeypatch):
    monkeypatch.setattr("options_executor.TRADES_FILE", str(tmp_path / "trades.csv"))
    ex = _bare_executor()
    pos = {
        "direction":   "long",
        "symbol":      "SPY260610C00595000",
        "entry_price": 1.50,
        "fill_time":   "2026-06-10T09:45:00",
        "contracts":   1,
    }
    ex._append_trade("SPY", pos, 2.30, "Trailing Stop")
    with open(str(tmp_path / "trades.csv")) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["ticker"]      == "SPY"
    assert rows[0]["exit_reason"] == "Trailing Stop"
    assert float(rows[0]["entry_price"]) == 1.50
    assert float(rows[0]["exit_price"])  == 2.30
    assert float(rows[0]["pnl_pct"])     == pytest.approx(53.33, abs=0.1)


def test_append_trade_appends_on_second_call(tmp_path, monkeypatch):
    monkeypatch.setattr("options_executor.TRADES_FILE", str(tmp_path / "trades.csv"))
    ex = _bare_executor()
    pos = {"direction": "long", "symbol": "X", "entry_price": 1.00,
           "fill_time": "2026-06-10T09:45:00", "contracts": 1}
    ex._append_trade("SPY",  pos, 1.50, "Trailing Stop")
    ex._append_trade("TSLA", pos, 0.50, "Stop Loss (50%)")
    with open(str(tmp_path / "trades.csv")) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[1]["ticker"] == "TSLA"
```

- [ ] **Step 2: Run to confirm FAIL**

```
pytest tests/test_options_executor.py::test_save_positions_excludes_non_serialisable_fields -v
```
Expected: `AttributeError: type object 'OptionsExecutor' has no attribute '_save_positions'`.

- [ ] **Step 3: Append `OptionsExecutor` class to `options_executor.py`**

Add after the `_check_exit` function:

```python
class OptionsExecutor:

    def __init__(self):
        self._ib      = IB()
        self._positions = {}    # ticker -> position state dict
        self._lock    = threading.Lock()
        self._running = False
        self._thread  = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def connect(self, host: str = "127.0.0.1", port: int = 4002, client_id: int = 111):
        self._ib.connect(host, port, clientId=client_id, timeout=10)
        self._load_positions()
        self._running = True
        self._thread  = threading.Thread(target=self._tracking_loop, daemon=True)
        self._thread.start()

    def disconnect(self):
        self._running = False
        if self._ib.isConnected():
            self._ib.disconnect()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_positions(self):
        data = {}
        with self._lock:
            for ticker, pos in self._positions.items():
                data[ticker] = {
                    k: v for k, v in pos.items()
                    if k not in ("contract", "ticker_obj")
                }
        try:
            with open(POSITIONS_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[executor] save error: {e}")

    def _load_positions(self):
        if not os.path.exists(POSITIONS_FILE):
            return
        try:
            with open(POSITIONS_FILE) as f:
                data = json.load(f)
        except Exception:
            return

        today = dt.date.today()
        for ticker, state in data.items():
            fill_dt = dt.datetime.fromisoformat(state["fill_time"])
            if fill_dt.date() < today:
                print(f"[executor] Skipping expired position for {ticker}")
                continue
            try:
                contract = Option(**state["contract_params"])
                self._ib.qualifyContracts(contract)
                ticker_obj = self._ib.reqMktData(contract, "", False, False)
                self._positions[ticker] = {
                    **state,
                    "contract":   contract,
                    "ticker_obj": ticker_obj,
                }
                print(f"[executor] Recovered open position: {ticker}")
            except Exception as e:
                print(f"[executor] Could not recover {ticker}: {e}")

    def _append_trade(self, ticker: str, pos: dict, exit_price: float, reason: str):
        file_exists = os.path.exists(TRADES_FILE)
        fill_dt     = dt.datetime.fromisoformat(pos["fill_time"])
        hold_mins   = int((dt.datetime.now() - fill_dt).total_seconds() / 60)
        pnl_pct     = round((exit_price - pos["entry_price"]) / pos["entry_price"] * 100, 2)
        fieldnames  = [
            "ticker", "direction", "option_symbol", "entry_price",
            "exit_price", "pnl_pct", "hold_minutes", "exit_reason",
            "entry_time", "exit_time",
        ]
        try:
            with open(TRADES_FILE, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow({
                    "ticker":        ticker,
                    "direction":     pos["direction"],
                    "option_symbol": pos["symbol"],
                    "entry_price":   pos["entry_price"],
                    "exit_price":    exit_price,
                    "pnl_pct":       pnl_pct,
                    "hold_minutes":  hold_mins,
                    "exit_reason":   reason,
                    "entry_time":    pos["fill_time"],
                    "exit_time":     dt.datetime.now().isoformat(),
                })
        except Exception as e:
            print(f"[executor] trade log error: {e}")

    # ── Stubs — implemented in Tasks 4-6 ─────────────────────────────────────

    def on_signal(self, ticker: str, direction: str) -> None:
        raise NotImplementedError

    def _handle_signal(self, ticker: str, direction: str) -> None:
        raise NotImplementedError

    def _open_position(self, ticker: str, direction: str, now: dt.datetime) -> None:
        raise NotImplementedError

    def _select_option(self, ticker: str, contract, direction: str, current_price: float):
        raise NotImplementedError

    def _place_order(self, ticker: str, opt_contract, direction: str, current_price: float):
        raise NotImplementedError

    def _tracking_loop(self):
        raise NotImplementedError

    def _exit_position(self, ticker: str, reason: str) -> None:
        raise NotImplementedError

    def _reconnect(self):
        raise NotImplementedError
```

- [ ] **Step 4: Run all tests to confirm PASS**

```
pytest tests/ -v
```
Expected: all 20 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add options_executor.py tests/test_options_executor.py
git commit -m "feat: add OptionsExecutor skeleton with persistence and trade log"
```

---

## Task 4: on_signal guard checks

**Files:**
- Modify: `options_executor.py` (replace `on_signal` and `_handle_signal` stubs)
- Modify: `tests/test_options_executor.py` (append guard tests)

- [ ] **Step 1: Append guard check tests to `tests/test_options_executor.py`**

```python
def test_handle_signal_ignores_unknown_ticker():
    ex = _bare_executor()
    ex._open_position  = MagicMock()
    ex._exit_position  = MagicMock()
    # AAPL is not in EXECUTOR_TICKERS — nothing should happen
    ex._handle_signal("AAPL", "long")
    ex._open_position.assert_not_called()
    ex._exit_position.assert_not_called()


def test_handle_signal_skips_same_direction():
    ex = _bare_executor()
    ex._positions["SPY"] = {
        "direction": "long", "entry_price": 1.50, "high_water_mark": 1.50,
        "trailing_active": False, "fill_time": dt.datetime.now().isoformat(),
        "contracts": 1, "symbol": "SPY...",
        "contract": MagicMock(), "ticker_obj": MagicMock(), "contract_params": {},
    }
    ex._open_position = MagicMock()
    ex._exit_position = MagicMock()
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("options_executor.dt.datetime",
                   type("_DT", (), {"now": staticmethod(lambda: dt.datetime(2026, 6, 10, 9, 45))})())
        ex._handle_signal("SPY", "long")
    ex._open_position.assert_not_called()


def test_handle_signal_exits_then_enters_on_reversal():
    ex = _bare_executor()
    ex._positions["SPY"] = {
        "direction": "long", "entry_price": 1.50, "high_water_mark": 1.50,
        "trailing_active": False, "fill_time": dt.datetime.now().isoformat(),
        "contracts": 1, "symbol": "SPY...",
        "contract": MagicMock(), "ticker_obj": MagicMock(), "contract_params": {},
    }
    ex._exit_position = MagicMock()
    ex._open_position = MagicMock()
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("options_executor.dt.datetime",
                   type("_DT", (), {"now": staticmethod(lambda: dt.datetime(2026, 6, 10, 9, 45))})())
        ex._handle_signal("SPY", "short")
    ex._exit_position.assert_called_once_with("SPY", "Signal Reversal")
    ex._open_position.assert_called_once()


def test_handle_signal_skips_after_entry_cutoff():
    ex = _bare_executor()
    ex._open_position = MagicMock()
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("options_executor.dt.datetime",
                   type("_DT", (), {"now": staticmethod(lambda: dt.datetime(2026, 6, 10, 14, 15))})())
        ex._handle_signal("SPY", "long")
    ex._open_position.assert_not_called()


def test_on_signal_swallows_exception():
    ex = _bare_executor()
    ex._handle_signal = MagicMock(side_effect=RuntimeError("boom"))
    ex.on_signal("SPY", "long")   # must not raise
```

- [ ] **Step 2: Run to confirm FAIL**

```
pytest tests/test_options_executor.py::test_handle_signal_ignores_unknown_ticker -v
```
Expected: `NotImplementedError`.

- [ ] **Step 3: Replace `on_signal` and `_handle_signal` stubs in `options_executor.py`**

```python
def on_signal(self, ticker: str, direction: str) -> None:
    """Called from alert engine. direction: 'long' | 'short'. Never raises."""
    try:
        self._handle_signal(ticker, direction)
    except Exception as e:
        print(f"[executor] on_signal error ({ticker} {direction}): {e}")


def _handle_signal(self, ticker: str, direction: str) -> None:
    if ticker not in EXECUTOR_TICKERS:
        return

    now = dt.datetime.now()
    if now.time() >= ENTRY_CUTOFF:
        print(f"[executor] {ticker}: past entry cutoff ({now.strftime('%H:%M')}), skipping")
        return

    with self._lock:
        pos = self._positions.get(ticker)

    if pos is not None:
        if pos["direction"] == direction:
            print(f"[executor] {ticker}: already {direction}, skipping")
            return
        self._exit_position(ticker, "Signal Reversal")

    self._open_position(ticker, direction, now)
```

- [ ] **Step 4: Run all tests to confirm PASS**

```
pytest tests/ -v
```
Expected: all tests PASS (stubs for `_open_position`/`_exit_position` are mocked in the tests).

- [ ] **Step 5: Commit**

```bash
git add options_executor.py tests/test_options_executor.py
git commit -m "feat: implement on_signal guard checks with full test coverage"
```

---

## Task 5: Option selection + order placement + _open_position

**Files:**
- Modify: `options_executor.py` (replace `_select_option`, `_place_order`, `_open_position` stubs)

- [ ] **Step 1: Replace `_select_option` stub**

```python
def _select_option(self, ticker: str, contract, direction: str, current_price: float):
    """Returns (Option, strike, expiry_str) or (None, None, None)."""
    try:
        cds = self._ib.reqContractDetails(contract)
        if not cds:
            print(f"[executor] {ticker}: no contract details from IB")
            return None, None, None
        con = cds[0].contract

        chains = self._ib.reqSecDefOptParams(
            con.symbol, con.exchange, con.secType, con.conId
        )
        if not chains:
            print(f"[executor] {ticker}: no option chain from IB")
            return None, None, None
        chain = chains[0]

        now    = dt.datetime.now()
        expiry = _select_expiry(list(chain.expirations), dt.date.today(), now.time())
        if expiry is None:
            print(f"[executor] {ticker}: no suitable expiry (need 0DTE or 1DTE)")
            return None, None, None

        strike = _find_atm_strike(list(chain.strikes), current_price)
        right  = "C" if direction == "long" else "P"

        opt = Option(
            symbol=con.symbol,
            lastTradeDateOrContractMonth=expiry,
            strike=strike,
            right=right,
            exchange="SMART",
            currency="USD",
            multiplier="100",
        )
        self._ib.qualifyContracts(opt)
        return opt, strike, expiry
    except Exception as e:
        print(f"[executor] _select_option error ({ticker}): {e}")
        return None, None, None
```

- [ ] **Step 2: Replace `_place_order` stub**

```python
def _place_order(self, ticker: str, opt_contract, direction: str, current_price: float):
    """
    Place limit order at bid/ask midpoint. Widen by 1 cent per retry.
    Returns fill price (float) or None if all retries exhausted.
    """
    for attempt in range(ORDER_MAX_RETRIES):
        snap = self._ib.reqMktData(opt_contract, "", True, False)
        self._ib.sleep(1.0)
        bid = snap.bid
        ask = snap.ask
        self._ib.cancelMktData(opt_contract)

        if not bid or not ask or bid <= 0 or ask <= 0:
            print(f"[executor] {ticker}: no bid/ask on attempt {attempt + 1}")
            time.sleep(ORDER_RETRY_SECS)
            continue

        limit_price = round((bid + ask) / 2 + attempt * 0.01, 2)
        order = LimitOrder("BUY", 1, limit_price)
        trade = self._ib.placeOrder(opt_contract, order)
        print(f"[executor] {ticker}: limit order ${limit_price:.2f} (attempt {attempt + 1})")

        deadline = time.time() + ORDER_RETRY_SECS
        while time.time() < deadline:
            self._ib.sleep(1)
            if trade.orderStatus.status == "Filled":
                fill = trade.orderStatus.avgFillPrice
                print(f"[executor] {ticker}: filled at ${fill:.2f}")
                return fill

        self._ib.cancelOrder(order)
        print(f"[executor] {ticker}: unfilled on attempt {attempt + 1}, retrying")

    return None
```

- [ ] **Step 3: Replace `_open_position` stub**

```python
def _open_position(self, ticker: str, direction: str, now: dt.datetime) -> None:
    contract = EXECUTOR_TICKERS[ticker]
    self._ib.qualifyContracts(contract)

    # Get current underlying price
    snap = self._ib.reqMktData(contract, "", True, False)
    self._ib.sleep(1.0)
    current_price = (
        snap.last
        or snap.close
        or ((snap.bid + snap.ask) / 2 if snap.bid and snap.ask else None)
    )
    self._ib.cancelMktData(contract)

    if not current_price:
        print(f"[executor] {ticker}: could not get underlying price")
        return

    opt_contract, strike, expiry = self._select_option(
        ticker, contract, direction, current_price
    )
    if opt_contract is None:
        return

    fill_price = self._place_order(ticker, opt_contract, direction, current_price)
    right = "CALL" if direction == "long" else "PUT"

    if fill_price is None:
        telegram.send_alert(
            f"*{ticker}* | {right} entry missed — unfilled after "
            f"{ORDER_MAX_RETRIES * ORDER_RETRY_SECS}s",
            channel="options",
        )
        return

    # Subscribe live streaming for position tracking
    ticker_obj   = self._ib.reqMktData(opt_contract, "", False, False)
    expiry_label = "0DTE" if expiry == now.strftime("%Y%m%d") else "1DTE"

    pos_state = {
        "contract": opt_contract,
        "contract_params": {
            "symbol":                       opt_contract.symbol,
            "lastTradeDateOrContractMonth": opt_contract.lastTradeDateOrContractMonth,
            "strike":                       opt_contract.strike,
            "right":                        opt_contract.right,
            "exchange":                     opt_contract.exchange,
            "currency":                     opt_contract.currency,
            "multiplier":                   opt_contract.multiplier,
        },
        "ticker_obj":      ticker_obj,
        "symbol":          opt_contract.localSymbol,
        "direction":       direction,
        "entry_price":     fill_price,
        "high_water_mark": fill_price,
        "trailing_active": False,
        "fill_time":       now.isoformat(),
        "contracts":       1,
    }

    with self._lock:
        self._positions[ticker] = pos_state
    self._save_positions()

    telegram.send_alert(
        f"*{ticker}* | {right} ATM | Strike: {strike:.0f} | {expiry_label} | "
        f"Fill: ${fill_price:.2f} | "
        f"Signal: {'ORB/MR Long' if direction == 'long' else 'ORB/MR Short'}",
        channel="options",
    )
```

- [ ] **Step 4: Run all tests to confirm no regression**

```
pytest tests/ -v
```
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add options_executor.py
git commit -m "feat: implement option selection, order placement, and position entry"
```

---

## Task 6: Tracking loop, exit, and IB reconnect

**Files:**
- Modify: `options_executor.py` (replace `_tracking_loop`, `_exit_position`, `_reconnect` stubs)

- [ ] **Step 1: Replace `_tracking_loop` stub**

```python
def _tracking_loop(self):
    while self._running:
        time.sleep(TRACK_INTERVAL)

        if not self._ib.isConnected():
            print("[executor] IB disconnected — attempting reconnect")
            self._reconnect()
            continue

        self._ib.sleep(0)   # flush incoming tick events from IB event loop

        now = dt.datetime.now()
        with self._lock:
            tickers = list(self._positions.keys())

        for ticker in tickers:
            with self._lock:
                pos = self._positions.get(ticker)
            if pos is None:
                continue

            # Read live streaming price (no new IB request needed)
            ticker_obj = pos["ticker_obj"]
            mid = ticker_obj.midpoint()
            if not mid or mid <= 0:
                if ticker_obj.bid and ticker_obj.ask and ticker_obj.bid > 0:
                    mid = (ticker_obj.bid + ticker_obj.ask) / 2
                else:
                    continue   # no price yet this cycle

            entry = pos["entry_price"]
            hwm   = pos["high_water_mark"]
            trail = pos["trailing_active"]

            # Update high-water mark
            if mid > hwm:
                with self._lock:
                    if self._positions.get(ticker):
                        self._positions[ticker]["high_water_mark"] = mid
                hwm = mid

            # Arm trailing stop once up 50%
            if not trail and mid >= entry * TRAIL_ACTIVATE:
                with self._lock:
                    if self._positions.get(ticker):
                        self._positions[ticker]["trailing_active"] = True
                trail = True
                self._save_positions()
                print(f"[executor] {ticker}: trailing stop armed at ${mid:.2f}")

            # EOD forced exit (highest priority — checked before rule exits)
            if now.time() >= EOD_CUTOFF:
                self._exit_position(ticker, "EOD")
                continue

            # Rule-based exits: stop loss then trailing
            should_exit, reason = _check_exit(entry, mid, hwm, trail)
            if should_exit:
                self._exit_position(ticker, reason)
```

- [ ] **Step 2: Replace `_exit_position` stub**

```python
def _exit_position(self, ticker: str, reason: str) -> None:
    with self._lock:
        pos = self._positions.get(ticker)
    if pos is None:
        return

    exit_price = pos["entry_price"]   # fallback if market order fails
    try:
        self._ib.cancelMktData(pos["contract"])
        order = MarketOrder("SELL", pos["contracts"])
        trade = self._ib.placeOrder(pos["contract"], order)

        deadline = time.time() + 30
        while time.time() < deadline:
            self._ib.sleep(1)
            if trade.orderStatus.status == "Filled":
                exit_price = trade.orderStatus.avgFillPrice
                break
    except Exception as e:
        print(f"[executor] exit order error ({ticker}): {e}")

    pnl_pct   = (exit_price - pos["entry_price"]) / pos["entry_price"] * 100
    fill_dt   = dt.datetime.fromisoformat(pos["fill_time"])
    hold_mins = int((dt.datetime.now() - fill_dt).total_seconds() / 60)
    right     = "CALL" if pos["direction"] == "long" else "PUT"

    telegram.send_alert(
        f"*{ticker}* | {right} EXIT | Reason: {reason}\n"
        f"Entry: ${pos['entry_price']:.2f} → Exit: ${exit_price:.2f} | "
        f"P&L: {pnl_pct:+.1f}% | Hold: {hold_mins}m",
        channel="options",
    )

    self._append_trade(ticker, pos, exit_price, reason)

    with self._lock:
        self._positions.pop(ticker, None)
    self._save_positions()
```

- [ ] **Step 3: Replace `_reconnect` stub**

```python
def _reconnect(self):
    attempts = 0
    while self._running and not self._ib.isConnected():
        attempts += 1
        try:
            self._ib.connect("127.0.0.1", 4002, clientId=111, timeout=10)
            print(f"[executor] Reconnected to IB (attempt {attempts})")
            # Re-subscribe streaming data for all open positions
            with self._lock:
                for ticker, pos in self._positions.items():
                    if pos.get("contract"):
                        ticker_obj = self._ib.reqMktData(pos["contract"], "", False, False)
                        self._positions[ticker]["ticker_obj"] = ticker_obj
            return
        except Exception as e:
            print(f"[executor] Reconnect attempt {attempts} failed: {e}")
            time.sleep(30)
```

- [ ] **Step 4: Run all tests to confirm no regression**

```
pytest tests/ -v
```
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add options_executor.py
git commit -m "feat: implement tracking loop, exit logic, and IB reconnect"
```

---

## Task 7: Modify alert_spicy_engine.py

**Files:**
- Modify: `alert_spicy_engine.py`

- [ ] **Step 1: Replace Telegram config + send_telegram with imports**

In `alert_spicy_engine.py`, remove these lines:

```python
import requests          # line 20 — remove

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]   # line 33 — remove
CHAT_ID   = os.environ["TELEGRAM_CHAT_ID"]     # line 34 — remove
```

Remove the entire `send_telegram` function (lines 54-58):
```python
def send_telegram(msg: str):
    url     = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    r = requests.post(url, json=payload, timeout=10)
    r.raise_for_status()
```

Also remove `import os` if it is now unused. Check: `os` is only used for env vars that are now removed. Confirm it's not used elsewhere in the file, then remove it.

Add these two imports after the existing imports block:

```python
import telegram
from options_executor import OptionsExecutor
```

- [ ] **Step 2: Initialise executor before the main loop**

Find the line (just before `print("INSANE Alert Engine started ..."`):

```python
print("INSANE Alert Engine started (5m) -- MR + ORB-Slope strategy")
```

Insert before it:

```python
# Options executor — connects to IB TWS/Gateway for SPY + TSLA order placement.
# If TWS is not running, the alert engine continues without options execution.
_executor = OptionsExecutor()
try:
    _executor.connect()
    print("[executor] Connected to IB TWS/Gateway")
except Exception as _e:
    print(f"[executor] Could not connect to IB ({_e}). Options execution disabled.")
    _executor = None
```

- [ ] **Step 3: Replace send_telegram call with telegram.send_alert**

Find (near the bottom of the main loop):

```python
            send_telegram(final_msg)
```

Replace with:

```python
            telegram.send_alert(final_msg)
```

- [ ] **Step 4: Add on_signal calls after the dedup block**

Find this block (signal dedup and message collect):

```python
            if signal:
                prev  = state["last_alert"]
                allow = (prev is None) or (prev[1] != signal_type)
                if allow:
                    combined_msgs.append(
                        f"*{'SPX' if ticker == '^GSPC' else ticker}*\n"
                        f"{signal}\n"
                        f"Time: {display_time}\n"
                        f"Price: {dv['Close'].iloc[-1]:.2f}"
                    )
                    state["last_alert"] = (bar_time, signal_type)
```

Replace with:

```python
            if signal:
                prev  = state["last_alert"]
                allow = (prev is None) or (prev[1] != signal_type)
                if allow:
                    combined_msgs.append(
                        f"*{'SPX' if ticker == '^GSPC' else ticker}*\n"
                        f"{signal}\n"
                        f"Time: {display_time}\n"
                        f"Price: {dv['Close'].iloc[-1]:.2f}"
                    )
                    state["last_alert"] = (bar_time, signal_type)
                    # Options execution: SPY and TSLA only, entry signals only
                    if (
                        _executor is not None
                        and ticker in ("SPY", "TSLA")
                        and signal_type in ("ORB_LONG", "MR_LONG", "ORB_SHORT", "MR_SHORT")
                    ):
                        exec_dir = "long" if "LONG" in signal_type else "short"
                        _executor.on_signal(ticker, exec_dir)
```

- [ ] **Step 5: Verify syntax**

```
python -c "import ast; ast.parse(open('alert_spicy_engine.py').read()); print('OK')"
```
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add alert_spicy_engine.py
git commit -m "feat: wire OptionsExecutor into alert engine for SPY/TSLA signals"
```

---

## Task 8: Retire old engine + final verification

**Files:**
- Rename: `alert_spicy_engine_options.py` → `_retired_alert_spicy_engine_options.py`

- [ ] **Step 1: Rename old engine**

```bash
git mv alert_spicy_engine_options.py _retired_alert_spicy_engine_options.py
```

- [ ] **Step 2: Run full test suite**

```
pytest tests/ -v
```
Expected: all tests PASS with zero failures.

- [ ] **Step 3: Verify alert_spicy_engine.py imports cleanly (no IB needed)**

```
python -c "
import sys
from unittest.mock import MagicMock
sys.modules['ib_insync'] = MagicMock()
import ast, builtins
# Just parse — don't execute top-level IB connect
ast.parse(open('alert_spicy_engine.py').read())
print('alert_spicy_engine.py: syntax OK')
"
```
Expected: `alert_spicy_engine.py: syntax OK`

- [ ] **Step 4: Final commit**

```bash
git add _retired_alert_spicy_engine_options.py
git commit -m "chore: retire alert_spicy_engine_options.py — replaced by options_executor.py"
```

---

## Spec Coverage Checklist

| Spec Requirement | Task |
|----------------|------|
| `telegram.py` with `send_alert(msg, channel)` | Task 1 |
| `channel="main"` → main bot, `channel="options"` → options bot | Task 1 |
| `EXECUTOR_TICKERS` allowlist (SPY + TSLA only) | Task 2 |
| ATM strike: `min(strikes, key=lambda s: abs(s-price))` | Task 2, 5 |
| 0DTE preferred, 1DTE fallback, switch at 13:30 | Task 2, 5 |
| Guard: unknown ticker → skip | Task 4 |
| Guard: time > 14:00 → skip | Task 4 |
| Guard: same direction → skip | Task 4 |
| Guard: opposite direction → exit then enter | Task 4 |
| Limit order at midpoint, widen 1¢ per retry, 3 retries | Task 5 |
| Entry Telegram via `channel="options"` | Task 5 |
| Live streaming via `reqMktData` snapshot=False | Task 5, 6 |
| 5s tracking loop | Task 6 |
| 50% stop loss | Task 2 (tested), Task 6 (executed) |
| Trailing: arm at 150%, trail at 75% HWM | Task 2 (tested), Task 6 (executed) |
| EOD hard exit at 14:55 | Task 6 |
| Signal reversal exit | Task 4 |
| Market order on exit | Task 6 |
| Exit Telegram with P&L + hold time | Task 6 |
| `option_positions.json` persistence | Task 3 |
| `option_trades.csv` append-only log | Task 3 |
| Crash recovery (re-subscribe streaming on restart) | Task 3 (`_load_positions`) |
| IB reconnect loop every 30s | Task 6 |
| Re-subscribe streaming after reconnect | Task 6 |
| `alert_spicy_engine.py`: `on_signal` for SPY/TSLA ORB/MR only | Task 7 |
| `alert_spicy_engine.py`: uses `telegram.send_alert` | Task 7 |
| `alert_spicy_engine.py`: graceful if IB not running | Task 7 |
| Retire `alert_spicy_engine_options.py` | Task 8 |
