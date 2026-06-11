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


# ── Persistence tests ─────────────────────────────────────────────────────────

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


def test_handle_signal_ignores_unknown_ticker():
    ex = _bare_executor()
    ex._open_position  = MagicMock()
    ex._exit_position  = MagicMock()
    # AAPL is not in EXECUTOR_TICKERS — nothing should happen
    ex._handle_signal("AAPL", "long")
    ex._open_position.assert_not_called()
    ex._exit_position.assert_not_called()


def test_handle_signal_skips_same_direction(monkeypatch):
    ex = _bare_executor()
    ex._positions["SPY"] = {
        "direction": "long", "entry_price": 1.50, "high_water_mark": 1.50,
        "trailing_active": False, "fill_time": dt.datetime.now().isoformat(),
        "contracts": 1, "symbol": "SPY...",
        "contract": MagicMock(), "ticker_obj": MagicMock(), "contract_params": {},
    }
    ex._open_position = MagicMock()
    ex._exit_position = MagicMock()
    monkeypatch.setattr("options_executor.dt.datetime", MagicMock(
        now=MagicMock(return_value=dt.datetime(2026, 6, 10, 9, 45))
    ))
    ex._handle_signal("SPY", "long")
    ex._open_position.assert_not_called()


def test_handle_signal_exits_then_enters_on_reversal(monkeypatch):
    ex = _bare_executor()
    ex._positions["SPY"] = {
        "direction": "long", "entry_price": 1.50, "high_water_mark": 1.50,
        "trailing_active": False, "fill_time": dt.datetime.now().isoformat(),
        "contracts": 1, "symbol": "SPY...",
        "contract": MagicMock(), "ticker_obj": MagicMock(), "contract_params": {},
    }
    ex._exit_position = MagicMock()
    ex._open_position = MagicMock()
    monkeypatch.setattr("options_executor.dt.datetime", MagicMock(
        now=MagicMock(return_value=dt.datetime(2026, 6, 10, 9, 45))
    ))
    ex._handle_signal("SPY", "short")
    ex._exit_position.assert_called_once_with("SPY", "Signal Reversal")
    ex._open_position.assert_called_once()


def test_handle_signal_skips_after_entry_cutoff(monkeypatch):
    ex = _bare_executor()
    ex._open_position = MagicMock()
    monkeypatch.setattr("options_executor.dt.datetime", MagicMock(
        now=MagicMock(return_value=dt.datetime(2026, 6, 10, 14, 15))
    ))
    ex._handle_signal("SPY", "long")
    ex._open_position.assert_not_called()


def test_on_signal_swallows_exception():
    ex = _bare_executor()
    ex._handle_signal = MagicMock(side_effect=RuntimeError("boom"))
    ex.on_signal("SPY", "long")   # must not raise
