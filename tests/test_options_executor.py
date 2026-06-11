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
