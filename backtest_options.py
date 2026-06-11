"""
Options strategy backtest.

Replays intraday signals (same logic as alert_spicy_engine.py) over the last 55 days,
fetches historical option prices from IBKR for each signal, then simulates the
options_executor exit rules:
  - Stop loss at 50% of entry premium
  - Trailing stop: arms at 150% of entry, trails at 75% of HWM
  - Signal reversal exit
  - Hard EOD exit at 14:55 CT

Output: option_backtest_{TICKER}.json  (read by INSANE-PRO UI)
"""

import datetime as dt
import json
import logging
import os
import time

import pandas as pd
from dotenv import load_dotenv
from ib_insync import IB, Option, util

from build_dataset import build_feature_dataset
from intraday_signals import (
    add_intraday_features,
    run_new, run_orb_slope, run_orb_sliding,
    mr_orb_upgrade_qualifies,
    _orb_reversal_confirmed, REVERSAL_ANGLE,
    S_THR, ATR_FLOOR,
    ORB_SLOPE_DEG, ORB_SLOPE_DEG_DEFAULT,
    ORB_R2_THR, ORB_R2_THR_DEFAULT,
    ORB_ACCUM_START, ORB_ACCUM_START_DEFAULT,
    ORB_USE_SLIDING, ORB_USE_SLIDING_DEFAULT,
)

load_dotenv()

# ── LOGGING ───────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"logs/backtest_options_{dt.date.today():%Y-%m-%d}.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────
TICKERS        = ["SPY", "TSLA"]
TIMEFRAME      = "5m"
SIGNAL_DAYS    = 31   # match live engine history so TOS_Trail warmup is identical
OPTION_DAYS    = 2    # IB retains expired 0DTE contracts for ~1 trading day

# Mirror options_executor.py constants exactly
STOP_LOSS_PCT  = 0.50    # exit if premium drops to 50% of entry
TRAIL_ACTIVATE = 1.50    # arm trailing once premium hits 150% of entry
TRAIL_FROM_HWM = 0.75    # trail at 75% of high-water mark
ENTRY_CUTOFF   = dt.time(14,  0)
EOD_CUTOFF     = dt.time(14, 55)
OTM_CUTOFF     = dt.time(13, 30)   # switch from 0DTE to 1DTE after this

# Nearest standard strike increment per ticker
STRIKE_INC = {"SPY": 1.0, "TSLA": 2.5}   # SPY uses $1 whole-dollar strikes; TSLA uses $2.50

SESSION_START = pd.Timestamp("08:30").time()
SESSION_END   = pd.Timestamp("15:00").time()

IB_HOST      = os.getenv("IB_HOST", "127.0.0.1")
IB_PORT      = int(os.getenv("IB_PORT", "4002"))
IB_CLIENT_ID = 113   # separate from live executor (111) and any chart feed (112)

# Pace IBKR historical requests (50 req/10s limit; 0.25s gap is conservative)
IB_REQ_PAUSE = 0.25


# ── SESSION STATE ─────────────────────────────────────────────────────────────

def _fresh_session() -> dict:
    return {
        "date":         None,
        "orb_fired":    False,
        "orb_dir":      None,
        "orb_bar_loc":  None,
        "position":     None,
        "position_src": None,
        "last_alert":   None,
        "mr_entry_bar": None,
    }


# ── SIGNAL REPLAY ─────────────────────────────────────────────────────────────

def generate_signals(ticker: str) -> list:
    """
    Walk-forward replay of alert_spicy_engine session state machine over 55 days.
    Returns list of signal dicts with bar_time, direction, signal_type, underlying_price.
    """
    end_date   = dt.date.today()
    start_date = end_date - dt.timedelta(days=SIGNAL_DAYS)   # 31 days matches live engine TOS_Trail warmup

    log.info(f"[{ticker}] fetching {SIGNAL_DAYS}-day 5m data from yfinance...")
    df = build_feature_dataset(
        ticker,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        timeframe=TIMEFRAME,
    )

    if df is None or len(df) == 0:
        log.error(f"[{ticker}] empty DataFrame from build_feature_dataset")
        return []

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("US/Central")

    df = add_intraday_features(df)

    # Per-ticker ORB params
    orb_deg     = ORB_SLOPE_DEG.get(ticker, ORB_SLOPE_DEG_DEFAULT)
    r2_thr      = ORB_R2_THR.get(ticker, ORB_R2_THR_DEFAULT)
    accum_start = ORB_ACCUM_START.get(ticker, ORB_ACCUM_START_DEFAULT)
    use_sliding = ORB_USE_SLIDING.get(ticker, ORB_USE_SLIDING_DEFAULT)
    _accum      = accum_start if accum_start is not None else ORB_ACCUM_START_DEFAULT

    # Signal generation: full SIGNAL_DAYS for correct TOS_Trail warmup
    # Option lookup: only last OPTION_DAYS (IB purges expired contracts fast)
    signal_cutoff = pd.Timestamp(end_date - dt.timedelta(days=SIGNAL_DAYS)).tz_localize("US/Central")
    option_cutoff = end_date - dt.timedelta(days=OPTION_DAYS)
    trading_days  = sorted(set(df[df.index >= signal_cutoff].index.date))

    signals = []

    for day in trading_days:
        dv_full = df[df.index.date == day]
        dv_full = dv_full[
            (dv_full.index.time >= SESSION_START) &
            (dv_full.index.time <= SESSION_END)
        ]
        if len(dv_full) < _accum:
            continue

        state = _fresh_session()
        state["date"] = day

        for i in range(len(dv_full)):
            dv       = dv_full.iloc[:i + 1]
            last_idx = len(dv) - 1
            bar_time = dv.index[-1]

            if bar_time.time() >= ENTRY_CUTOFF:
                break

            if len(dv) < _accum:
                continue
            mr_ready = len(dv) >= 14

            # ── Compute signals ────────────────────────────────────────────
            if mr_ready:
                lsig_mr, ssig_mr = run_new(
                    dv, S_THR.get(ticker, 0.60), ATR_FLOOR.get(ticker, 0.50))
                lsig_mr_rev, ssig_mr_rev = run_new(
                    dv, S_THR.get(ticker, 0.60), ATR_FLOOR.get(ticker, 0.50),
                    orb_reversal=True)
            else:
                lsig_mr = ssig_mr = pd.Series(False, index=dv.index)
                lsig_mr_rev = ssig_mr_rev = pd.Series(False, index=dv.index)

            lsig_orb, ssig_orb = run_orb_slope(
                dv, angle_deg=orb_deg, r2_thr=r2_thr, accum_start=accum_start)
            orb_now = bool(lsig_orb.iloc[-1]) or bool(ssig_orb.iloc[-1])

            lsig_slide = ssig_slide = pd.Series(False, index=dv.index)
            if not state["orb_fired"] and not orb_now and use_sliding:
                lsig_slide, ssig_slide = run_orb_sliding(dv, orb_deg, r2_thr)

            lsig_orb_c = lsig_orb | lsig_slide
            ssig_orb_c = ssig_orb | ssig_slide

            is_orb_long     = bool(lsig_orb_c.iloc[-1])
            is_orb_short    = bool(ssig_orb_c.iloc[-1])
            is_mr_long      = bool(lsig_mr.iloc[-1])
            is_mr_short     = bool(ssig_mr.iloc[-1])
            is_mr_long_rev  = bool(lsig_mr_rev.iloc[-1])
            is_mr_short_rev = bool(ssig_mr_rev.iloc[-1])

            signal_dir  = None
            signal_type = None

            # ── Same state-machine logic as alert_spicy_engine.py ──────────
            if is_orb_long and not state["orb_fired"]:
                state.update(orb_fired=True, orb_dir="long", orb_bar_loc=last_idx,
                             position="long", position_src="orb")
                signal_dir  = "long"
                signal_type = "ORB_LONG"

            elif is_orb_short and not state["orb_fired"]:
                state.update(orb_fired=True, orb_dir="short", orb_bar_loc=last_idx,
                             position="short", position_src="orb")
                signal_dir  = "short"
                signal_type = "ORB_SHORT"

            elif (is_mr_long_rev or is_mr_long) and state["position"] != "long":
                if state["position"] == "short" and state["position_src"] == "orb":
                    if is_mr_long_rev and _orb_reversal_confirmed(
                            dv, state["orb_bar_loc"], "short", last_idx,
                            threshold=REVERSAL_ANGLE, r2_thr=r2_thr):
                        state.update(position="long", position_src="mr", mr_entry_bar=last_idx)
                        signal_dir  = "long"
                        signal_type = "MR_LONG"
                elif is_mr_long:
                    state.update(position="long", position_src="mr", mr_entry_bar=last_idx)
                    signal_dir  = "long"
                    signal_type = "MR_LONG"

            elif (is_mr_short_rev or is_mr_short) and state["position"] != "short":
                if state["position"] == "long" and state["position_src"] == "orb":
                    if is_mr_short_rev and _orb_reversal_confirmed(
                            dv, state["orb_bar_loc"], "long", last_idx,
                            threshold=REVERSAL_ANGLE, r2_thr=r2_thr):
                        state.update(position="short", position_src="mr", mr_entry_bar=last_idx)
                        signal_dir  = "short"
                        signal_type = "MR_SHORT"
                elif is_mr_short:
                    state.update(position="short", position_src="mr", mr_entry_bar=last_idx)
                    signal_dir  = "short"
                    signal_type = "MR_SHORT"

            # Dedup: skip if same signal_type as last alert
            if signal_dir:
                prev = state["last_alert"]
                if prev is None or prev[1] != signal_type:
                    state["last_alert"] = (bar_time, signal_type)
                    signals.append({
                        "ticker":           ticker,
                        "date":             str(day),
                        "bar_time":         bar_time.isoformat(),
                        "direction":        signal_dir,
                        "signal_type":      signal_type,
                        "underlying_price": float(dv["Close"].iloc[-1]),
                    })

    log.info(f"[{ticker}] {len(signals)} signals found over {len(trading_days)} trading days")
    return signals


# ── OPTION CONTRACT DERIVATION ─────────────────────────────────────────────────

def derive_option_contract(ticker: str, direction: str,
                           underlying_price: float,
                           signal_time: pd.Timestamp) -> tuple:
    """
    Derive the ATM option contract that the executor would have traded.
    Returns (strike, expiry_yyyymmdd, right).
    """
    inc    = STRIKE_INC.get(ticker, 1.0)
    strike = round(round(underlying_price / inc) * inc, 2)
    right  = "C" if direction == "long" else "P"

    today = signal_time.date()
    if signal_time.time() <= OTM_CUTOFF:
        expiry = today.strftime("%Y%m%d")
    else:
        # Walk forward to next available expiry (skip weekends)
        next_day = today + dt.timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += dt.timedelta(days=1)
        expiry = next_day.strftime("%Y%m%d")

    return strike, expiry, right


# ── IBKR OPTION HISTORY ───────────────────────────────────────────────────────

def fetch_option_bars(ib: IB, ticker: str, strike: float,
                      expiry: str, right: str) -> pd.DataFrame | None:
    """
    Fetch 5m MIDPOINT bars for a specific option contract on its expiry date.
    Returns timezone-aware (US/Central) DataFrame or None on error.
    """
    opt = Option(
        symbol=ticker,
        lastTradeDateOrContractMonth=expiry,
        strike=strike,
        right=right,
        exchange="SMART",
        currency="USD",
        multiplier="100",
    )

    try:
        contracts = ib.qualifyContracts(opt)
        if not contracts:
            log.warning(f"  Cannot qualify {ticker} {right} {strike} {expiry}")
            return None
    except Exception as e:
        log.warning(f"  qualifyContracts error: {e}")
        return None

    # endDateTime = 4pm ET = 20:00 UTC on the expiry date, explicit UTC to silence IB warning
    end_dt = f"{expiry} 20:00:00 UTC"

    try:
        time.sleep(IB_REQ_PAUSE)   # pace requests
        bars = ib.reqHistoricalData(
            opt,
            endDateTime=end_dt,
            durationStr="1 D",
            barSizeSetting="5 mins",
            whatToShow="MIDPOINT",
            useRTH=True,
            formatDate=1,
        )
    except Exception as e:
        log.warning(f"  reqHistoricalData error for {ticker} {right} {strike} {expiry}: {e}")
        return None

    if not bars:
        log.warning(f"  No bars returned for {ticker} {right} {strike} {expiry}")
        return None

    records = []
    for b in bars:
        records.append({
            "time":  b.date,
            "open":  b.open,
            "high":  b.high,
            "low":   b.low,
            "close": b.close,
        })

    df = pd.DataFrame(records).set_index("time")
    df.index = pd.DatetimeIndex(df.index)

    if df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York")
    df.index = df.index.tz_convert("US/Central")

    return df


# ── TRADE SIMULATION ──────────────────────────────────────────────────────────

def _make_result(direction: str, entry_price: float, exit_price: float,
                 hwm: float, reason: str,
                 entry_time: pd.Timestamp, exit_time: pd.Timestamp) -> dict:
    hold_mins = int((exit_time - entry_time).total_seconds() / 60)
    pnl_pct   = round((exit_price - entry_price) / entry_price * 100, 2)
    return {
        "direction":     direction,
        "entry_price":   round(float(entry_price), 2),
        "exit_price":    round(float(exit_price),  2),
        "high_water_mark": round(float(hwm), 2),
        "pnl_pct":       pnl_pct,
        "hold_minutes":  hold_mins,
        "exit_reason":   reason,
        "entry_time":    entry_time.isoformat(),
        "exit_time":     exit_time.isoformat(),
    }


def simulate_trade(entry_time: pd.Timestamp, option_bars: pd.DataFrame,
                   direction: str, day_signals: list) -> dict | None:
    """
    Simulate executor tracking loop (5s cycle → mapped to 5m bars here).
    day_signals: all signals for the same day (to detect reversal exits).
    """
    # Normalise reversal signal times to same tz as entry_time
    reversal_times = set()
    for s in day_signals:
        t = pd.Timestamp(s["bar_time"])
        if t.tzinfo is None:
            t = t.tz_localize(entry_time.tzinfo)
        if s["direction"] != direction and t > entry_time:
            reversal_times.add(t)

    bars_after = option_bars[option_bars.index >= entry_time]
    if len(bars_after) == 0:
        return None

    # Entry at first available bar's close
    entry_price = float(bars_after.iloc[0]["close"])
    if entry_price <= 0 or pd.isna(entry_price):
        return None

    hwm             = entry_price
    trailing_active = False

    for bar_time, row in bars_after.iterrows():
        mid = float(row["close"])
        if mid <= 0 or pd.isna(mid):
            continue

        if mid > hwm:
            hwm = mid

        if not trailing_active and mid >= entry_price * TRAIL_ACTIVATE:
            trailing_active = True

        # EOD hard exit (highest priority)
        if bar_time.time() >= EOD_CUTOFF:
            return _make_result(direction, entry_price, mid, hwm, "EOD",
                                entry_time, bar_time)

        # Reversal signal
        if bar_time in reversal_times:
            return _make_result(direction, entry_price, mid, hwm, "Signal Reversal",
                                entry_time, bar_time)

        # Stop loss
        if mid <= entry_price * STOP_LOSS_PCT:
            return _make_result(direction, entry_price, mid, hwm, "Stop Loss (50%)",
                                entry_time, bar_time)

        # Trailing stop
        if trailing_active and mid <= hwm * TRAIL_FROM_HWM:
            return _make_result(direction, entry_price, mid, hwm, "Trailing Stop",
                                entry_time, bar_time)

    # Exhausted all bars — treat as EOD
    last_bar = bars_after.iloc[-1]
    return _make_result(direction, entry_price, float(last_bar["close"]), hwm, "EOD",
                        entry_time, bars_after.index[-1])


# ── MAIN BACKTEST RUNNER ──────────────────────────────────────────────────────

def run_backtest(ticker: str, ib: IB) -> list:
    all_signals = generate_signals(ticker)
    trades = []

    # Group signals by date for reversal detection
    by_date: dict[str, list] = {}
    for s in all_signals:
        by_date.setdefault(s["date"], []).append(s)

    option_cutoff = dt.date.today() - dt.timedelta(days=OPTION_DAYS)

    for sig in all_signals:
        bar_time = pd.Timestamp(sig["bar_time"])

        # Only fetch option data for signals within IB's retention window
        if dt.date.fromisoformat(sig["date"]) < option_cutoff:
            continue

        if bar_time.time() >= ENTRY_CUTOFF:
            continue

        strike, expiry, right = derive_option_contract(
            ticker, sig["direction"], sig["underlying_price"], bar_time)

        log.info(f"[{ticker}] {sig['signal_type']} @ {bar_time.strftime('%Y-%m-%d %H:%M')} "
                 f"| {right} {strike} exp {expiry}")

        opt_df = fetch_option_bars(ib, ticker, strike, expiry, right)
        if opt_df is None or len(opt_df) == 0:
            log.warning(f"  Skipped — no option data")
            continue

        day_signals = by_date.get(sig["date"], [])
        result = simulate_trade(bar_time, opt_df, sig["direction"], day_signals)
        if result is None:
            log.warning(f"  Skipped — simulation returned no result")
            continue

        trade = {
            "ticker":      ticker,
            "date":        sig["date"],
            "signal_type": sig["signal_type"],
            "strike":      strike,
            "expiry":      expiry,
            "right":       right,
            **result,
        }
        trades.append(trade)
        log.info(f"  PnL: {result['pnl_pct']:+.1f}%  exit: {result['exit_reason']}  "
                 f"hold: {result['hold_minutes']}m")

    return trades


def compute_summary(trades: list) -> dict:
    if not trades:
        return {}
    pnls  = [t["pnl_pct"] for t in trades]
    wins  = [p for p in pnls if p > 0]
    exits = {}
    for t in trades:
        exits[t["exit_reason"]] = exits.get(t["exit_reason"], 0) + 1
    return {
        "total_trades":  len(trades),
        "win_rate":      round(len(wins) / len(pnls) * 100, 1),
        "avg_pnl_pct":   round(sum(pnls) / len(pnls), 2),
        "total_pnl_pct": round(sum(pnls), 2),
        "best_trade":    round(max(pnls), 2),
        "worst_trade":   round(min(pnls), 2),
        "exit_reasons":  exits,
    }


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    util.startLoop()
    ib = IB()
    ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID, timeout=15)
    log.info(f"Connected to IB TWS/Gateway at {IB_HOST}:{IB_PORT}")

    try:
        for ticker in TICKERS:
            log.info(f"\n{'='*60}")
            log.info(f"  Running options backtest: {ticker}")
            log.info(f"{'='*60}")

            trades  = run_backtest(ticker, ib)
            summary = compute_summary(trades)

            fname = f"option_backtest_{ticker}.json"

            # Merge with existing results — deduplicate by entry_time
            existing_trades = []
            if os.path.exists(fname):
                try:
                    with open(fname) as f:
                        existing = json.load(f)
                    existing_trades = existing.get("trades", [])
                except Exception:
                    pass

            existing_keys = {t["entry_time"] for t in existing_trades}
            new_trades    = [t for t in trades if t["entry_time"] not in existing_keys]
            all_trades    = existing_trades + new_trades
            # Sort chronologically
            all_trades.sort(key=lambda t: t["entry_time"])
            merged_summary = compute_summary(all_trades)

            output = {
                "ticker":        ticker,
                "generated_at":  dt.datetime.now().isoformat(),
                "lookback_days": LOOKBACK_DAYS,
                "summary":       merged_summary,
                "trades":        all_trades,
            }

            with open(fname, "w") as f:
                json.dump(output, f, indent=2)
            log.info(f"Saved {fname}: +{len(new_trades)} new, {len(all_trades)} total trades")

            if merged_summary:
                log.info(f"  Total trades: {merged_summary['total_trades']}")
                log.info(f"  Win rate:     {merged_summary['win_rate']}%")
                log.info(f"  Avg PnL:      {merged_summary['avg_pnl_pct']:+.2f}%")
                log.info(f"  Total PnL:    {merged_summary['total_pnl_pct']:+.2f}%")
                log.info(f"  Best/Worst:   {merged_summary['best_trade']:+.1f}% / {merged_summary['worst_trade']:+.1f}%")

    finally:
        ib.disconnect()
        log.info("Disconnected from IB")
