"""
Options strategy backtest — Polygon.io data source.

- Signals:  same logic as alert_spicy_engine.py (31-day TOS_Trail warmup)
- Prices:   Polygon.io historical 5m OHLCV (cached to polygon_cache/)
- Fills:    NEXT BAR OPEN after signal / exit trigger (realistic slippage model)
- Rate:     5 API calls/minute — sleeps 12s between uncached requests
- Output:   option_backtest_{ticker}.json  (append mode, no re-fetch if cached)
"""

import datetime as dt
import json
import logging
import os
import sys
import time

import pandas as pd
import requests
from dotenv import load_dotenv

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
        logging.FileHandler(
            f"logs/backtest_options_{dt.date.today():%Y-%m-%d}.log",
            encoding="utf-8",
        ),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────
TICKERS      = ["SPY", "TSLA"]
TIMEFRAME    = "5m"
SIGNAL_DAYS  = 59       # within yfinance 60-day 5m limit; 31+ days for TOS_Trail warmup

POLYGON_API_KEY = os.getenv("MASSIVE_API_KEY", os.getenv("POLYGON_API_KEY", "")).strip()
CACHE_DIR       = "polygon_cache"
API_SLEEP       = 12    # seconds between Polygon calls (5 req/min limit)

# Mirror options_executor.py exactly
STOP_LOSS_PCT  = 0.50
TRAIL_ACTIVATE = 1.50
TRAIL_FROM_HWM = 0.75
ENTRY_CUTOFF   = dt.time(14,  0)
EOD_CUTOFF     = dt.time(14, 55)
OTM_CUTOFF     = dt.time(13, 30)

STRIKE_INC = {"SPY": 1.0, "TSLA": 2.5}

SESSION_START = pd.Timestamp("08:30").time()
SESSION_END   = pd.Timestamp("15:00").time()


# ── SESSION STATE ─────────────────────────────────────────────────────────────

def _fresh_session() -> dict:
    return {
        "date": None, "orb_fired": False, "orb_dir": None, "orb_bar_loc": None,
        "position": None, "position_src": None, "last_alert": None, "mr_entry_bar": None,
    }


# ── SIGNAL REPLAY ─────────────────────────────────────────────────────────────

def generate_signals(ticker: str) -> list:
    """Walk-forward replay of the live alert engine session state over SIGNAL_DAYS."""
    today      = dt.date.today()
    end_date   = today + dt.timedelta(days=1)             # +1 so yfinance includes today
    start_date = today - dt.timedelta(days=SIGNAL_DAYS)   # anchor from today, not end_date

    log.info(f"[{ticker}] fetching {SIGNAL_DAYS}-day 5m data from yfinance...")
    df = build_feature_dataset(
        ticker,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        timeframe=TIMEFRAME,
    )
    if df is None or len(df) == 0:
        log.error(f"[{ticker}] empty DataFrame")
        return []

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("US/Central")
    df = add_intraday_features(df)

    orb_deg     = ORB_SLOPE_DEG.get(ticker, ORB_SLOPE_DEG_DEFAULT)
    r2_thr      = ORB_R2_THR.get(ticker, ORB_R2_THR_DEFAULT)
    accum_start = ORB_ACCUM_START.get(ticker, ORB_ACCUM_START_DEFAULT)
    use_sliding = ORB_USE_SLIDING.get(ticker, ORB_USE_SLIDING_DEFAULT)
    _accum      = accum_start if accum_start is not None else ORB_ACCUM_START_DEFAULT

    # Replay full SIGNAL_DAYS; all signals are eligible for Polygon lookup
    cutoff       = pd.Timestamp(start_date).tz_localize("US/Central")
    trading_days = sorted(set(df[df.index >= cutoff].index.date))
    signals      = []

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

            if mr_ready:
                lsig_mr,     ssig_mr     = run_new(dv, S_THR.get(ticker, 0.60), ATR_FLOOR.get(ticker, 0.50))
                lsig_mr_rev, ssig_mr_rev = run_new(dv, S_THR.get(ticker, 0.60), ATR_FLOOR.get(ticker, 0.50), orb_reversal=True)
            else:
                lsig_mr = ssig_mr = lsig_mr_rev = ssig_mr_rev = pd.Series(False, index=dv.index)

            lsig_orb, ssig_orb = run_orb_slope(dv, angle_deg=orb_deg, r2_thr=r2_thr, accum_start=accum_start)
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

            signal_dir = signal_type = None

            if is_orb_long and not state["orb_fired"]:
                state.update(orb_fired=True, orb_dir="long", orb_bar_loc=last_idx,
                             position="long", position_src="orb")
                signal_dir, signal_type = "long", "ORB_LONG"

            elif is_orb_short and not state["orb_fired"]:
                state.update(orb_fired=True, orb_dir="short", orb_bar_loc=last_idx,
                             position="short", position_src="orb")
                signal_dir, signal_type = "short", "ORB_SHORT"

            elif (is_mr_long_rev or is_mr_long) and state["position"] != "long":
                if state["position"] == "short" and state["position_src"] == "orb":
                    if is_mr_long_rev and _orb_reversal_confirmed(
                            dv, state["orb_bar_loc"], "short", last_idx,
                            threshold=REVERSAL_ANGLE, r2_thr=r2_thr):
                        state.update(position="long", position_src="mr", mr_entry_bar=last_idx)
                        signal_dir, signal_type = "long", "MR_LONG"
                elif is_mr_long:
                    state.update(position="long", position_src="mr", mr_entry_bar=last_idx)
                    signal_dir, signal_type = "long", "MR_LONG"

            elif (is_mr_short_rev or is_mr_short) and state["position"] != "short":
                if state["position"] == "long" and state["position_src"] == "orb":
                    if is_mr_short_rev and _orb_reversal_confirmed(
                            dv, state["orb_bar_loc"], "long", last_idx,
                            threshold=REVERSAL_ANGLE, r2_thr=r2_thr):
                        state.update(position="short", position_src="mr", mr_entry_bar=last_idx)
                        signal_dir, signal_type = "short", "MR_SHORT"
                elif is_mr_short:
                    state.update(position="short", position_src="mr", mr_entry_bar=last_idx)
                    signal_dir, signal_type = "short", "MR_SHORT"

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

    log.info(f"[{ticker}] {len(signals)} signals across {len(trading_days)} trading days")
    return signals, df


# ── OPTION CONTRACT ───────────────────────────────────────────────────────────

def derive_option_contract(ticker: str, direction: str,
                           underlying_price: float,
                           signal_time: pd.Timestamp) -> tuple:
    inc    = STRIKE_INC.get(ticker, 1.0)
    strike = round(round(underlying_price / inc) * inc, 2)
    right  = "C" if direction == "long" else "P"
    today  = signal_time.date()

    if signal_time.time() <= OTM_CUTOFF:
        expiry = today.strftime("%Y%m%d")
    else:
        next_day = today + dt.timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += dt.timedelta(days=1)
        expiry = next_day.strftime("%Y%m%d")

    return strike, expiry, right


def polygon_symbol(ticker: str, expiry: str, right: str, strike: float) -> str:
    """Build Polygon option ticker: O:SPY260610C00736000"""
    exp_yy   = expiry[2:]                     # "20260610" -> "260610"
    strike_i = int(round(strike * 1000))      # 736.0 -> 736000
    return f"O:{ticker}{exp_yy}{right}{strike_i:08d}"


# ── POLYGON DATA FETCH (CACHED) ───────────────────────────────────────────────

def _cache_path(symbol: str, date_str: str) -> str:
    safe = symbol.replace(":", "_")
    return os.path.join(CACHE_DIR, f"{safe}_{date_str}.json")


def fetch_option_bars_polygon(ticker: str, strike: float,
                              expiry: str, right: str) -> pd.DataFrame | None:
    symbol   = polygon_symbol(ticker, expiry, right, strike)
    date_str = f"{expiry[:4]}-{expiry[4:6]}-{expiry[6:]}"   # "2026-06-10"
    cp       = _cache_path(symbol, date_str)

    # ── Cache hit ─────────────────────────────────────────────────────────────
    if os.path.exists(cp):
        with open(cp) as f:
            data = json.load(f)
        df = _polygon_to_df(data)
        if df is not None:
            log.debug(f"  [cache] {symbol}")
        return df

    # ── API call ──────────────────────────────────────────────────────────────
    log.info(f"  [polygon] fetching {symbol} ...")
    time.sleep(API_SLEEP)   # respect 5 req/min limit

    url = (f"https://api.polygon.io/v2/aggs/ticker/{symbol}"
           f"/range/5/minute/{date_str}/{date_str}")
    params = {"adjusted": "false", "sort": "asc", "limit": 500,
              "apiKey": POLYGON_API_KEY}

    try:
        resp = requests.get(url, params=params, timeout=20)
    except Exception as e:
        log.warning(f"  Request error: {e}")
        return None

    if resp.status_code != 200:
        log.warning(f"  Polygon HTTP {resp.status_code}: {resp.text[:200]}")
        return None

    data = resp.json()

    if data.get("resultsCount", 0) == 0 or not data.get("results"):
        log.warning(f"  No data from Polygon for {symbol}")
        # Cache the empty response so we don't retry next run
        with open(cp, "w") as f:
            json.dump(data, f)
        return None

    # Cache successful response
    with open(cp, "w") as f:
        json.dump(data, f)

    return _polygon_to_df(data)


def _polygon_to_df(data: dict) -> pd.DataFrame | None:
    results = data.get("results")
    if not results:
        return None

    records = []
    for r in results:
        records.append({
            "time":   pd.Timestamp(r["t"], unit="ms", tz="UTC").tz_convert("US/Central"),
            "open":   r["o"],
            "high":   r["h"],
            "low":    r["l"],
            "close":  r["c"],
            "volume": r.get("v", 0),
        })

    df = pd.DataFrame(records).set_index("time")
    return df


# ── TRADE SIMULATION (NEXT-BAR-OPEN FILLS) ────────────────────────────────────

def _make_result(direction: str, entry_price: float, exit_price: float,
                 hwm: float, reason: str,
                 entry_time: pd.Timestamp, exit_time: pd.Timestamp) -> dict:
    hold_mins  = int((exit_time - entry_time).total_seconds() / 60)
    pnl_pct    = round((exit_price - entry_price) / entry_price * 100, 2)
    pnl_dollar = round((exit_price - entry_price) * 100, 2)  # 1 contract = 100 shares
    return {
        "direction":       direction,
        "entry_price":     round(float(entry_price), 2),
        "exit_price":      round(float(exit_price),  2),
        "high_water_mark": round(float(hwm), 2),
        "pnl_pct":         pnl_pct,
        "pnl_dollar":      pnl_dollar,
        "hold_minutes":    hold_mins,
        "exit_reason":     reason,
        "entry_time":      entry_time.isoformat(),
        "exit_time":       exit_time.isoformat(),
    }


def simulate_trade(signal_bar_time: pd.Timestamp, option_bars: pd.DataFrame,
                   direction: str, day_signals: list,
                   use_stop_loss: bool = True,
                   dv_session: pd.DataFrame = None,
                   mr_entry_bar: int = None,
                   ticker: str = None) -> dict | None:
    """
    Entry at NEXT BAR OPEN after signal bar.
    Exit fill also at NEXT BAR OPEN after the trigger bar.
    EOD (14:55) exits at that bar's close (no next bar).
    use_stop_loss=False: ORB signals — let trade run to EOD, flip, or trail.
    """
    # Bars strictly after signal bar
    bars = option_bars[option_bars.index > signal_bar_time]
    if len(bars) == 0:
        return None

    entry_time  = bars.index[0]
    entry_price = float(bars.iloc[0]["open"])
    if entry_price <= 0 or pd.isna(entry_price):
        return None

    # Opposite-direction signals on same day (reversal exit triggers)
    reversal_times = set()
    for s in day_signals:
        t = pd.Timestamp(s["bar_time"])
        if t.tzinfo is None:
            t = t.tz_localize(signal_bar_time.tzinfo)
        if s["direction"] != direction and t > signal_bar_time:
            reversal_times.add(t)

    hwm             = entry_price
    trailing_active = False
    is_upgraded     = False   # MR→ORB upgrade state (disables stop loss once confirmed)

    # Pre-compute ORB upgrade params (only used for MR_UPGRADED trades)
    if dv_session is not None and ticker is not None:
        _orb_deg = ORB_SLOPE_DEG.get(ticker, ORB_SLOPE_DEG_DEFAULT)
        _r2_thr  = ORB_R2_THR.get(ticker, ORB_R2_THR_DEFAULT)
        _acc     = ORB_ACCUM_START.get(ticker, ORB_ACCUM_START_DEFAULT)
        _acc     = _acc if _acc is not None else ORB_ACCUM_START_DEFAULT
    else:
        _orb_deg = _r2_thr = _acc = None

    for i, (bar_time, row) in enumerate(bars.iterrows()):
        mid = float(row["close"])
        if mid <= 0 or pd.isna(mid):
            continue

        if mid > hwm:
            hwm = mid

        if not trailing_active and mid >= entry_price * TRAIL_ACTIVATE:
            trailing_active = True
            log.debug(f"  Trailing stop armed at ${mid:.2f}")

        # For MR_UPGRADED trades: check if upgrade has fired at this bar
        # Once upgraded, stop loss is disabled for the remainder of the trade
        if dv_session is not None and not is_upgraded:
            matching = dv_session[dv_session.index <= bar_time]
            last_idx = len(matching) - 1
            if last_idx > mr_entry_bar:
                try:
                    if mr_orb_upgrade_qualifies(
                            matching, mr_entry_bar, last_idx,
                            _orb_deg, _r2_thr, direction, _acc):
                        is_upgraded = True
                        log.debug(f"  MR→ORB upgrade at {bar_time.strftime('%H:%M')} — stop loss off")
                except Exception:
                    pass

        # EOD: exit at this bar's close (no next bar makes sense here)
        if bar_time.time() >= EOD_CUTOFF:
            return _make_result(direction, entry_price, mid, hwm,
                                "EOD", entry_time, bar_time)

        # Stop loss: active for MR_ONLY always; for MR_UPGRADED only before upgrade bar
        sl_active = use_stop_loss and not is_upgraded

        # Determine if an exit should trigger
        reason = None
        if bar_time in reversal_times:
            reason = "Signal Reversal"
        elif sl_active and mid <= entry_price * STOP_LOSS_PCT:
            reason = "Stop Loss (50%)"
        elif trailing_active and mid <= hwm * TRAIL_FROM_HWM:
            reason = "Trailing Stop"

        if reason:
            # Fill at NEXT bar open
            remaining = bars.iloc[i + 1:]
            if len(remaining) == 0:
                return _make_result(direction, entry_price, mid, hwm,
                                    reason, entry_time, bar_time)
            exit_time  = remaining.index[0]
            exit_price = float(remaining.iloc[0]["open"])
            return _make_result(direction, entry_price, exit_price, hwm,
                                reason, entry_time, exit_time)

    # Ran out of bars without hitting an exit — treat as EOD
    last = bars.iloc[-1]
    return _make_result(direction, entry_price, float(last["close"]), hwm,
                        "EOD", entry_time, bars.index[-1])


# ── BACKTEST RUNNER ───────────────────────────────────────────────────────────

def run_backtest(ticker: str) -> list:
    """
    Models executor behavior: ONE position at a time per day.
    - Same-direction signal while position open → SKIP (executor already holds that direction)
    - Opposite-direction signal while position open → FLIP (previous trade exits via Signal Reversal)
    """
    all_signals, full_df = generate_signals(ticker)
    by_date: dict[str, list] = {}
    for s in all_signals:
        by_date.setdefault(s["date"], []).append(s)

    trades = []

    for date_str, day_sigs in sorted(by_date.items()):
        day_date = pd.Timestamp(date_str).date()
        dv_full  = full_df[full_df.index.date == day_date]
        dv_full  = dv_full[(dv_full.index.time >= SESSION_START) &
                           (dv_full.index.time <= SESSION_END)]

        orb_deg = ORB_SLOPE_DEG.get(ticker, ORB_SLOPE_DEG_DEFAULT)
        r2_thr  = ORB_R2_THR.get(ticker, ORB_R2_THR_DEFAULT)
        acc     = ORB_ACCUM_START.get(ticker, ORB_ACCUM_START_DEFAULT)
        acc     = acc if acc is not None else ORB_ACCUM_START_DEFAULT

        # Position state for this day
        last_exit_time = None   # pd.Timestamp when last trade exited
        last_direction = None   # 'long' | 'short'

        for sig in sorted(day_sigs, key=lambda s: s["bar_time"]):
            bar_time = pd.Timestamp(sig["bar_time"])

            if bar_time.time() >= OTM_CUTOFF:
                # After 13:30 CT: executor uses 1DTE options (next-day expiry).
                # Cross-day bar simulation is unreliable — stop processing for today.
                break

            # ── Executor rule: skip if same direction while still in a position ──
            if (last_exit_time is not None
                    and bar_time < last_exit_time
                    and sig["direction"] == last_direction):
                log.info(f"  SKIP {sig['signal_type']} {sig['direction'].upper()} "
                         f"— already {last_direction} open until {last_exit_time.strftime('%H:%M')}")
                continue
            # Opposite direction while open → this is a flip; allowed through.
            # simulate_trade will exit the previous position via Signal Reversal.

            strike, expiry, right = derive_option_contract(
                ticker, sig["direction"], sig["underlying_price"], bar_time)

            log.info(f"[{ticker}] {sig['signal_type']} @ {bar_time.strftime('%Y-%m-%d %H:%M')} "
                     f"| {right} {strike} exp {expiry}")

            opt_df = fetch_option_bars_polygon(ticker, strike, expiry, right)
            if opt_df is None or len(opt_df) == 0:
                log.warning(f"  Skipped - no Polygon data")
                continue

            is_orb = "ORB" in sig["signal_type"]

            if is_orb:
                trade_type    = "ORB"
                use_stop_loss = False
                dv_session    = None
                mr_entry_bar  = None
            else:
                before_entry = dv_full[dv_full.index <= bar_time]
                mr_bar       = max(0, len(before_entry) - 1)
                upgraded     = False
                for last_idx in range(mr_bar + 1, len(dv_full)):
                    try:
                        if mr_orb_upgrade_qualifies(dv_full, mr_bar, last_idx,
                                                    orb_deg, r2_thr, sig["direction"], acc):
                            upgraded = True
                            break
                    except Exception:
                        pass
                if upgraded:
                    trade_type    = "MR_UPGRADED_ORB"
                    use_stop_loss = True
                    dv_session    = dv_full
                    mr_entry_bar  = mr_bar
                else:
                    trade_type    = "MR_ONLY"
                    use_stop_loss = True
                    dv_session    = None
                    mr_entry_bar  = None

            result = simulate_trade(
                bar_time, opt_df, sig["direction"], day_sigs,
                use_stop_loss=use_stop_loss,
                dv_session=dv_session,
                mr_entry_bar=mr_entry_bar,
                ticker=ticker,
            )
            if result is None:
                log.warning(f"  Skipped - no valid bars after signal")
                continue

            # Update position tracker
            last_exit_time = pd.Timestamp(result["exit_time"])
            last_direction = sig["direction"]

            trades.append({
                "ticker":      ticker,
                "date":        date_str,
                "signal_type": sig["signal_type"],
                "trade_type":  trade_type,
                "strike":      strike,
                "expiry":      expiry,
                "right":       right,
                **result,
            })
            log.info(f"  [{trade_type}] PnL: {result['pnl_pct']:+.1f}%  ${result['pnl_dollar']:+.2f} "
                     f"| exit: {result['exit_reason']}  hold: {result['hold_minutes']}m")

    return trades


def compute_summary(trades: list) -> dict:
    if not trades:
        return {}
    pnls_pct    = [t["pnl_pct"]    for t in trades]
    pnls_dollar = [t["pnl_dollar"] for t in trades]
    wins        = [p for p in pnls_pct if p > 0]
    exits       = {}
    for t in trades:
        exits[t["exit_reason"]] = exits.get(t["exit_reason"], 0) + 1
    return {
        "total_trades":     len(trades),
        "win_rate":         round(len(wins) / len(pnls_pct) * 100, 1),
        "avg_pnl_pct":      round(sum(pnls_pct)    / len(pnls_pct),    2),
        "total_pnl_pct":    round(sum(pnls_pct),                        2),
        "avg_pnl_dollar":   round(sum(pnls_dollar)  / len(pnls_dollar), 2),
        "total_pnl_dollar": round(sum(pnls_dollar),                     2),
        "best_trade":       round(max(pnls_pct), 2),
        "worst_trade":      round(min(pnls_pct), 2),
        "exit_reasons":     exits,
    }


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not POLYGON_API_KEY:
        log.error("POLYGON_API_KEY not set in .env")
        sys.exit(1)

    os.makedirs(CACHE_DIR, exist_ok=True)

    for ticker in TICKERS:
        log.info(f"\n{'='*60}\n  Options backtest: {ticker}\n{'='*60}")

        trades  = run_backtest(ticker)
        fname   = f"option_backtest_{ticker}.json"

        # Merge with existing (append mode)
        existing_trades: list = []
        if os.path.exists(fname):
            try:
                with open(fname) as f:
                    existing_trades = json.load(f).get("trades", [])
            except Exception:
                pass

        existing_keys = {t["entry_time"] for t in existing_trades}
        new_trades    = [t for t in trades if t["entry_time"] not in existing_keys]
        all_trades    = sorted(existing_trades + new_trades, key=lambda t: t["entry_time"])
        summary       = compute_summary(all_trades)

        with open(fname, "w") as f:
            json.dump({
                "ticker":        ticker,
                "generated_at":  dt.datetime.now().isoformat(),
                "lookback_days": SIGNAL_DAYS,
                "summary":       summary,
                "trades":        all_trades,
            }, f, indent=2)

        log.info(f"Saved {fname}: +{len(new_trades)} new / {len(all_trades)} total")
        if summary:
            log.info(f"  Trades:     {summary['total_trades']}")
            log.info(f"  Win rate:   {summary['win_rate']}%")
            log.info(f"  Avg PnL:    {summary['avg_pnl_pct']:+.2f}%  ${summary['avg_pnl_dollar']:+.2f}")
            log.info(f"  Total PnL:  {summary['total_pnl_pct']:+.2f}%  ${summary['total_pnl_dollar']:+.2f}")
            log.info(f"  Best/Worst: {summary['best_trade']:+.1f}% / {summary['worst_trade']:+.1f}%")
