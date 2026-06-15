"""
backtest_slope_r2.py

Backtests a new entry logic based on slope_sq + R² peak exhaustion:

  SHORT entry: slope_sq held > +0.5 for 3+ bars, R² peaked,
               slope_sq now declining by > 0.10 from peak, R² declining by > 0.05
  LONG  entry: slope_sq held < -0.5 for 3+ bars, R² peaked,
               slope_sq now rising by > 0.10 from trough, R² declining by > 0.05

Exit: opposite entry condition fires OR trailing stop on option OR EOD.

Run: python backtest_slope_r2.py
"""

import os, json, datetime as dt, logging
import numpy as np
import pandas as pd
from pathlib import Path

from build_dataset import build_feature_dataset
from intraday_signals import add_intraday_features
from backtest_options import (
    derive_option_contract, fetch_option_bars_polygon,
    SESSION_START, EOD_CUTOFF, OTM_CUTOFF,
    TRAIL_ACTIVATE, TRAIL_FROM_HWM, STOP_LOSS_PCT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# ── PARAMETERS ────────────────────────────────────────────────────────────────
TICKERS       = ["TSLA"]
SIGNAL_DAYS   = 59
SQ_ZONE_THR   = 0.50    # slope_sq must exceed ±this to be in the "strong zone"
SQ_BARS_MIN   = 5       # must hold in zone for at least this many bars (tightened 3→5)
SQ_REVERSAL   = 0.20    # slope_sq must reverse this much from extreme (tightened 0.10→0.20)
R2_PEAK_MIN   = 0.50    # R² must reach at least this during the zone (tightened 0.40→0.50)
R2_DECLINE    = 0.08    # R² must have declined this much from zone peak (tightened 0.05→0.08)
COOLDOWN_BARS = 8       # bars between signals (40 min on 5m)

# ── SIGNAL GENERATOR ──────────────────────────────────────────────────────────
def compute_slope_sq(dv):
    atr = dv["ATR"].replace(0, np.nan)
    return np.tanh(5 * dv["lr_slope"] / atr).fillna(0)

def generate_signals(ticker: str):
    today      = dt.date.today()
    start_date = (today - dt.timedelta(days=SIGNAL_DAYS)).strftime("%Y-%m-%d")
    end_date   = (today + dt.timedelta(days=1)).strftime("%Y-%m-%d")

    log.info(f"[{ticker}] loading {SIGNAL_DAYS}-day 5m data...")
    df = build_feature_dataset(ticker, start_date=start_date,
                               end_date=end_date, timeframe="5m")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("US/Central")
    df = add_intraday_features(df)

    trading_days = sorted(set(df.index.date))
    signals = []

    for day in trading_days:
        dv = df[df.index.date == day]
        dv = dv[(dv.index.time >= SESSION_START) &
                (dv.index.time <= pd.Timestamp("15:00").time())]
        if len(dv) < SQ_BARS_MIN + 2:
            continue

        sq   = compute_slope_sq(dv)
        r2   = dv["lr_r2"]
        px   = dv["Close"]

        # State machine per day
        state          = "neutral"   # neutral | long_zone | short_zone
        bars_in_zone   = 0
        sq_extreme     = None        # peak (short) or trough (long)
        r2_zone_peak   = None        # max R² seen while in zone
        cooldown       = 0

        for i in range(len(dv)):
            if cooldown > 0:
                cooldown -= 1
                continue

            t      = dv.index[i]
            sq_val = float(sq.iloc[i])
            r2_val = float(r2.iloc[i]) if not pd.isna(r2.iloc[i]) else 0.0
            price  = float(px.iloc[i])

            if t.time() >= OTM_CUTOFF:
                break

            # ── SHORT ZONE: slope_sq rising to positive extreme ──────────────
            if sq_val >= SQ_ZONE_THR:
                if state != "short_zone":
                    state = "short_zone"
                    bars_in_zone = 0
                    sq_extreme   = sq_val
                    r2_zone_peak = r2_val
                else:
                    bars_in_zone += 1
                    if sq_val > sq_extreme:
                        sq_extreme = sq_val
                    if r2_val > (r2_zone_peak or 0):
                        r2_zone_peak = r2_val

            # ── LONG ZONE: slope_sq falling to negative extreme ───────────────
            elif sq_val <= -SQ_ZONE_THR:
                if state != "long_zone":
                    state = "long_zone"
                    bars_in_zone = 0
                    sq_extreme   = sq_val
                    r2_zone_peak = r2_val
                else:
                    bars_in_zone += 1
                    if sq_val < sq_extreme:
                        sq_extreme = sq_val
                    if r2_val > (r2_zone_peak or 0):
                        r2_zone_peak = r2_val

            # ── MIDDLE ZONE: check for exit from extreme ──────────────────────
            else:
                if state == "short_zone" and bars_in_zone >= SQ_BARS_MIN:
                    sq_fall  = (sq_extreme - sq_val) if sq_extreme is not None else 0
                    r2_fall  = ((r2_zone_peak or 0) - r2_val)
                    r2_ok    = (r2_zone_peak or 0) >= R2_PEAK_MIN and r2_fall >= R2_DECLINE

                    if sq_fall >= SQ_REVERSAL and r2_ok:
                        signals.append({
                            "date":             day.strftime("%Y-%m-%d"),
                            "bar_time":         t.isoformat(),
                            "direction":        "short",
                            "signal_type":      "SQ_SHORT",
                            "underlying_price": price,
                            "sq_extreme":       round(sq_extreme, 4),
                            "sq_now":           round(sq_val, 4),
                            "sq_fall":          round(sq_fall, 4),
                            "r2_peak":          round(r2_zone_peak or 0, 4),
                            "r2_now":           round(r2_val, 4),
                            "bars_in_zone":     bars_in_zone,
                        })
                        log.info(f"  SQ_SHORT @ {t.strftime('%H:%M')} | "
                                 f"sq_peak={sq_extreme:.2f} fall={sq_fall:.2f} "
                                 f"r2_peak={r2_zone_peak:.2f} r2_now={r2_val:.2f}")
                        cooldown = COOLDOWN_BARS

                elif state == "long_zone" and bars_in_zone >= SQ_BARS_MIN:
                    sq_rise  = (sq_val - sq_extreme) if sq_extreme is not None else 0
                    r2_fall  = ((r2_zone_peak or 0) - r2_val)
                    r2_ok    = (r2_zone_peak or 0) >= R2_PEAK_MIN and r2_fall >= R2_DECLINE

                    if sq_rise >= SQ_REVERSAL and r2_ok:
                        signals.append({
                            "date":             day.strftime("%Y-%m-%d"),
                            "bar_time":         t.isoformat(),
                            "direction":        "long",
                            "signal_type":      "SQ_LONG",
                            "underlying_price": price,
                            "sq_extreme":       round(sq_extreme, 4),
                            "sq_now":           round(sq_val, 4),
                            "sq_rise":          round(sq_rise, 4),
                            "r2_peak":          round(r2_zone_peak or 0, 4),
                            "r2_now":           round(r2_val, 4),
                            "bars_in_zone":     bars_in_zone,
                        })
                        log.info(f"  SQ_LONG  @ {t.strftime('%H:%M')} | "
                                 f"sq_trough={sq_extreme:.2f} rise={sq_rise:.2f} "
                                 f"r2_peak={r2_zone_peak:.2f} r2_now={r2_val:.2f}")
                        cooldown = COOLDOWN_BARS

                # Reset state
                state = "neutral"
                bars_in_zone = 0
                sq_extreme   = None
                r2_zone_peak = None

    log.info(f"[{ticker}] {len(signals)} SQ signals across {len(trading_days)} days")
    return signals, df

# ── TRADE SIMULATOR ───────────────────────────────────────────────────────────
def simulate_trade(signal_bar_time, option_bars, direction, day_signals):
    bars = option_bars[option_bars.index > signal_bar_time]
    if len(bars) == 0:
        return None

    entry_time  = bars.index[0]
    entry_price = float(bars.iloc[0]["open"])
    if entry_price <= 0 or pd.isna(entry_price):
        return None

    # Opposite-direction signals as exit triggers
    reversal_times = set()
    for s in day_signals:
        t = pd.Timestamp(s["bar_time"])
        if t.tzinfo is None:
            t = t.tz_localize(signal_bar_time.tzinfo)
        if s["direction"] != direction and t > signal_bar_time:
            reversal_times.add(t)

    hwm            = entry_price
    trailing_active = False

    for i, (bar_time, row) in enumerate(bars.iterrows()):
        mid = float(row["close"])
        if mid <= 0 or pd.isna(mid):
            continue
        if mid > hwm:
            hwm = mid
        if not trailing_active and mid >= entry_price * TRAIL_ACTIVATE:
            trailing_active = True

        if bar_time.time() >= EOD_CUTOFF:
            pnl_pct = round((mid - entry_price) / entry_price * 100, 2)
            pnl_dol = round((mid - entry_price) * 100, 2)
            return dict(entry_time=entry_time.isoformat(),
                        exit_time=bar_time.isoformat(),
                        entry_price=entry_price, exit_price=mid,
                        high_water_mark=hwm, pnl_pct=pnl_pct,
                        pnl_dollar=pnl_dol, hold_minutes=int(
                            (bar_time - entry_time).total_seconds() / 60),
                        exit_reason="EOD", direction=direction)

        reason = None
        if bar_time in reversal_times:
            reason = "Signal Reversal"
        elif mid <= entry_price * STOP_LOSS_PCT:
            reason = "Stop Loss (50%)"
        elif trailing_active and mid <= hwm * TRAIL_FROM_HWM:
            reason = "Trailing Stop"

        if reason:
            remaining = bars.iloc[i + 1:]
            if len(remaining) == 0:
                exit_price = mid
                exit_time  = bar_time
            else:
                exit_price = float(remaining.iloc[0]["open"])
                exit_time  = remaining.index[0]
            pnl_pct = round((exit_price - entry_price) / entry_price * 100, 2)
            pnl_dol = round((exit_price - entry_price) * 100, 2)
            return dict(entry_time=entry_time.isoformat(),
                        exit_time=exit_time.isoformat(),
                        entry_price=entry_price, exit_price=exit_price,
                        high_water_mark=hwm, pnl_pct=pnl_pct,
                        pnl_dollar=pnl_dol, hold_minutes=int(
                            (exit_time - entry_time).total_seconds() / 60),
                        exit_reason=reason, direction=direction)

    last = bars.iloc[-1]
    pnl_pct = round((float(last["close"]) - entry_price) / entry_price * 100, 2)
    pnl_dol = round((float(last["close"]) - entry_price) * 100, 2)
    return dict(entry_time=entry_time.isoformat(),
                exit_time=bars.index[-1].isoformat(),
                entry_price=entry_price, exit_price=float(last["close"]),
                high_water_mark=hwm, pnl_pct=pnl_pct, pnl_dollar=pnl_dol,
                hold_minutes=int((bars.index[-1]-entry_time).total_seconds()/60),
                exit_reason="EOD", direction=direction)

# ── BACKTEST RUNNER ───────────────────────────────────────────────────────────
def run_backtest(ticker: str):
    all_signals, _ = generate_signals(ticker)

    by_date = {}
    for s in all_signals:
        by_date.setdefault(s["date"], []).append(s)

    trades      = []
    last_exit   = {}   # date -> last exit time (position tracker)
    last_dir    = {}

    for date_str, day_sigs in sorted(by_date.items()):
        for sig in sorted(day_sigs, key=lambda s: s["bar_time"]):
            bar_time = pd.Timestamp(sig["bar_time"])
            if bar_time.time() >= OTM_CUTOFF:
                break

            # One position at a time
            le = last_exit.get(date_str)
            ld = last_dir.get(date_str)
            if le is not None and bar_time < le and sig["direction"] == ld:
                log.info(f"  SKIP {sig['signal_type']} {sig['direction']} — position open")
                continue

            strike, expiry, right = derive_option_contract(
                ticker, sig["direction"], sig["underlying_price"], bar_time)

            log.info(f"[{ticker}] {sig['signal_type']} @ "
                     f"{bar_time.strftime('%Y-%m-%d %H:%M')} "
                     f"| {right}{strike} exp {expiry} "
                     f"sq={sig['sq_now']:.2f} r2={sig['r2_now']:.2f}")

            opt_df = fetch_option_bars_polygon(ticker, strike, expiry, right)
            if opt_df is None or len(opt_df) == 0:
                log.warning("  Skipped — no Polygon data")
                continue

            result = simulate_trade(bar_time, opt_df, sig["direction"], day_sigs)
            if result is None:
                log.warning("  Skipped — no valid bars")
                continue

            last_exit[date_str] = pd.Timestamp(result["exit_time"])
            last_dir[date_str]  = sig["direction"]

            trade = {
                "ticker":      ticker,
                "date":        sig["date"],
                "signal_type": sig["signal_type"],
                "strike":      strike,
                "expiry":      expiry,
                "right":       right,
                "sq_extreme":  sig["sq_extreme"],
                "r2_peak":     sig["r2_peak"],
                **result,
            }
            trades.append(trade)
            log.info(f"  PnL: {result['pnl_pct']:+.1f}%  ${result['pnl_dollar']:+.2f} "
                     f"| {result['exit_reason']}  {result['hold_minutes']}m")

    return trades

# ── SUMMARY ───────────────────────────────────────────────────────────────────
def summarise(trades):
    if not trades:
        return {}
    pnls  = [t["pnl_pct"]    for t in trades]
    dlrs  = [t["pnl_dollar"] for t in trades]
    wins  = [p for p in pnls if p > 0]
    exits = {}
    for t in trades:
        exits[t["exit_reason"]] = exits.get(t["exit_reason"], 0) + 1
    return dict(total_trades=len(trades),
                win_rate=round(len(wins)/len(pnls)*100, 1),
                avg_pnl_pct=round(sum(pnls)/len(pnls), 2),
                total_pnl_pct=round(sum(pnls), 2),
                avg_pnl_dollar=round(sum(dlrs)/len(dlrs), 2),
                total_pnl_dollar=round(sum(dlrs), 2),
                best_trade=round(max(pnls), 2),
                worst_trade=round(min(pnls), 2),
                exit_reasons=exits)

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    all_trades = []
    for ticker in TICKERS:
        log.info(f"\n{'='*55}\n  SQ+R² Backtest: {ticker}\n{'='*55}")
        trades = run_backtest(ticker)
        s = summarise(trades)
        all_trades.extend(trades)

        print(f"\n{ticker}: {s.get('total_trades',0)} trades | "
              f"win {s.get('win_rate',0):.1f}% | "
              f"avg {s.get('avg_pnl_pct',0):+.1f}% | "
              f"total ${s.get('total_pnl_dollar',0):+,.0f}")
        for reason, n in sorted(s.get("exit_reasons",{}).items(), key=lambda x:-x[1]):
            print(f"  {reason:<22} {n}")

    # Save results
    out = Path("option_backtest_sq_r2.json")
    with open(out, "w") as f:
        json.dump({"trades": all_trades,
                   "summary": summarise(all_trades),
                   "params": dict(SQ_ZONE_THR=SQ_ZONE_THR, SQ_BARS_MIN=SQ_BARS_MIN,
                                  SQ_REVERSAL=SQ_REVERSAL, R2_PEAK_MIN=R2_PEAK_MIN,
                                  R2_DECLINE=R2_DECLINE, COOLDOWN_BARS=COOLDOWN_BARS)},
                  f, indent=2)
    log.info(f"Saved to {out}")

    s = summarise(all_trades)
    print(f"\nCOMBINED: {s.get('total_trades',0)} trades | "
          f"win {s.get('win_rate',0):.1f}% | "
          f"total ${s.get('total_pnl_dollar',0):+,.0f}")
