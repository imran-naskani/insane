import time
import pandas as pd
from build_dataset import build_feature_dataset
# from build_dataset import floor_5_or_int   # OLD: used for VWAP/range thresholds — no longer needed
# from model import spicy_sauce              # OLD: 2D Kalman — replaced by OLS slope in intraday_signals
from intraday_signals import (
    add_intraday_features,
    run_new, run_orb_slope, run_orb_sliding, sliding_qualifies_direction,
    _orb_reversal_confirmed, REVERSAL_ANGLE,
    S_THR, ATR_FLOOR, ORB_SLOPE_BARS, ORB_REVERSAL_DEG,
    ORB_SLOPE_DEG, ORB_SLOPE_DEG_DEFAULT,
    ORB_R2_THR, ORB_R2_THR_DEFAULT,
    ORB_ACCUM_START, ORB_ACCUM_START_DEFAULT,
    ORB_USE_SLIDING, ORB_USE_SLIDING_DEFAULT,
)
import datetime as dt
from dotenv import load_dotenv
import os
import requests

load_dotenv()

end_date   = dt.date.today() + dt.timedelta(days=1)
start_date = end_date - dt.timedelta(days=31)

# ── CONFIG ────────────────────────────────────────────────────────────────────
# OLD: TICKERS = ["^GSPC", "TSLA", "AAPL"]
# ^GSPC swapped for SPY: real volume needed for correct ATR; validated in SIGNAL_LOGIC.md backtest
TICKERS   = ["TSLA", "SPY", "^GSPC", "QQQ", "TQQQ"]
TIMEFRAME = "5m"

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID   = os.environ["TELEGRAM_CHAT_ID"]

# ── SESSION STATE ─────────────────────────────────────────────────────────────
# Tracks position, ORB state, and alert dedup per ticker. Resets each trading day.
# OLD: last_alert = {}   — simple (bar_time, signal_type) dict, replaced by richer state below
def _fresh_session():
    return {
        "date":         None,
        "orb_fired":    False,
        "orb_dir":      None,         # 'long' | 'short'
        "orb_bar_loc":  None,         # iloc of ORB/sliding/accum signal bar (dynamic)
        "position":     None,         # 'long' | 'short' | None
        "position_src": None,         # 'orb' | 'mr'
        "last_alert":   None,         # (bar_time, signal_type) for dedup
    }

session_state = {t: _fresh_session() for t in TICKERS}

# ── HELPERS ───────────────────────────────────────────────────────────────────
def send_telegram(msg: str):
    url     = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    r = requests.post(url, json=payload, timeout=10)
    r.raise_for_status()

def get_last_closed_bar(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the currently forming 5m candle."""
    return df.iloc[:-1]

def sleep_until_next_5m(offset_seconds=2):
    """Sleep until the next 5-minute boundary + offset."""
    now      = time.time()
    interval = 300
    next_run = ((now // interval) + 1) * interval + offset_seconds
    time.sleep(max(0, next_run - now))

SESSION_START = pd.Timestamp("08:30").time()
SESSION_END   = pd.Timestamp("15:00").time()
EOD_CUTOFF    = pd.Timestamp("14:30").time()

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
print("INSANE Alert Engine started (5m) -- MR + ORB-Slope strategy")
MAX_RETRIES = 10
RETRY_SLEEP = 2

while True:
    try:
        sleep_until_next_5m(offset_seconds=2)
        combined_msgs    = []
        current_bar_time = None

        for ticker in TICKERS:
            # ── p_win was only used by the old quantile logic — kept here for reference
            # OLD: if ticker == "^GSPC": p_win = 84
            # OLD: else:                 p_win = 84

            # ── Fetch ─────────────────────────────────────────────────────
            df = None
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    df = build_feature_dataset(
                        ticker,
                        start_date=start_date.strftime("%Y-%m-%d"),
                        end_date=end_date.strftime("%Y-%m-%d"),
                        timeframe=TIMEFRAME
                    )
                    break
                except Exception as e:
                    print(f"{ticker} attempt {attempt}/{MAX_RETRIES} failed: {e}")
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_SLEEP)

            if df is None:
                print(f"{ticker} skipped after {MAX_RETRIES} failures")
                continue

            # ── Timezone ──────────────────────────────────────────────────
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            df.index = df.index.tz_convert("US/Central")

            # ── Add OLS features (lr_slope, lr_r2) on FULL df ─────────────
            # Must run before session filtering: TOS_Trail warmup carries across days
            df = add_intraday_features(df)
            df = get_last_closed_bar(df)

            # ── Session state: reset on new trading day ────────────────────
            today = dt.date.today()
            state = session_state[ticker]
            if state["date"] != today:
                session_state[ticker] = _fresh_session()
                session_state[ticker]["date"] = today
                state = session_state[ticker]

            # ── Filter to today's closed session bars ─────────────────────
            dv = df[df.index.date == today]
            dv = dv[(dv.index.time >= SESSION_START) &
                    (dv.index.time <= SESSION_END)]

            if len(dv) < 14:  # need 14 bars for OLS warmup
                print(f"{ticker}: {len(dv)} session bars — waiting for warmup")
                continue

            bar_time     = dv.index[-1]
            last_idx     = len(dv) - 1
            display_time = bar_time.strftime("%H:%M CT")
            current_bar_time = bar_time

            # ── EOD force-exit at 14:30 CT ────────────────────────────────
            if bar_time.time() >= EOD_CUTOFF and state["position"]:
                direction = state["position"].upper()
                combined_msgs.append(
                    f"*{'SPX' if ticker == '^GSPC' else ticker}*\n"
                    f"EOD Exit -- close {direction} position\n"
                    f"Time: {display_time}\n"
                    f"Price: {dv['Close'].iloc[-1]:.2f}"
                )
                state["position"]     = None
                state["position_src"] = None
                continue

            # ── Run new signal logic ───────────────────────────────────────
            lsig_mr,  ssig_mr  = run_new(dv, S_THR.get(ticker, 0.60),
                                          ATR_FLOOR.get(ticker, 0.50))

            # Per-ticker ORB parameters (locked strategy)
            orb_deg    = ORB_SLOPE_DEG.get(ticker, ORB_SLOPE_DEG_DEFAULT)
            r2_thr     = ORB_R2_THR.get(ticker, ORB_R2_THR_DEFAULT)
            accum_start = ORB_ACCUM_START.get(ticker, ORB_ACCUM_START_DEFAULT)
            use_sliding = ORB_USE_SLIDING.get(ticker, ORB_USE_SLIDING_DEFAULT)

            # ORB: accumulation window + R2 gate, or fixed 6-bar
            lsig_orb, ssig_orb = run_orb_slope(dv, angle_deg=orb_deg,
                                                r2_thr=r2_thr, accum_start=accum_start)
            orb_fired = lsig_orb.iloc[-1] or ssig_orb.iloc[-1]

            # Sliding window: only if ORB hasn't fired yet today
            lsig_slide, ssig_slide = pd.Series(False, index=dv.index), pd.Series(False, index=dv.index)
            if not state["orb_fired"] and not orb_fired and use_sliding:
                lsig_slide, ssig_slide = run_orb_sliding(dv, orb_deg, r2_thr)

            # Combined ORB signals (accum/fixed + sliding)
            lsig_orb_combined = lsig_orb | lsig_slide
            ssig_orb_combined = ssig_orb | ssig_slide

            is_orb_long  = bool(lsig_orb_combined.iloc[-1])
            is_orb_short = bool(ssig_orb_combined.iloc[-1])
            is_mr_long   = bool(lsig_mr.iloc[-1])
            is_mr_short  = bool(ssig_mr.iloc[-1])

            signal      = None
            signal_type = None

            # ── ORB: accumulation / fixed / sliding (fires once per session) ─
            if is_orb_long and not state["orb_fired"]:
                state.update(
                    orb_fired=True, orb_dir="long",
                    orb_bar_loc=last_idx,   # dynamic: bar 5, 6, or sliding bar 7-18
                    position="long", position_src="orb"
                )
                signal      = "ORB Signal -- LONG (hold to EOD)"
                signal_type = "ORB_LONG"

            elif is_orb_short and not state["orb_fired"]:
                state.update(
                    orb_fired=True, orb_dir="short",
                    orb_bar_loc=last_idx,   # dynamic
                    position="short", position_src="orb"
                )
                signal      = "ORB Signal -- SHORT (hold to EOD)"
                signal_type = "ORB_SHORT"

            # ── MR long ───────────────────────────────────────────────────
            elif is_mr_long and state["position"] != "long":
                # Reversal gate: MR wants to flip an active ORB short
                if state["position"] == "short" and state["position_src"] == "orb":
                    if _orb_reversal_confirmed(dv, state["orb_bar_loc"], "short", last_idx,
                                               threshold=REVERSAL_ANGLE, r2_thr=r2_thr):
                        state.update(position="long", position_src="mr")
                        signal      = "MR Flip -- LONG (ORB reversal confirmed)"
                        signal_type = "MR_LONG"
                    # else: gate blocks — suppress silently
                else:
                    state.update(position="long", position_src="mr")
                    signal      = "MR Signal -- LONG"
                    signal_type = "MR_LONG"

            # ── MR short ──────────────────────────────────────────────────
            elif is_mr_short and state["position"] != "short":
                if state["position"] == "long" and state["position_src"] == "orb":
                    if _orb_reversal_confirmed(dv, state["orb_bar_loc"], "long", last_idx,
                                               threshold=REVERSAL_ANGLE, r2_thr=r2_thr):
                        state.update(position="short", position_src="mr")
                        signal      = "MR Flip -- SHORT (ORB reversal confirmed)"
                        signal_type = "MR_SHORT"
                else:
                    state.update(position="short", position_src="mr")
                    signal      = "MR Signal -- SHORT"
                    signal_type = "MR_SHORT"

            # ── OLD SIGNAL LOGIC (2D Kalman + quantile bands) ─────────────
            # Summary
            # -------
            # spicy_sauce() 2D Kalman filter -> Smooth (price), Slope (slope component)
            # price_delta       = Close - Smooth  (deviation from Kalman trend)
            # price_delta_shift = price_delta - price_delta.shift(1)  (momentum of deviation)
            #
            # Slope_Pos (long entry):
            #   price_delta_shift > rolling(84).quantile(0.75)
            #   AND Close > TOS_Trail
            #   AND TOS_RSI > 50
            #   AND (vwap_range >= vwap_thr OR today_range >= daily_thr)
            #
            # Slope_Neg (short entry): symmetric conditions, quantile 0.25, RSI < 50
            #
            # Turn_Up / Turn_Down: edge detection — first bar where Slope_Pos/Neg flips True
            #
            # Exit (Sell_Long):
            #   VWAP_Upper cross-down OR TOS_Trail cross-down OR TOS_RSI drops below 70
            # Exit (Sell_Short): symmetric (VWAP_Lower, TOS_Trail, RSI rises above 30)
            #
            # Dedup: block repeated same-type signals or back-to-back EXITs
            #
            # p_win = 84
            # df["Smooth"], df["Slope"] = spicy_sauce(df["Close"])
            # df["price_delta"] = df["Close"] - df["Smooth"]
            # df["q05"] = df["price_delta"].rolling(p_win).quantile(0.05)
            # df["q95"] = df["price_delta"].rolling(p_win).quantile(0.95)
            # df["date"] = df.index.date
            # day_high = df.groupby("date")["High"].cummax()
            # day_low  = df.groupby("date")["Low"].cummin()
            # df["today_range"] = day_high - day_low
            # df['price_delta_shift'] = df['price_delta'] - df['price_delta'].shift(1)
            # df['price_delta_shift'] = df['price_delta_shift'].fillna(0)
            # df["q01"] = df["price_delta_shift"].rolling(p_win).quantile(0.25)
            # df["q99"] = df["price_delta_shift"].rolling(p_win).quantile(0.75)
            # df['vwap_range'] = round(df["VWAP_Upper"] - df["VWAP_Lower"])
            # daily_thr = floor_5_or_int(df['today_range'].median())
            # vwap_thr  = floor_5_or_int(df['vwap_range'].median())
            # df["Slope_Neg"] = (
            #     (df['price_delta_shift'] < df["q01"])
            #     & (df["Close"] < df["TOS_Trail"])
            #     & ((df['vwap_range'] >= vwap_thr) | (df["today_range"] >= daily_thr))
            #     & (df['TOS_RSI'] < 50)
            # )
            # df["Slope_Pos"] = (
            #     (df['price_delta_shift'] > df["q99"])
            #     & (df["Close"] > df["TOS_Trail"])
            #     & ((df['vwap_range'] >= vwap_thr) | (df["today_range"] >= daily_thr))
            #     & (df['TOS_RSI'] > 50)
            # )
            # df["Turn_Up"]   = df["Slope_Pos"] & (~df["Slope_Pos"].shift(1).fillna(False))
            # df["Turn_Down"] = df["Slope_Neg"] & (~df["Slope_Neg"].shift(1).fillna(False))
            # df["Sell_Long"] = (df["Position"] == 1) & (
            #     ((df["Close"].shift(1) >= df["VWAP_Upper"].shift(1)) &
            #      (df["Close"] < df["VWAP_Upper"]) & (df['vwap_range'] >= vwap_thr)) |
            #     ((df["Low"].shift(1) >= df["TOS_Trail"].shift(1)) &
            #      (df["Low"] < df["TOS_Trail"])) |
            #     ((df["TOS_RSI"].shift(1) > 70) & (df["TOS_RSI"] < 70))
            # )
            # df["Sell_Short"] = (df["Position"] == -1) & (
            #     ((df["Close"].shift(1) <= df["VWAP_Lower"].shift(1)) &
            #      (df["Close"] > df["VWAP_Lower"]) & (df['vwap_range'] >= vwap_thr)) |
            #     ((df["High"].shift(1) <= df["TOS_Trail"].shift(1)) &
            #      (df["High"] > df["TOS_Trail"])) |
            #     ((df["TOS_RSI"].shift(1) < 30) & (df["TOS_RSI"] > 30))
            # )
            # last = df.iloc[-1]; bar_time = df.index[-1]; display_time = bar_time.strftime("%H:%M CT")
            # signal = None; signal_type = None
            # if last["Turn_Up"]:    signal = "Momentum Rising - Potential Long";    signal_type = "TURN_UP"
            # elif last["Turn_Down"]:signal = "Momentum Declining - Potential Short";signal_type = "TURN_DOWN"
            # elif last["Sell_Long"]: signal = "Exit Warning - Close Long";  signal_type = "EXIT"
            # elif last["Sell_Short"]:signal = "Exit Warning - Close Short"; signal_type = "EXIT"
            # prev = last_alert.get(ticker)
            # if signal:
            #     allow = prev is None or not (
            #         (signal_type == "EXIT" and prev[1] == "EXIT") or
            #         (signal_type == prev[1] and signal_type in ("TURN_UP", "TURN_DOWN"))
            #     )
            # ── END OLD SIGNAL LOGIC ──────────────────────────────────────

            # ── MR→ORB upgrade: if sliding confirms same direction ──────────
            # When in an MR position, check if the current 6-bar window qualifies
            # as an ORB sliding signal in the same direction. If so, upgrade to
            # pos_src='orb' to add reversal gate protection (effective immediately).
            if state["position"] == "long" and state["position_src"] == "mr":
                if sliding_qualifies_direction(dv, last_idx, orb_deg, r2_thr, "long"):
                    state.update(position_src="orb", orb_bar_loc=last_idx)
                    # Note: no signal update — upgrade happens silently

            elif state["position"] == "short" and state["position_src"] == "mr":
                if sliding_qualifies_direction(dv, last_idx, orb_deg, r2_thr, "short"):
                    state.update(position_src="orb", orb_bar_loc=last_idx)

            # ── Dedup and collect ─────────────────────────────────────────
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
            else:
                print(f"{ticker} @ {display_time}: no signal")

        # ── Send ONE combined Telegram message ────────────────────────────
        if combined_msgs:
            final_msg = (
                "INSANE ALERT\n\n"
                + "\n\n".join(combined_msgs)
            )
            send_telegram(final_msg)
            print(f"[{dt.datetime.now()}] Combined alert sent")

    except Exception as e:
        print("Alert engine error:", e)
        time.sleep(60)
