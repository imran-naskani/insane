from ib_insync import *
import asyncio
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import numpy as np
from dotenv import load_dotenv
from build_dataset import add_features, floor_5_or_int
from model import spicy_sauce
import os
import requests
import time
import datetime as dt
from collections import deque

load_dotenv()

ACCOUNT_CAPITAL = 100_000      # paper capital
RISK_PCT = 0.01                # risk 1% per trade
MAX_CONTRACTS = 1             # hard cap

BOT_TOKEN = os.environ["TELEGRAM_OPTION_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_OPTION_CHAT_ID"]

last_alert = {}   # { ticker: (bar_timestamp, signal_type) }

feature_params = {
    "sma_short": 20,
    "sma_mid": 50,
    "sma_long": 200,
    "trend_slope_short": 5,
    "trend_slope_long": 20,
    "rsi_length": 14,
    "stoch_length": 14,
    "stoch_signal": 3,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "adx_length": 14,
    "atr_length": 14,
    "bollinger_length": 20,
    "volatility_length": 20,
    "volume_zscore_length": 20,
    "volume_change_length": 1,
}

INSTRUMENTS = {
    # "MES": {
    #     "contract": Future(
    #         symbol="MES",
    #         lastTradeDateOrContractMonth="202603",
    #         exchange="CME",
    #         currency="USD"
    #     ),
    #     "option_symbol": "MES",
    #     "multiplier": "5",
    #     "p_win": 252
    # },
    "TSLA": {
        "contract": Stock("TSLA", "SMART", "USD"),
        "option_symbol": "TSLA",
        "multiplier": "100",
        "p_win": 84
    },
    "TMUS": {
        "contract": Stock("TMUS", "SMART", "USD"),
        "option_symbol": "TMUS",
        "multiplier": "100",
        "p_win": 84
    },
    "SPX": {
        "contract": Index("SPX", "CBOE"),
        "option_symbol": "SPX",
        "multiplier": "100",
        "p_win": 84
    }
}

active_trades = {
    sym: {
        "active": False,
        "option_contract": None,
        "option_symbol": None,
        "direction": None,
        "entry_price": None,
        "prices": deque(maxlen=10),
        "last_check": None
    }
    for sym in INSTRUMENTS.keys()
}

BAR_SIZE_stock = "5 mins"
LOOKBACK_stock = "10 D"
BAR_SIZE_option = "1 min"
LOOKBACK_option = "2 D"

FORCE_EXIT_HOUR = 14
FORCE_EXIT_MINUTE = 55

TRAIL_PCT = 0.25
VWAP_LOOKBACK = 20

def get_next_otm_strike(price, ladder, direction):
    if direction == "CALL":
        return np.ceil(price / ladder) * ladder
    else:  # PUT
        return np.floor(price / ladder) * ladder

def sleep_until_next_5m(offset_seconds=2):
    """
    Sleeps until the next 5-minute boundary + offset.
    Offset ensures bar is fully closed and data is available.
    """
    now = time.time()

    interval = 300  # 300 5 minutes in seconds
    next_run = ((now // interval) + 1) * interval + offset_seconds

    sleep_for = max(0, next_run - now)
    time.sleep(sleep_for)

def fetch_next_otm_option_price(
    ib,
    underlying_contract,
    underlying_price,
    direction,            # "CALL" or "PUT"
    option_symbol=None    # only needed for futures (e.g. "MES")
):
    """
    Auto-detects STOCK vs FUTURE and fetches next OTM option price.
    Returns: (localSymbol, price) or (None, None)
    """

    sec_type = underlying_contract.secType
    right = "C" if direction == "CALL" else "P"

    # -------------------------------
    # 1️⃣ Get option chain
    # -------------------------------
    cds = ib.reqContractDetails(underlying_contract)
    if not cds:
        return None, None, None

    con = cds[0].contract

    chains = ib.reqSecDefOptParams(
        con.symbol,
        con.exchange,
        con.secType,
        con.conId
    )

    if not chains:
        return None, None, None

    chain = chains[0]
    expiries = sorted(chain.expirations)
    strikes = sorted(chain.strikes)

    if not expiries or not strikes:
        return None, None, None

    expiry = expiries[0]  # nearest expiry

    # -------------------------------
    # 2️⃣ Pick next valid OTM strike
    # -------------------------------
    try:
        if right == "C":
            strike = min(s for s in strikes if s > underlying_price)
        else:
            strike = max(s for s in strikes if s < underlying_price)
    except ValueError:
        return None, None, None

    strike = float(strike)

    # -------------------------------
    # 3️⃣ Build correct option contract
    # -------------------------------
    if sec_type == "FUT":
        opt = FuturesOption(
            symbol=option_symbol or con.symbol,
            lastTradeDateOrContractMonth=expiry,
            strike=strike,
            right=right,
            exchange=con.exchange,
            currency=con.currency
        )
    else:  # STOCK
        opt = Option(
            symbol=con.symbol,
            lastTradeDateOrContractMonth=expiry,
            strike=strike,
            right=right,
            exchange=con.exchange,
            currency=con.currency,
            multiplier="100"
        )

    # -------------------------------
    # 4️⃣ Fetch price snapshot
    # -------------------------------
    try:
        ib.qualifyContracts(opt)
        ticker = ib.reqMktData(opt, "", False, False)
        ib.sleep(1.5)
        
        print("OPT SNAPSHOT:", opt.localSymbol, ticker.last, ticker.bid, ticker.ask)

        price = (
            ticker.last
            or ticker.close
            or ticker.marketPrice()
            or (
                (ticker.bid + ticker.ask) / 2
                if ticker.bid is not None and ticker.ask is not None
                else None
            )
        )
        ib.cancelMktData(opt)

        if price and price > 0:
            return opt, opt.localSymbol, price
        else:
            return None, None, None

    except Exception:
        return None, None, None

def fetch_option_snapshot_price(ib, option_contract):
    ticker = ib.reqMktData(option_contract, "", False, False)
    ib.sleep(1.0)
    price = (
        ticker.last
        or ticker.close
        or ticker.marketPrice()
        or (
            (ticker.bid + ticker.ask) / 2
            if ticker.bid is not None and ticker.ask is not None
            else None
        )
    )
    ib.cancelMktData(option_contract)
    return price

def evaluate_option_exit(prices, max_dd=-0.05, stall_bars=3):
    if len(prices) < 4:
        return False

    peak = max(prices[:-1])
    drawdown = (prices[-1] - peak) / peak

    # Count consecutive failures to make a new high
    stall = 0
    for p in reversed(prices[:-1]):
        if p < peak:
            stall += 1
        else:
            break

    slope = np.mean(np.diff(prices[-4:]))

    return (
        drawdown <= max_dd and
        stall >= stall_bars and
        slope <= 0
    )

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": msg,
        "parse_mode": "Markdown"
    }
    requests.post(url, data=payload, timeout=5)


# reuse your existing connection style
util.startLoop()

# ---- SAFE RECONNECT ----
if 'ib' in globals() and ib.isConnected():
    ib.disconnect()

ib = IB()
ib.connect('127.0.0.1', 4002, clientId=111, timeout=5)
print("Connected:", ib.isConnected())
print("Server time:", ib.reqCurrentTime())

bars_5m = {}
# bars_1m_opt = {}

print("🚨 INSANE Spicy Alert Option Engine started (5m)")
while True:
    try:
        # sleep_until_next_5m(offset_seconds=10)
        now = time.time()

        combined_msgs = []
        current_bar_time = None
        # ==================================
        # 🔴 TRACK MODE (runs every minute)
        # ==================================
        for sym, trade in active_trades.items():
            if not trade["active"]:
                continue

            if trade["last_check"] is None or now - trade["last_check"] >= 60:
                opt_price = fetch_option_snapshot_price(
                    ib, trade["option_contract"]
                )

                if opt_price:
                    trade["prices"].append(opt_price)

                    if evaluate_option_exit(trade["prices"]):
                        pnl_pct = ((opt_price - trade["entry_price"]) / trade["entry_price"]) * 100

                        exit_msg = (
                            f"*EXIT ALERT*\n"
                            f"{sym}\n"
                            f"Option: {trade['option_symbol']}\n"
                            f"Exit Price: {opt_price:.2f}\n"
                            f"P/L: {pnl_pct:.1f}%"
                        )

                        print(exit_msg)
                        send_telegram(exit_msg)

                        # reset this ticker only
                        trade.update({
                            "active": False,
                            "option_contract": None,
                            "option_symbol": None,
                            "direction": None,
                            "entry_price": None,
                            "prices": deque(maxlen=10),
                            "last_check": None
                        })

                trade["last_check"] = now

        # ==================================
        # 🟢 SCAN MODE (5-minute boundary)
        # ==================================
        if not any(t["active"] for t in active_trades.values()):
            sleep_until_next_5m(offset_seconds=10)
        else:
            time.sleep(1)

        for sym, cfg in INSTRUMENTS.items():
            # 🔒 If this ticker is tracking, skip scan
            if active_trades[sym]["active"]:
                continue

            ib.qualifyContracts(cfg["contract"])
            bars_5m[sym] = ib.reqHistoricalData(
                cfg["contract"],
                "", LOOKBACK_stock, BAR_SIZE_stock,
                "TRADES", 
                useRTH=True, 
                keepUpToDate=False
            )
            df = pd.DataFrame([{
                "Date": b.date,
                "Open": b.open,
                "High": b.high,
                "Low": b.low,
                "Close": b.close,
                "Volume": b.volume,
                "Average": b.average,
                "BarCount": b.barCount
            } for b in bars_5m[sym][:-1]])

            df.set_index("Date", inplace=True)
            df = add_features(df, feature_params, timeframe='5m')
            if len(df) < cfg["p_win"]:
                continue
            df["Smooth"], df["Slope"] = spicy_sauce(df["Close"])
            df["price_delta"] = df["Close"] - df["Smooth"]
                    
            df["q05"] = df["price_delta"].rolling(cfg["p_win"]).quantile(0.05)
            df["q95"] = df["price_delta"].rolling(cfg["p_win"]).quantile(0.95)

            df["date"] = df.index.date
            day_high = df.groupby("date")["High"].cummax()
            day_low  = df.groupby("date")["Low"].cummin()
            df["today_range"] = day_high - day_low
            df['price_delta_shift'] = df['price_delta'] - df['price_delta'].shift(1)
            df['price_delta_shift'] = df['price_delta_shift'].fillna(0)
            df["q01"] = df["price_delta_shift"].rolling(cfg["p_win"]).quantile(0.25) #0.05
            df["q99"] = df["price_delta_shift"].rolling(cfg["p_win"]).quantile(0.75) #0.95
            df['vwap_range'] = round(df["VWAP_Upper"] - df["VWAP_Lower"])
            daily_thr = floor_5_or_int(df['today_range'].median())
            vwap_thr  = floor_5_or_int(df['vwap_range'].median())

            df["Slope_Neg"] = ((df['price_delta_shift'] <  df["q01"] ) ) & (df["Close"] < df["TOS_Trail"]) & ((df['vwap_range'] >= vwap_thr) | (df["today_range"]  >= daily_thr )) & (df['TOS_RSI'] < 50)
            df["Slope_Pos"] = ((df['price_delta_shift'] >  df["q99"] ) ) & (df["Close"] > df["TOS_Trail"]) & ((df['vwap_range'] >= vwap_thr) | (df["today_range"]  >= daily_thr )) & (df['TOS_RSI'] > 50)
            
            df["Turn_Up"] = df["Slope_Pos"] & (~df["Slope_Pos"].shift(1, fill_value=False))
            df["Turn_Down"] = df["Slope_Neg"] & (~df["Slope_Neg"].shift(1, fill_value=False))           
            print(df.tail())

            last = df.iloc[-1]
            bar_time = df.index[-1]
            display_time = bar_time.strftime("%H:%M CT")

            signal = None
            signal_type = None

            if last["Turn_Up"]:
                signal = "Momentum Rising - Potential Long"
                signal_type = "TURN_UP"
            elif last["Turn_Down"]:
                signal = "Momentum Declining - Potential Short"
                signal_type = "TURN_DOWN"

            print(signal, signal_type, "\n\n")
            
            current_bar_time = bar_time

            prev = last_alert.get(sym)
            if signal:
                option_info = "" 
                direction = "CALL" if signal_type == "TURN_UP" else "PUT"
                opt_contract, opt_symbol, opt_price = fetch_next_otm_option_price(
                    ib,
                    cfg["contract"],
                    last["Close"],
                    direction,
                    option_symbol=cfg.get("option_symbol")
                )
                print("Option Contract:", opt_contract, "Sym:", opt_symbol, "Price:", opt_price)
                if opt_price:
                    option_info = f"\nOption: {opt_symbol}\nOpt Price: {opt_price:.2f}"
                    print(option_info)
                    
                    active_trades[sym].update({
                    "active": True,
                    "option_contract": opt_contract,  
                    "option_symbol": opt_symbol,
                    "direction": direction,
                    "entry_price": opt_price,
                    "prices": deque([opt_price], maxlen=10),
                    "last_check": None
                })
                
                if prev is None and signal_type != 'EXIT':
                    allow = True
                else:
                    prev_time, prev_type = prev
                    # Block repeated EXITs
                    allow = not (
                        (signal_type == "EXIT" and prev_type == "EXIT") or
                        (signal_type == prev_type and signal_type in ("TURN_UP", "TURN_DOWN"))
                    )

                if allow:
                    combined_msgs.append(
                        f"*{sym}*\n"
                        # f"*{'SPX' if bars_5m[sym] == '^GSPC' else bars_5m[sym]}*\n"
                        f"{signal}\n"
                        f"Time: {display_time}\n"
                        f"Price: {last['Close']:.2f}"
                        f"{option_info}"
                    )
                    last_alert[sym] = (bar_time, signal_type)
        
        if combined_msgs:
            final_msg = (
                "🚨 INSANE ALERT 🚨\n\n"
                + "\n\n".join(combined_msgs)
            )
            send_telegram(final_msg)
            print(f"[{dt.datetime.now()}] Combined alert sent")

    except Exception as e:
        print("❌ Alert engine error:", e)
        time.sleep(60)
