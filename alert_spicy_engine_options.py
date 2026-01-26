from ib_insync import *
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from build_dataset import add_features
from model import spicy_sauce
import os
import requests

load_dotenv()

# ================= CONFIG =================

# -------- POSITION SIZING --------
ACCOUNT_CAPITAL = 100_000      # paper capital
RISK_PCT = 0.01                # risk 1% per trade
MAX_CONTRACTS = 1             # hard cap

BOT_TOKEN = os.environ["TELEGRAM_OPTION_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_OPTION_CHAT_ID"]

STRIKE_LADDERS = {
    "MES": 5,
    "SPX": 5,
    "TSLA": 2.5
}

INSTRUMENTS = {
    # "MES": {
    #     "contract": Future("MES", "202603", "CME", "USD"),
    #     "option_symbol": "MES",
    #     "multiplier": "5",
    #     "p_win": 252
    # },
    # Uncomment when ready
    "TSLA": {
        "contract": Stock("TSLA", "SMART", "USD"),
        "option_symbol": "TSLA",
        "multiplier": "100",
        "p_win": 84
    },
    "SPX": {
        "contract": Index("SPX", "CBOE"),
        "option_symbol": "SPX",
        "multiplier": "100",
        "p_win": 252
    }
}

BAR_SIZE = "5 mins"
LOOKBACK = "10 D"

FORCE_EXIT_HOUR = 14
FORCE_EXIT_MINUTE = 55

TRAIL_PCT = 0.25
VWAP_LOOKBACK = 20

# ================= IBKR =================
util.startLoop()
ib = IB()
ib.connect("127.0.0.1", 4002, clientId=103)

# ================= STATE =================
signal_state = {}
option_state = {}

# -------- SIGNAL ALERT STATE --------
last_alert = {}   # symbol -> (timestamp, signal_type)
daily_stats = {
    "trades": 0,
    "wins": 0,
    "losses": 0,
    "pnl": 0.0
}

daily_summary_sent = False

for s in INSTRUMENTS:
    signal_state[s] = {"in_position": False}
    option_state[s] = {
        "contract": None,
        "entry_price": None,
        "peak_price": None,
        "vwap": None,
        "bars": [],
        "order": None
    }

# ================= HELPERS =================
def send_daily_summary():
    global daily_summary_sent
    if daily_summary_sent:
        return

    send_telegram(
        f"📊 *DAILY SUMMARY*\n"
        f"Trades: `{daily_stats['trades']}`\n"
        f"Wins: `{daily_stats['wins']}`\n"
        f"Losses: `{daily_stats['losses']}`\n"
        f"Net PnL: *{daily_stats['pnl']:+.2f}* USD"
    )

    daily_summary_sent = True

def on_trade_filled(trade, sym, direction):
    if not trade.fills:
        return

    fill = trade.fills[-1]
    price = fill.execution.price
    qty = fill.execution.shares
    option_state[sym]["entry_fill_price"] = price

    send_telegram(
        f"✅ *TRADE FILLED*\n"
        f"Symbol: `{sym}`\n"
        f"Direction: *{direction}*\n"
        f"Qty: `{qty}`\n"
        f"Price: `{price:.2f}`"
    )

def on_exit_filled(trade, sym, reason):
    if not trade.fills:
        return

    fill = trade.fills[-1]
    price = fill.execution.price
    qty = fill.execution.shares

    entry_price = option_state[sym].get("entry_fill_price")
    mult = float(INSTRUMENTS[sym]["multiplier"])

    pnl = None
    if entry_price is not None:
        pnl = (price - entry_price) * qty * mult

    daily_stats["trades"] += 1
    if pnl is not None:
        daily_stats["pnl"] += pnl

    if pnl is not None:
        if pnl > 0:
            daily_stats["wins"] += 1
        else:
            daily_stats["losses"] += 1

    msg = (
        f"🔴 *TRADE CLOSED*\n"
        f"Symbol: `{sym}`\n"
        f"Reason: *{reason}*\n"
        f"Qty: `{qty}`\n"
        f"Entry: `{entry_price:.2f}`\n"
        f"Exit: `{price:.2f}`"
    )

    if pnl is not None:
        msg += f"\nPnL: *{pnl:+.2f}* USD"

    send_telegram(msg)
    reset(sym)


def send_signal_alert(symbol, signal, signal_type, bar_time, price):
    prev = last_alert.get(symbol)

    if prev is None:
        allow = True
    else:
        prev_time, prev_type = prev
        # Block repeated EXITs or same TURN signals
        allow = not (
            (signal_type == "EXIT" and prev_type == "EXIT") or
            (signal_type == prev_type and signal_type in ("TURN_UP", "TURN_DOWN"))
        )

    if allow:
        send_telegram(
            f"*{symbol}*\n"
            f"{signal}\n"
            f"Price: {price:.2f}"
        )
        last_alert[symbol] = (bar_time, signal_type)


def send_telegram(msg):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": msg,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"[TELEGRAM ERROR] {e}")


def calc_position_size(option_price, multiplier):
    """
    Risk-based sizing.
    Risk = ACCOUNT_CAPITAL * RISK_PCT
    Contracts = floor(risk / (option_price * multiplier))
    """
    risk_dollars = ACCOUNT_CAPITAL * RISK_PCT
    cost_per_contract = option_price * float(multiplier)

    if cost_per_contract <= 0:
        return 0

    qty = int(risk_dollars // cost_per_contract)
    return max(1, min(qty, MAX_CONTRACTS))

def get_next_strike(symbol, price, direction):
    step = STRIKE_LADDERS.get(symbol)
    if step is None:
        raise ValueError(f"No strike ladder defined for {symbol}")

    if direction == "LONG":
        return int(np.ceil(price / step) * step)
    else:
        return int(np.floor(price / step) * step)


def nearest_expiry():
    now = pd.Timestamp.now(tz="US/Central")
    if now.weekday() < 5:
        return now.strftime("%Y%m%d")
    days_ahead = 7 - now.weekday()
    return (now + pd.Timedelta(days=days_ahead)).strftime("%Y%m%d")


def to_ohlcv(df):
    df = df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    })
    df.index = pd.to_datetime(df["date"], utc=True).tz_convert("US/Central")
    return df[["Open", "High", "Low", "Close", "Volume"]]


def select_option(symbol, direction, underlying_price):
    cfg = INSTRUMENTS[symbol]
    strike = get_next_strike(symbol, underlying_price, direction)
    expiry = nearest_expiry()
    right = "C" if direction == "LONG" else "P"

    opt = Option(
        symbol=cfg["option_symbol"],
        lastTradeDateOrContractMonth=expiry,
        strike=strike,
        right=right,
        exchange="SMART",
        currency="USD",
        multiplier=cfg["multiplier"]
    )
    ib.qualifyContracts(opt)
    return opt


def calc_vwap(df):
    pv = (df["Close"] * df["Volume"]).sum()
    vol = df["Volume"].sum()
    return pv / vol if vol > 0 else df["Close"].iloc[-1]


def reset(sym):
    signal_state[sym]["in_position"] = False
    option_state[sym] = {
        "contract": None,
        "entry_price": None,
        "peak_price": None,
        "vwap": None,
        "bars": [],
        "order": None,
        "qty": None,
        "entry_fill_price": None
    }

# ================= OPTION 1-MIN EXIT ENGINE =================
def on_option_1m(bars, hasNewBar, sym):
    if not hasNewBar:
        return

    # sym = bars.contract.symbol
    ts = pd.Timestamp(bars[-2].date, tz="UTC").tz_convert("US/Central")
    bar = bars[-2]

    st = option_state[sym]
    price = bar.close
    vol = bar.volume

    st["bars"].append({"Close": price, "Volume": vol})
    df = pd.DataFrame(st["bars"][-VWAP_LOOKBACK:])

    st["vwap"] = calc_vwap(df)

    if st["entry_price"] is None:
        st["entry_price"] = price
        st["peak_price"] = price
        return

    st["peak_price"] = max(st["peak_price"], price)
    trail = st["peak_price"] * (1 - TRAIL_PCT)

    if price < trail or price < st["vwap"]:
        exit_trade = ib.placeOrder(
            st["contract"],
            MarketOrder("SELL", st["qty"])
        )
        exit_trade.filledEvent += lambda _: on_exit_filled(exit_trade, sym, "TRAIL/VWAP")
        # print(f"[{sym}] EXIT OPTION @ {price:.2f}")
        # reset(sym)
        return

    if ts.hour == FORCE_EXIT_HOUR and ts.minute >= FORCE_EXIT_MINUTE:
        exit_trade = ib.placeOrder(
            st["contract"],
            MarketOrder("SELL", st["qty"])
        )
        exit_trade.filledEvent += lambda _: on_exit_filled(exit_trade, sym, "FORCE_2:55")
        send_daily_summary()
        # print(f"[{sym}] FORCE EXIT 2:55")
        # reset(sym)

# ================= DATA =================
bars_5m = {}
bars_1m_opt = {}

for sym, cfg in INSTRUMENTS.items():
    ib.qualifyContracts(cfg["contract"])
    bars_5m[sym] = ib.reqHistoricalData(
        cfg["contract"],
        "", LOOKBACK, BAR_SIZE,
        "TRADES", False, True
    )

# ================= 5-MIN SIGNAL =================
def on_5m_bar(bars, hasNewBar):
    if not hasNewBar:
        return

    sym = bars.contract.symbol
    cfg = INSTRUMENTS[sym]

    df = to_ohlcv(util.df(bars).iloc[:-1])
    df = add_features(df, {}, timeframe="5m")

    if len(df) < cfg["p_win"]:
        return

    df["Smooth"], _ = spicy_sauce(df["Close"])
    delta = df["Close"] - df["Smooth"]

    q05 = delta.rolling(cfg["p_win"]).quantile(0.05).iloc[-1]
    q95 = delta.rolling(cfg["p_win"]).quantile(0.95).iloc[-1]
    last = df.iloc[-1]
    if delta.iloc[-1] > q95:
        send_signal_alert(
            symbol=sym,
            signal="TURN UP",
            signal_type="TURN_UP",
            bar_time=last.name,
            price=last["Close"]
        )
    elif delta.iloc[-1] < q05:
        send_signal_alert(
            symbol=sym,
            signal="TURN DOWN",
            signal_type="TURN_DOWN",
            bar_time=last.name,
            price=last["Close"]
        )


    if signal_state[sym]["in_position"]:
        return

    if delta.iloc[-1] > q95:
        direction = "LONG"
    elif delta.iloc[-1] < q05:
        direction = "SHORT"
    else:
        return

    opt = select_option(sym, direction, last["Close"])
    ticker = ib.reqMktData(opt, "", False, False)
    ib.sleep(0.5)  # allow price to populate

    opt_price = ticker.last or ticker.close or ticker.marketPrice()
    if opt_price is None or opt_price <= 0:
        print(f"[{sym}] SKIP — no option price")
        return

    qty = calc_position_size(opt_price, INSTRUMENTS[sym]["multiplier"])
    if qty <= 0:
        print(f"[{sym}] SKIP — size=0")
        return

    trade = ib.placeOrder(opt, MarketOrder("BUY", qty))
    trade.filledEvent += lambda _: on_trade_filled(trade, sym, direction)


    option_state[sym].update({
    "contract": opt,
    "entry_price": None,
    "peak_price": None,
    "bars": [],
    "order": trade,
    "qty": qty,
    "entry_fill_price": None,
    })


    bars_1m_opt[sym] = ib.reqHistoricalData(
        opt, "", "1 D", "1 min",
        "TRADES", False, True
    )
    # bars_1m_opt[sym].updateEvent += on_option_1m
    bars_1m_opt[sym].updateEvent += lambda bars, hasNewBar, s=sym: on_option_1m(bars, hasNewBar, s)

    signal_state[sym]["in_position"] = True
    print(f"[{sym}] ENTER {direction} → {opt}")
    

# ================= ATTACH =================
for b in bars_5m.values():
    b.updateEvent += on_5m_bar

print("▶ FULL OPTIONS ENGINE RUNNING")
ib.run()
