import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import ta

# Backtest
def backtest_intraday_next_open(df, trade_capital=10000):

    close = df["Close"].values
    openp = df["Open"].values

    turn_up = df["Turn_Up"].values
    turn_down = df["Turn_Down"].values

    sell_long = df["Sell_Long_Plot"].values
    sell_short = df["Sell_Short_Plot"].values
    
    dates = df.index

    position = 0
    entry_price = None
    entry_date = None
    reserve_profit = 0

    trade_log = []

    for i in range(len(df)-1):  # because we use next-open at i+1
        next_open = openp[i+1]
        date_next = dates[i+1]

        # -------------------------------------------------------
        #   LONG ENTRY (next open)
        # -------------------------------------------------------
        if turn_up[i] and position == 0:
            position = 1
            entry_price = next_open
            entry_date = date_next
            continue

        # -------------------------------------------------------
        #   SHORT ENTRY (next open)
        # -------------------------------------------------------
        if turn_down[i] and position == 0:
            position = -1
            entry_price = next_open
            entry_date = date_next
            continue

        # =====================================================================
        # 🔥 NEW RULE 1: TURN DOWN closes LONG (exit at next open) AND FLIP
        # =====================================================================
        if position == 1 and turn_down[i]:
            shares = trade_capital / entry_price
            pnl = (next_open - entry_price) * shares
            reserve_profit += pnl

            # Log closing LONG
            trade_log.append({
                "Trade": len(trade_log)+1,
                "Direction": "LONG",
                "Entry_Date": entry_date,
                "Exit_Date": date_next,
                "Entry_Price": entry_price,
                "Exit_Price": next_open,
                "PnL_$": pnl,
                "Return_%": (pnl/trade_capital)*100,
                "Total_Equity": trade_capital + reserve_profit
            })

            # FLIP to SHORT immediately at next open
            position = -1
            entry_price = next_open
            entry_date = date_next
            continue

        # =====================================================================
        # 🔥 NEW RULE 2: TURN UP closes SHORT (exit at next open) AND FLIP
        # =====================================================================
        if position == -1 and turn_up[i]:
            shares = trade_capital / entry_price
            pnl = (entry_price - next_open) * shares
            reserve_profit += pnl

            # Log closing SHORT
            trade_log.append({
                "Trade": len(trade_log)+1,
                "Direction": "SHORT",
                "Entry_Date": entry_date,
                "Exit_Date": date_next,
                "Entry_Price": entry_price,
                "Exit_Price": next_open,
                "PnL_$": pnl,
                "Return_%": (pnl/trade_capital)*100,
                "Total_Equity": trade_capital + reserve_profit
            })

            # FLIP to LONG immediately at next open
            position = 1
            entry_price = next_open
            entry_date = date_next
            continue

        # -------------------------------------------------------
        #   LONG EXIT BASED ON SELL SIGNAL (next open)
        # -------------------------------------------------------
        if position == 1 and sell_long[i]:
            shares = trade_capital / entry_price
            pnl = (next_open - entry_price) * shares
            reserve_profit += pnl

            trade_log.append({
                "Trade": len(trade_log)+1,
                "Direction": "LONG",
                "Entry_Date": entry_date,
                "Exit_Date": date_next,
                "Entry_Price": entry_price,
                "Exit_Price": next_open,
                "PnL_$": pnl,
                "Return_%": (pnl/trade_capital)*100,
                "Total_Equity": trade_capital + reserve_profit
            })

            position = 0
            continue

        # -------------------------------------------------------
        #   SHORT EXIT BASED ON SELL SIGNAL (next open)
        # -------------------------------------------------------
        if position == -1 and sell_short[i]:
            shares = trade_capital / entry_price
            pnl = (entry_price - next_open) * shares
            reserve_profit += pnl

            trade_log.append({
                "Trade": len(trade_log)+1,
                "Direction": "SHORT",
                "Entry_Date": entry_date,
                "Exit_Date": date_next,
                "Entry_Price": entry_price,
                "Exit_Price": next_open,
                "PnL_$": pnl,
                "Return_%": (pnl/trade_capital)*100,
                "Total_Equity": trade_capital + reserve_profit
            })

            position = 0
            continue

    # -------------------------------------------------------
    #   CLOSE LAST TRADE ON FINAL CLOSE
    # -------------------------------------------------------
    if position != 0:
        final_price = close[-1]
        final_date = dates[-1]
        shares = trade_capital / entry_price

        pnl = (final_price - entry_price)*shares if position==1 else (entry_price - final_price)*shares
        reserve_profit += pnl

        trade_log.append({
            "Trade": len(trade_log)+1,
            "Direction": "LONG" if position==1 else "SHORT",
            "Entry_Date": entry_date,
            "Exit_Date": final_date,
            "Entry_Price": entry_price,
            "Exit_Price": final_price,
            "PnL_$": pnl,
            "Return_%": (pnl/trade_capital)*100,
            "Total_Equity": trade_capital + reserve_profit
        })

    return pd.DataFrame(trade_log), trade_capital + reserve_profit


def backtest_intraday_next_open_sell_only(df, capital):
    position = 0
    entry_price = None
    entry_time = None

    equity = capital
    trades = []

    for i in range(len(df) - 1):
        row = df.iloc[i]
        next_row = df.iloc[i + 1]
        bar_time = row.name.time()

        # ----- SKIP EXTENDED HOURS -----
        if bar_time < pd.Timestamp("09:30").time() or bar_time > pd.Timestamp("15:55").time():
            continue

        # ----- FORCED EOD EXIT (2:55 PM — SAME BAR CLOSE) -----
        if position != 0 and bar_time >= pd.Timestamp("15:55").time():
            exit_price = row["Close"]
            pnl = (exit_price - entry_price) if position == 1 else (entry_price - exit_price)
            equity += pnl

            trades.append({
                "Entry_Date": entry_time,
                "Exit_Date": row.name,
                "Direction": "LONG" if position == 1 else "SHORT",
                "Entry_Price": entry_price,
                "Exit_Price": exit_price,
                "PnL_$": pnl,
                "Return_%": (pnl / entry_price) * 100,
                "Total_Equity": equity
            })

            position = 0
            entry_price = None
            entry_time = None
            continue

        # ----- ENTRY -----
        if position == 0:
            if row["Turn_Up"]:
                position = 1
                entry_price = next_row["Open"]
                entry_time = next_row.name

            elif row["Turn_Down"]:
                position = -1
                entry_price = next_row["Open"]
                entry_time = next_row.name

        # ----- EXIT (SELL SIGNALS ONLY) -----
        elif position == 1 and row["Sell_Long"]:
            exit_price = next_row["Open"]
            pnl = exit_price - entry_price
            equity += pnl

            trades.append({
                "Entry_Date": entry_time,
                "Exit_Date": next_row.name,
                "Direction": "LONG",
                "Entry_Price": entry_price,
                "Exit_Price": exit_price,
                "PnL_$": pnl,
                "Return_%": (pnl / entry_price) * 100,
                "Total_Equity": equity
            })

            position = 0
            entry_price = None
            entry_time = None

        elif position == -1 and row["Sell_Short"]:
            exit_price = next_row["Open"]
            pnl = entry_price - exit_price
            equity += pnl

            trades.append({
                "Entry_Date": entry_time,
                "Exit_Date": next_row.name,
                "Direction": "SHORT",
                "Entry_Price": entry_price,
                "Exit_Price": exit_price,
                "PnL_$": pnl,
                "Return_%": (pnl / entry_price) * 100,
                "Total_Equity": equity
            })

            position = 0
            entry_price = None
            entry_time = None

    trade_df = pd.DataFrame(trades)
    return trade_df, equity
