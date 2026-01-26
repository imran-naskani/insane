import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import ta

# Backtest
def backtest_intraday_close(df, trade_capital=10000):

    print(
            df[
                (df['Turn_Up'] == True) |
                (df['Turn_Down'] == True) |
                (df['Sell_Long_Plot'] == True) |
                (df['Sell_Short_Plot'] == True)
            ][['Turn_Up', 'Turn_Down', 'Sell_Long', 'Sell_Short', 'Sell_Long_Plot', 'Sell_Short_Plot', 'Position']]
        )

    close = df["Close"].values
    turn_up = df["Turn_Up"].values
    turn_down = df["Turn_Down"].values

    sell_long = df["Sell_Long_Plot"].values
    sell_short = df["Sell_Short_Plot"].values

    dates = df.index

    position = 0       # +1 long, -1 short, 0 flat
    entry_price = None
    entry_date = None
    reserve_profit = 0

    trade_log = []

    for i in range(len(df)):
        price = close[i]
        date  = dates[i]

        # -------------------------------------------------------
        #   LONG ENTRY
        # -------------------------------------------------------
        if turn_up[i] and position == 0:
            position = 1
            entry_price = price
            entry_date = date
            continue

        # -------------------------------------------------------
        #   SHORT ENTRY
        # -------------------------------------------------------
        if turn_down[i] and position == 0:
            position = -1
            entry_price = price
            entry_date = date
            continue

        # =====================================================================
        # 🔥 NEW RULE 1: TURN DOWN closes LONG immediately (flip optional)
        # =====================================================================
        if position == 1 and turn_down[i]:
            shares = trade_capital / entry_price
            pnl = (price - entry_price) * shares
            reserve_profit += pnl

            trade_log.append({
                "Trade": len(trade_log)+1,
                "Direction": "LONG",
                "Entry_Date": entry_date,
                "Exit_Date": date,
                "Entry_Price": entry_price,
                "Exit_Price": price,
                "PnL_$": pnl,
                "Return_%": (pnl/trade_capital)*100,
                "Total_Equity": trade_capital + reserve_profit
            })

            # Flip to short same bar
            position = -1
            entry_price = price
            entry_date = date
            continue

        # =====================================================================
        # 🔥 NEW RULE 2: TURN UP closes SHORT immediately (flip optional)
        # =====================================================================
        if position == -1 and turn_up[i]:
            shares = trade_capital / entry_price
            pnl = (entry_price - price) * shares
            reserve_profit += pnl

            trade_log.append({
                "Trade": len(trade_log)+1,
                "Direction": "SHORT",
                "Entry_Date": entry_date,
                "Exit_Date": date,
                "Entry_Price": entry_price,
                "Exit_Price": price,
                "PnL_$": pnl,
                "Return_%": (pnl/trade_capital)*100,
                "Total_Equity": trade_capital + reserve_profit
            })

            # Flip to long same bar
            position = 1
            entry_price = price
            entry_date = date
            continue

        # -------------------------------------------------------
        #   LONG EXIT BASED ON SELL SIGNAL
        # -------------------------------------------------------
        if position == 1 and sell_long[i]:
            shares = trade_capital / entry_price
            pnl = (price - entry_price) * shares
            reserve_profit += pnl

            trade_log.append({
                "Trade": len(trade_log)+1,
                "Direction": "LONG",
                "Entry_Date": entry_date,
                "Exit_Date": date,
                "Entry_Price": entry_price,
                "Exit_Price": price,
                "PnL_$": pnl,
                "Return_%": (pnl/trade_capital)*100,
                "Total_Equity": trade_capital + reserve_profit
            })

            position = 0
            continue

        # -------------------------------------------------------
        #   SHORT EXIT BASED ON SELL SIGNAL
        # -------------------------------------------------------
        if position == -1 and sell_short[i]:
            shares = trade_capital / entry_price
            pnl = (entry_price - price) * shares
            reserve_profit += pnl

            trade_log.append({
                "Trade": len(trade_log)+1,
                "Direction": "SHORT",
                "Entry_Date": entry_date,
                "Exit_Date": date,
                "Entry_Price": entry_price,
                "Exit_Price": price,
                "PnL_$": pnl,
                "Return_%": (pnl/trade_capital)*100,
                "Total_Equity": trade_capital + reserve_profit
            })

            position = 0
            continue

    # -------------------------------------------------------
    #   CLOSE LAST OPEN POSITION ON FINAL BAR
    # -------------------------------------------------------
    if position != 0:
        price = close[-1]
        date = dates[-1]

        shares = trade_capital / entry_price
        pnl = (price - entry_price) * shares if position == 1 else (entry_price - price) * shares
        reserve_profit += pnl

        trade_log.append({
            "Trade": len(trade_log)+1,
            "Direction": "LONG" if position == 1 else "SHORT",
            "Entry_Date": entry_date,
            "Exit_Date": date,
            "Entry_Price": entry_price,
            "Exit_Price": price,
            "PnL_$": pnl,
            "Return_%": (pnl/trade_capital)*100,
            "Total_Equity": trade_capital + reserve_profit
        })

    return pd.DataFrame(trade_log), trade_capital + reserve_profit

def backtest_intraday_close_sell_only(df, capital):
    position = 0          # 1 = LONG, -1 = SHORT, 0 = FLAT
    entry_price = None
    entry_time = None

    equity = capital
    trades = []

    for i in range(len(df)):
        row = df.iloc[i]
        bar_time = row.name.time()

        # ----- SKIP EXTENDED HOURS -----
        if bar_time < pd.Timestamp("09:30").time() or bar_time > pd.Timestamp("15:55").time():
            continue

        # ----- FORCED EOD EXIT (2:55 PM) -----
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
                entry_price = row["Close"]
                entry_time = row.name

            elif row["Turn_Down"]:
                position = -1
                entry_price = row["Close"]
                entry_time = row.name

        # ----- EXIT (SELL SIGNALS ONLY) -----
        elif position == 1 and row["Sell_Long"]:
            exit_price = row["Close"]
            pnl = exit_price - entry_price
            equity += pnl

            trades.append({
                "Entry_Date": entry_time,
                "Exit_Date": row.name,
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
            exit_price = row["Close"]
            pnl = entry_price - exit_price
            equity += pnl

            trades.append({
                "Entry_Date": entry_time,
                "Exit_Date": row.name,
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
