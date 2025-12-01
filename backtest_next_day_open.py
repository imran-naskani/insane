import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import ta

# Backtest
def generate_trade_log_next_open(df, trade_capital=10000):

    close = df["Close"].values
    openp = df["Open"].values
    turn_up = df["Turn_Up"].values
    turn_down = df["Turn_Down"].values
    dates = df.index
    
    reserve_profit = 0
    position = 0
    entry_price = None
    entry_date = None

    trade_log = []

    for i in range(len(df)-1):  # stop at len-1 because fill happens at i+1 open
        date = dates[i]
        next_open = openp[i+1]

        # --------------------------------------------
        # TURN UP → Enter LONG at next bar's open
        # --------------------------------------------
        if turn_up[i]:

            if position == -1:
                # close short at next bar open
                shares = trade_capital / entry_price
                pnl = (entry_price - next_open) * shares
                reserve_profit += pnl

                trade_log.append({
                    "Trade": len(trade_log)+1,
                    "Direction": "SHORT",
                    "Entry_Date": entry_date,
                    "Exit_Date": dates[i+1],
                    "Entry_Price": entry_price,
                    "Exit_Price": next_open,
                    "PnL_$": pnl,
                    "Return_%": (pnl/trade_capital)*100,
                    "Total_Equity": trade_capital + reserve_profit
                })

            elif position == +1:
                continue

            # open new long at next bar open
            entry_price = next_open
            entry_date = dates[i+1]
            position = +1

        # --------------------------------------------
        # TURN DOWN → Enter SHORT at next bar's open
        # --------------------------------------------
        elif turn_down[i]:

            if position == +1:
                # close long at next bar open
                shares = trade_capital / entry_price
                pnl = (next_open - entry_price) * shares
                reserve_profit += pnl

                trade_log.append({
                    "Trade": len(trade_log)+1,
                    "Direction": "LONG",
                    "Entry_Date": entry_date,
                    "Exit_Date": dates[i+1],
                    "Entry_Price": entry_price,
                    "Exit_Price": next_open,
                    "PnL_$": pnl,
                    "Return_%": (pnl/trade_capital)*100,
                    "Total_Equity": trade_capital + reserve_profit
                })

            elif position == -1:
                continue
            
            # open new short at next bar open
            entry_price = next_open
            entry_date = dates[i+1]
            position = -1

    # --------------------------------------------
    # Close final position on the LAST bar's close
    # --------------------------------------------
    if position != 0:
        final_price = close[-1]
        final_date = dates[-1]
        shares = trade_capital / entry_price

        pnl = (final_price - entry_price) * shares if position == +1 \
              else (entry_price - final_price) * shares

        reserve_profit += pnl

        trade_log.append({
            "Trade": len(trade_log)+1,
            "Direction": "LONG" if position == +1 else "SHORT",
            "Entry_Date": entry_date,
            "Exit_Date": final_date,
            "Entry_Price": entry_price,
            "Exit_Price": final_price,
            "PnL_$": pnl,
            "Return_%": (pnl/trade_capital)*100,
            "Total_Equity": trade_capital + reserve_profit
        })

    return pd.DataFrame(trade_log), trade_capital + reserve_profit
