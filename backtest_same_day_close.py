import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import ta

# Backtest

def generate_trade_log(df, trade_capital=10000):
    
    close = df["Close"].values
    turn_up = df["Turn_Up"].values
    turn_down = df["Turn_Down"].values
    dates = df.index
    
    reserve_profit = 0      # accumulated profit
    position = 0            # +1 long, -1 short
    entry_price = None
    entry_date = None

    trade_log = []

    for i in range(len(df)):
        price = close[i]
        date = dates[i]

        # --------------------------------------------------
        # ENTRY: Turn Up (Long Entry)
        # --------------------------------------------------
        if turn_up[i]:

            if position == -1:  
                # closing short
                shares = trade_capital / entry_price
                pnl = (entry_price - price) * shares
                reserve_profit += pnl

                trade_log.append({
                    "Trade": len(trade_log) + 1,
                    "Direction": "SHORT",
                    "Entry_Date": entry_date,
                    "Exit_Date": date,
                    "Entry_Price": entry_price,
                    "Exit_Price": price,
                    "PnL_$": pnl,
                    "Return_%": (pnl / trade_capital) * 100,
                    "Total_Equity": trade_capital + reserve_profit
                })

            elif position == +1:
                continue
 
            # open new long position
            entry_price = price
            entry_date = date
            position = +1

        # --------------------------------------------------
        # ENTRY: Turn Down (Short Entry)
        # --------------------------------------------------
        elif turn_down[i]:

            if position == +1:
                # closing long
                shares = trade_capital / entry_price
                pnl = (price - entry_price) * shares
                reserve_profit += pnl

                trade_log.append({
                    "Trade": len(trade_log) + 1,
                    "Direction": "LONG",
                    "Entry_Date": entry_date,
                    "Exit_Date": date,
                    "Entry_Price": entry_price,
                    "Exit_Price": price,
                    "PnL_$": pnl,
                    "Return_%": (pnl / trade_capital) * 100,
                    "Total_Equity": trade_capital + reserve_profit
                })

            elif position == -1:
                continue

            # open new short position
            entry_price = price
            entry_date = date
            position = -1

    # --------------------------------------------------
    # CLOSE FINAL OPEN TRADE (optional)
    # --------------------------------------------------
    if position != 0:
        price = close[-1]
        date = dates[-1]
        shares = trade_capital / entry_price

        pnl = (price - entry_price) * shares if position == +1 \
              else (entry_price - price) * shares

        reserve_profit += pnl

        trade_log.append({
            "Trade": len(trade_log) + 1,
            "Direction": "LONG" if position == +1 else "SHORT",
            "Entry_Date": entry_date,
            "Exit_Date": date,
            "Entry_Price": entry_price,
            "Exit_Price": price,
            "PnL_$": pnl,
            "Return_%": (pnl / trade_capital) * 100,
            "Total_Equity": trade_capital + reserve_profit
        })

    final_equity = trade_capital + reserve_profit
    return pd.DataFrame(trade_log), final_equity



# Compute Trade Stats
def compute_trade_stats(trade_log, trade_capital=10000):

    # -----------------------------------------
    # Add HOLD PERIOD column
    # -----------------------------------------
    trade_log = trade_log.copy()
    trade_log["Hold_Days"] = (
        pd.to_datetime(trade_log["Exit_Date"]) -
        pd.to_datetime(trade_log["Entry_Date"])
    ).dt.days

    longs  = trade_log[trade_log["Direction"] == "LONG"]
    shorts = trade_log[trade_log["Direction"] == "SHORT"]

    # -----------------------------------------
    # Win rates
    # -----------------------------------------
    def win_rate(df):
        if len(df) == 0:
            return np.nan
        return (df["PnL_$"] > 0).mean() * 100

    total_win_rate = win_rate(trade_log)
    long_win_rate  = win_rate(longs)
    short_win_rate = win_rate(shorts)

    # -----------------------------------------
    # Profit Factor
    # -----------------------------------------
    def profit_factor(df):
        if len(df) == 0:
            return np.nan
        gp = df[df["PnL_$"] > 0]["PnL_$"].sum()
        gl = -df[df["PnL_$"] < 0]["PnL_$"].sum()
        return np.inf if gl == 0 else gp / gl

    pf_total = profit_factor(trade_log)
    pf_long  = profit_factor(longs)
    pf_short = profit_factor(shorts)

    # -----------------------------------------
    # Equity curve & MDD
    # -----------------------------------------
    equity_curve = trade_log["Total_Equity"].values

    def max_drawdown(equity):
        if len(equity) == 0:
            return np.nan
        peak = -1e9
        mdd = 0
        for x in equity:
            peak = max(peak, x)
            mdd = max(mdd, (peak - x) / peak)
        return mdd * 100

    mdd_total = max_drawdown(equity_curve)
    mdd_long  = max_drawdown(longs["Total_Equity"].values)
    mdd_short = max_drawdown(shorts["Total_Equity"].values)

    # -----------------------------------------
    # Sharpe Ratio (trade-based)
    # -----------------------------------------
    trade_returns = trade_log["Return_%"] / 100

    def sharpe(r):
        if len(r) <= 1 or r.std() == 0:
            return np.nan
        return (r.mean() / r.std()) * np.sqrt(252)

    sharpe_total = sharpe(trade_returns)
    sharpe_long  = sharpe(longs["Return_%"] / 100)
    sharpe_short = sharpe(shorts["Return_%"] / 100)

    # -----------------------------------------
    # Hold period metrics
    # -----------------------------------------
    def avg_or_nan(series):
        return np.nan if len(series) == 0 else series.mean()

    def max_or_nan(series):
        return np.nan if len(series) == 0 else series.max()

    avg_hold_total  = avg_or_nan(trade_log["Hold_Days"])
    avg_hold_long   = avg_or_nan(longs["Hold_Days"])
    avg_hold_short  = avg_or_nan(shorts["Hold_Days"])

    max_hold_total  = max_or_nan(trade_log["Hold_Days"])
    max_hold_long   = max_or_nan(longs["Hold_Days"])
    max_hold_short  = max_or_nan(shorts["Hold_Days"])

    # -----------------------------------------
    # Assemble result
    # -----------------------------------------
    stats = {
        "Trades_Total": len(trade_log),
        "Trades_Long": len(longs),
        "Trades_Short": len(shorts),

        "WinRate_Total_%": total_win_rate,
        "WinRate_Long_%": long_win_rate,
        "WinRate_Short_%": short_win_rate,

        "ProfitFactor_Total": pf_total,
        "ProfitFactor_Long": pf_long,
        "ProfitFactor_Short": pf_short,

        "MaxDD_Total_%": mdd_total,
        "MaxDD_Long_%": mdd_long,
        "MaxDD_Short_%": mdd_short,

        "Sharpe_Total": sharpe_total,
        "Sharpe_Long": sharpe_long,
        "Sharpe_Short": sharpe_short,

        "AvgHold_Total_Days": avg_hold_total,
        "AvgHold_Long_Days": avg_hold_long,
        "AvgHold_Short_Days": avg_hold_short,

        "MaxHold_Total_Days": max_hold_total,
        "MaxHold_Long_Days": max_hold_long,
        "MaxHold_Short_Days": max_hold_short,

        "Final_Equity": trade_log["Total_Equity"].iloc[-1],
        "Net_Profit_$": trade_log["Total_Equity"].iloc[-1] - trade_capital,
    }

    return stats, equity_curve

