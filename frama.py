import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import ta

def compute_frama(series, window=16, fc=1, sc=200):
    """
    Compute FRAMA (Fractal Adaptive Moving Average)
    John Ehlers original algorithm.
    series : pandas Series of prices
    window : lookback length for fractal dimension (default 16)
    fc : fast constant (default 1)
    sc : slow constant (default 200)
    """
    close = series.values
    n = window

    frama = np.zeros(len(close))
    frama[:] = np.nan

    for i in range(n * 2, len(close)):
        # First half
        hl1 = max(close[i-n:i]) - min(close[i-n:i])
        # Second half
        hl2 = max(close[i-2*n:i-n]) - min(close[i-2*n:i-n])
        # Whole window
        hl = max(close[i-2*n:i]) - min(close[i-2*n:i])

        # Avoid division errors
        if hl1 == 0 or hl2 == 0 or hl == 0:
            continue

        # Fractal dimension
        dim = (np.log(hl1 + hl2) - np.log(hl)) / np.log(2)

        # Smoothing factor
        alpha = np.exp(-4.6 * (dim - 1))
        alpha = max(min(alpha, 1), 0.01)  # Bound between 0.01 and 1

        # FRAMA recursive formula
        if np.isnan(frama[i-1]):
            frama[i] = close[i]
        else:
            frama[i] = alpha * close[i] + (1 - alpha) * frama[i-1]

    return pd.Series(frama, index=series.index)