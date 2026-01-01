### INSANE.PY
# Imran Naskani's Statistical Analysis for Navigating Equities (INSANE)

import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np
import ta
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
import mplfinance as mpf
# from frama import compute_frama
from build_dataset import build_feature_dataset
from technical_features import add_all_labels
from backtest_same_day_close import generate_trade_log, compute_trade_stats
from backtest_next_day_open import generate_trade_log_next_open

ticker = "NVDA"

# Pull Data from Yahoo finance
data = build_feature_dataset(ticker, start_date="2020-01-01") # , end_date="2025-10-18"
data = data.dropna().copy()
data.tail()

# Create Technical Features
main_df = add_all_labels(data)
main_df.tail()

df = main_df.copy()

# ------ Kalman Filter Smoothing - Hyperparameter Tuning

# ------ transition_covariance 
# ------ Lower (0.0001 or 0.001) -> smoother trend -> fewer signals -> slower but cleaner flips
# ------ Higher (0.01 or 0.1) -> more reactive trend -> more signals -> catches reversals faster but increases noise
# ------ Default 0.001

# ------ observation_covariance 
# ------ Higher (5) -> assumes price is noisy -> smooths more -> fewer signals
# ------ Lower (0.5) -> trusts price more -> smoother sticks closer to actual price -> more signals
# ------ Default 1

# ------ initial_state_covariance - Only affects the very beginning of the series
# ------ Set it high (1–10) so the model adapts quickly
# ------ Has almost no impact on long-run trading performance
# ------ Default 1

# ------ transition_matrices & observation_matrices
# ------ Currently using the simplest system: Trend today ≈ trend yesterday & Observation = true price

# def kalman_basic(close):
#     kf = KalmanFilter(
#         transition_matrices=[1],
#         observation_matrices=[1],
#         initial_state_mean=close.iloc[0],
#         initial_state_covariance=1,
#         observation_covariance=5,  # R 5
#         transition_covariance=1  # Q 0.01
#     )

#     state_means, _ = kf.filter(close.values)
#     smooth = state_means.flatten()
#     slope = np.gradient(smooth)
#     return smooth, slope

def secret_sauce(close):
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=close.iloc[0],
        initial_state_covariance=1,
        observation_covariance=5,
        transition_covariance=1 #0.1
    )

    # Forward filter
    filtered_state_means = []
    filtered_state_covs  = []

    state_mean = close.iloc[0]
    state_cov  = 1

    for z in close.values:
        state_mean, state_cov = kf.filter_update(
            filtered_state_mean = state_mean,
            filtered_state_covariance = state_cov,
            observation = z
        )
        filtered_state_means.append(state_mean)
        filtered_state_covs.append(state_cov)

    smooth = np.array(filtered_state_means).flatten()
    slope  = np.gradient(smooth)

    return smooth, slope


def spicy_sauce(close):
    """
    2-D Kalman filter (price + slope)
    Aligned with your 1-D Kalman:
      R = 5
      Q_price = 1
      Q_slope = 0.01
    """
    dt = 0.7
    kf = KalmanFilter(
           # slows slope contribution - default 1.0
        transition_matrices=np.array([
            [1.0, dt],   # price = price + slope
            [0.0, 1.0]    # slope = slope
        ]),
        observation_matrices=np.array([[1.0, 0.0]]),
        initial_state_mean=[close.iloc[0], 0.0],
        initial_state_covariance=np.eye(2),
        observation_covariance=8,          # SAME as your 1-D
        transition_covariance=np.array([
            [1,  0.0],   # SAME as your 1-D Q
            [0.0,  0.01]   # Q / 100 → stable slope
        ])
    )

    state_means, state_covs = kf.filter(close.values)

    smooth = state_means[:, 0]   # Kalman-smoothed price
    slope  = state_means[:, 1]   # Kalman-estimated slope

    return smooth, slope
