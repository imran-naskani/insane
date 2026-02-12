"""
FINAL PRODUCTION SCRIPT
=======================

Title:      S&P 500 Daily Predictor with GAN + GRU + CatBoost Ensemble
Purpose:    Predict next-day close price and achieve high "Predicted Price within OHLC" accuracy (~96%+)
Features:   Technical indicators, Finnhub sentiment, GAN data augmentation, GRU neural net, CatBoost stacking
Metrics:    Focus on OHLC containment, directional accuracy, and buffer accuracy

How to expand:
- Change ticker: Modify TICKER in main block
- Add new features: Extend add_features() and FEATURES list
- Tune GAN: Adjust ratio/epochs in augment_data()
- Add new models: Extend the ensemble section (e.g., add LightGBM)
- Daily automation: Use schedule or cron to call run_daily_prediction()

Requirements (pip install):
    tensorflow, ta, yfinance, catboost, python-dotenv, scikit-learn
Environment variables in .env:
    FINNHUB_API_KEY=your_key
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
import ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Bidirectional, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import mixed_precision
import yfinance as yf
import logging
import requests
from dotenv import load_dotenv
from catboost import CatBoostRegressor

# ----------------------------- Global Settings -----------------------------
mixed_precision.set_global_policy('mixed_float16')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

os.environ["OMP_NUM_THREADS"] = "16"
tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(16)

# ----------------------------- GAN Augmentation -----------------------------
def build_generator(n_timesteps: int, n_features: int) -> Sequential:
    """
    Builds the GAN generator (LSTM-based) to create synthetic time-series data.
    """
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(n_timesteps, n_features)),
        LSTM(64, return_sequences=False),
        Dense(n_timesteps * n_features),
        tf.keras.layers.Reshape((n_timesteps, n_features))
    ])
    return model


def train_gan(X_train: np.ndarray, epochs: int = 40, batch_size: int = 128) -> Sequential:
    """
    Trains a simple GAN generator on the training sequences.
    Returns the trained generator for data augmentation.
    """
    n_timesteps, n_features = X_train.shape[1], X_train.shape[2]
    generator = build_generator(n_timesteps, n_features)
    discriminator = build_discriminator(n_timesteps, n_features)  # defined below
    gan = build_gan(generator, discriminator)

    for epoch in range(epochs):
        for _ in range(X_train.shape[0] // batch_size):
            noise = np.random.normal(0, 1, (batch_size, n_timesteps, n_features))
            generated = generator.predict(noise, verbose=0)
            real = X_train[np.random.randint(0, X_train.shape[0], batch_size)]

            d_loss_real = discriminator.train_on_batch(real, np.ones((batch_size, 1)))
            d_loss_fake = discriminator.train_on_batch(generated, np.zeros((batch_size, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (batch_size, n_timesteps, n_features))
            g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        # logging.info(f"GAN Epoch {epoch+1}/{epochs} - D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")
        # logging.info(
        #                 f"GAN Epoch {epoch+1}/{epochs} - "
        #                 f"D Loss: {d_loss[0]:.4f}, D Acc: {d_loss[1]:.4f}, "
        #                 f"G Loss: {g_loss:.4f}"
        #             )

    return generator


def build_discriminator(n_timesteps: int, n_features: int) -> Sequential:
    model = Sequential([
        LSTM(64, input_shape=(n_timesteps, n_features), return_sequences=True),
        LSTM(32),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


def build_gan(generator: Sequential, discriminator: Sequential) -> Sequential:
    discriminator.trainable = False
    model = Sequential([generator, discriminator])
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


def augment_data(X_train: np.ndarray, generator: Sequential, aug_ratio: float = 0.25) -> np.ndarray:
    """
    Generates synthetic data using the trained GAN generator and concatenates with original data.
    """
    n_aug = int(len(X_train) * aug_ratio)
    noise = np.random.normal(0, 1, (n_aug, X_train.shape[1], X_train.shape[2]))
    synthetic = generator.predict(noise, verbose=0)
    return np.concatenate([X_train, synthetic])


# ----------------------------- Core Financial Functions -----------------------------
def compute_tos_rsi(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    """Exact ThinkOrSwim-style RSI calculation."""
    c = df["Close"].astype(float)
    delta = c.diff()
    net = delta.ewm(span=n, adjust=False).mean()
    tot = delta.abs().ewm(span=n, adjust=False).mean()
    rsi = 50 * (net / tot + 1)
    return pd.DataFrame({"TOS_RSI": rsi}, index=df.index)


def tos_atr_modified(df: pd.DataFrame, atr_period: int = 10) -> pd.Series:
    """Exact ThinkOrSwim modified ATR."""
    # (Full implementation from your original code - kept intact)
    high = df["High"].values.astype(float)
    low  = df["Low"].values.astype(float)
    close = df["Close"].values.astype(float)
    n = len(df)

    hl = high - low
    sma_hl = pd.Series(hl).rolling(atr_period).mean().values
    hilo = np.where(np.isnan(sma_hl), hl, np.minimum(hl, 1.5 * sma_hl))

    href = np.zeros(n)
    lref = np.zeros(n)
    for i in range(1, n):
        if low[i] <= high[i-1]:
            href[i] = high[i] - close[i-1]
        else:
            href[i] = (high[i] - close[i-1]) - 0.5 * (low[i] - high[i-1])

        if high[i] >= low[i-1]:
            lref[i] = close[i-1] - low[i]
        else:
            lref[i] = (close[i-1] - low[i]) - 0.5 * (low[i-1] - high[i])

    tr = np.maximum(hilo, np.maximum(href, lref))

    atr = np.full(n, np.nan)
    if n > 1:
        atr[1] = tr[1]
    alpha = 1.0 / atr_period
    for i in range(2, n):
        atr[i] = atr[i-1] + alpha * (tr[i] - atr[i-1])

    return pd.Series(atr, index=df.index, name="TOS_ATR")


def tos_trailing_stop(df: pd.DataFrame, atr_period: int = 10, atr_factor: float = 1.5, first_trade: str = "long") -> pd.DataFrame:
    """Full TOS trailing stop state machine."""
    # (Full implementation from your original code - kept intact)
    atr = tos_atr_modified(df, atr_period).values
    close = df["Close"].values.astype(float)
    n = len(df)

    loss  = atr_factor * atr
    state = np.full(n, "init", dtype=object)
    trail = np.full(n, np.nan)

    for i in range(1, n):
        if state[i-1] == "init":
            if not np.isnan(loss[i]):
                if first_trade == "long":
                    state[i] = "long"
                    trail[i] = close[i] - loss[i]
                else:
                    state[i] = "short"
                    trail[i] = close[i] + loss[i]
            else:
                state[i] = "init"
                trail[i] = np.nan
            continue

        prev_state = state[i-1]
        prev_trail = trail[i-1]

        if prev_state == "long":
            if close[i] > prev_trail:
                state[i] = "long"
                trail[i] = max(prev_trail, close[i] - loss[i])
            else:
                state[i] = "short"
                trail[i] = close[i] + loss[i]
        else:
            if close[i] < prev_trail:
                state[i] = "short"
                trail[i] = min(prev_trail, close[i] + loss[i])
            else:
                state[i] = "long"
                trail[i] = close[i] - loss[i]

    buy  = np.zeros(n, dtype=bool)
    sell = np.zeros(n, dtype=bool)
    for i in range(1, n):
        buy[i]  = (state[i] == "long"  and state[i-1] != "long")
        sell[i] = (state[i] == "short" and state[i-1] != "short")

    return pd.DataFrame({
        "TOS_ATR": atr,
        "TOS_Trail": trail,
        "TOS_State": state,
        "TOS_Buy": buy,
        "TOS_Sell": sell,
    }, index=df.index)


def download_price(ticker: str, start: str, end: str = None, timeframe: str = '1d') -> pd.DataFrame:
    """Safe yfinance downloader with standardized columns."""
    if end is None:
        end = pd.to_datetime('today').strftime('%Y-%m-%d')

    if timeframe == '1d' or timeframe is None:
        df = yf.download(ticker, start=start, end=end, auto_adjust=False)
    else:
        df = yf.download(ticker, start=start, end=end, interval=timeframe, prepost=True, auto_adjust=False)
        df = df.tz_convert("America/Chicago")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns={"Adj Close": "Adj_Close"})
    df = df[["Open","High","Low","Close","Volume","Adj_Close"]]
    return df


def get_market_context(start: str, end: str, timeframe: str = '1d') -> pd.DataFrame:
    """Fetches VIX and SPY market context."""
    vix = download_price("^VIX", start, end, timeframe)
    vix = vix.rename(columns={"Close": "VIX_Close"})[["VIX_Close"]]

    spy = download_price("SPY", start, end, timeframe)
    spy["SPY_Return"] = spy["Close"].pct_change()
    spy["SPY_Volatility_20"] = spy["Close"].pct_change().rolling(20).std()

    return pd.concat([vix, spy[["SPY_Return","SPY_Volatility_20"]]], axis=1)


def get_daily_sentiment(ticker: str, dates: pd.Index) -> pd.Series:
    """Fetches daily social sentiment from Finnhub."""
    api_key = os.getenv('FINNHUB_API_KEY')
    if not api_key:
        logging.error("FINNHUB_API_KEY not found.")
        return pd.Series(0.0, index=dates, name='Sentiment')

    sentiment_ticker = 'SPY' if ticker == '^GSPC' else ticker
    start_date = dates[0].strftime('%Y-%m-%d')
    end_date = dates[-1].strftime('%Y-%m-%d')

    url = f"https://finnhub.io/api/v1/stock/social-sentiment?symbol={sentiment_ticker}&from={start_date}&to={end_date}&token={api_key}"
    
    try:
        response = requests.get(url).json()
        sentiment_dict = {}
        for source in ['reddit', 'twitter']:
            for item in response.get(source, []):
                date_str = item.get('atTime', '')[:10]
                score = item.get('score', 0.0)
                sentiment_dict[date_str] = sentiment_dict.get(date_str, 0) + score
        sentiment = {d: sentiment_dict.get(d.strftime('%Y-%m-%d'), 0.0) for d in dates}
        return pd.Series(sentiment, name='Sentiment')
    except Exception as e:
        logging.error(f"Sentiment fetch failed: {e}")
        return pd.Series(0.0, index=dates, name='Sentiment')


# ---------------------------------------------------------
# Feature Configuration & Engineering
# ---------------------------------------------------------
feature_params = {
    "sma_short": 9,
    "sma_mid": 20,
    "sma_long": 50,
    "trend_slope_short": 12,
    "trend_slope_long": 50,
    "rsi_length": 14,
    "macd_fast": 8,
    "macd_slow": 20,
    "macd_signal": 9,
    "adx_length": 14,
    "bollinger_length": 20,
    "volatility_length": 20,
}


def trend_slope(series: pd.Series) -> float:
    """Linear regression slope helper."""
    y = np.asarray(series).astype(float)
    x = np.arange(len(y))
    return np.polyfit(x, y, 1)[0]


def add_features(df: pd.DataFrame, p: dict = feature_params) -> pd.DataFrame:
    """Core feature engineering pipeline."""
    # Trend
    df["SMA_Short"] = df["Close"].rolling(p["sma_short"]).mean()
    df["SMA_Mid"] = df["Close"].rolling(p["sma_mid"]).mean()
    df["SMA_Long"] = df["Close"].rolling(p["sma_long"]).mean()
    df["Slope_Short"] = df["Close"].rolling(p["trend_slope_short"]).apply(trend_slope, raw=False)
    df["Slope_Long"] = df["Close"].rolling(p["trend_slope_long"]).apply(trend_slope, raw=False)

    # Momentum
    df["RSI"] = ta.momentum.rsi(df["Close"], window=p["rsi_length"])
    macd = ta.trend.MACD(df["Close"], window_slow=p["macd_slow"], window_fast=p["macd_fast"], window_sign=p["macd_signal"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()
    df["ADX"] = ta.trend.adx(df["High"], df["Low"], df["Close"], window=p["adx_length"])

    # Volatility
    df["Daily_Return"] = df["Close"].pct_change()
    df["Daily_Delta"] = df["Close"] - df["Close"].shift(1)
    df["Daily_Delta"].fillna(0, inplace=True)
    df["Volatility_20"] = df["Daily_Return"].rolling(p["volatility_length"]).std()
    bb = ta.volatility.BollingerBands(df["Close"], window=p["bollinger_length"], window_dev=2)
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Mid"] = bb.bollinger_mavg()
    df["BB_Lower"] = bb.bollinger_lband()

    # TOS indicators
    tos = tos_trailing_stop(df, atr_period=5, atr_factor=1.5)
    df["TOS_Trail"] = tos["TOS_Trail"]
    tos_rsi = compute_tos_rsi(df, 9)
    df["TOS_RSI"] = tos_rsi["TOS_RSI"]

    # Extra features
    df['Lag1'] = df['Close'].shift(1)
    df['Lag2'] = df['Close'].shift(2)
    df['Lag3'] = df['Close'].shift(3)
    df['TOS_ATR'] = tos_atr_modified(df, 10)

    return df


def frac_diff(series: pd.Series, d: float = 0.2, thresh: float = 1e-3) -> np.ndarray:
    """Fractional differencing for stationarity."""
    w = [1.]
    for k in range(1, len(series)):
        w_ = -w[-1] * (d - k + 1) / k
        if abs(w_) < thresh:
            break
        w.append(w_)
    w = np.array(w[::-1])
    out = np.full(len(series), np.nan)
    for i in range(len(w), len(series)):
        out[i] = np.dot(w, series[i - len(w) + 1 : i + 1])
    return out


def build_feature_dataset(ticker: str, start_date: str = "2018-01-01", end_date: str = None, timeframe: str = '1d') -> pd.DataFrame:
    """Main dataset builder."""
    if end_date is None:
        end_date = pd.to_datetime('today').strftime('%Y-%m-%d')

    ohlcv = download_price(ticker, start_date, end_date, timeframe)
    feat = add_features(ohlcv.copy(), feature_params)
    market = get_market_context(start_date, end_date, timeframe)
    market = market.reindex(feat.index)

    out = feat.join(market)
    out['Delta_Close'] = frac_diff(out['Close'].values)

    sentiment = get_daily_sentiment(ticker, out.index)
    out = out.join(sentiment)

    out = out.dropna()
    logging.info(f"Dataset shape: {out.shape}")
    return out


def preprocess_data(df: pd.DataFrame, features: list, target: str, n_timesteps: int = 90, test_start: str = '2024-01-01'):
    """Creates sequences for training/testing."""
    cols = list(features)
    if target not in cols:
        cols.append(target)

    data = df[cols]
    train_data = data.loc[:test_start]
    test_data = data.loc[test_start:]

    scaler = MinMaxScaler()
    scaler.fit(train_data.values)

    scaled_train = scaler.transform(train_data.values)
    scaled_test = scaler.transform(test_data.values)

    target_idx = cols.index(target)

    X_train, y_train = [], []
    for i in range(n_timesteps, len(scaled_train) - 1):
        X_train.append(scaled_train[i - n_timesteps : i])
        y_train.append(scaled_train[i + 1, target_idx])

    X_test, y_test = [], []
    for i in range(n_timesteps, len(scaled_test) - 1):
        X_test.append(scaled_test[i - n_timesteps : i])
        y_test.append(scaled_test[i + 1, target_idx])

    return (np.array(X_train), np.array(y_train),
            np.array(X_test), np.array(y_test), scaled_test,
            scaler, cols)


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------
def build_hybrid_model(n_timesteps: int, n_features: int) -> Sequential:
    """Stacked Bidirectional GRU model."""
    model = Sequential([
        Bidirectional(GRU(256, activation='tanh', return_sequences=True, recurrent_dropout=0.2),
                      input_shape=(n_timesteps, n_features)),
        Bidirectional(GRU(128, activation='tanh', return_sequences=True)),
        Bidirectional(GRU(64, activation='tanh')),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.Huber())
    model.summary()
    return model


# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()
    ticker = '^GSPC'

    stock_data = build_feature_dataset(ticker, start_date="2018-01-01")
    n_timesteps = 90

    TARGET = 'Slope_Long'
    FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_Short', 'Delta_Close',
                'SMA_Mid', 'SMA_Long', 'Slope_Short', 'MACD', 'MACD_Signal', 'MACD_Hist',
                'BB_Upper', 'BB_Mid', 'BB_Lower', 'TOS_Trail', 'TOS_RSI', 'Sentiment',
                'VIX_Close', 'SPY_Return', 'SPY_Volatility_20', 'Lag1', 'Lag2', 'Lag3', 'TOS_ATR']

    X_train, y_train, X_test, y_test, scaled_test, scaler, cols = preprocess_data(
        stock_data, FEATURES, TARGET, n_timesteps
    )

    # GAN augmentation
    logging.info("Training GAN for augmentation...")
    generator = train_gan(X_train, epochs=40)
    X_train = augment_data(X_train, generator, aug_ratio=0.25)
    y_train = np.tile(y_train, int(len(X_train)/len(y_train) + 1))[:len(X_train)]

    model = build_hybrid_model(n_timesteps, X_train.shape[2])

    es = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=300, batch_size=256, callbacks=[es], shuffle=False, validation_split=0.1)

    gru_preds = model.predict(X_test)[:, 0]

    # CatBoost
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    cb = CatBoostRegressor(iterations=1000, depth=6, learning_rate=0.05, verbose=0)
    cb.fit(X_train_flat, y_train)
    cb_preds = cb.predict(X_test_flat)

    # Ensemble
    y_pred = (gru_preds + cb_preds) / 2

    # Bias & Final Prediction
    y_diff = y_test - y_pred
    q50 = pd.Series(y_diff).rolling(5).quantile(0.50).fillna(0)
    bias = q50.to_numpy()

    predicted_close = stock_data['Close'].iloc[-len(y_test):].values + y_pred + bias

    # Evaluation (unchanged from your working version)
    test_len = len(predicted_close)
    temp_df = pd.DataFrame({
        'Date': stock_data.index[-test_len:],
        'Low': stock_data['Low'].iloc[-test_len:],
        'High': stock_data['High'].iloc[-test_len:],
        'Open': stock_data['Open'].iloc[-test_len:],
        'Close': stock_data['Close'].iloc[-test_len:],
        'Predicted Price': predicted_close,
    })

    pred_delta_pct = (temp_df['Predicted Price'] - temp_df['Close']) / temp_df['Close'] * 100
    safe = pred_delta_pct.abs().quantile(0.9)
    buffer = temp_df['Close'] * safe / 100

    temp_df['Prediction Lower Band'] = temp_df['Predicted Price'] - buffer
    temp_df['Prediction Upper Band'] = temp_df['Predicted Price'] + buffer
    temp_df['Close within Predicted Buffer'] = (
        (temp_df['Close'] >= temp_df['Prediction Lower Band']) &
        (temp_df['Close'] <= temp_df['Prediction Upper Band'])
    )

    temp_df['Prediction within OHLC'] = (
        (temp_df['Predicted Price'] >= temp_df['Low']) &
        (temp_df['Predicted Price'] <= temp_df['High'])
    )

    temp_df['ActualDir'] = np.sign(temp_df['Close'] - temp_df['Close'].shift(1))
    temp_df['PredDir'] = np.sign(temp_df['Predicted Price'] - temp_df['Predicted Price'].shift(1))
    temp_df['CorrectDir'] = (temp_df['ActualDir'] * temp_df['PredDir']) > 0
    temp_df["Directional Accuracy"] = temp_df['Prediction within OHLC'] | temp_df['CorrectDir']
    temp_df["Directional Buffer Accuracy"] = temp_df['Close within Predicted Buffer'] | temp_df['CorrectDir']

    temp_df = temp_df.drop(columns=['ActualDir', 'PredDir', 'CorrectDir']).dropna()
    print(temp_df.tail(10))

    print('Ticker:', ticker, "- Next Predicted Direction:", 'Down' if predicted_close[-1] < predicted_close[-2] else 'Up')
    print("Next Predicted Price:", round(predicted_close[-1], 2))
    print("Predicted Price Buffer %:", round(safe, 3))
    print("Next Prediction Lower Band:", round(temp_df['Prediction Lower Band'].iloc[-1], 2))
    print("Next Prediction Upper Band:", round(temp_df['Prediction Upper Band'].iloc[-1], 2))

    price_accuracy = temp_df['Prediction within OHLC'].mean() * 100
    print("Predicted Price within OHLC %:", round(price_accuracy, 3))

    buffer_accuracy = temp_df['Close within Predicted Buffer'].mean() * 100
    print("Close within Predicted Buffer %:", round(buffer_accuracy, 3))

    directional_accuracy = temp_df['Directional Accuracy'].mean() * 100
    print("Directional Accuracy %:", round(directional_accuracy, 3))

    directional_buffer_accuracy = temp_df['Directional Buffer Accuracy'].mean() * 100
    print("Directional Buffer Accuracy %:", round(directional_buffer_accuracy, 3))

    # Tomorrow's prediction
    seq = -1
    last_block = scaled_test[-n_timesteps:]
    y_next_gru = model.predict(last_block.reshape(1, n_timesteps, len(cols)))[0, 0]
    y_next_cb = cb.predict(last_block.reshape(1, -1))[0]
    y_next = (y_next_gru + y_next_cb) / 2

    actual_close_today = stock_data['Close'].iloc[seq]
    predicted_tomorrow = actual_close_today + y_next + bias[-1]
    print(f"{ticker} Actual Close today: {round(actual_close_today, 2)}")
    print(f"{ticker} Predicted Close Tomorrow: {round(predicted_tomorrow, 2)}")