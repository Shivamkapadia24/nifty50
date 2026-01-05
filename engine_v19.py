import pandas as pd
import numpy as np
import joblib

model = joblib.load("v19_model.pkl")

def ai_predict_next(df):
    df = df.copy()

    # Ensure enough candles exist
    # after features are created
    df = df.tail(200)
    df = df.fillna(method="bfill").fillna(method="ffill")

    if len(df) < 60:
        return {
            "direction": "NOT ENOUGH DATA",
            "confidence": 0
        }

    # ================= FEATURES =================
    df["return"] = df["Close"].pct_change()
    df["body"] = df["Close"] - df["Open"]
    df["hl_range"] = df["High"] - df["Low"]
    df["co_range"] = df["Close"] - df["Open"]

    # RSI
    change = df["Close"].diff()
    gain = np.where(change > 0, change, 0)
    loss = np.where(change < 0, abs(change), 0)
    avg_gain = pd.Series(gain).rolling(14).mean()
    avg_loss = pd.Series(loss).rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    # EMA
    df["ema20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["Close"].ewm(span=50, adjust=False).mean()

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26

    # ATR
    df["atr"] = (df["High"] - df["Low"]).rolling(14).mean()

    # Training extras
    df["volatility"] = df["return"].rolling(10).std()
    df["trend_strength"] = df["ema20"] - df["ema50"]
    df["momentum"] = df["Close"] - df["Close"].shift(10)

    # Only drop NA on last 100 candles, not whole dataset
    df = df.tail(200)
    df = df.fillna(method="bfill").fillna(method="ffill")

    # If STILL empty -> bail safely
    if df.empty:
        return {
            "direction": "NOT ENOUGH DATA",
            "confidence": 0
        }

    latest = df.tail(1)

    required = list(model.feature_names_in_)

    for c in required:
        if c not in latest.columns:
            latest[c] = 0

    X = latest[required]

    probs = model.predict_proba(X)[0]
    pred = model.predict(X)[0]

    return {
        "direction": "UP" if pred == 1 else "DOWN",
        "confidence": round(float(probs.max()) * 100, 2)

    }
