import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# ---------------- LOAD TRAINED MODEL ----------------
# (For now we rebuild inside file — later we will load saved .pkl)
model = None

def load_model(df):
    global model
    if model is not None:
        return model
    
    X = df.drop(columns=['datetime','direction','next_close'])
    y = df['direction']

    split = int(len(df)*0.8)

    X_train = X[:split]
    y_train = y[:split]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=42
    )

    model.fit(X_train,y_train)
    return model


# ---------------- BASIC INDICATORS ----------------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_atr(df, period=14):
    df["hl"] = df["High"] - df["Low"]
    df["hc"] = (df["High"] - df["Close"].shift()).abs()
    df["lc"] = (df["Low"] - df["Close"].shift()).abs()
    df["tr"] = df[["hl","hc","lc"]].max(axis=1)
    df["atr"] = df["tr"].rolling(period).mean()
    return df


# ---------------- AI DECISION ENGINE ----------------
def generate_signal(df):

    df = df.copy()

    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

    df["RSI"] = compute_rsi(df["Close"])
    df = compute_atr(df)

    latest = df.iloc[-1]

    price = float(latest["Close"])
    ema20 = float(latest["EMA20"])
    ema50 = float(latest["EMA50"])
    rsi = float(latest["RSI"])
    atr = float(latest["atr"])

    signal = "NO TRADE"
    confidence = 0
    trend = "Neutral"
    risk = "Normal"
    sl = 0
    tp = 0

    # -------- TREND --------
    if ema20 > ema50:
        trend = "Bullish"
    elif ema20 < ema50:
        trend = "Bearish"

    # -------- BUY --------
    if (ema20 > ema50) and (rsi > 55):
        signal = "BUY"
        confidence = 0.65
        sl = price - (atr * 0.7)
        tp = price + (atr * 1.25)

    # -------- SELL --------
    elif (ema20 < ema50) and (rsi < 45):
        signal = "SELL"
        confidence = 0.65
        sl = price + (atr * 0.7)
        tp = price - (atr * 1.25)

    # -------- RISK --------
    if atr > df["atr"].mean():
        risk = "High"

    return {
        "signal": signal,
        "confidence": round(confidence, 2),
        "trend": trend,
        "risk": risk,
        "sl": round(sl, 2),
        "tp": round(tp, 2),
        "price": price
    }
