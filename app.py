import streamlit as st
import yfinance as yf
import pandas as pd
import time
import pytz
import plotly.graph_objects as go


# ==============================
#  DASHBOARD CONFIG
# ==============================
st.set_page_config(page_title="NIFTY AI Dashboard", layout="wide")

st.title("📈 NIFTY AI Dashboard (V18.3)")
st.write("Live Market Monitor + Institutional AI Signal Engine")


# ==============================
#  SIDEBAR
# ==============================
st.sidebar.header("Settings")

refresh_rate = st.sidebar.slider(
    "Auto Refresh (seconds)",
    min_value=10,
    max_value=120,
    value=30
)
st.sidebar.write(f"Updating every {refresh_rate} seconds ⏳")

symbol = st.sidebar.selectbox(
    "Choose Instrument",
    ["^NSEI", "^NSEBANK"],
    index=0
)

interval = st.sidebar.selectbox(
    "Timeframe",
    ["1m", "5m", "15m"],
    index=2
)

st.sidebar.write("Status: 🟢 Running")


# session memory
if "last_signal" not in st.session_state:
    st.session_state.last_signal = "NO TRADE"

placeholder = st.empty()


# ==============================
#  AUTO REFRESH LOOP
# ==============================
while True:
    with placeholder.container():

        # ==============================
        #  MARKET DATA SECTION
        # ==============================
        st.subheader("Market Data")

        try:
            df = yf.download(
                tickers=symbol,
                period="5d",
                interval=interval
            )

            # Convert to India time
            df.index = df.index.tz_convert('Asia/Kolkata')
            df.index = df.index.tz_localize(None)

            if df.empty:
                st.error("No data received from Yahoo Finance")
            else:
                st.success("Data Loaded Successfully")

                st.dataframe(df.tail())

                # ========== CANDLE CHART ==========
                df_plot = df.tail(80)

                fig = go.Figure()

                fig.add_trace(go.Candlestick(
                    x=df_plot.index,
                    open=df_plot["Open"],
                    high=df_plot["High"],
                    low=df_plot["Low"],
                    close=df_plot["Close"],
                    name="Price"
                ))

                # EMA20
                df_plot["EMA20"] = df_plot["Close"].ewm(span=20, adjust=False).mean()
                fig.add_trace(go.Scatter(
                    x=df_plot.index,
                    y=df_plot["EMA20"],
                    line=dict(color="yellow", width=2),
                    name="EMA 20"
                ))

                # EMA50
                df_plot["EMA50"] = df_plot["Close"].ewm(span=50, adjust=False).mean()
                fig.add_trace(go.Scatter(
                    x=df_plot.index,
                    y=df_plot["EMA50"],
                    line=dict(color="cyan", width=2),
                    name="EMA 50"
                ))

                fig.update_layout(
                    title="NIFTY Live Price (TradingView Style)",
                    xaxis_title="Time (India)",
                    yaxis_title="Price",
                    template="plotly_dark",
                    height=500,
                    xaxis_rangeslider_visible=True
                )

                st.plotly_chart(fig, width='stretch')

        except Exception as e:
            st.error("Error fetching data")
            st.code(str(e))
            df = None


# ==============================
        #  AI ENGINE
        # ==============================
        if df is not None:
            from engine_v17 import generate_signal
            from engine_v19 import ai_predict_next   # <-- V19 Added

            # ==============================
            #  V18 TRADE ENGINE
            # ==============================
            st.subheader("AI Trade Signal")

            try:
                result = generate_signal(df)

                # ---------------- ROW 1 ----------------
                st.markdown("### 📊 Market Decision Panel")

                r1c1, r1c2, r1c3 = st.columns(3)

                with r1c1:
                    st.metric("Signal", result["signal"])

                with r1c2:
                    st.metric("Confidence", f"{int(result['confidence']*100)} %")

                with r1c3:
                    st.metric("Trend", result["trend"])

                # ---------------- ROW 2 ----------------
                st.markdown("### 🎯 Execution Guidance")

                r2c1, r2c2, r2c3 = st.columns(3)

                with r2c1:
                    st.metric("Suggested SL", result["sl"])

                with r2c2:
                    st.metric("Suggested TP", result["tp"])

                with r2c3:
                    st.metric("Risk Level", result["risk"])

            except Exception as e:
                st.error("AI Engine Error")
                st.code(str(e))


            # ==============================
            #  V19 NEXT CANDLE AI
            # ==============================
            st.subheader("🧠 V19 AI — Next Candle Prediction")

            try:
                if interval != "5m":
                    st.warning("V19 AI works only on 5-minute timeframe. Switch to 5m.")
                else:
                    from engine_v19 import ai_predict_next
                    
                    df_clean = df.copy()

                    # Fix multi-index dataframe from Yahoo
                    if isinstance(df_clean.columns, pd.MultiIndex):
                        df_clean.columns = [c[0] for c in df_clean.columns]

                    df_clean = df_clean.tail(200)

                    ai = ai_predict_next(df_clean)

                    colA, colB = st.columns(2)

                    with colA:
                        st.metric("Next Candle Direction", ai["direction"])

                    with colB:
                        st.metric("AI Confidence", f"{ai['confidence']} %")

            except Exception as e:
                st.error("V19 AI Error")
                st.exception(e)

        st.caption("V18.3 + V19 Hybrid — Institutional Trade Mode Active")

    time.sleep(refresh_rate)
    st.rerun()