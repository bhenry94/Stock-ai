import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="BlueCollar Stock AI", layout="wide")
st.title("🛠 BlueCollar Stock AI")
st.markdown("**Built for guys who pour concrete all day and still want to understand this Wall Street bullshit.**")
st.caption("Educational toy only. Not financial advice. You can lose money.")

ticker = st.text_input("Enter stock ticker (example: AAPL, TSLA, NVDA)", "AAPL").upper().strip()

if ticker:
    # Fetch data
    stock = yf.Ticker(ticker)
    hist = stock.history(period="2y")
    
    if hist.empty:
        st.error("Couldn't find that ticker. Try AAPL or TSLA.")
    else:
        # Basic chart
        st.subheader(f"{ticker} Price History")
        fig = go.Figure(data=[go.Candlestick(x=hist.index,
                                            open=hist['Open'],
                                            high=hist['High'],
                                            low=hist['Low'],
                                            close=hist['Close'])])
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Current stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", f"${hist['Close'].iloc[-1]:.2f}")
        with col2:
            st.metric("52-Week High", f"${hist['High'].max():.2f}")
        with col3:
            st.metric("Volume Today", f"{hist['Volume'].iloc[-1]:,}")
        
        # Institutional holders
        st.subheader("Big Money Owners (Current Holdings)")
        try:
            holders = stock.get_institutional_holders()
            if holders is not None and not holders.empty:
                st.dataframe(holders[['Holder', 'Shares', '% Out']], use_container_width=True)
                st.caption("This shows who the big funds currently own and how much of the company they have.")
            else:
                st.info("No institutional holder data available right now.")
        except:
            st.info("Holder data temporarily unavailable.")
        
        # AI Prediction Section
        st.subheader("🧠 Your Random Forest AI Predictor")
        st.write("This AI looks at past price moves, volume, and momentum to guess if the stock goes **UP or DOWN** in the next 5 trading days.")
        
        # Prepare data for model
        df = hist.copy()
        df['Return'] = df['Close'].pct_change()
        df['MA10'] = df['Close'].rolling(10).mean()
        df['MA50'] = df['Close'].rolling(50).mean()
        df['Volatility'] = df['Return'].rolling(20).std()
        df['Target'] = (df['Close'].shift(-5) > df['Close']).astype(int)  # 1 = up in 5 days
        df = df.dropna()
        
        if
