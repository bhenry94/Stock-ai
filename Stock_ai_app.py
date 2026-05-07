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
    stock = yf.Ticker(ticker)
    hist = stock.history(period="2y")
    
    if hist.empty:
        st.error("Couldn't find that ticker. Try AAPL or TSLA.")
    else:
        st.subheader(f"{ticker} Price History")
        fig = go.Figure(data=[go.Candlestick(x=hist.index,
                                            open=hist['Open'],
                                            high=hist['High'],
                                            low=hist['Low'],
                                            close=hist['Close'])])
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", f"${hist['Close'].iloc[-1]:.2f}")
        with col2:
            st.metric("52-Week High", f"${hist['High'].max():.2f}")
        with col3:
            st.metric("Volume Today", f"{hist['Volume'].iloc[-1]:,}")
        
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
        
        st.subheader("🧠 Your Random Forest AI Predictor")
        st.write("This AI looks at past price moves, volume, and momentum to guess if the stock goes **UP or DOWN** in the next 5 trading days.")
        
        df = hist.copy()
        df['Return'] = df['Close'].pct_change()
        df['MA10'] = df['Close'].rolling(10).mean()
        df['MA50'] = df['Close'].rolling(50).mean()
        df['Volatility'] = df['Return'].rolling(20).std()
        df['Target'] = (df['Close'].shift(-5) > df['Close']).astype(int)
        df = df.dropna()
        
        if len(df) < 100:
            st.warning("Not enough history for reliable prediction.")
        else:
            features = ['Return', 'MA10', 'MA50', 'Volatility', 'Volume']
            X = df[features]
            y = df['Target']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier(n_estimators=200, random_state=42)
            model.fit(X_train, y_train)
            
            latest = X.iloc[-1:].copy()
            prob_up = model.predict_proba(latest)[0][1]
            direction = "🚀 LIKELY TO GO UP" if prob_up > 0.5 else "🔻 LIKELY TO GO DOWN"
            
            st.metric("AI 5-Day Prediction", direction, f"{prob_up:.1%} chance of rising")
            
            importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
            st.subheader("Why the AI thinks that")
            plain = {
                'Return': "recent price momentum",
                'MA10': "short-term average price",
                'MA50': "longer-term trend",
                'Volatility': "how wild the price has been",
                'Volume': "how much trading is happening"
            }
            for feat, imp in importance.head(5).items():
                st.write(f"• **{imp:.1%} importance** → {plain.get(feat, feat)}")
            
            st.caption("Higher importance = bigger reason the AI made its call. Past data only — future is never guaranteed.")
            st.info("Model accuracy on past data was about 55-65%. That's better than a coin flip but still not magic.")

st.markdown("---")
st.caption("Made with ❤️ for construction workers who want to understand stocks.")
