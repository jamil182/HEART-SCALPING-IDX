import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="HEART SCALPING IDX ^JKSE", page_icon="❤️", layout="wide")

# CSS
st.markdown("""
<style>
.main-header {font-size:2.4rem; color:#00d4ff; font-weight:bold; text-align:center; margin-bottom:1rem;}
.status-box {padding:1.2rem; border-radius:10px; color:white; font-weight:bold; text-align:center; margin:1rem 0;}
.buy-box {background:linear-gradient(135deg, #11998e, #38ef7d);}
.hold-box {background:linear-gradient(135deg, #feca57, #ff9ff3);}
</style>
""", unsafe_allow_html=True)

# Indicator functions
def calculate_atr(high, low, close, length):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(length).mean()

def calculate_rsi(close, length=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(length).mean()
    loss = -delta.where(delta < 0, 0).rolling(length).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

# Sidebar
st.sidebar.header("HEART SCALPING IDX")
style_options = ["Super Aggressive (a=0.5, c=5)", "Stealing Profit (a=1.0, c=10)", "Fast Scalping (a=0.8, c=8)", "Relaxing Swing (a=2.0, c=15)"]
selected_style = st.sidebar.selectbox("Trading Style", style_options, index=0)
if "Super" in selected_style: a, c = 0.5, 5
elif "Stealing" in selected_style: a, c = 1.0, 10
elif "Fast" in selected_style: a, c = 0.8, 8
else: a, c = 2.0, 15

use_confirmed_bar = st.sidebar.checkbox("Use confirmed bar", True)
timeframe = st.sidebar.selectbox("Timeframe", ["5m", "15m", "30m"], index=0)
auto_refresh = st.sidebar.checkbox("Auto-refresh every 60 seconds", value=False)

# Load data
@st.cache_data(ttl=300)
def load_data():
    try:
        df = yf.download("^JKSE", period="8d", interval=timeframe, progress=False)
        if df.empty:
            return None
        return df[['Open','High','Low','Close','Volume']].copy()
    except:
        return None

# Core logic
def run_heart_logic(df, a, c, use_confirmed):
    if df is None or len(df) < 30:
        return None, None
    df = df.copy()

    df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'], c)
    df['EMA21'] = calculate_ema(df['Close'], 21)
    df['EMA50'] = calculate_ema(df['Close'], 50)
    df['RSI14'] = calculate_rsi(df['Close'], 14)
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()

    # Fill NaNs
    for col in ['EMA21', 'EMA50', 'RSI14', 'ATR']:
        df[col] = df[col].ffill()

    df = df.dropna().reset_index(drop=True)
    if len(df) < 10:
        return None, None

    # Use .gt() instead of > to avoid alignment error
    df['trend_ok'] = df['Close'].gt(df['EMA50']) & df['EMA21'].gt(df['EMA50'])
    df['rsi_ok'] = df['RSI14'] > 48
    df['vol_ok'] = df['Volume'] > df['Vol_MA20'] * 1.3
    df['candle_ok'] = (df['Close'] > df['Open']) & (df['Close'].shift(1) < df['Open'].shift(1))
    df['filter_ok'] = df['trend_ok'] & df['rsi_ok'] & df['vol_ok'] & df['candle_ok']

    # Trailing Stop
    src = df['Close']
    nloss = a * df['ATR']
    trail = np.full(len(df), np.nan)
    trail[0] = src.iloc[0] - nloss.iloc[0] if not np.isnan(nloss.iloc[0]) else src.iloc[0]
    for i in range(1, len(df)):
        prev_trail = trail[i-1]
        curr_src = src.iloc[i]
        prev_src = src.iloc[i-1]
        if curr_src > prev_trail:
            trail[i] = max(prev_trail, curr_src - nloss.iloc[i])
        elif prev_src < prev_trail:
            trail[i] = min(prev_trail, curr_src + nloss.iloc[i])
        else:
            trail[i] = curr_src - nloss.iloc[i] if curr_src > prev_trail else curr_src + nloss.iloc[i]
    df['Trail'] = trail

    # Signals
    if use_confirmed:
        df['Buy'] = (df['Close'].shift(2) < df['Trail'].shift(2)) & (df['Close'].shift(1) > df['Trail'].shift(2)) & df['filter_ok'].shift(1)
    else:
        df['Buy'] = (src.shift(1) < df['Trail'].shift(1)) & (src > df['Trail'])
    df['Sell'] = (src.shift(1) > df['Trail'].shift(1)) & (src < df['Trail'])

    # Position tracking
    pos = 0
    positions, entries = [], []
    for i in range(len(df)):
        if df['Buy'].iloc[i] and pos == 0:
            pos = 1
            entries.append(df['Close'].iloc[i])
        elif df['Sell'].iloc[i] and pos == 1:
            pos = 0
            entries.append(np.nan)
        else:
            entries.append(np.nan if pos == 0 else entries[-1] if entries else np.nan)
        positions.append(pos)
    df['Position'] = positions
    df['Entry'] = entries

    return df, df.iloc[-1] if not df.empty else None

# Main UI
st.markdown('<div class="main-header">❤️ HEART SCALPING ^JKSE</div>', unsafe_allow_html=True)

df_raw = load_data()

if df_raw is None:
    st.error("Cannot load ^JKSE data right now (market closed?)")
else:
    df, latest = run_heart_logic(df_raw, a, c, use_confirmed_bar)

    if latest is None:
        st.warning("Not enough data yet. Waiting for more bars...")
    else:
        # Status
        if latest.get('Position', 0) == 1:
            st.markdown(f'<div class="status-box buy-box">LONG ACTIVE | Entry ≈ {latest.get("Entry", "?"):.0f} | Trail {latest.get("Trail", "?"):.0f}</div>', unsafe_allow_html=True)
        elif latest.get('Buy', False):
            st.markdown('<div class="status-box buy-box">BUY SIGNAL !</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-box hold-box">WAIT • NO CLEAR SETUP</div>', unsafe_allow_html=True)

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Close", f"{latest.get('Close', 0):.0f}")
        col2.metric("ATR", f"{latest.get('ATR', 0):.0f}")
        col3.metric("RSI 14", f"{latest.get('RSI14', 0):.1f}")
        col4.metric("Trail Stop", f"{latest.get('Trail', 0):.0f}")

        # Simple chart
        try:
            if len(df) > 5:
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.25, 0.15])
                fig.add_trace(go.Candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df.Trail, name="Trail", line=dict(color="orange")), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df.EMA21, name="EMA21"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df.RSI14, name="RSI"), row=2, col=1)
                fig.add_trace(go.Bar(x=df.index, y=df.Volume, name="Volume"), row=3, col=1)
                fig.update_layout(height=780, template="plotly_dark", title=f"^JKSE {timeframe} • {datetime.now().strftime('%H:%M WIB')}")
                st.plotly_chart(fig, use_container_width=True)
        except:
            st.info("Chart not available yet")

st.caption("HEART Scalping IDX • Educational use only")

if auto_refresh:
    import time
    time.sleep(60)
    st.rerun()
