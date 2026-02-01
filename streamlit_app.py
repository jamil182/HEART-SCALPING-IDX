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

# Custom CSS
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
    return tr.rolling(window=length).mean()

def calculate_rsi(close, length):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=length).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=length).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

# Sidebar
st.sidebar.header("HEART SCALPING IDX")
style_options = ["Super Aggressive (a=0.5, c=5)", "Stealing Profit (a=1.0, c=10)", "Fast Scalping (a=0.8, c=8)", "Relaxing Swing (a=2.0, c=15)"]
selected_style = st.sidebar.selectbox("Trading Style", style_options, index=0)

if selected_style.startswith("Super"): a, c = 0.5, 5
elif selected_style.startswith("Stealing"): a, c = 1.0, 10
elif selected_style.startswith("Fast"): a, c = 0.8, 8
else: a, c = 2.0, 15

use_confirmed_bar = st.sidebar.checkbox("Use confirmed bar (less repaint)", True)
timeframe = st.sidebar.selectbox("Timeframe", ["5m", "15m", "30m", "1h"], index=0)
auto_refresh = st.sidebar.checkbox("Auto-refresh every 60 seconds", value=False)
refresh_now = st.sidebar.button("🔄 Refresh Now")

# Data loader
@st.cache_data(ttl=60 if auto_refresh else 300)
def load_data():
    try:
        df = yf.download("^JKSE", period="10d", interval=timeframe, progress=False)
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

    # Safe NaN handling
    df['EMA21'] = df['EMA21'].ffill()
    df['EMA50'] = df['EMA50'].ffill()
    df['RSI14'] = df['RSI14'].ffill()
    df['ATR']   = df['ATR'].ffill()

    df = df.dropna().reset_index(drop=True)
    if len(df) < 10:
        return None, None

    # Filters
    df['trend_ok'] = (df['Close'] > df['EMA50']) & (df['EMA21'] > df['EMA50'])
    df['rsi_ok'] = df['RSI14'] > 48
    df['vol_ok'] = df['Volume'] > df['Vol_MA20'] * 1.3
    df['candle_ok'] = (df['Close'] > df['Open']) & (df['Close'].shift(1) < df['Open'].shift(1))
    df['filter_ok'] = df['trend_ok'] & df['rsi_ok'] & df['vol_ok'] & df['candle_ok']

    # HEART Trailing Stop
    src = df['Close']
    nloss = a * df['ATR']
    trail = np.full(len(df), np.nan)
    trail[0] = src.iloc[0] - nloss.iloc[0] if not np.isnan(nloss.iloc[0]) else src.iloc[0]

    for i in range(1, len(df)):
        prev = trail[i-1]
        curr = src.iloc[i]
        prev_src = src.iloc[i-1]
        if curr > prev:
            trail[i] = max(prev, curr - nloss.iloc[i])
        elif prev_src < prev:
            trail[i] = min(prev, curr + nloss.iloc[i])
        else:
            trail[i] = curr - nloss.iloc[i] if curr > prev else curr + nloss.iloc[i]

    df['Trail'] = trail

    # Signals
    cross_up_raw = (src.shift(1) < df['Trail'].shift(1)) & (src > df['Trail'])
    if use_confirmed:
        df['Buy'] = (df['Close'].shift(2) < df['Trail'].shift(2)) & (df['Close'].shift(1) > df['Trail'].shift(2)) & df['filter_ok'].shift(1)
    else:
        df['Buy'] = cross_up_raw & df['filter_ok']
    df['Sell'] = (src.shift(1) > df['Trail'].shift(1)) & (src < df['Trail'])

    # Position tracking
    pos = 0
    positions = []
    entries = []
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

# Main app
st.markdown('<div class="main-header">❤️ HEART SCALPING ^JKSE</div>', unsafe_allow_html=True)

df_raw = load_data()

if df_raw is None:
    st.error("Cannot load ^JKSE data right now.")
else:
    df, latest = run_heart_logic(df_raw, a, c, use_confirmed_bar)

    if latest is None:
        st.warning("Not enough data yet.")
    else:
        # Safe status
        pos = latest.get('Position', 0)
        buy_sig = latest.get('Buy', False)
        if pos == 1:
            entry = f"{latest['Entry']:.0f}" if pd.notna(latest.get('Entry')) else "?"
            trail = f"{latest['Trail']:.0f}" if pd.notna(latest.get('Trail')) else "?"
            st.markdown(f'<div class="status-box buy-box">LONG ACTIVE | Entry ≈ {entry} | Trail {trail}</div>', unsafe_allow_html=True)
        elif buy_sig:
            st.markdown('<div class="status-box buy-box">BUY SIGNAL !</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-box hold-box">WAIT • NO CLEAR SETUP</div>', unsafe_allow_html=True)

        # Safe metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Last Close", f"{latest['Close']:.0f}" if pd.notna(latest.get('Close')) else "-")
        col2.metric("ATR", f"{latest['ATR']:.0f}" if pd.notna(latest.get('ATR')) else "-")
        col3.metric("RSI 14", f"{latest['RSI14']:.1f}" if pd.notna(latest.get('RSI14')) else "-")
        col4.metric("Trail Stop", f"{latest['Trail']:.0f}" if pd.notna(latest.get('Trail')) else "-")

        # Safe chart
        try:
            df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Trail'])
            if len(df_clean) > 5:
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.6, 0.25, 0.15])
                fig.add_trace(go.Candlestick(x=df_clean.index, open=df_clean.Open, high=df_clean.High, low=df_clean.Low, close=df_clean.Close), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['Trail'], line=dict(color='#f59e0b', width=2), name="Trail", fill='tonexty'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean.EMA21, name="EMA21"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean.EMA50, name="EMA50"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean.RSI14, name="RSI"), row=2, col=1)
                fig.add_trace(go.Bar(x=df_clean.index, y=df_clean.Volume, name="Volume"), row=3, col=1)
                fig.update_layout(height=780, template='plotly_dark', title=f"^JKSE {timeframe} • {datetime.now().strftime('%H:%M WIB')}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Waiting for more data...")
        except Exception as e:
            st.error(f"Chart rendering issue: {str(e)[:80]}")
            st.dataframe(df.tail(5))

st.caption("HEART Scalping IDX • Long-only • Educational only")

if auto_refresh and refresh_now is False:
    time.sleep(60)
    st.rerun()
