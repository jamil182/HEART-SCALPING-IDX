import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="HEART SCALPING IDX", page_icon="❤️", layout="wide")

st.markdown("""
<style>
.main-header {font-size:2.3rem; color:#00d4ff; font-weight:bold; text-align:center; margin-bottom:1rem;}
.status-box {padding:1rem; border-radius:10px; color:white; font-weight:bold; text-align:center; margin:1rem 0;}
.buy-box {background:linear-gradient(135deg, #11998e, #38ef7d);}
.hold-box {background:linear-gradient(135deg, #feca57, #ff9ff3);}
</style>
""", unsafe_allow_html=True)

def calculate_atr(h, l, c, length):
    tr1 = h - l
    tr2 = (h - c.shift()).abs()
    tr3 = (l - c.shift()).abs()
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
style = st.sidebar.selectbox("Style", ["Super Aggressive (a=0.5,c=5)", "Stealing Profit (a=1.0,c=10)", "Fast (a=0.8,c=8)", "Swing (a=2.0,c=15)"], index=0)
a = 0.5 if "Super" in style else 1.0 if "Stealing" in style else 0.8 if "Fast" in style else 2.0
c = 5 if "Super" in style else 10 if "Stealing" in style else 8 if "Fast" in style else 15

use_confirmed = st.sidebar.checkbox("Use confirmed bar", True)
tf = st.sidebar.selectbox("Timeframe", ["5m", "15m", "30m"], index=0)
auto_refresh = st.sidebar.checkbox("Auto refresh (60s)", False)

@st.cache_data(ttl=60)
def load_data():
    try:
        df = yf.download("^JKSE", period="8d", interval=tf, progress=False)
        return df[['Open','High','Low','Close','Volume']].copy() if not df.empty else None
    except:
        return None

def run_heart_logic(df, a, c, use_confirmed):
    if df is None or len(df) < 30:
        return None, None
    df = df.copy()

    df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'], c)
    df['EMA21'] = calculate_ema(df['Close'], 21)
    df['EMA50'] = calculate_ema(df['Close'], 50)
    df['RSI'] = calculate_rsi(df['Close'])
    df['VolMA'] = df['Volume'].rolling(20).mean()

    df['EMA21'] = df['EMA21'].ffill()
    df['EMA50'] = df['EMA50'].ffill()
    df['ATR'] = df['ATR'].ffill()
    df['RSI'] = df['RSI'].ffill()

    df = df.dropna().reset_index(drop=True)
    if len(df) < 10:
        return None, None

    df['trend_ok'] = (df['Close'] > df['EMA50']) & (df['EMA21'] > df['EMA50'])
    df['rsi_ok'] = df['RSI'] > 48
    df['vol_ok'] = df['Volume'] > df['VolMA'] * 1.3
    df['candle_ok'] = (df['Close'] > df['Open']) & (df['Close'].shift(1) < df['Open'].shift(1))
    df['filter_ok'] = df['trend_ok'] & df['rsi_ok'] & df['vol_ok'] & df['candle_ok']

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

    if use_confirmed:
        df['Buy'] = (df['Close'].shift(2) < df['Trail'].shift(2)) & (df['Close'].shift(1) > df['Trail'].shift(2)) & df['filter_ok'].shift(1)
    else:
        df['Buy'] = (src.shift(1) < df['Trail'].shift(1)) & (src > df['Trail'])

    df['Sell'] = (src.shift(1) > df['Trail'].shift(1)) & (src < df['Trail'])

    # Position
    pos, positions, entries = 0, [], []
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

# Main
st.markdown('<div class="main-header">❤️ HEART SCALPING ^JKSE</div>', unsafe_allow_html=True)

df_raw = load_data()

if df_raw is None:
    st.error("Cannot load ^JKSE data")
else:
    df, latest = run_heart_logic(df_raw, a, c, use_confirmed)

    if latest is None:
        st.warning("Not enough data yet")
    else:
        # Safe status
        if latest.get('Position', 0) == 1:
            st.markdown(f'<div class="status-box buy-box">LONG ACTIVE | Entry ≈ {latest.get("Entry", "?"):.0f} | Trail {latest.get("Trail", "?"):.0f}</div>', unsafe_allow_html=True)
        elif latest.get('Buy', False):
            st.markdown('<div class="status-box buy-box">BUY SIGNAL !</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-box hold-box">WAIT • NO CLEAR SETUP</div>', unsafe_allow_html=True)

        # Safe metrics
        cols = st.columns(4)
        cols[0].metric("Close", f"{latest.get('Close', 0):.0f}")
        cols[1].metric("ATR", f"{latest.get('ATR', 0):.0f}")
        cols[2].metric("RSI", f"{latest.get('RSI', 0):.1f}")
        cols[3].metric("Trail", f"{latest.get('Trail', 0):.0f}")

        # Safe chart
        try:
            df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Trail'])
            if len(df_clean) > 5:
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.25, 0.15])
                fig.add_trace(go.Candlestick(x=df_clean.index, open=df_clean.Open, high=df_clean.High, low=df_clean.Low, close=df_clean.Close), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean.Trail, line=dict(color='orange'), name='Trail'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean.EMA21, name='EMA21'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean.RSI, name='RSI'), row=2, col=1)
                fig.add_trace(go.Bar(x=df_clean.index, y=df_clean.Volume, name='Volume'), row=3, col=1)
                fig.update_layout(height=750, template='plotly_dark', title=f"^JKSE {tf} • {datetime.now().strftime('%H:%M')}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Loading chart...")
        except Exception as e:
            st.warning("Chart temporarily unavailable")
            st.dataframe(df.tail(5))

st.caption("Educational tool only • No financial advice")

if auto_refresh and not refresh_now:
    import time
    time.sleep(60)
    st.rerun()
