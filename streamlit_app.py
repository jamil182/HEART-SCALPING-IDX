import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# --- CONFIG ---
st.set_page_config(page_title="HEART SCALPING IDX ^JKSE", page_icon="❤️", layout="wide")

st.markdown("""
<style>
.main-header {font-size:2.4rem; color:#00d4ff; font-weight:bold; text-align:center;}
.status-box {padding:1rem; border-radius:10px; color:white; font-weight:bold; text-align:center; margin:1rem 0;}
.buy-box {background:linear-gradient(135deg, #11998e, #38ef7d);}
.hold-box {background:linear-gradient(135deg, #feca57, #ff9ff3);}
</style>
""", unsafe_allow_html=True)

# --- INDICATORS ---
def calculate_atr(high, low, close, length):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(length).mean()

def calculate_rsi(close, length=14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / (loss + 1e-9) # Hindari division by zero
    return 100 - (100 / (1 + rs))

def calculate_ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

# --- SIDEBAR ---
st.sidebar.header("HEART SCALPING IDX")
style = st.sidebar.selectbox("Trading Style", 
    ["Super Aggressive (a=0.5, c=5)", "Stealing Profit (a=1.0, c=10)", "Fast Scalping (a=0.8, c=8)", "Relaxing Swing (a=2.0, c=15)"], 
    index=0)

a = 0.5 if "Super" in style else 1.0 if "Stealing" in style else 0.8 if "Fast" in style else 2.0
c = 5 if "Super" in style else 10 if "Stealing" in style else 8 if "Fast" in style else 15

use_confirmed = st.sidebar.checkbox("Use confirmed bar", True)
timeframe = st.sidebar.selectbox("Timeframe", ["5m", "15m", "30m"], index=0)

if st.sidebar.button("Refresh Now"):
    st.rerun()

# --- DATA LOADING ---
@st.cache_data(ttl=60)
def load_data():
    try:
        # Gunakan auto_adjust=True agar kolom Close, High, Low bersih
        df = yf.download("^JKSE", period="8d", interval=timeframe, progress=False, auto_adjust=True)
        if df.empty:
            return None
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# --- LOGIC ---
def run_heart_logic(df_input, a, c, use_confirmed):
    if df_input is None or len(df_input) < 50:
        return None, None
    
    df = df_input.copy()
    
    # Perhitungan Indikator
    df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'], c)
    df['EMA21'] = calculate_ema(df['Close'], 21)
    df['EMA50'] = calculate_ema(df['Close'], 50)
    df['RSI14'] = calculate_rsi(df['Close'], 14)
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()

    # Drop bar awal yang masih kosong (NaN) agar perhitungan numpy sinkron
    df = df.dropna().copy()
    if len(df) < 10: return None, None

    # Ekstrak values ke Numpy untuk menghindari Index Alignment Error
    c_val = df['Close'].values
    o_val = df['Open'].values
    ema21 = df['EMA21'].values
    ema50 = df['EMA50'].values
    rsi   = df['RSI14'].values
    vol   = df['Volume'].values
    vma20 = df['Vol_MA20'].values
    atr   = df['ATR'].values

    # Kondisi Filter Utama
    trend_ok  = (c_val > ema50) & (ema21 > ema50)
    rsi_ok    = rsi > 48
    vol_ok    = vol > (vma20 * 1.2)
    candle_ok = (c_val > o_val) & (np.roll(c_val, 1) < np.roll(o_val, 1)) # Bullish Engulfing simpel
    
    filter_ok = trend_ok & rsi_ok & vol_ok & candle_ok
    df['filter_ok'] = filter_ok

    # Trail Stop Logic
    nloss = a * atr
    trail = np.zeros(len(df))
    trail[0] = c_val[0] - nloss[0]

    for i in range(1, len(df)):
        prev_t = trail[i-1]
        curr_c = c_val[i]
        prev_c = c_val[i-1]
        
        if curr_c > prev_t and prev_c > prev_t:
            trail[i] = max(prev_t, curr_c - nloss[i])
        elif curr_c < prev_t and prev_c < prev_t:
            trail[i] = min(prev_t, curr_c + nloss[i])
        else:
            trail[i] = (curr_c - nloss[i]) if curr_c > prev_t else (curr_c + nloss[i])

    df['Trail'] = trail

    # Buy & Sell Signals
    if use_confirmed:
        # Menggunakan shift agar sinyal muncul setalah candle tutup di atas trail
        df['Buy'] = (df['Close'].shift(1) > df['Trail'].shift(1)) & \
                    (df['Close'].shift(2) < df['Trail'].shift(2)) & \
                    (df['filter_ok'].shift(1) == True)
    else:
        df['Buy'] = (c_val > trail) & (np.roll(c_val, 1) < np.roll(trail, 1)) & filter_ok

    df['Sell'] = (c_val < trail) & (np.roll(c_val, 1) > np.roll(trail, 1))

    # Simulasi Posisi
    pos = 0
    positions = []
    entry_prices = []
    current_entry = np.nan

    for i in range(len(df)):
        if df['Buy'].iloc[i] and pos == 0:
            pos = 1
            current_entry = df['Close'].iloc[i]
        elif df['Sell'].iloc[i] and pos == 1:
            pos = 0
            current_entry = np.nan
        
        positions.append(pos)
        entry_prices.append(current_entry)

    df['Position'] = positions
    df['Entry'] = entry_prices
    
    return df, df.iloc[-1]

# --- UI RENDER ---
st.markdown('<div class="main-header">❤️ HEART SCALPING ^JKSE</div>', unsafe_allow_html=True)

df_raw = load_data()

if df_raw is None:
    st.error("Gagal menarik data dari Yahoo Finance.")
else:
    df, latest = run_heart_logic(df_raw, a, c, use_confirmed)
    
    if latest is None:
        st.warning("Data tidak cukup untuk kalkulasi indikator.")
    else:
        # Status Box
        if latest['Position'] == 1:
            st.markdown(f'<div class="status-box buy-box">LONG ACTIVE | Entry: {latest["Entry"]:.0f} | Stop: {latest["Trail"]:.0f}</div>', unsafe_allow_html=True)
        elif latest['Buy']:
            st.markdown('<div class="status-box buy-box">🚀 SIGNAL BUY!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-box hold-box">WAITING FOR SETUP...</div>', unsafe_allow_html=True)

        # Dashboard Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Close", f"{latest['Close']:.0f}")
        c2.metric("ATR", f"{latest['ATR']:.1f}")
        c3.metric("RSI", f"{latest['RSI14']:.1f}")
        c4.metric("Trail", f"{latest['Trail']:.0f}")

        # Plotly Chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
        
        # Main Candle & Trail
        fig.add_trace(go.Candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close, name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df.Trail, line=dict(color='orange', width=2), name="Trail Stop"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df.EMA50, line=dict(color='gray', dash='dash'), name="EMA50"), row=1, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=df.index, y=df.RSI14, line=dict(color='#ff9ff3'), name="RSI"), row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", row=2, col=1, line_color="white", opacity=0.3)

        fig.update_layout(height=700, template='plotly_dark', xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

st.caption(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data: Yahoo Finance")
