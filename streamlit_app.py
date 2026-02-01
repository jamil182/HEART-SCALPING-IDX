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

# --- INDICATORS (Calculated on Series) ---
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
    rs = gain / (loss + 1e-9)
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

# --- DATA LOADING (The Fix) ---
@st.cache_data(ttl=60)
def load_data():
    try:
        # Download data
        raw_df = yf.download("^JKSE", period="8d", interval=timeframe, progress=False, auto_adjust=True)
        
        if raw_df.empty:
            return None
        
        # FIX: Hilangkan MultiIndex jika ada
        df = raw_df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Pastikan kolom standar dan reset index agar bersih
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        return df
    except Exception as e:
        st.error(f"Data Error: {e}")
        return None

# --- LOGIC ---
def run_heart_logic(df, a, c, use_confirmed):
    if df is None or len(df) < 50:
        return None, None
    
    # Kalkulasi indikator
    df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'], c)
    df['EMA21'] = calculate_ema(df['Close'], 21)
    df['EMA50'] = calculate_ema(df['Close'], 50)
    df['RSI14'] = calculate_rsi(df['Close'], 14)
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()
    
    df = df.dropna().copy()
    if len(df) < 5: return None, None

    # --- FIX: Konversi ke Numpy untuk Komparasi (Mencegah Alignment Error) ---
    closes = df['Close'].values
    opens = df['Open'].values
    e21 = df['EMA21'].values
    e50 = df['EMA50'].values
    rsis = df['RSI14'].values
    vols = df['Volume'].values
    vmas = df['Vol_MA20'].values
    atrs = df['ATR'].values

    # Kondisi menggunakan Numpy logic
    trend_ok = (closes > e50) & (e21 > e50)
    rsi_ok = rsis > 48
    vol_ok = vols > (vmas * 1.2)
    candle_ok = (closes > opens) & (np.roll(closes, 1) < np.roll(opens, 1))
    
    # Simpan kembali ke DataFrame
    df['filter_ok'] = trend_ok & rsi_ok & vol_ok & candle_ok

    # Trail Stop calculation
    nloss = a * atrs
    trail = np.zeros(len(df))
    trail[0] = closes[0] - nloss[0]

    for i in range(1, len(df)):
        p_t = trail[i-1]
        if closes[i] > p_t and closes[i-1] > p_t:
            trail[i] = max(p_t, closes[i] - nloss[i])
        elif closes[i] < p_t and closes[i-1] < p_t:
            trail[i] = min(p_t, closes[i] + nloss[i])
        else:
            trail[i] = (closes[i] - nloss[i]) if closes[i] > p_t else (closes[i] + nloss[i])

    df['Trail'] = trail

    # Signals
    if use_confirmed:
        df['Buy'] = (df['Close'].shift(1) > df['Trail'].shift(1)) & \
                    (df['Close'].shift(2) < df['Trail'].shift(2)) & \
                    (df['filter_ok'].shift(1) == True)
    else:
        df['Buy'] = (df['Close'] > df['Trail']) & \
                    (df['Close'].shift(1) < df['Trail'].shift(1)) & \
                    (df['filter_ok'] == True)

    df['Sell'] = (df['Close'] < df['Trail']) & (df['Close'].shift(1) > df['Trail'].shift(1))

    # Position Logic
    pos = 0
    positions, entries = [], []
    curr_entry = np.nan

    for i in range(len(df)):
        if df['Buy'].iloc[i] and pos == 0:
            pos = 1
            curr_entry = df['Close'].iloc[i]
        elif df['Sell'].iloc[i] and pos == 1:
            pos = 0
            curr_entry = np.nan
        positions.append(pos)
        entries.append(curr_entry)

    df['Position'] = positions
    df['Entry'] = entries
    
    return df, df.iloc[-1]

# --- UI ---
st.markdown('<div class="main-header">❤️ HEART SCALPING ^JKSE</div>', unsafe_allow_html=True)

df_raw = load_data()

if df_raw is not None:
    df, latest = run_heart_logic(df_raw, a, c, use_confirmed)
    
    if latest is not None:
        if latest['Position'] == 1:
            st.markdown(f'<div class="status-box buy-box">LONG ACTIVE | Entry: {latest["Entry"]:.0f} | Trail: {latest["Trail"]:.0f}</div>', unsafe_allow_html=True)
        elif latest['Buy']:
            st.markdown('<div class="status-box buy-box">🚀 SIGNAL BUY!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-box hold-box">WAITING...</div>', unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Price", f"{latest['Close']:.0f}")
        m2.metric("ATR", f"{latest['ATR']:.1f}")
        m3.metric("RSI", f"{latest['RSI14']:.1f}")
        m4.metric("Trail", f"{latest['Trail']:.0f}")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close, name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df.Trail, line=dict(color='orange'), name="Trail"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df.RSI14, name="RSI"), row=2, col=1)
        fig.update_layout(height=600, template='plotly_dark', xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Menghitung indikator... butuh lebih banyak bar.")
else:
    st.error("Gagal memuat data dari Yahoo Finance.")
