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

# --- DATA LOADING (The Core Fix) ---
@st.cache_data(ttl=60)
def load_data():
    try:
        # Download dengan auto_adjust=True untuk menghindari kolom bertumpuk
        raw = yf.download("^JKSE", period="8d", interval=timeframe, progress=False, auto_adjust=True)
        
        if raw.empty:
            return None
        
        df = raw.copy()
        
        # 1. Hapus MultiIndex Kolom (Penyebab utama ValueError di screenshot)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # 2. Pastikan kolom dasar tersedia
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        # 3. Reset index agar perhitungan alignment berbasis integer (lebih aman)
        df = df.reset_index()
        return df
    except Exception as e:
        st.error(f"Data Fetch Error: {e}")
        return None

# --- LOGIC ---
def run_heart_logic(df_raw, a, c, use_confirmed):
    if df_raw is None or len(df_raw) < 50:
        return None, None
    
    df = df_raw.copy()
    
    # Kalkulasi Indikator ke Kolom Baru
    df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'], c)
    df['EMA21'] = calculate_ema(df['Close'], 21)
    df['EMA50'] = calculate_ema(df['Close'], 50)
    df['RSI14'] = calculate_rsi(df['Close'], 14)
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()
    
    # Hapus bar awal yang kosong akibat rolling window
    df = df.dropna().reset_index(drop=True)

    # --- FIX CRITICAL: Gunakan Array Numpy (.values) untuk komparasi ---
    # Ini menjamin tidak ada Index Alignment Error
    c_arr = df['Close'].values
    o_arr = df['Open'].values
    e21   = df['EMA21'].values
    e50   = df['EMA50'].values
    rsi   = df['RSI14'].values
    vol   = df['Volume'].values
    vma   = df['Vol_MA20'].values
    atr   = df['ATR'].values

    # Hitung Filter menggunakan logika Numpy
    trend_ok  = (c_arr > e50) & (e21 > e50)
    rsi_ok    = rsi > 48
    vol_ok    = vol > (vma * 1.2)
    candle_ok = (c_arr > o_arr) & (np.roll(c_arr, 1) < np.roll(o_arr, 1))
    
    # Simpan kembali ke DataFrame
    df['filter_ok'] = trend_ok & rsi_ok & vol_ok & candle_ok

    # Trail Stop Logic (Looping)
    nloss = a * atr
    trail = np.zeros(len(df))
    trail[0] = c_arr[0] - nloss[0]

    for i in range(1, len(df)):
        prev_t = trail[i-1]
        if c_arr[i] > prev_t and c_arr[i-1] > prev_t:
            trail[i] = max(prev_t, c_arr[i] - nloss[i])
        elif c_arr[i] < prev_t and c_arr[i-1] < prev_t:
            trail[i] = min(prev_t, c_arr[i] + nloss[i])
        else:
            trail[i] = (c_arr[i] - nloss[i]) if c_arr[i] > prev_t else (c_arr[i] + nloss[i])

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

    # Posisi Saat Ini
    pos = 0
    entries = []
    last_entry = np.nan
    for i in range(len(df)):
        if df['Buy'].iloc[i] and pos == 0:
            pos = 1
            last_entry = df['Close'].iloc[i]
        elif df['Sell'].iloc[i] and pos == 1:
            pos = 0
            last_entry = np.nan
        entries.append(last_entry)
    
    df['Entry'] = entries
    df['HasPos'] = pos # status terakhir
    
    return df, df.iloc[-1]

# --- UI ---
st.markdown('<div class="main-header">❤️ HEART SCALPING ^JKSE</div>', unsafe_allow_html=True)

df_raw = load_data()

if df_raw is not None:
    df, latest = run_heart_logic(df_raw, a, c, use_confirmed)
    
    if latest is not None:
        # Display Box
        if not np.isnan(latest['Entry']):
            st.markdown(f'<div class="status-box buy-box">LONG ACTIVE | Entry: {latest["Entry"]:.0f} | Trail: {latest["Trail"]:.0f}</div>', unsafe_allow_html=True)
        elif latest['Buy']:
            st.markdown('<div class="status-box buy-box">🚀 SIGNAL BUY!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-box hold-box">WAIT • NO CLEAR SIGNAL</div>', unsafe_allow_html=True)

        # Metric Columns
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Close", f"{latest['Close']:.0f}")
        c2.metric("ATR", f"{latest['ATR']:.1f}")
        c3.metric("RSI", f"{latest['RSI14']:.1f}")
        c4.metric("Trail", f"{latest['Trail']:.0f}")

        # Charting
        try:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
            # Pastikan sumbu X menggunakan waktu asli
            x_time = df['Datetime'] if 'Datetime' in df.columns else (df['Date'] if 'Date' in df.columns else df.index)
            
            fig.add_trace(go.Candlestick(x=x_time, open=df.Open, high=df.High, low=df.Low, close=df.Close, name="JKSE"), row=1, col=1)
            fig.add_trace(go.Scatter(x=x_time, y=df.Trail, line=dict(color='orange', width=2), name="Trail Stop"), row=1, col=1)
            fig.add_trace(go.Scatter(x=x_time, y=df.RSI14, line=dict(color='cyan'), name="RSI"), row=2, col=1)
            
            fig.update_layout(height=650, template='plotly_dark', xaxis_rangeslider_visible=False, margin=dict(t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.info("Chart sedang diproses...")
    else:
        st.warning("Data belum mencukupi untuk menghitung indikator.")
else:
    st.error("Gagal menarik data dari Yahoo Finance.")

st.caption(f"Update: {datetime.now().strftime('%H:%M:%S')} WIB")
