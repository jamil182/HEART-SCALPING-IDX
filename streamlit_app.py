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
    rs = gain / loss
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
        # Menambahkan auto_adjust agar tidak terjadi masalah MultiIndex kolom
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
    
    # Reset index untuk menghindari konflik alignment
    df = df_input.copy()
    
    # Kalkulasi Indikator
    df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'], c)
    df['EMA21'] = calculate_ema(df['Close'], 21)
    df['EMA50'] = calculate_ema(df['Close'], 50)
    df['RSI14'] = calculate_rsi(df['Close'], 14)
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()

    # Fillna agar tidak ada lubang di tengah data
    df = df.ffill().dropna()

    # Konversi ke numpy untuk komparasi cepat dan menghindari error index alignment
    close_vals = df['Close'].values
    open_vals = df['Open'].values
    ema21_vals = df['EMA21'].values
    ema50_vals = df['EMA50'].values
    rsi_vals = df['RSI14'].values
    vol_vals = df['Volume'].values
    vma_vals = df['Vol_MA20'].values
    atr_vals = df['ATR'].values

    # Kondisi Filter
    trend_ok = (close_vals > ema50_vals) & (ema21_vals > ema50_vals)
    rsi_ok = rsi_vals > 48
    vol_ok = vol_vals > (vma_vals * 1.3)
    # Candle Bullish Engulfing / Reversal sederhana
    candle_ok = (close_vals > open_vals) & (np.roll(close_vals, 1) < np.roll(open_vals, 1))
    
    filter_ok = trend_ok & rsi_ok & vol_ok & candle_ok
    df['filter_ok'] = filter_ok

    # Trail Stop Logic (SuperTrend Style)
    nloss = a * atr_vals
    trail = np.zeros(len(df))
    trail[0] = close_vals[0] - nloss[0]

    for i in range(1, len(df)):
        prev_trail = trail[i-1]
        if close_vals[i] > prev_trail and close_vals[i-1] > prev_trail:
            trail[i] = max(prev_trail, close_vals[i] - nloss[i])
        elif close_vals[i] < prev_trail and close_vals[i-1] < prev_trail:
            trail[i] = min(prev_trail, close_vals[i] + nloss[i])
        else:
            trail[i] = (close_vals[i] - nloss[i]) if close_vals[i] > prev_trail else (close_vals[i] + nloss[i])

    df['Trail'] = trail

    # Signal Generation
    if use_confirmed:
        # Signal muncul jika candle SEBELUMNYA closing di atas trail dan filter ok
        df['Buy'] = (np.roll(close_vals, 1) > np.roll(trail, 1)) & (np.roll(close_vals, 2) < np.roll(trail, 2)) & np.roll(filter_ok, 1)
    else:
        df['Buy'] = (close_vals > trail) & (np.roll(close_vals, 1) < np.roll(trail, 1)) & filter_ok

    df['Sell'] = (close_vals < trail) & (np.roll(close_vals, 1) > np.roll(trail, 1))

    # Position Tracker
    pos = 0
    positions = []
    entry_price = np.nan
    entries = []

    for i in range(len(df)):
        if df['Buy'].iloc[i] and pos == 0:
            pos = 1
            entry_price = df['Close'].iloc[i]
        elif df['Sell'].iloc[i] and pos == 1:
            pos = 0
            entry_price = np.nan
        
        positions.append(pos)
        entries.append(entry_price)

    df['Position'] = positions
    df['Entry'] = entries
    
    return df, df.iloc[-1]

# --- MAIN APP ---
st.markdown('<div class="main-header">❤️ HEART SCALPING ^JKSE</div>', unsafe_allow_html=True)

df_raw = load_data()

if df_raw is None:
    st.error("Gagal memuat data ^JKSE. Coba refresh beberapa saat lagi.")
else:
    df, latest = run_heart_logic(df_raw, a, c, use_confirmed)
    
    if latest is None:
        st.warning("Data belum cukup untuk kalkulasi (minimal butuh 50 bar).")
    else:
        # Status Display
        if latest['Position'] == 1:
            st.markdown(f'<div class="status-box buy-box">LONG ACTIVE | Entry ≈ {latest["Entry"]:.0f} | Trail {latest["Trail"]:.0f}</div>', unsafe_allow_html=True)
        elif latest['Buy']:
            st.markdown('<div class="status-box buy-box">🚀 BUY SIGNAL DETECTED!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-box hold-box">WAIT • NO CLEAR SETUP</div>', unsafe_allow_html=True)

        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Close", f"{latest['Close']:.0f}")
        m2.metric("ATR", f"{latest['ATR']:.1f}")
        m3.metric("RSI", f"{latest['RSI14']:.1f}")
        m4.metric("EMA 50", f"{latest['EMA50']:.0f}")

        # Charting
        try:
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.05,
                                row_heights=[0.6, 0.2, 0.2])

            # Price Chart
            fig.add_trace(go.Candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close, name="Price"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df.Trail, name="Heart Trail", line=dict(color="orange", width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df.EMA50, name="EMA50", line=dict(color="cyan", dash='dash')), row=1, col=1)

            # RSI Chart
            fig.add_trace(go.Scatter(x=df.index, y=df.RSI14, name="RSI", line=dict(color="magenta")), row=2, col=1)
            fig.add_hline(y=70, line_dash="dot", row=2, col=1, line_color="red")
            fig.add_hline(y=30, line_dash="dot", row=2, col=1, line_color="green")

            # Volume Chart
            colors = ['green' if df.Close.iloc[i] > df.Open.iloc[i] else 'red' for i in range(len(df))]
            fig.add_trace(go.Bar(x=df.index, y=df.Volume, name="Volume", marker_color=colors), row=3, col=1)

            fig.update_layout(height=800, template='plotly_dark', showlegend=False,
                            xaxis_rangeslider_visible=False,
                            title=f"^JKSE {timeframe} - Last Updated: {datetime.now().strftime('%H:%M:%S')}")
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering chart: {e}")

st.caption("HEART Scalping IDX • High Frequency Trading Educational Dashboard")
