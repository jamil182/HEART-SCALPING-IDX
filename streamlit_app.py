# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="HEART SCALPING IDX ^JKSE",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)
import time
if "last_ping" not in st.session_state:
    st.session_state.last_ping = time.time()
# ────────────────────────────────────────────────
#   Custom CSS
# ────────────────────────────────────────────────
st.markdown("""
    <style>
    .main-header {font-size:2.4rem; color:#00d4ff; font-weight:bold; text-align:center; margin-bottom:1rem;}
    .status-box {padding:1.2rem; border-radius:10px; color:white; font-weight:bold; text-align:center; margin:1rem 0;}
    .buy-box    {background:linear-gradient(135deg, #11998e, #38ef7d);}
    .hold-box   {background:linear-gradient(135deg, #feca57, #ff9ff3);}
    .metric     {background:#1e1e2f; padding:1rem; border-radius:8px; margin:0.5rem 0;}
    </style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────
#   Pure pandas indicator functions
# ────────────────────────────────────────────────
def calculate_atr(high, low, close, length):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=length).mean()

def calculate_rsi(close, length):
    delta = close.diff()
    gain  = delta.where(delta > 0, 0).rolling(window=length).mean()
    loss  = -delta.where(delta < 0, 0).rolling(window=length).mean()
    rs    = gain / loss
    rsi   = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

# ────────────────────────────────────────────────
#   Sidebar
# ────────────────────────────────────────────────
st.sidebar.header("HEART SCALPING IDX")

style_options = [
    "Super Aggressive (a=0.5, c=5)",
    "Stealing Profit (a=1.0, c=10)",
    "Fast Scalping (a=0.8, c=8)",
    "Relaxing Swing (a=2.0, c=15)"
]
selected_style = st.sidebar.selectbox("Trading Style", style_options, index=0)

if   selected_style.startswith("Super"):     a, c = 0.5, 5
elif selected_style.startswith("Stealing"):  a, c = 1.0, 10
elif selected_style.startswith("Fast"):      a, c = 0.8, 8
else:                                        a, c = 2.0, 15

use_confirmed_bar = st.sidebar.checkbox("Use confirmed bar (less repaint)", True)
timeframe = st.sidebar.selectbox("Timeframe", ["5m", "15m", "30m", "1h"], index=0)

auto_refresh = st.sidebar.checkbox("Auto-refresh every 60 seconds", True)
refresh_now  = st.sidebar.button("🔄 Refresh Now")

# ────────────────────────────────────────────────
#   Data loader
# ────────────────────────────────────────────────
@st.cache_data(ttl=60 if auto_refresh else 300)
def load_data():
    try:
        df = yf.download("^JKSE", period="10d", interval=timeframe, progress=False)
        if df.empty:
            return None
        df = df[['Open','High','Low','Close','Volume']].copy()
        return df
    except:
        return None

# ────────────────────────────────────────────────
#   HEART core logic
# ────────────────────────────────────────────────
def run_heart_logic(df, a, c, use_confirmed):
    if df is None or len(df) < 30:
        return None, None

    df = df.copy()

    # Indicators
    df['ATR']      = calculate_atr(df['High'], df['Low'], df['Close'], c)
    df['EMA21']    = calculate_ema(df['Close'], 21)
    df['EMA50']    = calculate_ema(df['Close'], 50)
    df['RSI14']    = calculate_rsi(df['Close'], 14)
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()

    # Basic filters (long only)
    df['trend_ok']   = (df['Close'] > df['EMA50']) & (df['EMA21'] > df['EMA50'])
    df['rsi_ok']     = df['RSI14'] > 48
    df['vol_ok']     = df['Volume'] > df['Vol_MA20'] * 1.3
    df['candle_ok']  = (df['Close'] > df['Open']) & (df['Close'].shift(1) < df['Open'].shift(1))

    df['filter_ok'] = df['trend_ok'] & df['rsi_ok'] & df['vol_ok'] & df['candle_ok']

    # ─── ATR Trailing Stop (HEART style) ────────────────────────────────
    src = df['Close']
    nloss = a * df['ATR']

    trail = np.zeros(len(df))
    trail[0] = src.iloc[0] - nloss.iloc[0]

    for i in range(1, len(df)):
        prev_trail = trail[i-1]
        curr_src   = src.iloc[i]
        prev_src   = src.iloc[i-1]

        if curr_src > prev_trail:
            trail[i] = max(prev_trail, curr_src - nloss.iloc[i])
        elif prev_src < prev_trail:
            trail[i] = min(prev_trail, curr_src + nloss.iloc[i])
        else:
            trail[i] = curr_src - nloss.iloc[i] if curr_src > prev_trail else curr_src + nloss.iloc[i]

    df['Trail'] = trail

    # ─── Signals ────────────────────────────────────────────────────────
    cross_up_raw   = (src.shift(1) < df['Trail'].shift(1)) & (src > df['Trail'])
    cross_down_raw = (src.shift(1) > df['Trail'].shift(1)) & (src < df['Trail'])

    if use_confirmed:
        # signal on current bar if previous bar crossed
        df['Buy']  = (df['Close'].shift(2) < df['Trail'].shift(2)) & \
                     (df['Close'].shift(1) > df['Trail'].shift(2)) & \
                     df['filter_ok'].shift(1)
    else:
        df['Buy']  = cross_up_raw & df['filter_ok']

    df['Sell'] = cross_down_raw

    # Simple position tracking
    pos = 0
    positions = []
    entries   = []

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
    df['Entry']    = entries

    return df, df.iloc[-1] if not df.empty else None

# ────────────────────────────────────────────────
#   Main page
# ────────────────────────────────────────────────
st.markdown('<div class="main-header">❤️ HEART SCALPING ^JKSE</div>', unsafe_allow_html=True)

df_raw = load_data()

if df_raw is None:
    st.error("Cannot load ^JKSE data right now. Market closed or connection issue.")
else:
    df, latest = run_heart_logic(df_raw, a, c, use_confirmed_bar)

    if latest is None:
        st.warning("Not enough data to calculate signals yet.")
    else:
        # ─── Status banner ─────────────────────────────────────────────
        if latest['Position'] == 1:
            txt = f"LONG ACTIVE  |  Entry ≈ {latest['Entry']:.0f}  |  Trail {latest['Trail']:.0f}"
            st.markdown(f'<div class="status-box buy-box">{txt}</div>', unsafe_allow_html=True)
        elif latest['Buy']:
            st.markdown('<div class="status-box buy-box">BUY SIGNAL !</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-box hold-box">WAIT • NO CLEAR SETUP</div>', unsafe_allow_html=True)

        # ─── Quick stats row ───────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Last Close", f"{latest['Close']:.0f}")
        col2.metric("ATR",        f"{latest['ATR']:.0f}")
        col3.metric("RSI 14",     f"{latest['RSI14']:.1f}")
        col4.metric("Trail Stop", f"{latest['Trail']:.0f}")

        # ─── Chart ─────────────────────────────────────────────────────
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            vertical_spacing=0.06,
                            row_heights=[0.60, 0.25, 0.15],
                            subplot_titles=("Price & HEART Trail", "RSI", "Volume"))

        # Candles
        fig.add_trace(go.Candlestick(
            x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close,
            name="^JKSE", increasing_line_color='#22c55e', decreasing_line_color='#ef4444'
        ), row=1, col=1)

        # Trail
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Trail'], line=dict(color='#f59e0b', width=2.2),
            name="HEART Trail", fill='tonexty', fillcolor='rgba(245,158,11,0.12)'
        ), row=1, col=1)

        # Buy markers
        buys = df[df['Buy']]
        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=buys.index, y=buys.Low * 0.997,
                mode='markers+text', text=['BUY']*len(buys),
                marker=dict(symbol='triangle-up', size=14, color='#22c55e'),
                textposition='bottom center', name="Buy"
            ), row=1, col=1)

        # EMA
        fig.add_trace(go.Scatter(x=df.index, y=df.EMA21, line=dict(color='#3b82f6'), name="EMA 21"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df.EMA50, line=dict(color='#8b5cf6'), name="EMA 50"), row=1, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=df.index, y=df.RSI14, line=dict(color='#a78bfa'), name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="#ef4444", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#22c55e", row=2, col=1)

        # Volume
        fig.add_trace(go.Bar(x=df.index, y=df.Volume, marker_color='rgba(100,116,255,0.4)', name="Vol"), row=3, col=1)

        fig.update_layout(
            height=780, showlegend=True, template='plotly_dark',
            title=f"^JKSE {timeframe} • {datetime.now().strftime('%Y-%m-%d %H:%M WIB')}",
            xaxis_rangeslider_visible=False,
            margin=dict(l=40,r=40,t=80,b=40)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Recent signals
        with st.expander("Last 12 signals / crosses"):
            sig = df[df['Buy'] | df['Sell']][['Close','RSI14','ATR','Trail','Buy','Sell']].tail(12)
            st.dataframe(sig.round(1))

# ─── Footer & auto-refresh ────────────────────────────────────────
st.markdown("---")
st.caption("HEART trailing logic • long-only bias • no guarantee • for education only")

if auto_refresh and refresh_now is False:
    time.sleep(60)
    st.rerun()
