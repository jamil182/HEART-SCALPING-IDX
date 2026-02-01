import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi Halaman
st.set_page_config(
    page_title="🔥 HEART SCALPING IDX (^JKSE)", 
    page_icon="💓", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Custom untuk IDX Theme
st.markdown("""
    <style>
    .main-header {font-size: 2.5rem; color: #00d4ff; font-weight: bold; text-shadow: 0 0 10px #00d4ff;}
    .metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px;}
    .status-buy {background: linear-gradient(135deg, #11998e, #38ef7d); color: white; padding: 0.5rem; border-radius: 8px;}
    .status-sell {background: linear-gradient(135deg, #ff6b6b, #ee5a52); color: white; padding: 0.5rem; border-radius: 8px;}
    .status-hold {background: linear-gradient(135deg, #feca57, #ff9ff3); color: white; padding: 0.5rem; border-radius: 8px;}
    </style>
""", unsafe_allow_html=True)

# ==============================================
# PART 1: SETTINGS (Sesuai Pine Script)
# ==============================================
@st.cache_data(ttl=300)  # Cache 5 menit
def load_idx_data(period="5d", interval="5m"):
    """Load data IHSG real-time menggunakan yfinance"""
    ticker = "^JKSE"
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty:
            st.error("❌ Data IHSG tidak tersedia. Pastikan bursa buka (09:00-15:00 WIB)!")
            return None
        return data
    except:
        st.error("❌ Gagal load data IDX. Coba refresh!")
        return None

# Trading Style Presets (Sesuai Pine v5)
TRADING_STYLES = {
    'Stealing Profit': {'a': 1.0, 'c': 10, 'rsi_len': 14, 'use_all_filters': False},
    'Fast Scalping': {'a': 0.8, 'c': 8, 'rsi_len': 7, 'use_all_filters': True},
    'Relaxing Swing': {'a': 2.0, 'c': 15, 'rsi_len': 21, 'use_all_filters': True},
    'Very Accurate': {'a': 2.0, 'c': 15, 'rsi_len': 21, 'use_all_filters': True},
    'Super Agresif IDX': {'a': 0.5, 'c': 5, 'rsi_len': 7, 'use_all_filters': True}  # Custom untuk IHSG
}

# Sidebar Settings
st.sidebar.header("⚙️ HEART SCALPING SETTINGS")
trading_style = st.sidebar.selectbox("Pilih Style", list(TRADING_STYLES.keys()), index=4)  # Default Super Agresif
use_confirmed = st.sidebar.checkbox("✅ Not Repaint (Wait Close)", value=True)
timeframe = st.sidebar.selectbox("Interval", ["1m", "5m", "15m"], index=1)

# Filter Settings
st.sidebar.subheader("🔍 Filters")
use_ema = st.sidebar.checkbox("EMA Trend", value=False)
use_rsi = st.sidebar.checkbox("RSI Momentum", value=True)
use_volume = st.sidebar.checkbox("Volume Spike", value=True)
use_adx = st.sidebar.checkbox("ADX Strength", value=True)

# Risk Management
st.sidebar.subheader("🎯 Risk Management")
atr_mult = st.sidebar.slider("ATR Multiplier (a)", 0.3, 3.0, TRADING_STYLES[trading_style]['a'], 0.1)
atr_period = st.sidebar.slider("ATR Period (c)", 5, 20, TRADING_STYLES[trading_style]['c'])
breakeven_r = st.sidebar.slider("Breakeven (R)", 0.5, 2.0, 1.0, 0.1)

# Auto Refresh
st.sidebar.subheader("🔄 Auto Refresh")
auto_refresh = st.sidebar.checkbox("Auto Update (30s)", value=True)
manual_refresh = st.sidebar.button("🔄 REFRESH NOW")

# ==============================================
# PART 2: CORE HEART LOGIC (Python Implementation)
# ==============================================
def heart_scalping_signals(df, style_params):
    """Implementasi lengkap HEART Trailing Stop + Filters"""
    
    # Parameters
    a = style_params['a']
    c = style_params['c']
    rsi_len = style_params['rsi_len']
    
    # Technical Indicators
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=c)
    df['EMA_fast'] = ta.ema(df['Close'], length=21)
    df['EMA_slow'] = ta.ema(df['Close'], length=50)
    df['RSI'] = ta.rsi(df['Close'], length=rsi_len)
    df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']
    df['DI+'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['DMP_14']
    df['DI-'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['DMN_14']
    df['Vol_MA'] = ta.sma(df['Volume'], length=20)
    
    # Filters (Sesuai Pine Script)
    df['EMA_OK'] = (df['Close'] > df['EMA_slow']) & (df['EMA_fast'] > df['EMA_slow'])
    df['RSI_OK'] = df['RSI'] > 50  # Long-only
    df['Vol_OK'] = df['Volume'] > df['Vol_MA'] * 1.5
    df['ADX_OK'] = (df['ADX'] > 20) & (df['DI+'] > df['DI-'])
    
    # Candlestick Patterns (Simplified)
    df['Bullish'] = (df['Close'] > df['Open']) & (df['Close'].shift(1) < df['Open'].shift(1))
    df['Candle_OK'] = df['Bullish']
    
    # Combined Filter (Long-only)
    df['FILTER_OK'] = (
        (use_ema == False) | df['EMA_OK'] &
        (use_rsi == False) | df['RSI_OK'] &
        (use_volume == False) | df['Vol_OK'] &
        (use_adx == False) | df['ADX_OK'] &
        df['Candle_OK']
    )
    
    # CORE: HEART ATR Trailing Stop
    src = df['Close'].copy()
    xatr = df['ATR'].copy()
    nloss = a * xatr
    
    # Initialize trailing stop
    xATRTrailingStop = pd.Series(0.0, index=df.index)
    
    for i in range(1, len(df)):
        prev_stop = xATRTrailingStop.iloc[i-1]
        curr_src = src.iloc[i]
        prev_src = src.iloc[i-1]
        
        if curr_src > prev_stop:
            xATRTrailingStop.iloc[i] = max(prev_stop, curr_src - nloss.iloc[i])
        elif curr_src < prev_stop and prev_src < prev_stop:
            xATRTrailingStop.iloc[i] = min(prev_stop, curr_src + nloss.iloc[i])
        else:
            xATRTrailingStop.iloc[i] = curr_src - nloss.iloc[i] if curr_src > prev_stop else curr_src + nloss.iloc[i]
    
    df['TrailStop'] = xATRTrailingStop
    
    # Signal Generation (Not Repaint)
    df['CrossUp_Raw'] = (src.shift(1) < df['TrailStop'].shift(1)) & (src > df['TrailStop'])
    df['CrossDown_Raw'] = (src.shift(1) > df['TrailStop'].shift(1)) & (src < df['TrailStop'])
    
    if use_confirmed:
        # Wait for close confirmation
        df['Buy_Signal'] = (df['Close'].shift(2) < df['TrailStop'].shift(2)) & \
                          (df['Close'].shift(1) > df['TrailStop'].shift(2)) & \
                          df['FILTER_OK'].shift(1)
    else:
        df['Buy_Signal'] = df['CrossUp_Raw'] & df['FILTER_OK']
    
    df['Sell_Signal'] = df['CrossDown_Raw']  # Exit signal
    
    # Position Management
    position = 0
    positions = []
    entry_price = []
    
    for i in range(len(df)):
        if df['Buy_Signal'].iloc[i]:
            position = 1
            entry_price.append(df['Close'].iloc[i])
        elif df['Sell_Signal'].iloc[i] and position == 1:
            position = 0
            entry_price.append(np.nan)
        else:
            entry_price.append(np.nan if position == 0 else entry_price[-1])
        positions.append(position)
    
    df['Position'] = positions
    df['Entry_Price'] = entry_price
    
    return df

# ==============================================
# PART 3: BACKTEST & STATISTICS
# ==============================================
def calculate_stats(df):
    """Hitung Winrate, Profit Factor, Max DD"""
    trades = []
    position = 0
    entry_price = 0
    
    for i in range(len(df)):
        if df['Buy_Signal'].iloc[i] and position == 0:
            position = 1
            entry_price = df['Close'].iloc[i]
        elif df['Sell_Signal'].iloc[i] and position == 1:
            exit_price = df['Close'].iloc[i]
            pnl_pct = (exit_price - entry_price) / entry_price * 100
            trades.append(pnl_pct)
            position = 0
    
    if not trades:
        return {"winrate": 0, "total_trades": 0, "profit_factor": 0, "avg_win": 0}
    
    trades = np.array(trades)
    wins = trades[trades > 0]
    losses = trades[trades < 0]
    
    winrate = len(wins) / len(trades) * 100
    avg_win = np.mean(wins) if len(wins) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0
    profit_factor = abs(np.sum(wins) / np.sum(losses)) if len(losses) > 0 else np.inf
    
    return {
        "winrate": winrate,
        "total_trades": len(trades),
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss
    }

# ==============================================
# PART 4: MAIN APP LOGIC
# ==============================================
st.markdown('<h1 class="main-header">💓 HEART SCALPING IDX COMPOSITE (^JKSE)</h1>', unsafe_allow_html=True)
st.caption("*Scalping IHSG Long-Only | Real-time | Winrate Target 80%+ | 100% Equity*")

# Load Data
if manual_refresh or auto_refresh:
    with st.spinner("📊 Loading data IHSG real-time..."):
        df = load_idx_data(period="10d", interval=timeframe)
    
    if df is not None and not df.empty:
        # Apply HEART Logic
        style_params = TRADING_STYLES[trading_style]
        df = heart_scalping_signals(df, style_params)
        
        # Update global state
        st.session_state['df'] = df
        st.session_state['stats'] = calculate_stats(df)
        st.session_state['latest'] = df.iloc[-1]
        
        st.success(f"✅ Data loaded: {len(df):,} bars | Style: **{trading_style}** | Interval: **{timeframe}**")
    else:
        st.session_state['df'] = pd.DataFrame()
        st.session_state['stats'] = {}

# Display Data
if 'df' in st.session_state and not st.session_state['df'].empty:
    df = st.session_state['df']
    latest = st.session_state['latest']
    stats = st.session_state['stats']
    
    # ==============================================
    # DASHBOARD PRO (Streamlit Version)
    # ==============================================
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💓 Status", "🟢 LONG" if latest['Position'] == 1 else "⚪ WAIT", 
                 f"{latest['Close']:,.0f}")
    
    with col2:
        color = "normal" if stats['winrate'] < 80 else "inverse"
        st.metric("🎯 Winrate", f"{stats['winrate']:.1f}%", delta=None, delta_color=color)
    
    with col3:
        st.metric("📈 ATR", f"{latest['ATR']:.0f}", f"{latest['ATR']/latest['Close']*100:.2f}%")
    
    with col4:
        st.metric("🔥 Signal", "BUY" if latest['Buy_Signal'] else "HOLD", 
                 "🚀" if latest['Buy_Signal'] else "⏳")
    
    # Status Advisor (Pine Script Logic)
    if latest['Position'] == 1:
        status_class = "status-buy"
        advice = "🚀 HOLD LONG! IHSG naik ke resistance berikutnya. Trail stop aktif."
    elif latest['Buy_Signal']:
        status_class = "status-buy"
        advice = "🟢 BUY SIGNAL! Masuk full position 100% equity. Target R:R 1:3."
    else:
        status_class = "status-hold"
        advice = "⏳ WAIT MARKET. Filter aktif, siapkan modal untuk breakout."
    
    st.markdown(f"""
    <div class="status-{status_class.split('-')[1]}">
        <b>💡 ADVISOR:</b> {advice} | SL: {latest['TrailStop']:,.0f} | R-Risk: {atr_mult:.1f}xATR
    </div>
    """, unsafe_allow_html=True)
    
    # Filter Status Table
    filter_status = pd.DataFrame({
        'Filter': ['EMA', 'RSI', 'Volume', 'ADX', 'Candle'],
        'Status': [
            '✅' if latest['EMA_OK'] else '❌',
            '✅' if latest['RSI_OK'] else '❌',
            '✅' if latest['Vol_OK'] else '❌',
            '✅' if latest['ADX_OK'] else '❌',
            '✅' if latest['Candle_OK'] else '❌'
        ],
        'Value': [
            f"RSI {latest['RSI']:.0f}", f"Vol x{latest['Volume']/latest['Vol_MA']:.1f}",
            f"ADX {latest['ADX']:.0f}", f"Pos: {latest['Position']}"
        ]
    })
    
    st.subheader("🔍 Filter Status")
    st.dataframe(filter_status, use_container_width=True)
    
    # Plotly Chart (Candles + Signals + TrailStop)
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('💓 IHSG ^JKSE | HEART Trailing', 'RSI + ADX', 'Volume'),
        row_heights=[0.6, 0.25, 0.15]
    )
    
    # Candlestick + Signals
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name="IHSG", increasing_line_color='#00ff88', decreasing_line_color='#ff4444'
        ), row=1, col=1
    )
    
    # Buy/Sell Signals
    buy_signals = df[df['Buy_Signal'] == True]
    sell_signals = df[df['Sell_Signal'] == True]
    
    fig.add_trace(
        go.Scatter(x=buy_signals.index, y=buy_signals['Low']*0.999, mode='markers+text',
                  marker=dict(color='#00ff00', size=12, symbol='triangle-up'),
                  text=['BUY↑']*len(buy_signals), textposition='bottom center',
                  name="🚀 BUY"), row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=sell_signals.index, y=sell_signals['High']*1.001, mode='markers+text',
                  marker=dict(color='#ff4444', size=12, symbol='triangle-down'),
                  text=['SELL↓']*len(sell_signals), textposition='top center',
                  name="📉 EXIT"), row=1, col=1
    )
    
    # Trailing Stop
    fig.add_trace(
        go.Scatter(x=df.index, y=df['TrailStop'], line=dict(color='#ffaa00', width=2),
                  name="Trail Stop", fill='tonexty', fillcolor='rgba(255,170,0,0.2)'),
        row=1, col=1
    )
    
    # EMAs
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_fast'], line=dict(color='#00aaff', width=1), name="EMA 21"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_slow'], line=dict(color='#ffaa00', width=1), name="EMA 50"), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#7d5fff'), name="RSI"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # ADX
    ax2 = fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], line=dict(color='#ff6b6b'), name="ADX"), row=2, col=1)
    
    # Volume
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color='rgba(100,100,255,0.3)', name="Volume"), row=3, col=1)
    
    fig.update_layout(
        height=800, showlegend=True, xaxis_rangeslider_visible=False,
        title=f"📊 HEART SCALPING IHSG | {datetime.now().strftime('%d/%m %H:%M WIB')}",
        template='plotly_dark'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Backtest Results
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎲 Total Trades", stats['total_trades'])
    with col2:
        st.metric("📊 Winrate", f"{stats['winrate']:.1f}%", 
                 delta=f"{stats['winrate']-75:+.1f}%" if stats['winrate'] > 0 else None)
    with col3:
        st.metric("💰 Profit Factor", f"{stats['profit_factor']:.2f}")
    with col4:
        st.metric("🏆 Avg Win", f"{stats['avg_win']:.2f}%")
    
    # Recent Signals Table
    recent_signals = df[df['Buy_Signal'] | df['Sell_Signal']].tail(10)[
        ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'ATR', 'Buy_Signal', 'Sell_Signal', 'TrailStop']
    ].round(2)
    
    st.subheader("📋 Recent Signals (Last 10)")
    st.dataframe(recent_signals, use_container_width=True)
    
    # Download CSV
    csv = df.to_csv().encode('utf-8')
    st.download_button(
        "💾 Download Full Data CSV", csv, f"heart_scalping_idx_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        "text/csv", key='download-csv'
    )
    
    # Auto Refresh Logic
    if auto_refresh:
        time.sleep(30)
        st.rerun()

else:
    st.info("⏳ Tunggu bursa buka (09:00-15:00 WIB) atau klik REFRESH. Data IHSG hanya tersedia saat market hours!")
    st.balloons()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    🔗 <a href='https://jamilstempel.com' target='_blank'>JAMIL STEMPEL</a> | 
    📱 WA: <a href='https://wa.me/6281255466019'>0812-5546-6019</a> | 
    💻 <a href='https://idx-screener-all.streamlit.app'>IDX Screener Pro</a>
</div>
""", unsafe_allow_html=True)
