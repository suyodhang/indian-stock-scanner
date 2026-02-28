"""
Streamlit Dashboard for Stock Scanner
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import config, scanner_config
from config.stock_universe import NIFTY_50, SECTORS
from data_collection.nse_fetcher import YahooFinanceFetcher
from analysis.technical_indicators import TechnicalIndicators
from analysis.pattern_recognition import CandlestickPatterns, ChartPatterns
from scanners.momentum_scanner import MomentumScanner
from scanners.breakout_scanner import BreakoutScanner
from scanners.custom_scanner import (
    CustomScanner, get_bullish_scanner_conditions,
    get_bearish_scanner_conditions, get_swing_trade_conditions
)

# Page config
st.set_page_config(
    page_title="AI Indian Stock Scanner",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
    html, body, [class*="css"], .stApp {
        font-family: 'Space Grotesk', sans-serif;
    }
    .signal-card {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid;
    }
    .bullish { border-left-color: #00ff00; background-color: #0a2a0a; }
    .bearish { border-left-color: #ff0000; background-color: #2a0a0a; }
    .metric-card {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background-color: #00ff00;
    }
    h1, h2, h3 {
        letter-spacing: 0.2px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)
def fetch_stock_data(symbols, period="1y"):
    """Fetch and cache stock data"""
    fetcher = YahooFinanceFetcher()
    data = fetcher.get_bulk_historical_data(symbols, period=period)
    
    # Calculate indicators
    ti = TechnicalIndicators()
    for symbol, df in data.items():
        data[symbol] = ti.calculate_all(df)
    
    return data


def get_live_index_snapshot(ticker: str) -> tuple[float, float] | tuple[None, None]:
    """Get best-effort live index value and change% with robust fallbacks."""
    try:
        tk = yf.Ticker(ticker)
        fi = getattr(tk, "fast_info", {}) or {}
        last = fi.get("last_price")
        prev = fi.get("previous_close")
        if last and prev:
            return float(last), ((float(last) - float(prev)) / float(prev)) * 100
    except Exception:
        pass

    # Fallback to recent intraday bars
    try:
        df = yf.download(ticker, period="2d", interval="5m", progress=False, auto_adjust=False)
        if not df.empty:
            closes = df["Close"].dropna()
            if len(closes) >= 2:
                current = float(closes.iloc[-1])
                prev = float(closes.iloc[-2])
                return current, ((current - prev) / prev) * 100
    except Exception:
        pass

    # Final fallback to daily close
    try:
        df = yf.download(ticker, period="5d", interval="1d", progress=False, auto_adjust=False)
        if not df.empty:
            closes = df["Close"].dropna()
            if len(closes) >= 2:
                current = float(closes.iloc[-1])
                prev = float(closes.iloc[-2])
                return current, ((current - prev) / prev) * 100
    except Exception:
        pass

    return None, None


def get_chart_links(symbol: str) -> Dict[str, str]:
    """Build external chart links for quick drill-down."""
    sym = symbol.upper().strip()
    tv_symbol = sym.replace("&", "_").replace("-", "_")
    return {
        "TradingView": f"https://www.tradingview.com/chart/?symbol=NSE:{tv_symbol}",
        "Yahoo": f"https://finance.yahoo.com/chart/{sym}.NS",
    }


def plot_candlestick_chart(df, symbol, show_indicators=True):
    """Create interactive candlestick chart"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{symbol} Price', 'Volume', 'RSI'),
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df['date'] if 'date' in df.columns else df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name="Price"
    ), row=1, col=1)
    
    if show_indicators:
        # Moving Averages
        for ma, color in [('SMA_20', '#FFD700'), ('SMA_50', '#00BFFF'), ('SMA_200', '#FF6347')]:
            if ma in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['date'] if 'date' in df.columns else df.index,
                    y=df[ma],
                    name=ma,
                    line=dict(color=color, width=1)
                ), row=1, col=1)
        
        # Bollinger Bands
        if 'BB_Upper' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['date'] if 'date' in df.columns else df.index,
                y=df['BB_Upper'], name='BB Upper',
                line=dict(color='gray', dash='dash', width=0.5)
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df['date'] if 'date' in df.columns else df.index,
                y=df['BB_Lower'], name='BB Lower',
                line=dict(color='gray', dash='dash', width=0.5),
                fill='tonexty', fillcolor='rgba(128,128,128,0.1)'
            ), row=1, col=1)
        
        # SuperTrend
        if 'SuperTrend' in df.columns:
            st_bullish = df[df['ST_Direction'] == 1]
            st_bearish = df[df['ST_Direction'] == -1]
            
            fig.add_trace(go.Scatter(
                x=st_bullish['date'] if 'date' in st_bullish.columns else st_bullish.index,
                y=st_bullish['SuperTrend'], name='SuperTrend (Buy)',
                line=dict(color='lime', width=1.5)
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=st_bearish['date'] if 'date' in st_bearish.columns else st_bearish.index,
                y=st_bearish['SuperTrend'], name='SuperTrend (Sell)',
                line=dict(color='red', width=1.5)
            ), row=1, col=1)
    
    # Volume
    colors = ['red' if df['close'].iloc[i] < df['open'].iloc[i] else 'green' 
              for i in range(len(df))]
    fig.add_trace(go.Bar(
        x=df['date'] if 'date' in df.columns else df.index,
        y=df['volume'],
        marker_color=colors,
        name="Volume",
        showlegend=False
    ), row=2, col=1)
    
    # RSI
    if 'RSI_14' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['date'] if 'date' in df.columns else df.index,
            y=df['RSI_14'],
            name="RSI(14)",
            line=dict(color='purple')
        ), row=3, col=1)
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    fig.update_layout(
        height=800,
        title=f"{symbol} - Technical Analysis",
        xaxis_rangeslider_visible=False,
        template="plotly_dark"
    )
    
    return fig


def main():
    """Main dashboard application"""
    
    # Sidebar
    st.sidebar.title("AI Stock Scanner")
    st.sidebar.markdown("---")
    
    # Page selection
    page = st.sidebar.selectbox(
        "Select Page",
        ["Dashboard", "Scanner", "Stock Analysis", "AI Predictions", "Settings"]
    )
    
    if page == "Dashboard":
        show_dashboard()
    elif page == "Scanner":
        show_scanner()
    elif page == "Stock Analysis":
        show_stock_analysis()
    elif page == "AI Predictions":
        show_ai_predictions()
    elif page == "Settings":
        show_settings()


def show_dashboard():
    """Main dashboard with market overview"""
    st.title("Market Dashboard")
    
    # Market indices
    col1, col2, col3, col4 = st.columns(4)
    
    with st.spinner("Fetching market data..."):
        indices = {
            "NIFTY 50": "^NSEI",
            "SENSEX": "^BSESN",
            "BANK NIFTY": "^NSEBANK",
            "NIFTY IT": "^CNXIT"
        }
        
        for (name, ticker), col in zip(indices.items(), [col1, col2, col3, col4]):
            try:
                current, change = get_live_index_snapshot(ticker)
                if current is not None and change is not None:
                    col.metric(name, f"{current:,.0f}", f"{change:+.2f}%")
            except:
                col.metric(name, "N/A", "")
    
    st.markdown("---")
    
    # Quick scan results
    st.subheader("Today's Top Signals")
    
    with st.spinner("Running quick scan..."):
        stock_data = fetch_stock_data(NIFTY_50[:20], period="6mo")
        
        momentum_scanner = MomentumScanner()
        breakout_scanner = BreakoutScanner()
        
        momentum_results = momentum_scanner.run_all_scans(stock_data)
        breakout_results = breakout_scanner.run_all_scans(stock_data)
        
        all_results = sorted(
            momentum_results + breakout_results,
            key=lambda x: x.strength,
            reverse=True
        )
    
    if all_results:
        # Display top signals
        for result in all_results[:10]:
            signal_type = "bullish" if "BUY" in result.signal or "BULLISH" in result.signal or "BREAKOUT" in result.signal else "bearish"
            
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 2, 1, 3])
                
                with col1:
                    st.markdown(f"**{result.symbol}**")
                    st.caption(result.signal.replace('_', ' '))
                    links = get_chart_links(result.symbol)
                    st.markdown(
                        f"[Chart]({links['TradingView']}) | [Yahoo]({links['Yahoo']})",
                        unsafe_allow_html=False,
                    )
                
                with col2:
                    change_color = "green" if result.change_pct > 0 else "red"
                    st.markdown(f"Rs {result.price:.2f}")
                    st.markdown(f"<span style='color:{change_color}'>{result.change_pct:+.2f}%</span>", unsafe_allow_html=True)
                
                with col3:
                    st.progress(result.strength)
                    st.caption(f"{result.strength:.0%}")
                
                with col4:
                    st.caption(" | ".join(result.reasons[:2]))
            
            st.markdown("---")
    else:
        st.info("No significant signals detected at the moment.")
    
    # Sector heatmap
    st.subheader("Sector Performance")
    show_sector_heatmap(stock_data)


def show_scanner():
    """Scanner page with multiple preset and custom scans"""
    st.title("Stock Scanner")
    
    # Scanner type selection
    scanner_type = st.selectbox(
        "Select Scanner",
        [
            "Momentum Scanner",
            "Breakout Scanner",
            "Bullish Setup Scanner",
            "Bearish Setup Scanner",
            "Swing Trade Scanner",
            "AI-Enhanced Scanner",
            "Custom Scanner"
        ]
    )
    
    # Universe selection
    universe = st.sidebar.multiselect(
        "Stock Universe",
        ["NIFTY 50", "NIFTY NEXT 50", "Custom"],
        default=["NIFTY 50"]
    )
    
    symbols = NIFTY_50
    if "Custom" in universe:
        custom_symbols = st.sidebar.text_area(
            "Enter symbols (comma separated)",
            "RELIANCE, TCS, INFY"
        )
        symbols = [s.strip().upper() for s in custom_symbols.split(",")]
    
    period = st.sidebar.selectbox("Data Period", ["3mo", "6mo", "1y", "2y"], index=2)
    
    if st.button("Run Scanner", type="primary"):
        with st.spinner(f"Scanning {len(symbols)} stocks..."):
            stock_data = fetch_stock_data(symbols, period=period)
            
            if scanner_type == "Momentum Scanner":
                scanner = MomentumScanner()
                results = scanner.run_all_scans(stock_data)
            
            elif scanner_type == "Breakout Scanner":
                scanner = BreakoutScanner()
                results = scanner.run_all_scans(stock_data)
            
            elif scanner_type == "Bullish Setup Scanner":
                scanner = CustomScanner()
                conditions = get_bullish_scanner_conditions()
                results = scanner.multi_condition_scan(stock_data, conditions, min_conditions_met=4)
            
            elif scanner_type == "Bearish Setup Scanner":
                scanner = CustomScanner()
                conditions = get_bearish_scanner_conditions()
                results = scanner.multi_condition_scan(stock_data, conditions, min_conditions_met=3)
            
            elif scanner_type == "Swing Trade Scanner":
                scanner = CustomScanner()
                conditions = get_swing_trade_conditions()
                results = scanner.multi_condition_scan(stock_data, conditions, min_conditions_met=3)
            
            elif scanner_type == "AI-Enhanced Scanner":
                scanner = CustomScanner()
                results = scanner.ai_enhanced_scan(stock_data)
            
            else:
                results = []
            
            # Display results
            if results:
                st.success(f"Found {len(results)} signals!")
                
                # Results table
                result_data = []
                for r in results:
                    result_data.append({
                        'Symbol': r.symbol,
                        'Signal': r.signal.replace('_', ' '),
                        'Price': f"Rs {r.price:.2f}",
                        'Change %': f"{r.change_pct:+.2f}%",
                        'Volume': f"{r.volume_ratio:.1f}x",
                        'Strength': f"{r.strength:.0%}",
                        'Reasons': ' | '.join(r.reasons[:2])
                    })
                
                df_results = pd.DataFrame(result_data)
                st.dataframe(df_results, width="stretch")
                
                # Detailed view
                st.markdown("---")
                st.subheader("Detailed Signals")
                
                for result in results[:20]:
                    with st.expander(f"{result.symbol} - {result.signal.replace('_', ' ')} ({result.strength:.0%})"):
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.metric("Price", f"Rs {result.price:.2f}", f"{result.change_pct:+.2f}%")
                            st.metric("Volume Ratio", f"{result.volume_ratio:.1f}x")
                            st.progress(result.strength)
                        
                        with col2:
                            for reason in result.reasons:
                                st.write(f"- {reason}")
                        
                        # Show chart for this stock
                        if result.symbol in stock_data:
                            fig = plot_candlestick_chart(
                                stock_data[result.symbol].tail(100),
                                result.symbol
                            )
                            st.plotly_chart(fig, width="stretch")
            else:
                st.warning("No signals found matching the criteria.")


def show_stock_analysis():
    """Individual stock analysis page"""
    st.title("Stock Analysis")
    
    symbol = st.text_input("Enter Stock Symbol", "RELIANCE").upper()
    links = get_chart_links(symbol)
    st.markdown(f"[Open TradingView Chart]({links['TradingView']}) | [Open Yahoo Chart]({links['Yahoo']})")
    period = st.selectbox("Period", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
    
    if st.button("Analyze", type="primary"):
        with st.spinner(f"Analyzing {symbol}..."):
            fetcher = YahooFinanceFetcher()
            df = fetcher.get_historical_data(symbol, period=period)
            
            if df.empty:
                st.error(f"No data found for {symbol}")
                return
            
            # Calculate indicators
            ti = TechnicalIndicators()
            df = ti.calculate_all(df)
            
            # Price chart
            fig = plot_candlestick_chart(df.tail(200), symbol)
            st.plotly_chart(fig, width="stretch")
            
            # Key metrics
            st.subheader("Key Metrics")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            col1.metric("Current Price", f"Rs {df['close'].iloc[-1]:.2f}")
            col2.metric("RSI (14)", f"{df['RSI_14'].iloc[-1]:.1f}")
            col3.metric("ADX", f"{df['ADX'].iloc[-1]:.1f}")
            
            st_dir = "Bullish Up" if df['ST_Direction'].iloc[-1] == 1 else "Bearish Down"
            col4.metric("SuperTrend", st_dir)
            
            vol_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
            col5.metric("Volume Ratio", f"{vol_ratio:.1f}x")
            
            # Support/Resistance
            st.subheader("Support & Resistance")
            sr = ChartPatterns.find_support_resistance(df)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Support Levels:**")
                for level in sr.get('support', []):
                    st.write(f"  Rs {level:.2f}")
            with col2:
                st.write("**Resistance Levels:**")
                for level in sr.get('resistance', []):
                    st.write(f"  Rs {level:.2f}")
            
            # Candlestick patterns
            st.subheader("Recent Candlestick Patterns")
            patterns = CandlestickPatterns.detect_all_patterns(df.tail(10))
            detected = patterns.iloc[-1]
            active_patterns = [col for col in patterns.columns if detected[col]]
            
            if active_patterns:
                for pattern in active_patterns:
                    st.write(f"OK {pattern.replace('_', ' ')}")
            else:
                st.info("No significant candlestick patterns detected in the last candle.")
            
            # Fundamental data
            st.subheader("Fundamentals")
            fundamentals = fetcher.get_fundamentals(symbol)
            
            if fundamentals:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("P/E Ratio", f"{fundamentals.get('pe_ratio', 'N/A'):.2f}" if fundamentals.get('pe_ratio') else "N/A")
                col2.metric("Market Cap", f"Rs {fundamentals.get('market_cap', 0)/1e7:,.0f} Cr")
                col3.metric("ROE", f"{fundamentals.get('roe', 0)*100:.1f}%")
                col4.metric("D/E Ratio", f"{fundamentals.get('debt_to_equity', 'N/A'):.2f}" if fundamentals.get('debt_to_equity') else "N/A")


def show_ai_predictions():
    """AI Predictions page"""
    st.title("AI Predictions")
    
    st.info("Train the AI model on historical data to get stock predictions.")
    
    symbol = st.text_input("Stock Symbol for Training", "RELIANCE").upper()
    
    col1, col2 = st.columns(2)
    with col1:
        training_period = st.selectbox("Training Period", ["2y", "3y", "5y"], index=1)
    with col2:
        prediction_horizon = st.slider("Prediction Horizon (days)", 1, 20, 5)
    
    if st.button("Train & Predict", type="primary"):
        with st.spinner("Training AI model..."):
            from ai_models.trend_predictor import TrendPredictor
            
            fetcher = YahooFinanceFetcher()
            df = fetcher.get_historical_data(symbol, period=training_period)
            
            if df.empty:
                st.error("No data available for training")
                return
            
            # Calculate indicators
            ti = TechnicalIndicators()
            df = ti.calculate_all(df)
            
            # Train model
            predictor = TrendPredictor()
            X, y = predictor.prepare_features(df, prediction_horizon)
            
            if len(X) < 100:
                st.error("Not enough data for training. Try a longer period.")
                return
            
            try:
                results = predictor.train(X, y)
            except Exception as e:
                st.error(f"Training failed: {e}")
                return
            
            # Display training results
            st.subheader("Model Performance")
            
            metrics_df = pd.DataFrame(results).T
            st.dataframe(metrics_df.style.highlight_max(axis=0), width="stretch")
            
            # Best model
            best_model = max(results, key=lambda x: results[x]['f1'])
            st.success(f"Best model: **{best_model}** (F1: {results[best_model]['f1']:.4f})")
            
            # Make prediction
            try:
                prediction = predictor.predict(X)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                return
            
            st.subheader("Current Prediction")
            
            col1, col2, col3 = st.columns(3)
            
            pred_color = "green" if prediction['prediction'] == 'BULLISH' else "red"
            col1.markdown(f"### <span style='color:{pred_color}'>{prediction['prediction']}</span>", unsafe_allow_html=True)
            col2.metric("Confidence", f"{prediction['confidence']:.0%}")
            col3.metric("Model Agreement", f"{prediction['bullish_votes']}/{prediction['total_models']}")
            
            # Feature importance
            st.subheader("Top Influential Features")
            if prediction.get('top_features'):
                fi_df = pd.DataFrame(
                    list(prediction['top_features'].items()),
                    columns=['Feature', 'Importance']
                ).sort_values('Importance', ascending=False)
                
                fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                           title="Feature Importance")
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, width="stretch")


def show_sector_heatmap(stock_data: Dict):
    """Show sector performance heatmap"""
    sector_perf = {}
    
    for sector, symbols in SECTORS.items():
        returns = []
        for symbol in symbols:
            if symbol in stock_data:
                df = stock_data[symbol]
                if len(df) > 1:
                    ret = df['close'].pct_change().iloc[-1] * 100
                    returns.append(ret)
        
        if returns:
            sector_perf[sector] = np.mean(returns)
    
    if sector_perf:
        df_sectors = pd.DataFrame(
            list(sector_perf.items()),
            columns=['Sector', 'Return %']
        ).sort_values('Return %', ascending=False)
        
        fig = px.bar(
            df_sectors, x='Sector', y='Return %',
            color='Return %',
            color_continuous_scale=['red', 'yellow', 'green'],
            title="Sector Performance (Today)"
        )
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, width="stretch")


def show_settings():
    """Settings page"""
    st.title("Settings")
    
    st.subheader("Scanner Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.number_input("RSI Oversold Level", value=30, min_value=10, max_value=50)
        st.number_input("RSI Overbought Level", value=70, min_value=50, max_value=90)
        st.number_input("Volume Spike Multiplier", value=2.0, min_value=1.0, max_value=5.0, step=0.1)
    
    with col2:
        st.number_input("Breakout Lookback (days)", value=20, min_value=5, max_value=100)
        st.number_input("Consolidation Range %", value=5.0, min_value=1.0, max_value=20.0, step=0.5)
        st.number_input("AI Confidence Threshold", value=0.7, min_value=0.5, max_value=0.95, step=0.05)
    
    st.subheader("Alert Settings")
    
    st.text_input("Telegram Bot Token", type="password")
    st.text_input("Telegram Chat ID")
    
    if st.button("Save Settings"):
        st.success("Settings saved!")


if __name__ == "__main__":
    main()
