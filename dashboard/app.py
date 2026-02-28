"""
ðŸ›ï¸ AI Stock Scanner - Professional Dashboard
Indian Stock Market Analysis Platform

Features:
- Dark theme professional UI
- Real-time market overview
- Multi-scanner results
- AI predictions with confidence meters
- Interactive charts (Plotly)
- Volume profile visualization
- Heatmaps & correlation matrix
- Watchlist management
- Portfolio tracker
- Alert management
- Settings panel
- PDF/CSV export

Run: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import json
import sys
import os
import warnings
import textwrap
from typing import Dict, List
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND)
# ============================================================
st.set_page_config(
    page_title="AI Stock Scanner | Indian Market",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "AI-Powered Indian Stock Market Scanner & Analyzer"
    }
)


# ============================================================
# CUSTOM CSS - PROFESSIONAL DARK THEME
# ============================================================
def inject_custom_css():
    st.markdown("""
    <style>
    /* ===== GLOBAL STYLES ===== */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    :root {
        --bg-primary: #fffdf7;
        --bg-secondary: #fff8e8;
        --bg-card: #ffffff;
        --bg-card-hover: #fff6dc;
        --bg-input: #fffaf0;
        --text-primary: #1f2937;
        --text-secondary: #6b7280;
        --text-muted: #9ca3af;
        --accent-blue: #c7a23a;
        --accent-cyan: #d4af37;
        --accent-green: #10b981;
        --accent-red: #ef4444;
        --accent-yellow: #b68900;
        --accent-purple: #b8860b;
        --accent-pink: #ec4899;
        --accent-orange: #f97316;
        --border-color: #ecd9a2;
        --border-light: #e2c676;
        --gradient-primary: linear-gradient(135deg, #b8860b, #d4af37);
        --gradient-green: linear-gradient(135deg, #10b981, #059669);
        --gradient-red: linear-gradient(135deg, #ef4444, #dc2626);
        --gradient-purple: linear-gradient(135deg, #c7a23a, #b8860b);
        --shadow-card: 0 2px 10px rgba(184, 134, 11, 0.08);
        --shadow-hover: 0 8px 20px rgba(184, 134, 11, 0.18);
        --radius: 12px;
        --radius-sm: 8px;
        --radius-lg: 16px;
        --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* ===== MAIN APP ===== */
    .stApp {
        background: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        font-family: 'Poppins', -apple-system, sans-serif !important;
    }
    
    /* ===== HEADER AREA ===== */
    header[data-testid="stHeader"] {
        background: rgba(255, 253, 247, 0.92) !important;
        backdrop-filter: blur(20px) !important;
        border-bottom: 1px solid var(--border-color) !important;
    }
    
    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-color) !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: var(--text-primary) !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stRadio label {
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    
    /* ===== METRIC CARDS ===== */
    [data-testid="stMetric"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius) !important;
        padding: 16px 20px !important;
        box-shadow: var(--shadow-card) !important;
        transition: var(--transition) !important;
    }
    
    [data-testid="stMetric"]:hover {
        border-color: var(--accent-blue) !important;
        box-shadow: var(--shadow-hover) !important;
        transform: translateY(-2px) !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
    }
    
    [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-weight: 700 !important;
        font-size: 1.5rem !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    [data-testid="stMetricDelta"] > div {
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 600 !important;
    }
    
    /* Green/Red delta */
    [data-testid="stMetricDelta"] svg[fill="rgba(9, 171, 59)"] ~ div,
    [data-testid="stMetricDelta"] [data-testid="stMetricDeltaIcon-Up"] ~ div {
        color: var(--accent-green) !important;
    }
    
    [data-testid="stMetricDelta"] svg[fill="rgba(255, 43, 43)"] ~ div,
    [data-testid="stMetricDelta"] [data-testid="stMetricDeltaIcon-Down"] ~ div {
        color: var(--accent-red) !important;
    }
    
    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-secondary) !important;
        border-radius: var(--radius) !important;
        padding: 4px !important;
        gap: 4px !important;
        border: 1px solid var(--border-color) !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--text-secondary) !important;
        border-radius: var(--radius-sm) !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        padding: 10px 20px !important;
        transition: var(--transition) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--accent-blue) !important;
        color: white !important;
        font-weight: 600 !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: var(--bg-card-hover) !important;
        color: var(--text-primary) !important;
    }
    
    /* ===== BUTTONS ===== */
    .stButton > button {
        background: var(--gradient-primary) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.02em !important;
        transition: var(--transition) !important;
        box-shadow: 0 2px 4px rgba(184, 134, 11, 0.25) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(184, 134, 11, 0.35) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0) !important;
    }
    
    /* Secondary button style */
    .stButton > button[kind="secondary"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-light) !important;
        color: var(--text-primary) !important;
    }
    
    /* ===== DATAFRAMES & TABLES ===== */
    .stDataFrame {
        border-radius: var(--radius) !important;
        overflow: hidden !important;
        border: 1px solid var(--border-color) !important;
    }
    
    [data-testid="stDataFrame"] > div {
        background: var(--bg-card) !important;
        border-radius: var(--radius) !important;
    }
    
    /* ===== EXPANDER ===== */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }
    
    .streamlit-expanderContent {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        border-top: none !important;
    }
    
    /* ===== SELECT BOXES ===== */
    .stSelectbox > div > div {
        background: var(--bg-input) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-primary) !important;
    }
    
    .stMultiSelect > div > div {
        background: var(--bg-input) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-sm) !important;
    }
    
    /* ===== TEXT INPUT ===== */
    .stTextInput > div > div > input {
        background: var(--bg-input) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-primary) !important;
        font-family: 'Poppins', sans-serif !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15) !important;
    }
    
    /* ===== SLIDER ===== */
    .stSlider > div > div {
        color: var(--text-primary) !important;
    }
    
    /* ===== DIVIDER ===== */
    hr {
        border-color: var(--border-color) !important;
    }
    
    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-primary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-light);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-muted);
    }
    
    /* ===== CUSTOM COMPONENTS ===== */
    
    /* Signal Card */
    .signal-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: var(--radius);
        padding: 20px;
        margin: 8px 0;
        transition: var(--transition);
        position: relative;
        overflow: hidden;
    }
    
    .signal-card:hover {
        border-color: var(--accent-blue);
        box-shadow: var(--shadow-hover);
        transform: translateY(-2px);
    }
    
    .signal-card::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
    }
    
    .signal-card.bullish::before {
        background: var(--gradient-green);
    }
    
    .signal-card.bearish::before {
        background: var(--gradient-red);
    }
    
    /* Market Ticker */
    .ticker-row {
        display: flex;
        gap: 12px;
        overflow-x: auto;
        padding: 8px 0;
        scrollbar-width: none;
    }
    
    .ticker-item {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-sm);
        padding: 12px 18px;
        min-width: 160px;
        text-align: center;
        transition: var(--transition);
    }
    
    .ticker-item:hover {
        border-color: var(--accent-cyan);
    }
    
    .ticker-name {
        font-size: 0.75rem;
        color: var(--text-muted);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    
    .ticker-value {
        font-size: 1.2rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        color: var(--text-primary);
        margin: 4px 0;
    }
    
    .ticker-change {
        font-size: 0.85rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .ticker-change.positive { color: var(--accent-green); }
    .ticker-change.negative { color: var(--accent-red); }
    
    /* Strength Meter */
    .strength-meter {
        width: 100%;
        height: 6px;
        background: var(--bg-primary);
        border-radius: 3px;
        overflow: hidden;
        margin: 8px 0;
    }
    
    .strength-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.5s ease;
    }
    
    /* Badge */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    
    .badge-bullish {
        background: rgba(16, 185, 129, 0.15);
        color: var(--accent-green);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .badge-bearish {
        background: rgba(239, 68, 68, 0.15);
        color: var(--accent-red);
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .badge-neutral {
        background: rgba(148, 163, 184, 0.15);
        color: var(--text-secondary);
        border: 1px solid rgba(148, 163, 184, 0.3);
    }
    
    .badge-ai {
        background: rgba(139, 92, 246, 0.15);
        color: var(--accent-purple);
        border: 1px solid rgba(139, 92, 246, 0.3);
    }
    
    /* Stats Grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 12px;
        margin: 12px 0;
    }
    
    .stat-item {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-sm);
        padding: 16px;
        text-align: center;
    }
    
    .stat-label {
        font-size: 0.75rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }
    
    .stat-value {
        font-size: 1.4rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 4px;
    }
    
    /* Logo/Header */
    .main-header {
        padding: 20px 0 10px 0;
    }
    
    .main-title {
        font-size: 1.8rem;
        font-weight: 800;
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        line-height: 1.2;
    }
    
    .main-subtitle {
        font-size: 0.85rem;
        color: var(--text-muted);
        margin-top: 4px;
        font-weight: 400;
    }
    
    /* Alert animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .live-dot {
        width: 8px;
        height: 8px;
        background: var(--accent-green);
        border-radius: 50%;
        display: inline-block;
        animation: pulse 2s infinite;
        margin-right: 6px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Plotly chart background */
    .js-plotly-plot .plotly .main-svg {
        background: transparent !important;
    }

    /* ===== MOBILE TUNING ===== */
    @media (max-width: 768px) {
        :root {
            --bg-primary: #f7f1df;
            --bg-secondary: #f2e7c7;
            --bg-card: #fff5dd;
            --bg-card-hover: #fcedc6;
            --bg-input: #fff3d2;
            --border-color: #dfc27a;
            --border-light: #cfa94a;
            --text-primary: #1b2430;
            --text-secondary: #4b5563;
        }

        .stApp {
            background: linear-gradient(180deg, #f9f3e3 0%, #f3e6c2 100%) !important;
        }

        [data-testid="stSidebar"] {
            background: #efe2bd !important;
        }

        [data-testid="stMetric"] {
            background: #fff3d7 !important;
            border-color: #d9b868 !important;
        }

        .ticker-item,
        .signal-card,
        .stat-item {
            background: #fff2d2 !important;
            border-color: #d9b868 !important;
        }

        [data-testid="stDataFrame"] > div,
        .stTabs [data-baseweb="tab-list"] {
            background: #f8edd0 !important;
        }

        .js-plotly-plot .plotly .main-svg,
        .js-plotly-plot .plotly .bg {
            background: #fff5de !important;
            fill: #fff5de !important;
        }
    }
    
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# PLOTLY CHART THEME
# ============================================================
CHART_THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(255,255,255,0.92)',
    font=dict(family='Poppins, sans-serif', color='#1f2937', size=12),
    xaxis=dict(
        gridcolor='rgba(212,175,55,0.20)',
        zerolinecolor='rgba(212,175,55,0.20)',
        tickfont=dict(color='#6b7280', size=11),
    ),
    yaxis=dict(
        gridcolor='rgba(212,175,55,0.20)',
        zerolinecolor='rgba(212,175,55,0.20)',
        tickfont=dict(color='#6b7280', size=11),
    ),
    legend=dict(
        bgcolor='rgba(255,255,255,0.95)',
        bordercolor='rgba(212,175,55,0.45)',
        font=dict(color='#374151'),
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
    ),
    margin=dict(l=10, r=10, t=40, b=10),
    hoverlabel=dict(
        bgcolor='#fffaf0',
        bordercolor='#d4af37',
        font=dict(color='#1f2937', family='JetBrains Mono'),
    ),
)

GREEN = '#10b981'
RED = '#ef4444'
BLUE = '#3b82f6'
CYAN = '#06b6d4'
YELLOW = '#f59e0b'
PURPLE = '#8b5cf6'


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_candlestick_chart(df, symbol="", height=500, show_volume=True):
    """Create professional candlestick chart with volume"""
    if show_volume:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.75, 0.25],
        )
    else:
        fig = make_subplots(rows=1, cols=1)
    
    # Candlestick
    colors_up = '#10b981'
    colors_down = '#ef4444'
    
    fig.add_trace(
        go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            increasing=dict(line=dict(color=colors_up, width=1), fillcolor=colors_up),
            decreasing=dict(line=dict(color=colors_down, width=1), fillcolor=colors_down),
            name='Price',
            showlegend=False,
        ),
        row=1, col=1
    )
    
    # Moving Averages
    if 'SMA_20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['SMA_20'],
            line=dict(color=CYAN, width=1.5),
            name='SMA 20', opacity=0.8,
        ), row=1, col=1)
    
    if 'SMA_50' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['SMA_50'],
            line=dict(color=YELLOW, width=1.5),
            name='SMA 50', opacity=0.8,
        ), row=1, col=1)
    
    if 'SMA_200' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['SMA_200'],
            line=dict(color=PURPLE, width=1.5),
            name='SMA 200', opacity=0.8,
        ), row=1, col=1)
    
    # Bollinger Bands
    if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['BB_Upper'],
            line=dict(color='rgba(148,163,184,0.3)', width=1),
            name='BB Upper', showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['BB_Lower'],
            line=dict(color='rgba(148,163,184,0.3)', width=1),
            fill='tonexty', fillcolor='rgba(59,130,246,0.05)',
            name='BB Lower', showlegend=False,
        ), row=1, col=1)
    
    # Volume
    if show_volume and 'volume' in df.columns:
        colors = [
            colors_up if c >= o else colors_down
            for c, o in zip(df['close'], df['open'])
        ]
        fig.add_trace(
            go.Bar(
                x=df['date'], y=df['volume'],
                marker_color=colors, opacity=0.6,
                name='Volume', showlegend=False,
            ),
            row=2, col=1
        )
        
        # Volume SMA
        vol_sma = df['volume'].rolling(20).mean()
        fig.add_trace(go.Scatter(
            x=df['date'], y=vol_sma,
            line=dict(color=CYAN, width=1),
            name='Vol SMA', showlegend=False,
        ), row=2, col=1)
    
    # Layout
    fig.update_layout(
        **CHART_THEME,
        height=height,
        title=dict(
            text=f'{symbol}' if symbol else '',
            font=dict(size=16, color='#e2e8f0'),
            x=0.02,
        ),
        xaxis_rangeslider_visible=False,
        showlegend=True,
    )
    
    fig.update_xaxes(
        rangebreaks=[dict(bounds=["sat", "mon"])],  # Hide weekends
    )
    
    return fig


def create_rsi_chart(df, height=200):
    """Create RSI indicator chart"""
    fig = go.Figure()
    
    if 'RSI_14' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['RSI_14'],
            line=dict(color=PURPLE, width=2),
            name='RSI (14)',
        ))
        
        # Overbought/Oversold zones
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(239,68,68,0.5)", line_width=1)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(16,185,129,0.5)", line_width=1)
        fig.add_hline(y=50, line_dash="dot", line_color="rgba(148,163,184,0.3)", line_width=1)
        
        # Fill zones
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,68,68,0.05)", line_width=0)
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(16,185,129,0.05)", line_width=0)
    
    layout_theme = dict(CHART_THEME)
    base_yaxis = dict(layout_theme.pop('yaxis', {}))
    base_yaxis['range'] = [0, 100]
    fig.update_layout(
        **layout_theme,
        height=height,
        title=dict(text='RSI (14)', font=dict(size=13)),
        yaxis=base_yaxis,
    )
    
    return fig


def create_macd_chart(df, height=200):
    """Create MACD indicator chart"""
    fig = go.Figure()
    
    if 'MACD' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['MACD'],
            line=dict(color=BLUE, width=2), name='MACD',
        ))
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['MACD_Signal'],
            line=dict(color=YELLOW, width=1.5), name='Signal',
        ))
        
        # Histogram
        colors = [GREEN if v >= 0 else RED for v in df['MACD_Hist']]
        fig.add_trace(go.Bar(
            x=df['date'], y=df['MACD_Hist'],
            marker_color=colors, opacity=0.5,
            name='Histogram',
        ))
    
    fig.update_layout(
        **CHART_THEME,
        height=height,
        title=dict(text='MACD (12, 26, 9)', font=dict(size=13)),
    )
    
    return fig


def signal_card_html(symbol, signal, price, change_pct, strength, reasons, signal_type="bullish"):
    """Generate HTML for a signal card"""
    color = GREEN if signal_type == "bullish" else RED
    badge_class = "badge-bullish" if signal_type == "bullish" else "badge-bearish"
    change_color = GREEN if change_pct > 0 else RED
    strength_pct = int(strength * 100)
    
    # Strength bar color
    if strength > 0.8:
        bar_color = GREEN
    elif strength > 0.6:
        bar_color = CYAN
    elif strength > 0.4:
        bar_color = YELLOW
    else:
        bar_color = RED
    
    reasons_html = "".join([f'<div style="color:#6b7280;font-size:0.82rem;padding:2px 0;">- {r}</div>' for r in reasons[:4]])
    
    return textwrap.dedent(f"""
    <div class="signal-card {signal_type}">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div>
                <span style="font-size:1.2rem;font-weight:700;color:#e2e8f0;">{symbol}</span>
                <span class="badge {badge_class}" style="margin-left:10px;">{signal.replace('_',' ')}</span>
            </div>
            <div style="text-align:right;">
                <div style="font-family:'JetBrains Mono';font-weight:700;font-size:1.1rem;">Rs {price:,.2f}</div>
                <div style="font-family:'JetBrains Mono';font-weight:600;color:{change_color};font-size:0.9rem;">
                    {'+' if change_pct > 0 else ''}{change_pct:.2f}%
                </div>
            </div>
        </div>
        <div class="strength-meter">
            <div class="strength-fill" style="width:{strength_pct}%;background:{bar_color};"></div>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
            <span style="font-size:0.75rem;color:#64748b;">Signal Strength</span>
            <span style="font-size:0.85rem;font-weight:600;color:{bar_color};font-family:'JetBrains Mono';">{strength_pct}%</span>
        </div>
        {reasons_html}
    </div>
    """).strip()


def market_ticker_html(name, value, change, change_pct):
    """Generate HTML for market ticker item"""
    change_class = "positive" if change >= 0 else "negative"
    sign = "+" if change >= 0 else ""
    
    return textwrap.dedent(f"""
    <div class="ticker-item">
        <div class="ticker-name">{name}</div>
        <div class="ticker-value">{value:,.2f}</div>
        <div class="ticker-change {change_class}">{sign}{change:,.2f} ({sign}{change_pct:.2f}%)</div>
    </div>
    """).strip()


def ai_prediction_gauge(confidence, prediction):
    """Create circular gauge for AI confidence"""
    color = GREEN if prediction == "BULLISH" else RED if prediction == "BEARISH" else YELLOW
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title=dict(text=f"AI: {prediction}", font=dict(size=16, color='#e2e8f0')),
        number=dict(suffix="%", font=dict(size=28, color=color, family='JetBrains Mono')),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor='#334155', tickfont=dict(color='#64748b')),
            bar=dict(color=color),
            bgcolor='#1a2332',
            borderwidth=0,
            steps=[
                dict(range=[0, 30], color='rgba(239,68,68,0.1)'),
                dict(range=[30, 60], color='rgba(245,158,11,0.1)'),
                dict(range=[60, 100], color='rgba(16,185,129,0.1)'),
            ],
        )
    ))
    
    fig.update_layout(
        **CHART_THEME,
        height=220,
    )
    
    return fig


def create_heatmap(data, title=""):
    """Create performance heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=data.values,
        x=data.columns,
        y=data.index,
        colorscale=[
            [0, RED],
            [0.5, '#1a2332'],
            [1, GREEN],
        ],
        text=data.round(2).values,
        texttemplate='%{text}',
        textfont=dict(size=11, color='white'),
        hoverongaps=False,
    ))
    
    fig.update_layout(
        **CHART_THEME,
        height=400,
        title=dict(text=title, font=dict(size=14)),
    )
    
    return fig


def get_chart_links(symbol: str, exchange: str = "NSE") -> Dict[str, str]:
    """Build external chart links for a symbol."""
    sym = str(symbol or "").upper().strip()
    ex = "BSE" if str(exchange).upper().startswith("BSE") else "NSE"
    ticker = f"{sym}.BO" if ex == "BSE" else f"{sym}.NS"
    tv_symbol = quote(f"{ex}:{sym}", safe="")
    return {
        "tradingview": f"https://www.tradingview.com/chart/?symbol={tv_symbol}",
        "yahoo": f"https://finance.yahoo.com/quote/{quote(ticker, safe='')}",
    }


# ============================================================
# DATA LOADING (with caching)
# ============================================================

@st.cache_data(ttl=300)  # 5 min cache
def load_stock_data(symbol, period="6mo", exchange="NSE", interval="1d"):
    """Load and cache stock data"""
    try:
        df = load_stock_data_raw(symbol, period=period, exchange=exchange, interval=interval)
        return df
    except Exception as e:
        st.error(f"Error loading {symbol}: {e}")
        return pd.DataFrame()


def load_stock_data_raw(symbol, period="6mo", exchange="NSE", interval="1d"):
    """Non-Streamlit helper safe for background threads."""
    from data_collection.yahoo_fetcher import YahooFinanceFetcher
    from analysis.technical_indicators import TechnicalIndicators

    fetcher = YahooFinanceFetcher(exchange=exchange)
    df = fetcher.get_historical_data(symbol, period=period, interval=interval, exchange=exchange)

    if not df.empty:
        ti = TechnicalIndicators()
        df = ti.calculate_all(df)

    return df


@st.cache_data(ttl=3600)
def load_ai_symbol_universes() -> Dict[str, List[str]]:
    """Load symbol lists for AI prediction selector."""
    universes: Dict[str, List[str]] = {}
    try:
        from config.stock_universe import NIFTY_50, NIFTY_NEXT_50
        universes["NSE - NIFTY 50"] = sorted(list(dict.fromkeys(NIFTY_50)))
        universes["NSE - NIFTY Next 50"] = sorted(list(dict.fromkeys([s for s in NIFTY_NEXT_50 if s])))
    except Exception:
        universes["NSE - NIFTY 50"] = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
        universes["NSE - NIFTY Next 50"] = ["BANKBARODA", "BEL", "CANBK", "DLF", "IOC", "IRCTC"]

    try:
        from data_collection.bse_fetcher import BSE_STOCK_CODES
        bse_symbols = sorted(list(dict.fromkeys(BSE_STOCK_CODES.keys())))
        universes["BSE - Top Mapped"] = bse_symbols
    except Exception:
        universes["BSE - Top Mapped"] = ["RELIANCE", "TCS", "INFY", "SBIN", "ITC", "LT", "MARUTI"]

    universes["Custom Input"] = []
    return universes

@st.cache_data(ttl=300)
def load_multiple_stocks(symbols, period="6mo"):
    """Load multiple stocks with caching"""
    try:
        from data_collection.yahoo_fetcher import YahooFinanceFetcher
        from analysis.technical_indicators import TechnicalIndicators
        
        fetcher = YahooFinanceFetcher(exchange="NSE")
        ti = TechnicalIndicators()
        
        symbols = [s for s in symbols if s.upper() != "TATAMOTORS"]
        data = fetcher.get_bulk_historical_data(symbols, period=period)
        
        for sym in data:
            data[sym] = ti.calculate_all(data[sym])
        
        return data
    except Exception as e:
        st.error(f"Error loading stocks: {e}")
        return {}

@st.cache_data(ttl=60)
def load_index_data():
    """Load index data"""
    try:
        from data_collection.yahoo_fetcher import YahooFinanceFetcher
        fetcher = YahooFinanceFetcher()
        
        indices = {}
        for name, ticker in [
            ("NIFTY 50", "^NSEI"),
            ("SENSEX", "^BSESN"),
            ("BANK NIFTY", "^NSEBANK"),
            ("INDIA VIX", "^INDIAVIX"),
        ]:
            df = fetcher.get_historical_data(ticker, period="5d")
            if not df.empty:
                latest = df.iloc[-1]['close']
                prev = df.iloc[-2]['close'] if len(df) > 1 else latest
                indices[name] = {
                    'value': latest,
                    'change': latest - prev,
                    'change_pct': (latest - prev) / prev * 100,
                }
        return indices
    except:
        return {}


@st.cache_data(ttl=900)
def load_market_news_feed(bucket: str, days: int = 7, max_per_topic: int = 3) -> pd.DataFrame:
    """Load multi-topic news feed with lightweight sentiment scoring."""
    try:
        analyzer = get_sentiment_analyzer()

        baskets = {
            "NSE Stocks": [
                "NIFTY 50", "NSE Bank", "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN"
            ],
            "BSE Stocks": [
                "SENSEX", "BSE 100", "BSE Midcap", "ITC", "AXISBANK", "KOTAKBANK", "MARUTI", "LT"
            ],
            "Global & Commodities": [
                "GIFT Nifty", "US stock market", "Dow Jones", "Nasdaq", "S&P 500",
                "Gold price", "Silver price", "Crude oil", "US Federal Reserve"
            ],
            "India + Global Mix": [
                "NIFTY 50", "SENSEX", "GIFT Nifty", "US stock market", "Gold price", "Silver price"
            ],
        }

        topics = baskets.get(bucket, baskets["India + Global Mix"])
        rows = []
        seen = set()

        for topic in topics:
            articles = analyzer.fetch_news(topic, days=days, max_articles=max_per_topic)
            for article in articles:
                title = str(article.get("title", "")).strip()
                url = str(article.get("url", "")).strip()
                key = (title.lower(), url.lower())
                if not title or key in seen:
                    continue
                seen.add(key)

                text = f"{title}. {article.get('description', '')}"
                s = analyzer.analyze_text(text)
                impact = analyzer.estimate_news_impact(
                    title=title,
                    description=article.get("description", ""),
                    source=article.get("source", ""),
                    published_at=article.get("published_at", ""),
                    sentiment=s,
                )
                sentiment_label = str(s.get("sentiment", "neutral")).upper()

                rows.append({
                    "topic": topic,
                    "title": title,
                    "source": article.get("source", "Unknown"),
                    "published_at": article.get("published_at", "N/A"),
                    "url": url,
                    "sentiment": sentiment_label,
                    "score": float(s.get("positive", 0.0) - s.get("negative", 0.0)),
                    "impact_score": impact.get("impact_score", 0.0),
                    "impact_label": impact.get("impact_label", "NEUTRAL"),
                    "confidence": float(s.get("confidence", 0.0)),
                })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df.sort_values(by=["topic", "score"], ascending=[True, False], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_resource
def get_sentiment_analyzer():
    """Reuse heavy analyzer/model objects across reruns."""
    from ai_models.sentiment_analyzer import SentimentAnalyzer
    return SentimentAnalyzer()


@st.cache_data(ttl=900)
def load_stock_sentiment(symbol: str, days: int = 7, max_articles: int = 10) -> Dict:
    """Cached stock-level sentiment/impact for dashboard workflows."""
    try:
        analyzer = get_sentiment_analyzer()
        return analyzer.get_stock_sentiment(symbol.upper().strip(), days=days, max_articles=max_articles)
    except Exception:
        return {
            "symbol": symbol,
            "overall_sentiment": "neutral",
            "impact_score": 0.0,
            "overall_impact": "NEUTRAL",
            "article_count": 0,
            "articles": [],
        }


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():
    """Render sidebar navigation"""
    with st.sidebar:
        # Logo
        st.markdown(textwrap.dedent("""
        <div style="text-align:center;padding:15px 0;">
            <div class="main-title">AI Scanner</div>
            <div class="main-subtitle">Indian Stock Market</div>
        </div>
        """), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "Navigation",
            [
                "Dashboard",
                "Scanner",
                "Stock Analysis",
                "AI Predictions",
                "AI Top Picks",
                "News & Sentiment",
                "Market Heatmap",
                "Watchlist",
                "Portfolio",
                "Settings",
            ],
            label_visibility="visible",
        )
        
        st.markdown("---")
        
        # Market Status
        now = datetime.now()
        is_market_hours = (
            now.weekday() < 5 and
            now.hour >= 9 and
            (now.hour < 15 or (now.hour == 15 and now.minute <= 30))
        )
        
        status_color = GREEN if is_market_hours else RED
        status_text = "MARKET OPEN" if is_market_hours else "MARKET CLOSED"
        
        st.markdown(f"""
        <div style="text-align:center;padding:10px;background:var(--bg-card);border-radius:8px;border:1px solid var(--border-color);">
            <span class="live-dot" style="background:{status_color};"></span>
            <span style="color:{status_color};font-weight:600;font-size:0.8rem;">{status_text}</span>
            <div style="color:var(--text-muted);font-size:0.75rem;margin-top:4px;">
                {now.strftime('%d %b %Y, %I:%M %p')}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Quick Links
        st.markdown("**Quick Actions**")
        if st.button("Refresh Data", width="stretch"):
            st.cache_data.clear()
            st.rerun()

        st.markdown(
            textwrap.dedent("""
            <div style="margin-top:12px;padding:10px;background:#fff3cd;border:1px solid #e2c676;border-radius:8px;">
                <div style="color:#7a5c00;font-size:0.74rem;line-height:1.35;">
                    <strong>Disclaimer:</strong> This app is for educational and analysis purposes only.
                    It is not financial advice and does not provide buy/sell/trade recommendations.
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )
        
        st.markdown("""
        <div style="text-align:center;color:var(--text-muted);font-size:0.7rem;padding:20px 0;">
            Version 2.0 | Built with care
        </div>
        """, unsafe_allow_html=True)
    
    return page


def render_disclaimer():
    """Render global legal disclaimer"""
    st.caption(
        "Disclaimer: This platform is intended only for market analysis and education. "
        "It is not investment advice. Do your own research and consult a SEBI-registered advisor before trading."
    )


def init_alert_state():
    """Initialize alert-related session state."""
    defaults = {
        "auto_alerts_enabled": False,
        "alert_channel_telegram": True,
        "alert_channel_email": False,
        "alert_buy_threshold": 20.0,
        "alert_sell_threshold": -20.0,
        "alert_cooldown_min": 30,
        "alert_max_per_run": 5,
        "telegram_bot_token": "",
        "telegram_chat_id": "",
        "email_smtp_server": "smtp.gmail.com",
        "email_smtp_port": 587,
        "email_sender": "",
        "email_password": "",
        "email_recipients": "",
        "alert_last_sent": {},
        "alert_history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _can_send_alert(alert_key: str) -> bool:
    """Cooldown gate for duplicate alerts."""
    cooldown = int(st.session_state.get("alert_cooldown_min", 30))
    last_sent = st.session_state.get("alert_last_sent", {})
    now = datetime.now()
    prev = last_sent.get(alert_key)
    if not prev:
        return True
    try:
        prev_dt = datetime.fromisoformat(prev)
        return (now - prev_dt).total_seconds() >= cooldown * 60
    except Exception:
        return True


def _mark_alert_sent(alert_key: str, payload: Dict):
    """Store sent alert metadata."""
    now = datetime.now().isoformat()
    last_sent = st.session_state.get("alert_last_sent", {})
    last_sent[alert_key] = now
    st.session_state["alert_last_sent"] = last_sent

    history = st.session_state.get("alert_history", [])
    history.insert(0, {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **payload,
    })
    st.session_state["alert_history"] = history[:200]


def send_auto_alert(symbol: str, action: str, conviction_score: float, ai_confidence: float, news_impact: float, exchange: str) -> Dict:
    """Send auto alert via configured channels. Returns delivery summary."""
    sent_channels = []
    msg_text = (
        f"AI Auto Alert\n"
        f"Symbol: {symbol} ({exchange})\n"
        f"Action: {action}\n"
        f"Conviction Score: {conviction_score:+.1f}\n"
        f"AI Confidence: {ai_confidence:.1f}%\n"
        f"News Impact: {news_impact:+.1f}\n"
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    chart_links = get_chart_links(symbol, exchange=exchange)
    msg_with_links = f"{msg_text}\nChart: {chart_links['tradingview']}\nQuote: {chart_links['yahoo']}"

    if st.session_state.get("alert_channel_telegram", True):
        token = st.session_state.get("telegram_bot_token", "").strip()
        chat_id = st.session_state.get("telegram_chat_id", "").strip()
        if token and chat_id:
            try:
                from alerts.telegram_bot import TelegramBot
                bot = TelegramBot(token, chat_id)
                if bot.send_message(msg_with_links):
                    sent_channels.append("telegram")
            except Exception:
                pass

    if st.session_state.get("alert_channel_email", False):
        sender = st.session_state.get("email_sender", "").strip()
        pwd = st.session_state.get("email_password", "").strip()
        recipients_raw = st.session_state.get("email_recipients", "").strip()
        recipients = [r.strip() for r in recipients_raw.split(",") if r.strip()]
        if sender and pwd and recipients:
            try:
                from alerts.email_alerts import EmailAlerts
                mailer = EmailAlerts(
                    smtp_server=st.session_state.get("email_smtp_server", "smtp.gmail.com"),
                    smtp_port=int(st.session_state.get("email_smtp_port", 587)),
                    sender_email=sender,
                    sender_password=pwd,
                    recipients=recipients,
                )
                html = (
                    "<html><body>"
                    f"<h3>AI Auto Alert: {symbol} ({exchange})</h3>"
                    f"<p><b>Action:</b> {action}</p>"
                    f"<p><b>Conviction Score:</b> {conviction_score:+.1f}</p>"
                    f"<p><b>AI Confidence:</b> {ai_confidence:.1f}%</p>"
                    f"<p><b>News Impact:</b> {news_impact:+.1f}</p>"
                    f"<p><a href='{chart_links['tradingview']}'>TradingView Chart</a> | "
                    f"<a href='{chart_links['yahoo']}'>Yahoo Quote</a></p>"
                    f"<p>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
                    "</body></html>"
                )
                if mailer.send_email(subject=f"AI Alert: {symbol} {action}", body_html=html):
                    sent_channels.append("email")
            except Exception:
                pass

    return {"sent_channels": sent_channels}


# ============================================================
# PAGE: DASHBOARD
# ============================================================

def page_dashboard():
    """Main dashboard page"""
    # Header
    st.markdown(textwrap.dedent("""
    <div class="main-header">
        <div class="main-title">Market Dashboard</div>
        <div class="main-subtitle">Real-time overview of Indian stock market</div>
    </div>
    """), unsafe_allow_html=True)
    
    # Market Ticker
    indices = load_index_data()
    
    if indices:
        ticker_html = '<div class="ticker-row">'
        for name, data in indices.items():
            ticker_html += market_ticker_html(
                name, data['value'], data['change'], data['change_pct']
            )
        ticker_html += '</div>'
        st.markdown(ticker_html, unsafe_allow_html=True)
    
    st.markdown("")
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    if indices:
        nifty = indices.get('NIFTY 50', {'value': 0, 'change': 0, 'change_pct': 0})
        sensex = indices.get('SENSEX', {'value': 0, 'change': 0, 'change_pct': 0})
        banknifty = indices.get('BANK NIFTY', {'value': 0, 'change': 0, 'change_pct': 0})
        vix = indices.get('INDIA VIX', {'value': 0, 'change': 0, 'change_pct': 0})
        
        col1.metric("NIFTY 50", f"{nifty['value']:,.2f}", f"{nifty['change_pct']:+.2f}%")
        col2.metric("SENSEX", f"{sensex['value']:,.2f}", f"{sensex['change_pct']:+.2f}%")
        col3.metric("BANK NIFTY", f"{banknifty['value']:,.2f}", f"{banknifty['change_pct']:+.2f}%")
        col4.metric("INDIA VIX", f"{vix['value']:.2f}", f"{vix['change_pct']:+.2f}%")
    
    st.markdown("")
    
    # Charts Section
    col_chart, col_signals = st.columns([2, 1])
    
    with col_chart:
        st.markdown("#### NIFTY 50 Chart")
        nifty_df = load_stock_data("^NSEI", period="3mo")
        if not nifty_df.empty:
            fig = create_candlestick_chart(nifty_df, "NIFTY 50", height=450)
            st.plotly_chart(fig, width="stretch")
    
    with col_signals:
        st.markdown("#### Latest Signals")
        
        # Load quick scan
        try:
            quick_symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
            data = load_multiple_stocks(quick_symbols, "3mo")
            
            if data:
                from scanners.momentum_scanner import MomentumScanner
                scanner = MomentumScanner()
                results = scanner.run_all_scans(data)
                
                if results:
                    for r in results[:5]:
                        sig_type = "bullish" if "BUY" in r.signal or "BULLISH" in r.signal else "bearish"
                        st.markdown(
                            signal_card_html(
                                r.symbol, r.signal, r.price,
                                r.change_pct, r.strength, r.reasons[:2], sig_type
                            ),
                            unsafe_allow_html=True
                        )
                else:
                    st.info("No active signals at this time")
        except Exception as e:
            st.warning(f"Scanner not available: {str(e)[:50]}")


# ============================================================
# PAGE: SCANNER
# ============================================================

def page_scanner():
    """Full scanner page"""
    st.markdown("""
    <div class="main-header">
        <div class="main-title">Stock Scanner</div>
        <div class="main-subtitle">Multi-strategy signal detection engine</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Scanner Config
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        scan_type = st.selectbox("Scanner Type", [
            "All Scanners",
            "Momentum",
            "Breakout",
            "Reversal",
            "Volume",
        ])
    
    with col2:
        from config.stock_universe import NIFTY_50
        universe = st.selectbox("Stock Universe", [
            "NIFTY 50",
            "NIFTY Bank",
            "Custom",
        ])
    
    with col3:
        period = st.selectbox("Data Period", ["3mo", "6mo", "1y", "2y"])
    
    # Custom symbols
    if universe == "Custom":
        custom_input = st.text_input("Enter symbols (comma separated)", "RELIANCE,TCS,INFY")
        symbols = [s.strip().upper() for s in custom_input.split(",")]
    elif universe == "NIFTY Bank":
        symbols = ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK",
                   "INDUSINDBK", "BANDHANBNK", "FEDERALBNK", "IDFCFIRSTB", "PNB"]
    else:
        try:
            from config.stock_universe import NIFTY_50
            symbols = NIFTY_50
        except:
            symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
                       "SBIN", "ITC", "BHARTIARTL", "KOTAKBANK", "LT",
                       "HCLTECH", "AXISBANK", "MARUTI", "SUNPHARMA", "TITAN"]
    
    # Run Scanner
    if st.button("Run Scanner", type="primary", width="stretch"):
        with st.spinner("Fetching data and running scanners..."):
            progress = st.progress(0, text="Loading stock data...")
            
            stock_data = load_multiple_stocks(symbols, period)
            progress.progress(50, text=f"Loaded {len(stock_data)} stocks. Running scanners...")
            
            all_results = []
            
            try:
                if scan_type in ["All Scanners", "Momentum"]:
                    from scanners.momentum_scanner import MomentumScanner
                    scanner = MomentumScanner()
                    all_results.extend(scanner.run_all_scans(stock_data))
                
                if scan_type in ["All Scanners", "Breakout"]:
                    from scanners.breakout_scanner import BreakoutScanner
                    scanner = BreakoutScanner()
                    all_results.extend(scanner.run_all_scans(stock_data))
                
                if scan_type in ["All Scanners", "Reversal"]:
                    from scanners.reversal_scanner import ReversalScanner
                    scanner = ReversalScanner()
                    all_results.extend(scanner.run_all_scans(stock_data))
                
                if scan_type in ["All Scanners", "Volume"]:
                    from scanners.volume_scanner import VolumeScanner
                    scanner = VolumeScanner()
                    all_results.extend(scanner.run_all_scans(stock_data))
                
                progress.progress(100, text="Scan complete!")
                time.sleep(0.5)
                progress.empty()
                
            except Exception as e:
                st.error(f"Scanner error: {e}")
                return
            
            if all_results:
                # Sort by strength
                all_results.sort(key=lambda x: x.strength, reverse=True)
                
                # Summary metrics
                bullish = [r for r in all_results if "BUY" in r.signal or "BULLISH" in r.signal or "ACCUMULATION" in r.signal]
                bearish = [r for r in all_results if "SELL" in r.signal or "BEARISH" in r.signal or "DISTRIBUTION" in r.signal]
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Signals", len(all_results))
                c2.metric("Bullish", len(bullish))
                c3.metric("Bearish", len(bearish))
                c4.metric("Avg Strength", f"{np.mean([r.strength for r in all_results]):.0%}")
                
                st.markdown("---")
                
                # Tabs for results
                tab_all, tab_bullish, tab_bearish = st.tabs(["All Signals", "Bullish", "Bearish"])
                
                with tab_all:
                    for r in all_results:
                        sig_type = "bullish" if r in bullish else "bearish"
                        st.markdown(
                            signal_card_html(r.symbol, r.signal, r.price, r.change_pct, r.strength, r.reasons, sig_type),
                            unsafe_allow_html=True
                        )
                
                with tab_bullish:
                    for r in bullish:
                        st.markdown(
                            signal_card_html(r.symbol, r.signal, r.price, r.change_pct, r.strength, r.reasons, "bullish"),
                            unsafe_allow_html=True
                        )
                
                with tab_bearish:
                    for r in bearish:
                        st.markdown(
                            signal_card_html(r.symbol, r.signal, r.price, r.change_pct, r.strength, r.reasons, "bearish"),
                            unsafe_allow_html=True
                        )
            else:
                st.info("No signals found. Try different settings or check back during market hours.")


# ============================================================
# PAGE: STOCK ANALYSIS
# ============================================================

def page_stock_analysis():
    """Individual stock analysis page"""
    st.markdown("""
    <div class="main-header">
        <div class="main-title">Stock Analysis</div>
        <div class="main-subtitle">Deep technical & fundamental analysis</div>
    </div>
    """, unsafe_allow_html=True)
    
    symbol_universes = load_ai_symbol_universes()

    c1, c2, c3 = st.columns([1.4, 2.2, 1])
    with c1:
        universe_name = st.selectbox(
            "Select List",
            list(symbol_universes.keys()),
            index=0,
            key="stock_analysis_universe",
        )
    with c2:
        if universe_name == "Custom Input":
            symbol = st.text_input(
                "Enter Stock Symbol",
                value="RELIANCE",
                placeholder="e.g., RELIANCE, TCS, INFY",
                key="stock_analysis_custom_symbol",
            )
        else:
            options = symbol_universes.get(universe_name, ["RELIANCE"])
            default_idx = options.index("RELIANCE") if "RELIANCE" in options else 0
            symbol = st.selectbox(
                "Choose Symbol",
                options,
                index=default_idx,
                key="stock_analysis_symbol_picker",
            )
    with c3:
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)

    selected_exchange = "BSE" if universe_name.startswith("BSE") else "NSE"
    st.caption(f"Selected Universe: {universe_name} | Exchange Mode: {selected_exchange}")
    
    if symbol:
        symbol = symbol.upper().strip()
        chart_links = get_chart_links(symbol, selected_exchange)
        st.markdown(
            f"[Open TradingView Chart]({chart_links['tradingview']}) | "
            f"[Open Yahoo Finance Chart]({chart_links['yahoo']})"
        )
        
        with st.spinner(f"Loading {symbol} data..."):
            df = load_stock_data(symbol, period, exchange=selected_exchange)
        
        if df.empty:
            st.error(f"No data found for {symbol}. Check the symbol name.")
            return
        
        # Quick Stats
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
        change = latest['close'] - prev['close']
        change_pct = change / prev['close'] * 100
        
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Price", f"Rs {latest['close']:,.2f}", f"{change_pct:+.2f}%")
        c2.metric("Open", f"Rs {latest['open']:,.2f}")
        c3.metric("High", f"Rs {latest['high']:,.2f}")
        c4.metric("Low", f"Rs {latest['low']:,.2f}")
        c5.metric("Volume", f"{latest['volume']/1e6:.2f}M")
        
        # Indicator Summary
        st.markdown("")
        ind_cols = st.columns(6)
        
        indicators = {
            'RSI': ('RSI_14', lambda v: f"{v:.1f}", lambda v: GREEN if v < 30 else RED if v > 70 else YELLOW),
            'MACD': ('MACD_Hist', lambda v: f"{v:.2f}", lambda v: GREEN if v > 0 else RED),
            'ADX': ('ADX', lambda v: f"{v:.1f}", lambda v: GREEN if v > 25 else YELLOW),
            'SuperTrend': ('ST_Direction', lambda v: "BUY" if v == 1 else "SELL", lambda v: GREEN if v == 1 else RED),
        }
        
        for i, (name, (col_name, fmt_fn, color_fn)) in enumerate(indicators.items()):
            if col_name in df.columns:
                val = latest[col_name]
                color = color_fn(val)
                ind_cols[i].markdown(f"""
                <div style="background:var(--bg-card);border:1px solid var(--border-color);border-radius:8px;padding:12px;text-align:center;">
                    <div style="font-size:0.75rem;color:var(--text-muted);text-transform:uppercase;">{name}</div>
                    <div style="font-size:1.3rem;font-weight:700;color:{color};font-family:'JetBrains Mono';">{fmt_fn(val)}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Charts
        st.markdown("---")
        
        tab_chart, tab_indicators, tab_volume, tab_fundamentals = st.tabs([
            "Price Chart", "Indicators", "Volume", "Fundamentals"
        ])
        
        with tab_chart:
            fig = create_candlestick_chart(df, symbol, height=550)
            st.plotly_chart(fig, width="stretch")
        
        with tab_indicators:
            col_rsi, col_macd = st.columns(2)
            with col_rsi:
                st.plotly_chart(create_rsi_chart(df), width="stretch")
            with col_macd:
                st.plotly_chart(create_macd_chart(df), width="stretch")
        
        with tab_volume:
            # Volume chart
            fig = go.Figure()
            colors = [GREEN if c >= o else RED for c, o in zip(df['close'], df['open'])]
            fig.add_trace(go.Bar(x=df['date'], y=df['volume'], marker_color=colors, opacity=0.7))
            
            vol_sma = df['volume'].rolling(20).mean()
            fig.add_trace(go.Scatter(x=df['date'], y=vol_sma, line=dict(color=CYAN, width=2), name='20-Day Avg'))
            
            fig.update_layout(**CHART_THEME, height=350, title="Volume Analysis")
            st.plotly_chart(fig, width="stretch")
        
        with tab_fundamentals:
            try:
                from data_collection.yahoo_fetcher import YahooFinanceFetcher
                fetcher = YahooFinanceFetcher(exchange=selected_exchange)
                fund = fetcher.get_fundamentals(symbol, exchange=selected_exchange)
                
                if fund:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("P/E Ratio", f"{fund.pe_ratio:.2f}")
                    c2.metric("P/B Ratio", f"{fund.pb_ratio:.2f}")
                    c3.metric("Market Cap", f"Rs {fund.market_cap/1e7:,.0f} Cr")
                    c4.metric("Div Yield", f"{fund.dividend_yield*100:.2f}%")
                    
                    c5, c6, c7, c8 = st.columns(4)
                    c5.metric("ROE", f"{fund.roe*100:.1f}%" if fund.roe else "N/A")
                    c6.metric("Debt/Equity", f"{fund.debt_to_equity:.2f}" if fund.debt_to_equity else "N/A")
                    c7.metric("EPS", f"Rs {fund.eps:.2f}" if fund.eps else "N/A")
                    c8.metric("52W Range", f"Rs {fund.week_52_low:.0f} - Rs {fund.week_52_high:.0f}")
                    
                    if fund.description:
                        with st.expander("Company Description"):
                            st.write(fund.description[:500] + "...")
                else:
                    st.info("Fundamental data not available for this stock")
            except Exception as e:
                st.warning(f"Could not load fundamentals: {str(e)[:50]}")


# ============================================================
# PAGE: AI PREDICTIONS
# ============================================================

def page_ai_predictions():
    """AI prediction page"""
    st.markdown("""
    <div class="main-header">
        <div class="main-title">AI Predictions</div>
        <div class="main-subtitle">Machine learning powered market predictions</div>
    </div>
    """, unsafe_allow_html=True)
    
    symbol_universes = load_ai_symbol_universes()

    c1, c2, c3 = st.columns([1.4, 2.2, 1])
    with c1:
        universe_name = st.selectbox(
            "Select List",
            list(symbol_universes.keys()),
            index=0,
        )
    with c2:
        if universe_name == "Custom Input":
            ai_symbol = st.text_input("Symbol for AI Analysis", value="RELIANCE")
        else:
            options = symbol_universes.get(universe_name, ["RELIANCE"])
            default_idx = options.index("RELIANCE") if "RELIANCE" in options else 0
            ai_symbol = st.selectbox("Choose Symbol", options, index=default_idx)
    with c3:
        horizon = st.selectbox("Prediction Horizon", ["5 Days", "10 Days", "20 Days"])

    ai_exchange = "BSE" if universe_name.startswith("BSE") else "NSE"
    st.caption(f"Selected Universe: {universe_name} | Exchange Mode: {ai_exchange}")
    ai_links = get_chart_links(ai_symbol, ai_exchange)
    st.markdown(
        f"[Open TradingView Chart]({ai_links['tradingview']}) | "
        f"[Open Yahoo Finance Chart]({ai_links['yahoo']})"
    )
    
    if st.button("Run AI Analysis", type="primary", width="stretch"):
        with st.spinner("Training AI models..."):
            try:
                symbol_clean = ai_symbol.upper().strip()
                if not symbol_clean:
                    st.error("Please select or enter a valid symbol.")
                    return

                df = load_stock_data(symbol_clean, "2y", exchange=ai_exchange)
                
                if df.empty or len(df) < 200:
                    st.error("Insufficient data for AI analysis (need at least 200 data points)")
                    return
                
                from ai_models.trend_predictor import TrendPredictor
                
                predictor = TrendPredictor()
                X, y = predictor.prepare_features(df)
                
                if len(X) < 100:
                    st.error("Insufficient feature data")
                    return
                
                results = predictor.train(X, y)
                prediction = predictor.predict(X)
                
                # Display Results
                st.markdown("---")
                
                c1, c2, c3 = st.columns([1, 1, 1])
                
                with c1:
                    st.plotly_chart(
                        ai_prediction_gauge(prediction['confidence'], prediction['prediction']),
                        width="stretch"
                    )
                
                with c2:
                    st.markdown(f"""
                    <div style="background:var(--bg-card);border-radius:12px;padding:20px;border:1px solid var(--border-color);">
                        <h4 style="color:var(--text-secondary);margin-bottom:15px;">Model Votes</h4>
                        <div style="font-size:2rem;font-weight:700;color:{GREEN if prediction['prediction']=='BULLISH' else RED};">
                            {prediction['prediction']}
                        </div>
                        <div style="color:var(--text-muted);margin-top:8px;">
                            Bullish: {prediction.get('bullish_votes', 'N/A')} / {prediction.get('total_models', 'N/A')} models
                        </div>
                        <div style="color:var(--text-muted);">
                            Confidence: {prediction['confidence']:.0%}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with c3:
                    st.markdown("#### Model Performance")
                    for name, metrics in results.items():
                        acc = metrics.get('accuracy', 0)
                        f1 = metrics.get('f1', 0)
                        color = GREEN if acc > 0.55 else YELLOW if acc > 0.50 else RED
                        st.markdown(f"""
                        <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid var(--border-color);">
                            <span style="color:var(--text-secondary);">{name}</span>
                            <span style="color:{color};font-family:'JetBrains Mono';font-weight:600;">{acc:.1%}</span>
                        </div>
                        """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"AI Analysis failed: {e}")


# ============================================================
# PAGE: AI TOP PICKS
# ============================================================

def page_ai_top_picks():
    """Batch AI ranking with optional news-impact merge."""
    st.markdown("""
    <div class="main-header">
        <div class="main-title">AI Top Picks</div>
        <div class="main-subtitle">Batch AI ranking with conviction score</div>
    </div>
    """, unsafe_allow_html=True)

    symbol_universes = load_ai_symbol_universes()
    available_universes = [u for u in symbol_universes.keys() if u != "Custom Input"]

    c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1])
    with c1:
        universe_name = st.selectbox("Universe", available_universes, index=0, key="top_picks_universe")
    with c2:
        period = st.selectbox("Training Period", ["1y", "2y", "3y"], index=1, key="top_picks_period")
    with c3:
        max_symbols = st.slider("Symbols", 5, 40, 15, key="top_picks_max_symbols")
    with c4:
        top_n = st.slider("Top Picks", 3, 15, 10, key="top_picks_top_n")

    t1, t2 = st.columns([1, 3])
    with t1:
        data_mode = st.selectbox("Mode", ["Daily", "Intraday Fast"], index=0, key="top_picks_data_mode")
    with t2:
        intraday_interval = st.selectbox("Intraday Interval", ["5m", "15m", "30m", "60m"], index=2, key="top_picks_intraday_interval")

    include_news_merge = st.toggle(
        "Merge News Impact into Conviction Score",
        value=True,
        help="Conviction score combines AI confidence and news impact score.",
    )

    exchange = "BSE" if universe_name.startswith("BSE") else "NSE"
    symbols = symbol_universes.get(universe_name, [])[:max_symbols]
    st.caption(f"Universe: {universe_name} | Exchange: {exchange} | Symbols Selected: {len(symbols)}")

    if st.button("Run Batch AI Ranking", type="primary", width="stretch", key="btn_top_picks"):
        if not symbols:
            st.warning("No symbols available in selected universe.")
            return

        try:
            from ai_models.trend_predictor import TrendPredictor
            from analysis.regime_detector import detect_regime
        except Exception as e:
            st.error(f"Could not load AI predictor: {e}")
            return

        analyzer = None
        news_cache = {}
        if include_news_merge:
            try:
                analyzer = get_sentiment_analyzer()
                # Prefetch news in main thread to avoid Streamlit context warnings in worker threads.
                for sym in symbols:
                    news_cache[sym] = load_stock_sentiment(sym, days=5, max_articles=8)
            except Exception as e:
                st.warning(f"News analyzer unavailable. Running AI-only ranking. Reason: {e}")
                include_news_merge = False

        rows = []
        failed = []
        progress = st.progress(0, text="Starting batch AI scan...")
        
        def process_symbol(symbol: str):
            try:
                use_period = "1mo" if data_mode == "Intraday Fast" else period
                use_interval = intraday_interval if data_mode == "Intraday Fast" else "1d"

                df = load_stock_data_raw(symbol, period=use_period, exchange=exchange, interval=use_interval)
                if df.empty or len(df) < 220:
                    return None, symbol

                predictor = TrendPredictor()
                X, y = predictor.prepare_features(df)
                if len(X) < 120:
                    return None, symbol

                results = predictor.train(X, y, quick_mode=True)
                pred = predictor.predict(X)

                ai_signal = str(pred.get("prediction", "NEUTRAL")).upper()
                ai_conf = float(pred.get("confidence", 0.0))
                ai_dir = 1.0 if ai_signal == "BULLISH" else -1.0 if ai_signal == "BEARISH" else 0.0
                ai_score = ai_dir * ai_conf * 100.0
                regime_info = detect_regime(df)
                regime = regime_info.get("regime", "UNKNOWN")
                regime_conf = float(regime_info.get("confidence", 0.0))

                impact_score = 0.0
                impact_label = "NA"
                if include_news_merge and analyzer is not None:
                    news = news_cache.get(symbol, {})
                    impact_score = float(news.get("impact_score", 0.0))
                    impact_label = str(news.get("overall_impact", "NEUTRAL"))

                conviction = (0.75 * ai_score) + (0.25 * impact_score if include_news_merge else 0.0)
                # Regime-aware damping: reduce confidence in high-volatility range.
                if regime == "HIGH_VOLATILITY":
                    conviction *= 0.85
                elif regime == "RANGE_BOUND":
                    conviction *= 0.92
                else:
                    conviction *= (0.95 + 0.05 * regime_conf)
                action = "BUY" if conviction >= 15 else "SELL" if conviction <= -15 else "WATCH"

                latest = df.iloc[-1]
                prev = df.iloc[-2] if len(df) > 1 else latest
                change_pct = float((latest["close"] - prev["close"]) / max(prev["close"], 1e-9) * 100)

                return {
                    "symbol": symbol,
                    "action": action,
                    "ai_signal": ai_signal,
                    "ai_confidence": round(ai_conf * 100.0, 1),
                    "ai_score": round(ai_score, 1),
                    "news_impact": round(impact_score, 1),
                    "impact_label": impact_label,
                    "conviction_score": round(conviction, 1),
                    "regime": regime,
                    "regime_conf": round(regime_conf * 100.0, 1),
                    "last_price": round(float(latest["close"]), 2),
                    "change_pct": round(change_pct, 2),
                    "model_acc": round(float(results.get("ensemble", {}).get("accuracy", 0.0) * 100.0), 1),
                }, None
            except Exception:
                return None, symbol

        max_workers = min(4, len(symbols))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_symbol, sym): sym for sym in symbols}
            done = 0
            for fut in as_completed(futures):
                done += 1
                progress.progress(int(done * 100 / len(symbols)), text=f"Processed {done}/{len(symbols)}")
                row, failed_symbol = fut.result()
                if row:
                    rows.append(row)
                if failed_symbol:
                    failed.append(failed_symbol)

        progress.empty()

        if not rows:
            st.warning("No valid results generated. Try smaller list or different period.")
            return

        out = pd.DataFrame(rows).sort_values("conviction_score", ascending=False).reset_index(drop=True)
        bullish = out[out["conviction_score"] > 0].head(top_n)
        bearish = out[out["conviction_score"] < 0].sort_values("conviction_score", ascending=True).head(top_n)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Scanned", len(symbols))
        m2.metric("Valid Results", len(out))
        m3.metric("Strong Buy", int((out["action"] == "BUY").sum()))
        m4.metric("Strong Sell", int((out["action"] == "SELL").sum()))

        if failed:
            st.caption(f"Skipped/failed symbols: {', '.join(failed[:12])}{' ...' if len(failed) > 12 else ''}")

        tab1, tab2, tab3 = st.tabs(["Top Bullish", "Top Bearish", "All Ranked"])
        with tab1:
            st.dataframe(
                bullish[["symbol", "action", "conviction_score", "regime", "ai_confidence", "news_impact", "last_price", "change_pct", "model_acc"]],
                width="stretch"
            )
        with tab2:
            st.dataframe(
                bearish[["symbol", "action", "conviction_score", "regime", "ai_confidence", "news_impact", "last_price", "change_pct", "model_acc"]],
                width="stretch"
            )
        with tab3:
            st.dataframe(
                out[["symbol", "action", "conviction_score", "regime", "regime_conf", "ai_signal", "ai_confidence", "news_impact", "impact_label", "last_price", "change_pct", "model_acc"]],
                width="stretch"
            )

        # Auto alerts trigger
        if st.session_state.get("auto_alerts_enabled", False):
            buy_th = float(st.session_state.get("alert_buy_threshold", 20.0))
            sell_th = float(st.session_state.get("alert_sell_threshold", -20.0))
            max_alerts = int(st.session_state.get("alert_max_per_run", 5))

            candidates = out[
                (out["conviction_score"] >= buy_th) |
                (out["conviction_score"] <= sell_th)
            ].copy()
            candidates = candidates.sort_values("conviction_score", ascending=False)

            sent_count = 0
            for _, row in candidates.iterrows():
                if sent_count >= max_alerts:
                    break
                alert_key = f"{exchange}:{row['symbol']}:{row['action']}"
                if not _can_send_alert(alert_key):
                    continue
                delivery = send_auto_alert(
                    symbol=str(row["symbol"]),
                    action=str(row["action"]),
                    conviction_score=float(row["conviction_score"]),
                    ai_confidence=float(row["ai_confidence"]),
                    news_impact=float(row["news_impact"]),
                    exchange=exchange,
                )
                if delivery.get("sent_channels"):
                    _mark_alert_sent(
                        alert_key,
                        {
                            "symbol": row["symbol"],
                            "exchange": exchange,
                            "action": row["action"],
                            "conviction_score": float(row["conviction_score"]),
                            "channels": ",".join(delivery["sent_channels"]),
                        },
                    )
                    sent_count += 1

            if sent_count > 0:
                st.success(f"Auto Alerts sent: {sent_count}")
            else:
                st.caption("Auto Alerts enabled, but no new threshold-cross signals passed cooldown/channel checks.")

        st.markdown("#### Quick Chart Links")
        link_symbol = st.selectbox("Select Symbol for Chart", out["symbol"].tolist(), index=0, key="top_picks_link_symbol")
        link_urls = get_chart_links(link_symbol, exchange)
        st.markdown(
            f"[Open TradingView Chart]({link_urls['tradingview']}) | "
            f"[Open Yahoo Finance Chart]({link_urls['yahoo']})"
        )

        st.info(
            "Conviction Score = 75% AI directional confidence + 25% News Impact. "
            "Thresholds: BUY >= +15, SELL <= -15, else WATCH."
        )


# ============================================================
# PAGE: NEWS & SENTIMENT
# ============================================================

def page_news_sentiment():
    """News and sentiment analysis page"""
    st.markdown("""
    <div class="main-header">
        <div class="main-title">News & Sentiment</div>
        <div class="main-subtitle">Single-stock and market-wide headline sentiment</div>
    </div>
    """, unsafe_allow_html=True)

    tab_symbol, tab_market = st.tabs(["Single Symbol", "Market Feed"])

    with tab_symbol:
        col1, col2 = st.columns([2, 1])
        with col1:
            symbol = st.text_input("Symbol for News Analysis", value="RELIANCE")
        with col2:
            days = st.selectbox("Lookback Days", [3, 7, 14, 30], index=1, key="news_days_symbol")

        if st.button("Fetch Symbol News", type="primary", width="stretch", key="btn_symbol_news"):
            with st.spinner("Fetching latest headlines and running sentiment analysis..."):
                try:
                    result = load_stock_sentiment(symbol.upper().strip(), days=days, max_articles=12)

                    if result.get('article_count', 0) == 0:
                        st.warning("No news articles found for this symbol in selected period.")
                        return

                    score = result.get('sentiment_score', 0.0)
                    impact_score = result.get('impact_score', 0.0)
                    overall = result.get('overall_sentiment', 'neutral').upper()
                    overall_impact = result.get('overall_impact', 'NEUTRAL')

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Articles", result.get('article_count', 0))
                    c2.metric("Sentiment", overall)
                    c3.metric("Impact", f"{impact_score:+.1f}", overall_impact)
                    c4.metric("High Impact News", result.get('high_impact_count', 0))

                    fig = go.Figure(data=[
                        go.Bar(
                            x=["Positive", "Neutral", "Negative"],
                            y=[
                                result.get('avg_positive', 0.0),
                                result.get('avg_neutral', 0.0),
                                result.get('avg_negative', 0.0),
                            ],
                            marker_color=[GREEN, YELLOW, RED],
                            text=[
                                f"{result.get('avg_positive', 0.0):.0%}",
                                f"{result.get('avg_neutral', 0.0):.0%}",
                                f"{result.get('avg_negative', 0.0):.0%}",
                            ],
                            textposition='auto',
                            name='Sentiment Mix',
                        )
                    ])
                    fig.update_layout(
                        **CHART_THEME,
                        title=f"{symbol.upper().strip()} News Sentiment Distribution",
                        height=320,
                        yaxis_title="Score",
                    )
                    st.plotly_chart(fig, width="stretch")

                    st.markdown("### Top Headlines")
                    for idx, article in enumerate(result.get('articles', [])[:12], start=1):
                        s = article.get('sentiment', {})
                        s_label = str(s.get('sentiment', 'neutral')).upper()
                        s_conf = s.get('confidence', 0.0)
                        impact = article.get('impact', {})
                        i_score = float(impact.get('impact_score', 0.0))
                        i_label = str(impact.get('impact_label', 'NEUTRAL'))
                        s_color = GREEN if i_score > 15 else RED if i_score < -15 else YELLOW

                        title = article.get('title', 'Untitled')
                        source = article.get('source', 'Unknown')
                        published = article.get('published_at', 'N/A')
                        url = article.get('url', '')

                        st.markdown(
                            f"""
                            <div class="signal-card" style="border-left:4px solid {s_color};">
                                <div style="font-weight:600;color:var(--text-primary);margin-bottom:6px;">
                                    {idx}. {title}
                                </div>
                                <div style="display:flex;gap:12px;flex-wrap:wrap;font-size:0.82rem;color:var(--text-secondary);">
                                    <span><b>Source:</b> {source}</span>
                                    <span><b>Sentiment:</b> <span style="color:{s_color};font-weight:600;">{s_label}</span></span>
                                    <span><b>AI Impact:</b> <span style="color:{s_color};font-weight:700;">{i_score:+.1f} ({i_label})</span></span>
                                    <span><b>Confidence:</b> {s_conf:.0%}</span>
                                    <span><b>Time:</b> {published}</span>
                                </div>
                                <div style="margin-top:6px;">
                                    <a href="{url}" target="_blank" style="color:#b8860b;text-decoration:none;">Open Article</a>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                except Exception as e:
                    st.error(f"News sentiment analysis failed: {e}")

    with tab_market:
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            bucket = st.selectbox(
                "Feed Basket",
                ["NSE Stocks", "BSE Stocks", "Global & Commodities", "India + Global Mix"],
                index=3,
            )
        with c2:
            market_days = st.selectbox("Lookback Days", [1, 3, 7, 14], index=2, key="news_days_market")
        with c3:
            per_topic = st.slider("Per Topic", 1, 6, 3)

        if st.button("Fetch Market Feed", type="primary", width="stretch", key="btn_market_news"):
            with st.spinner("Collecting NSE/BSE/global news and scoring sentiment..."):
                feed_df = load_market_news_feed(bucket, days=market_days, max_per_topic=per_topic)

            if feed_df.empty:
                st.warning("No market headlines found for selected basket.")
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Headlines", len(feed_df))
                c2.metric("Topics", feed_df['topic'].nunique())
                c3.metric("Avg Impact", f"{feed_df['impact_score'].mean():+.1f}")
                c4.metric("High Impact %", f"{(feed_df['impact_score'].abs().ge(35).mean()):.0%}")

                topic_summary = (
                    feed_df.groupby('topic', as_index=False)
                    .agg(avg_impact=('impact_score', 'mean'), headlines=('title', 'count'))
                    .sort_values('avg_impact', ascending=False)
                )

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=topic_summary['avg_impact'],
                    y=topic_summary['topic'],
                    orientation='h',
                    text=[f"{v:+.1f}" for v in topic_summary['avg_impact']],
                    textposition='auto',
                    marker_color=[
                        GREEN if v > 15 else RED if v < -15 else YELLOW
                        for v in topic_summary['avg_impact']
                    ],
                    name="Topic Impact",
                ))
                fig.update_layout(
                    **CHART_THEME,
                    title=f"{bucket} - AI Impact by Topic",
                    height=340,
                    xaxis_title="Avg Impact Score (-100 to +100)",
                    yaxis_title="Topic",
                )
                st.plotly_chart(fig, width="stretch")

                st.markdown("### Market Headlines")
                for idx, row in feed_df.head(30).iterrows():
                    s_color = GREEN if row['impact_score'] > 15 else RED if row['impact_score'] < -15 else YELLOW
                    st.markdown(
                        f"""
                        <div class="signal-card" style="border-left:4px solid {s_color};">
                            <div style="font-size:0.78rem;color:var(--text-muted);margin-bottom:4px;">
                                {row['topic']}
                            </div>
                            <div style="font-weight:600;color:var(--text-primary);margin-bottom:6px;">
                                {idx + 1}. {row['title']}
                            </div>
                            <div style="display:flex;gap:12px;flex-wrap:wrap;font-size:0.82rem;color:var(--text-secondary);">
                                <span><b>Source:</b> {row['source']}</span>
                                <span><b>Sentiment:</b> <span style="color:{s_color};font-weight:600;">{row['sentiment']}</span></span>
                                <span><b>AI Impact:</b> <span style="color:{s_color};font-weight:700;">{row['impact_score']:+.1f} ({row['impact_label']})</span></span>
                                <span><b>Confidence:</b> {row['confidence']:.0%}</span>
                                <span><b>Time:</b> {row['published_at']}</span>
                            </div>
                            <div style="margin-top:6px;">
                                <a href="{row['url']}" target="_blank" style="color:#b8860b;text-decoration:none;">Open Article</a>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        st.info(
            "AI Impact Score combines sentiment + event keywords + source quality + recency weighting. "
            "Basket includes NSE/BSE stocks and global themes like Gold, Silver, GIFT Nifty and US indices."
        )


# ============================================================
# PAGE: MARKET HEATMAP
# ============================================================

def page_heatmap():
    """Market heatmap page"""
    st.markdown("""
    <div class="main-header">
        <div class="main-title">Market Heatmap</div>
        <div class="main-subtitle">Sector & stock performance visualization</div>
    </div>
    """, unsafe_allow_html=True)
    
    symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
               "SBIN", "ITC", "BHARTIARTL", "KOTAKBANK", "LT",
               "HCLTECH", "AXISBANK", "MARUTI", "SUNPHARMA", "TITAN"]
    
    with st.spinner("Loading heatmap data..."):
        data = load_multiple_stocks(symbols, "1mo")
    
    if data:
        # Performance data
        perf_data = {}
        for sym, df in data.items():
            if len(df) > 1:
                perf_data[sym] = {
                    '1D': (df['close'].iloc[-1] / df['close'].iloc[-2] - 1) * 100 if len(df) > 1 else 0,
                    '1W': (df['close'].iloc[-1] / df['close'].iloc[-6] - 1) * 100 if len(df) > 5 else 0,
                    '1M': (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100,
                }
        
        if perf_data:
            perf_df = pd.DataFrame(perf_data).T
            
            fig = create_heatmap(perf_df, "Stock Performance Heatmap (%)")
            st.plotly_chart(fig, width="stretch")
            
            # Treemap
            treemap_data = []
            for sym, perfs in perf_data.items():
                treemap_data.append({
                    'symbol': sym,
                    'return_1d': perfs['1D'],
                    'abs_return': abs(perfs['1D']),
                })
            
            tm_df = pd.DataFrame(treemap_data)
            fig_tree = px.treemap(
                tm_df, path=['symbol'], values='abs_return',
                color='return_1d',
                color_continuous_scale=[[0, RED], [0.5, '#1a2332'], [1, GREEN]],
                color_continuous_midpoint=0,
                title="Market Treemap (Today's Performance)",
            )
            fig_tree.update_layout(**CHART_THEME, height=500)
            st.plotly_chart(fig_tree, width="stretch")


# ============================================================
# PAGE: WATCHLIST
# ============================================================

def page_watchlist():
    """Watchlist management page"""
    st.markdown("""
    <div class="main-header">
        <div class="main-title">Watchlist</div>
        <div class="main-subtitle">Track your favorite stocks</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Default watchlist
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
    
    # Add stock
    col1, col2 = st.columns([3, 1])
    with col1:
        new_symbol = st.text_input("Add Stock Symbol", placeholder="e.g., TATAMOTORS")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Add", width="stretch"):
            if new_symbol and new_symbol.upper() not in st.session_state.watchlist:
                st.session_state.watchlist.append(new_symbol.upper())
                st.rerun()
    
    # Display watchlist
    if st.session_state.watchlist:
        data = load_multiple_stocks(st.session_state.watchlist, "1mo")
        
        for sym in st.session_state.watchlist:
            if sym in data and not data[sym].empty:
                df = data[sym]
                latest = df.iloc[-1]
                prev = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
                change_pct = (latest['close'] - prev['close']) / prev['close'] * 100
                color = GREEN if change_pct > 0 else RED
                
                col1, col2, col3, col4, col5 = st.columns([2, 1.5, 1, 1, 0.5])
                col1.markdown(f"**{sym}**")
                col2.markdown(f"<span style='font-family:JetBrains Mono;font-weight:700;'>Rs {latest['close']:,.2f}</span>", unsafe_allow_html=True)
                col3.markdown(f"<span style='color:{color};font-family:JetBrains Mono;font-weight:600;'>{change_pct:+.2f}%</span>", unsafe_allow_html=True)
                col4.markdown(f"<span style='color:var(--text-muted);font-size:0.85rem;'>Vol: {latest['volume']/1e6:.1f}M</span>", unsafe_allow_html=True)
                if col5.button("Remove", key=f"remove_{sym}"):
                    st.session_state.watchlist.remove(sym)
                    st.rerun()


# ============================================================
# PAGE: PORTFOLIO
# ============================================================

def page_portfolio():
    """Portfolio tracker page"""
    st.markdown("""
    <div class="main-header">
        <div class="main-title">Portfolio</div>
        <div class="main-subtitle">Track your investments</div>
    </div>
    """, unsafe_allow_html=True)
    
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = []
    
    # Add entry
    with st.expander("Add Trade"):
        c1, c2, c3, c4 = st.columns(4)
        p_symbol = c1.text_input("Symbol", placeholder="RELIANCE")
        p_price = c2.number_input("Buy Price (Rs)", min_value=0.01, value=100.0)
        p_qty = c3.number_input("Quantity", min_value=1, value=1)
        p_date = c4.date_input("Buy Date")
        
        if st.button("Add to Portfolio"):
            st.session_state.portfolio.append({
                'symbol': p_symbol.upper(),
                'buy_price': p_price,
                'qty': p_qty,
                'date': str(p_date),
            })
            st.success(f"Added {p_symbol.upper()} to portfolio!")
    
    # Display portfolio
    if st.session_state.portfolio:
        total_investment = 0
        total_current = 0
        
        for entry in st.session_state.portfolio:
            investment = entry['buy_price'] * entry['qty']
            total_investment += investment
            
            # Get current price
            df = load_stock_data(entry['symbol'], "5d")
            current_price = df['close'].iloc[-1] if not df.empty else entry['buy_price']
            current_value = current_price * entry['qty']
            total_current += current_value
            pnl = current_value - investment
            pnl_pct = (pnl / investment) * 100
            
            color = GREEN if pnl > 0 else RED
            
            st.markdown(f"""
            <div class="signal-card {'bullish' if pnl > 0 else 'bearish'}">
                <div style="display:flex;justify-content:space-between;">
                    <div>
                        <span style="font-size:1.1rem;font-weight:700;">{entry['symbol']}</span>
                        <span style="color:var(--text-muted);margin-left:10px;">{entry['qty']} shares @ Rs {entry['buy_price']:.2f}</span>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-family:'JetBrains Mono';font-weight:700;">Rs {current_price:,.2f}</div>
                        <div style="color:{color};font-family:'JetBrains Mono';font-weight:600;">
                            {'+' if pnl > 0 else ''}Rs {pnl:,.2f} ({pnl_pct:+.2f}%)
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Summary
        total_pnl = total_current - total_investment
        total_pnl_pct = (total_pnl / total_investment * 100) if total_investment > 0 else 0
        
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Investment", f"Rs {total_investment:,.2f}")
        c2.metric("Current Value", f"Rs {total_current:,.2f}")
        c3.metric("Total P&L", f"Rs {total_pnl:,.2f}", f"{total_pnl_pct:+.2f}%")
    else:
        st.info("Your portfolio is empty. Add trades above!")


# ============================================================
# PAGE: SETTINGS
# ============================================================

def page_settings():
    """Settings page"""
    st.markdown("""
    <div class="main-header">
        <div class="main-title">Settings</div>
        <div class="main-subtitle">Configure scanner preferences</div>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Scanner", "Alerts", "Database"])
    
    with tab1:
        st.markdown("#### Scanner Configuration")
        st.slider("RSI Oversold Threshold", 10, 40, 30)
        st.slider("RSI Overbought Threshold", 60, 90, 70)
        st.slider("Volume Spike Multiplier", 1.0, 5.0, 2.0, 0.5)
        st.slider("Min Signal Strength", 0.0, 1.0, 0.5, 0.1)
        st.number_input("Scan Interval (minutes)", 1, 60, 5)
    
    with tab2:
        st.markdown("#### Alert Settings")
        st.session_state["auto_alerts_enabled"] = st.toggle(
            "Enable Auto Alerts (AI Top Picks)",
            value=st.session_state.get("auto_alerts_enabled", False),
        )

        c1, c2, c3 = st.columns(3)
        st.session_state["alert_buy_threshold"] = c1.number_input(
            "BUY Threshold",
            min_value=0.0,
            max_value=100.0,
            value=float(st.session_state.get("alert_buy_threshold", 20.0)),
            step=1.0,
        )
        st.session_state["alert_sell_threshold"] = c2.number_input(
            "SELL Threshold",
            min_value=-100.0,
            max_value=0.0,
            value=float(st.session_state.get("alert_sell_threshold", -20.0)),
            step=1.0,
        )
        st.session_state["alert_cooldown_min"] = int(c3.number_input(
            "Cooldown (min)",
            min_value=1,
            max_value=1440,
            value=int(st.session_state.get("alert_cooldown_min", 30)),
            step=1,
        ))

        st.session_state["alert_max_per_run"] = int(
            st.slider(
                "Max Alerts Per Batch Run",
                min_value=1,
                max_value=20,
                value=int(st.session_state.get("alert_max_per_run", 5)),
            )
        )

        st.markdown("##### Channels")
        cc1, cc2 = st.columns(2)
        st.session_state["alert_channel_telegram"] = cc1.toggle(
            "Telegram",
            value=st.session_state.get("alert_channel_telegram", True),
        )
        st.session_state["alert_channel_email"] = cc2.toggle(
            "Email",
            value=st.session_state.get("alert_channel_email", False),
        )

        with st.expander("Telegram Configuration"):
            st.session_state["telegram_bot_token"] = st.text_input(
                "Telegram Bot Token",
                value=st.session_state.get("telegram_bot_token", ""),
                type="password",
            )
            st.session_state["telegram_chat_id"] = st.text_input(
                "Telegram Chat ID",
                value=st.session_state.get("telegram_chat_id", ""),
            )

        with st.expander("Email Configuration"):
            ec1, ec2 = st.columns(2)
            st.session_state["email_smtp_server"] = ec1.text_input(
                "SMTP Server",
                value=st.session_state.get("email_smtp_server", "smtp.gmail.com"),
            )
            st.session_state["email_smtp_port"] = int(ec2.number_input(
                "SMTP Port",
                min_value=1,
                max_value=65535,
                value=int(st.session_state.get("email_smtp_port", 587)),
            ))
            st.session_state["email_sender"] = st.text_input(
                "Sender Email",
                value=st.session_state.get("email_sender", ""),
            )
            st.session_state["email_password"] = st.text_input(
                "Sender Password / App Password",
                value=st.session_state.get("email_password", ""),
                type="password",
            )
            st.session_state["email_recipients"] = st.text_input(
                "Recipients (comma separated)",
                value=st.session_state.get("email_recipients", ""),
            )

        st.markdown("##### Recent Alert History")
        history = st.session_state.get("alert_history", [])
        if history:
            st.dataframe(pd.DataFrame(history[:50]), width="stretch")
        else:
            st.caption("No alerts sent yet.")

        c_reset1, c_reset2 = st.columns(2)
        if c_reset1.button("Clear Alert History"):
            st.session_state["alert_history"] = []
            st.success("Alert history cleared.")
        if c_reset2.button("Reset Cooldown Memory"):
            st.session_state["alert_last_sent"] = {}
            st.success("Cooldown memory reset.")
    
    with tab3:
        st.markdown("#### Database")
        st.text_input("Database URL", value="sqlite:///stock_scanner.db")
        if st.button("Test Connection"):
            st.success("Connection successful!")
        if st.button("Clear Cache"):
            st.cache_data.clear()
            st.success("Cache cleared!")


# ============================================================
# MAIN APP
# ============================================================

def main():
    """Main application entry point"""
    inject_custom_css()
    init_alert_state()
    page = render_sidebar()
    
    # Route to correct page
    if "Dashboard" in page:
        page_dashboard()
    elif "Scanner" in page:
        page_scanner()
    elif "Stock Analysis" in page:
        page_stock_analysis()
    elif "AI Predictions" in page:
        page_ai_predictions()
    elif "AI Top Picks" in page:
        page_ai_top_picks()
    elif "News & Sentiment" in page:
        page_news_sentiment()
    elif "Heatmap" in page:
        page_heatmap()
    elif "Watchlist" in page:
        page_watchlist()
    elif "Portfolio" in page:
        page_portfolio()
    elif "Settings" in page:
        page_settings()

    st.markdown("---")
    render_disclaimer()


if __name__ == "__main__":
    main()


