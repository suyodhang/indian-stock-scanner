"""
Market regime detection utilities.
"""

from typing import Dict
import pandas as pd
import numpy as np


def detect_regime(df: pd.DataFrame) -> Dict:
    """
    Detect market regime from technical columns.
    Returns dict with regime label and confidence [0,1].
    """
    if df is None or df.empty:
        return {"regime": "UNKNOWN", "confidence": 0.0}

    latest = df.iloc[-1]

    adx = float(latest.get("ADX", np.nan))
    atr = float(latest.get("ATR_14", np.nan))
    close = float(latest.get("close", np.nan))
    bb_width = float(latest.get("BB_Width", np.nan))
    rsi = float(latest.get("RSI_14", np.nan))

    # Volatility proxy
    vol_ratio = 0.0
    if np.isfinite(atr) and np.isfinite(close) and close > 0:
        vol_ratio = atr / close
    if np.isfinite(bb_width) and bb_width > 0:
        vol_ratio = max(vol_ratio, bb_width / 100.0)

    # Trend strength proxy
    trend_strength = 0.0
    if np.isfinite(adx):
        trend_strength = min(max(adx / 50.0, 0.0), 1.0)
    if np.isfinite(rsi):
        trend_strength = max(trend_strength, min(abs(rsi - 50.0) / 30.0, 1.0))

    if vol_ratio >= 0.035:
        regime = "HIGH_VOLATILITY"
        confidence = min(1.0, 0.6 + vol_ratio * 8)
    elif trend_strength >= 0.55:
        regime = "TRENDING"
        confidence = min(1.0, 0.5 + trend_strength * 0.5)
    else:
        regime = "RANGE_BOUND"
        confidence = min(1.0, 0.55 + (0.55 - trend_strength) * 0.6)

    return {
        "regime": regime,
        "confidence": float(round(confidence, 3)),
        "vol_ratio": float(round(vol_ratio, 4)),
        "trend_strength": float(round(trend_strength, 4)),
    }

