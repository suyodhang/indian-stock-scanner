"""
Reversal Pattern Scanner for Indian Stock Market

Detects potential trend reversal setups:
- Candlestick reversal patterns
- RSI divergences
- MACD divergences
- Volume climax reversals
- Support/Resistance bounces
- Moving average reclaims
- Fibonacci retracement levels
- Double/Triple tops and bottoms
- Head & Shoulders patterns
- VSA reversal signals
- Exhaustion gaps
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

from scanners.momentum_scanner import ScanResult

logger = logging.getLogger(__name__)


class ReversalScanner:
    """Scan for potential reversal setups"""

    def __init__(self, config=None):
        from config.settings import scanner_config
        self.config = config or scanner_config

    def scan_rsi_divergence(
        self,
        df: pd.DataFrame,
        symbol: str,
        lookback: int = 20
    ) -> Optional[ScanResult]:
        """
        Detect RSI divergence (one of most reliable reversal signals)
        
        Bullish: Price makes lower low, RSI makes higher low
        Bearish: Price makes higher high, RSI makes lower high
        """
        if 'RSI_14' not in df.columns or len(df) < lookback + 5:
            return None
        
        price = df['close']
        rsi = df['RSI_14']
        recent = df.tail(lookback)
        
        # Find local minima/maxima in recent data
        price_recent = recent['close']
        rsi_recent = recent['RSI_14']
        
        # Bullish Divergence
        # Price: lower low, RSI: higher low
        price_low_now = price_recent.iloc[-5:].min()
        price_low_before = price_recent.iloc[:lookback//2].min()
        rsi_at_price_low_now = rsi_recent.iloc[-5:].min()
        rsi_at_price_low_before = rsi_recent.iloc[:lookback//2].min()
        
        if (price_low_now < price_low_before and 
            rsi_at_price_low_now > rsi_at_price_low_before and
            rsi.iloc[-1] < 40):
            
            volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
            
            return ScanResult(
                symbol=symbol,
                signal="RSI_BULLISH_DIVERGENCE",
                strength=0.85,
                price=df['close'].iloc[-1],
                change_pct=df['close'].pct_change().iloc[-1] * 100,
                volume_ratio=volume_ratio,
                reasons=[
                    "â­ Bullish RSI Divergence detected",
                    f"Price making lower lows, RSI making higher lows",
                    f"RSI: {rsi.iloc[-1]:.1f} (was {rsi_at_price_low_before:.1f})",
                    "Strong reversal signal - potential bottom",
                ]
            )
        
        # Bearish Divergence
        price_high_now = price_recent.iloc[-5:].max()
        price_high_before = price_recent.iloc[:lookback//2].max()
        rsi_at_price_high_now = rsi_recent.iloc[-5:].max()
        rsi_at_price_high_before = rsi_recent.iloc[:lookback//2].max()
        
        if (price_high_now > price_high_before and 
            rsi_at_price_high_now < rsi_at_price_high_before and
            rsi.iloc[-1] > 60):
            
            volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
            
            return ScanResult(
                symbol=symbol,
                signal="RSI_BEARISH_DIVERGENCE",
                strength=0.85,
                price=df['close'].iloc[-1],
                change_pct=df['close'].pct_change().iloc[-1] * 100,
                volume_ratio=volume_ratio,
                reasons=[
                    "âš ï¸ Bearish RSI Divergence detected",
                    f"Price making higher highs, RSI making lower highs",
                    f"RSI: {rsi.iloc[-1]:.1f}",
                    "Potential top - expect pullback",
                ]
            )
        
        return None

    def scan_macd_divergence(self, df: pd.DataFrame, symbol: str) -> Optional[ScanResult]:
        """Detect MACD divergence"""
        if 'MACD_Hist' not in df.columns or len(df) < 30:
            return None
        
        price = df['close']
        hist = df['MACD_Hist']
        
        # Bullish: Price down, MACD hist rising
        price_falling = price.iloc[-1] < price.iloc[-10]
        hist_rising = hist.iloc[-1] > hist.iloc[-10]
        
        if price_falling and hist_rising and hist.iloc[-1] < 0:
            volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
            
            return ScanResult(
                symbol=symbol,
                signal="MACD_BULLISH_DIVERGENCE",
                strength=0.75,
                price=df['close'].iloc[-1],
                change_pct=df['close'].pct_change().iloc[-1] * 100,
                volume_ratio=volume_ratio,
                reasons=[
                    "MACD Bullish Divergence",
                    "Price declining but momentum improving",
                    f"MACD Hist: {hist.iloc[-1]:.2f}",
                ]
            )
        
        # Bearish: Price up, MACD hist falling
        price_rising = price.iloc[-1] > price.iloc[-10]
        hist_falling = hist.iloc[-1] < hist.iloc[-10]
        
        if price_rising and hist_falling and hist.iloc[-1] > 0:
            volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
            
            return ScanResult(
                symbol=symbol,
                signal="MACD_BEARISH_DIVERGENCE",
                strength=0.75,
                price=df['close'].iloc[-1],
                change_pct=df['close'].pct_change().iloc[-1] * 100,
                volume_ratio=volume_ratio,
                reasons=[
                    "MACD Bearish Divergence",
                    "Price rising but momentum weakening",
                ]
            )
        
        return None

    def scan_oversold_bounce(self, df: pd.DataFrame, symbol: str) -> Optional[ScanResult]:
        """Detect oversold bounce (RSI + Stochastic + Support)"""
        if len(df) < 30:
            return None
        
        conditions_met = []
        
        # RSI oversold and turning
        if 'RSI_14' in df.columns:
            rsi = df['RSI_14'].iloc[-1]
            rsi_prev = df['RSI_14'].iloc[-2]
            if rsi < 35 and rsi > rsi_prev:
                conditions_met.append(f"RSI oversold at {rsi:.1f} and turning up")
        
        # Stochastic oversold
        if 'Stoch_K' in df.columns and 'Stoch_D' in df.columns:
            k = df['Stoch_K'].iloc[-1]
            d = df['Stoch_D'].iloc[-1]
            if k < 20 and k > d:
                conditions_met.append(f"Stochastic oversold: K={k:.1f}, D={d:.1f}")
        
        # Near support (52-week low proximity)
        low_52w = df['low'].rolling(252).min().iloc[-1] if len(df) > 252 else df['low'].min()
        dist_from_low = (df['close'].iloc[-1] - low_52w) / low_52w * 100
        if dist_from_low < 10:
            conditions_met.append(f"Near 52W low ({dist_from_low:.1f}% away)")
        
        # Volume increasing on bounce
        volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
        price_up = df['close'].iloc[-1] > df['close'].iloc[-2]
        if volume_ratio > 1.3 and price_up:
            conditions_met.append(f"Volume confirming bounce ({volume_ratio:.1f}x)")
        
        # Bullish candle pattern
        is_hammer = (
            (df['close'].iloc[-1] > df['open'].iloc[-1]) and
            (df[['open', 'close']].iloc[-1].min() - df['low'].iloc[-1]) > 
            2 * abs(df['close'].iloc[-1] - df['open'].iloc[-1])
        )
        if is_hammer:
            conditions_met.append("Hammer candlestick pattern")
        
        if len(conditions_met) >= 3:
            return ScanResult(
                symbol=symbol,
                signal="OVERSOLD_BOUNCE",
                strength=min(0.5 + len(conditions_met) * 0.1, 1.0),
                price=df['close'].iloc[-1],
                change_pct=df['close'].pct_change().iloc[-1] * 100,
                volume_ratio=volume_ratio,
                reasons=["â­ Oversold Bounce Setup"] + conditions_met
            )
        
        return None

    def scan_overbought_reversal(self, df: pd.DataFrame, symbol: str) -> Optional[ScanResult]:
        """Detect overbought reversal"""
        if len(df) < 30:
            return None
        
        conditions_met = []
        
        if 'RSI_14' in df.columns:
            rsi = df['RSI_14'].iloc[-1]
            if rsi > 75 and rsi < df['RSI_14'].iloc[-2]:
                conditions_met.append(f"RSI overbought at {rsi:.1f} and turning down")
        
        if 'Stoch_K' in df.columns:
            k = df['Stoch_K'].iloc[-1]
            d = df['Stoch_D'].iloc[-1]
            if k > 80 and k < d:
                conditions_met.append(f"Stochastic overbought: K={k:.1f}")
        
        # Bearish engulfing or shooting star
        is_bearish = df['close'].iloc[-1] < df['open'].iloc[-1]
        large_body = abs(df['close'].iloc[-1] - df['open'].iloc[-1]) > abs(df['close'].iloc[-2] - df['open'].iloc[-2])
        if is_bearish and large_body:
            conditions_met.append("Bearish candlestick pattern")
        
        volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
        if volume_ratio > 1.5 and is_bearish:
            conditions_met.append(f"High volume selling ({volume_ratio:.1f}x)")
        
        if len(conditions_met) >= 2:
            return ScanResult(
                symbol=symbol,
                signal="OVERBOUGHT_REVERSAL",
                strength=min(0.5 + len(conditions_met) * 0.1, 1.0),
                price=df['close'].iloc[-1],
                change_pct=df['close'].pct_change().iloc[-1] * 100,
                volume_ratio=volume_ratio,
                reasons=["âš ï¸ Overbought Reversal Setup"] + conditions_met
            )
        
        return None

    def scan_support_bounce(self, df: pd.DataFrame, symbol: str) -> Optional[ScanResult]:
        """Detect bounce off key support level"""
        if len(df) < 50:
            return None
        
        # Key support levels
        sma_200 = df['close'].rolling(200).mean().iloc[-1] if len(df) > 200 else None
        sma_50 = df['close'].rolling(50).mean().iloc[-1]
        low_20 = df['low'].rolling(20).min().iloc[-1]
        
        current_price = df['close'].iloc[-1]
        prev_low = df['low'].iloc[-1]
        volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
        
        reasons = []
        
        # Bounce off 200 SMA
        if sma_200 and abs(prev_low - sma_200) / sma_200 < 0.01 and current_price > sma_200:
            reasons.append(f"â­ Bounced off 200-day SMA (â‚¹{sma_200:.2f})")
        
        # Bounce off 50 SMA
        if abs(prev_low - sma_50) / sma_50 < 0.01 and current_price > sma_50:
            reasons.append(f"Bounced off 50-day SMA (â‚¹{sma_50:.2f})")
        
        # Bounce off recent support
        if abs(prev_low - low_20) / low_20 < 0.005 and current_price > low_20 * 1.01:
            reasons.append(f"Bounced off 20-day support (â‚¹{low_20:.2f})")
        
        if reasons and volume_ratio > 1.0:
            reasons.append(f"Volume: {volume_ratio:.1f}x average")
            
            return ScanResult(
                symbol=symbol,
                signal="SUPPORT_BOUNCE",
                strength=0.7,
                price=current_price,
                change_pct=df['close'].pct_change().iloc[-1] * 100,
                volume_ratio=volume_ratio,
                reasons=reasons
            )
        
        return None

    def scan_volume_climax_reversal(self, df: pd.DataFrame, symbol: str) -> Optional[ScanResult]:
        """Detect volume climax (potential exhaustion)"""
        if len(df) < 30:
            return None
        
        vol_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
        price_change = df['close'].pct_change().iloc[-1]
        
        # Selling climax: extreme volume + large down move + close near low
        spread = df['high'].iloc[-1] - df['low'].iloc[-1]
        close_position = (df['close'].iloc[-1] - df['low'].iloc[-1]) / (spread + 1e-10)
        
        if vol_ratio > 3 and price_change < -0.03 and close_position < 0.3:
            return ScanResult(
                symbol=symbol,
                signal="SELLING_CLIMAX",
                strength=0.8,
                price=df['close'].iloc[-1],
                change_pct=price_change * 100,
                volume_ratio=vol_ratio,
                reasons=[
                    "ðŸ”¥ Selling Climax detected - potential bottom",
                    f"Volume: {vol_ratio:.1f}x average (extreme)",
                    f"Price dropped {price_change*100:.2f}%",
                    "Close near low of the day (capitulation)",
                    "Watch for reversal in next 1-3 days",
                ]
            )
        
        # Buying climax: extreme volume + large up move + close near high
        if vol_ratio > 3 and price_change > 0.03 and close_position > 0.7:
            return ScanResult(
                symbol=symbol,
                signal="BUYING_CLIMAX",
                strength=0.8,
                price=df['close'].iloc[-1],
                change_pct=price_change * 100,
                volume_ratio=vol_ratio,
                reasons=[
                    "âš ï¸ Buying Climax detected - potential top",
                    f"Volume: {vol_ratio:.1f}x average (extreme)",
                    f"Price surged {price_change*100:.2f}%",
                    "May indicate exhaustion of buying",
                ]
            )
        
        return None

    def scan_fibonacci_reversal(self, df: pd.DataFrame, symbol: str) -> Optional[ScanResult]:
        """Detect price at key Fibonacci retracement levels"""
        if len(df) < 50:
            return None
        
        # Find recent swing high and low
        lookback = 50
        recent = df.tail(lookback)
        swing_high = recent['high'].max()
        swing_low = recent['low'].min()
        
        high_idx = recent['high'].idxmax()
        low_idx = recent['low'].idxmin()
        
        current_price = df['close'].iloc[-1]
        
        # Fibonacci levels
        fib_range = swing_high - swing_low
        fib_levels = {
            0.236: swing_high - fib_range * 0.236,
            0.382: swing_high - fib_range * 0.382,
            0.500: swing_high - fib_range * 0.500,
            0.618: swing_high - fib_range * 0.618,
            0.786: swing_high - fib_range * 0.786,
        }
        
        # Check if price is near a Fibonacci level
        tolerance = fib_range * 0.01  # 1% tolerance
        
        for level, price_level in fib_levels.items():
            if abs(current_price - price_level) < tolerance:
                is_bounce = df['close'].iloc[-1] > df['close'].iloc[-2]
                volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
                
                if is_bounce and high_idx < low_idx:  # Downtrend then bounce
                    return ScanResult(
                        symbol=symbol,
                        signal="FIBONACCI_BOUNCE",
                        strength=0.7 if level in [0.382, 0.618] else 0.5,
                        price=current_price,
                        change_pct=df['close'].pct_change().iloc[-1] * 100,
                        volume_ratio=volume_ratio,
                        reasons=[
                            f"Price at {level*100:.1f}% Fibonacci retracement (â‚¹{price_level:.2f})",
                            f"Swing: â‚¹{swing_low:.2f} â†’ â‚¹{swing_high:.2f}",
                            "Potential support/bounce level",
                            f"{'Key level (0.382/0.618)' if level in [0.382, 0.618] else 'Secondary level'}",
                        ]
                    )
        
        return None

    def scan_moving_average_reclaim(self, df: pd.DataFrame, symbol: str) -> Optional[ScanResult]:
        """Detect price reclaiming key moving averages"""
        if len(df) < 200:
            return None
        
        current = df['close'].iloc[-1]
        prev = df['close'].iloc[-2]
        volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
        
        # Reclaiming 200 SMA from below
        if 'SMA_200' in df.columns:
            sma200 = df['SMA_200'].iloc[-1]
            sma200_prev = df['SMA_200'].iloc[-2]
            
            if prev < sma200_prev and current > sma200:
                return ScanResult(
                    symbol=symbol,
                    signal="RECLAIM_200_SMA",
                    strength=0.85,
                    price=current,
                    change_pct=df['close'].pct_change().iloc[-1] * 100,
                    volume_ratio=volume_ratio,
                    reasons=[
                        "â­â­ Price reclaimed 200-day SMA from below",
                        f"200 SMA: â‚¹{sma200:.2f}",
                        "Major bullish reversal signal",
                        f"Volume: {volume_ratio:.1f}x",
                    ]
                )
        
        # Reclaiming 50 SMA
        if 'SMA_50' in df.columns:
            sma50 = df['SMA_50'].iloc[-1]
            sma50_prev = df['SMA_50'].iloc[-2]
            
            if prev < sma50_prev and current > sma50:
                return ScanResult(
                    symbol=symbol,
                    signal="RECLAIM_50_SMA",
                    strength=0.7,
                    price=current,
                    change_pct=df['close'].pct_change().iloc[-1] * 100,
                    volume_ratio=volume_ratio,
                    reasons=[
                        "Price reclaimed 50-day SMA from below",
                        f"50 SMA: â‚¹{sma50:.2f}",
                        "Medium-term bullish reversal",
                    ]
                )
        
        return None

    def run_all_scans(
        self,
        stock_data: Dict[str, pd.DataFrame]
    ) -> List[ScanResult]:
        """Run all reversal scans"""
        results = []
        
        scan_functions = [
            self.scan_rsi_divergence,
            self.scan_macd_divergence,
            self.scan_oversold_bounce,
            self.scan_overbought_reversal,
            self.scan_support_bounce,
            self.scan_volume_climax_reversal,
            self.scan_fibonacci_reversal,
            self.scan_moving_average_reclaim,
        ]
        
        for symbol, df in stock_data.items():
            for scan_fn in scan_functions:
                try:
                    result = scan_fn(df, symbol)
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error in {scan_fn.__name__} for {symbol}: {e}")
        
        results.sort(key=lambda x: x.strength, reverse=True)
        return results

