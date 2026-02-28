"""
Breakout Scanner - Finds stocks breaking out of consolidation/resistance
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

from scanners.momentum_scanner import ScanResult

logger = logging.getLogger(__name__)


class BreakoutScanner:
    """Scan for breakout setups"""
    
    def __init__(self, config=None):
        from config.settings import scanner_config
        self.config = config or scanner_config
    
    def scan_52week_high_breakout(self, df: pd.DataFrame, symbol: str) -> Optional[ScanResult]:
        """Detect stocks breaking 52-week highs"""
        if len(df) < 252:
            return None
        
        high_52w = df['high'].rolling(252).max()
        current_price = df['close'].iloc[-1]
        prev_high_52w = high_52w.iloc[-2]
        
        if current_price > prev_high_52w:
            volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
            
            return ScanResult(
                symbol=symbol,
                signal="52_WEEK_HIGH_BREAKOUT",
                strength=0.9 if volume_ratio > 1.5 else 0.7,
                price=current_price,
                change_pct=df['close'].pct_change().iloc[-1] * 100,
                volume_ratio=volume_ratio,
                reasons=[
                    f"â­â­ Breaking 52-week high! Price: â‚¹{current_price:.2f}",
                    f"Previous 52W high: â‚¹{prev_high_52w:.2f}",
                    f"Volume: {volume_ratio:.1f}x average" + (" (strong)" if volume_ratio > 1.5 else ""),
                ]
            )
        
        return None
    
    def scan_resistance_breakout(
        self,
        df: pd.DataFrame,
        symbol: str,
        lookback: int = 20
    ) -> Optional[ScanResult]:
        """Detect price breaking above recent resistance"""
        resistance = df['high'].rolling(lookback).max()
        current_close = df['close'].iloc[-1]
        prev_resistance = resistance.iloc[-2]
        
        # Check if price was below resistance and now breaks above
        was_below = df['close'].iloc[-2] <= prev_resistance
        now_above = current_close > prev_resistance
        
        if was_below and now_above:
            volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
            
            if volume_ratio >= self.config.BREAKOUT_VOLUME_FACTOR:
                return ScanResult(
                    symbol=symbol,
                    signal="RESISTANCE_BREAKOUT",
                    strength=min(0.5 + volume_ratio * 0.1, 1.0),
                    price=current_close,
                    change_pct=df['close'].pct_change().iloc[-1] * 100,
                    volume_ratio=volume_ratio,
                    reasons=[
                        f"Breaking {lookback}-day resistance at â‚¹{prev_resistance:.2f}",
                        f"Volume confirmation: {volume_ratio:.1f}x average",
                        f"Price: â‚¹{current_close:.2f}",
                    ]
                )
        
        return None
    
    def scan_bollinger_squeeze_breakout(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> Optional[ScanResult]:
        """Detect Bollinger Band squeeze breakout"""
        if 'BB_Width' not in df.columns:
            return None
        
        # Squeeze: BB width at multi-week low
        bb_width = df['BB_Width']
        bb_width_percentile = bb_width.rolling(100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )
        
        if bb_width_percentile.iloc[-2] < 0.1:  # Was in squeeze
            current_close = df['close'].iloc[-1]
            bb_upper = df['BB_Upper'].iloc[-1]
            bb_lower = df['BB_Lower'].iloc[-1]
            
            if current_close > bb_upper:
                volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
                
                return ScanResult(
                    symbol=symbol,
                    signal="BB_SQUEEZE_BULLISH",
                    strength=0.8,
                    price=current_close,
                    change_pct=df['close'].pct_change().iloc[-1] * 100,
                    volume_ratio=volume_ratio,
                    reasons=[
                        "â­ Bollinger Band Squeeze Breakout (Bullish)",
                        f"BB Width was in bottom 10% (tight squeeze)",
                        f"Price broke above upper BB: â‚¹{bb_upper:.2f}",
                        "Expect increased volatility and momentum",
                    ]
                )
            
            elif current_close < bb_lower:
                volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
                
                return ScanResult(
                    symbol=symbol,
                    signal="BB_SQUEEZE_BEARISH",
                    strength=0.8,
                    price=current_close,
                    change_pct=df['close'].pct_change().iloc[-1] * 100,
                    volume_ratio=volume_ratio,
                    reasons=[
                        "âš ï¸ Bollinger Band Squeeze Breakdown (Bearish)",
                        f"Price broke below lower BB: â‚¹{bb_lower:.2f}",
                    ]
                )
        
        return None
    
    def scan_volume_breakout(self, df: pd.DataFrame, symbol: str) -> Optional[ScanResult]:
        """Detect stocks with unusual volume and price movement"""
        volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
        price_change = df['close'].pct_change().iloc[-1] * 100
        
        if volume_ratio >= self.config.VOLUME_SPIKE_MULTIPLIER and price_change > 2:
            return ScanResult(
                symbol=symbol,
                signal="VOLUME_BREAKOUT",
                strength=min(volume_ratio / 5, 1.0),
                price=df['close'].iloc[-1],
                change_pct=price_change,
                volume_ratio=volume_ratio,
                reasons=[
                    f"ðŸ”¥ Volume spike: {volume_ratio:.1f}x average",
                    f"Price up {price_change:.2f}%",
                    f"Volume: {df['volume'].iloc[-1]:,.0f}",
                    f"Average volume: {df['volume'].rolling(20).mean().iloc[-1]:,.0f}",
                ]
            )
        
        return None
    
    def scan_consolidation_breakout(
        self,
        df: pd.DataFrame,
        symbol: str,
        lookback: int = 20
    ) -> Optional[ScanResult]:
        """Detect breakout from tight consolidation range"""
        recent = df.tail(lookback)
        high = recent['high'].max()
        low = recent['low'].min()
        range_pct = (high - low) / low * 100
        
        current_close = df['close'].iloc[-1]
        volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
        
        # Was in tight range and now breaking out
        if range_pct < self.config.CONSOLIDATION_RANGE_PCT:
            if current_close > high and volume_ratio > 1.3:
                return ScanResult(
                    symbol=symbol,
                    signal="CONSOLIDATION_BREAKOUT",
                    strength=0.85,
                    price=current_close,
                    change_pct=df['close'].pct_change().iloc[-1] * 100,
                    volume_ratio=volume_ratio,
                    reasons=[
                        f"â­ Breaking out of {lookback}-day consolidation",
                        f"Consolidation range: {range_pct:.1f}% (tight)",
                        f"Breakout above â‚¹{high:.2f} with volume",
                        f"Volume: {volume_ratio:.1f}x average",
                    ]
                )
        
        return None
    
    def scan_gap_up(self, df: pd.DataFrame, symbol: str) -> Optional[ScanResult]:
        """Detect significant gap-up openings"""
        gap = (df['open'].iloc[-1] / df['close'].iloc[-2] - 1) * 100
        
        if gap > 2:  # More than 2% gap up
            volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
            
            return ScanResult(
                symbol=symbol,
                signal="GAP_UP",
                strength=min(gap / 10, 1.0),
                price=df['close'].iloc[-1],
                change_pct=df['close'].pct_change().iloc[-1] * 100,
                volume_ratio=volume_ratio,
                reasons=[
                    f"Gap up: {gap:.2f}%",
                    f"Open: â‚¹{df['open'].iloc[-1]:.2f}, Prev Close: â‚¹{df['close'].iloc[-2]:.2f}",
                    f"Volume: {volume_ratio:.1f}x average",
                ]
            )
        
        return None
    
    def run_all_scans(
        self,
        stock_data: Dict[str, pd.DataFrame]
    ) -> List[ScanResult]:
        """Run all breakout scans"""
        results = []
        
        scan_functions = [
            self.scan_52week_high_breakout,
            self.scan_resistance_breakout,
            self.scan_bollinger_squeeze_breakout,
            self.scan_volume_breakout,
            self.scan_consolidation_breakout,
            self.scan_gap_up,
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
