"""
Momentum & Trend Scanner
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    """Individual scan result"""
    symbol: str
    signal: str
    strength: float  # 0.0 to 1.0
    price: float
    change_pct: float
    volume_ratio: float
    reasons: List[str]
    ai_prediction: Optional[Dict] = None
    timestamp: str = ""


class MomentumScanner:
    """Scan for momentum-based setups"""
    
    def __init__(self, config=None):
        from config.settings import scanner_config
        self.config = config or scanner_config
    
    def scan_rsi_oversold(self, df: pd.DataFrame, symbol: str) -> Optional[ScanResult]:
        """Find RSI oversold stocks (potential bounce)"""
        if 'RSI_14' not in df.columns or len(df) < 20:
            return None
        
        current_rsi = df['RSI_14'].iloc[-1]
        prev_rsi = df['RSI_14'].iloc[-2]
        
        if current_rsi < self.config.RSI_OVERSOLD and current_rsi > prev_rsi:
            # RSI oversold and turning up
            reasons = [
                f"RSI({current_rsi:.1f}) below {self.config.RSI_OVERSOLD}",
                "RSI showing reversal (turning up)",
            ]
            
            # Check for bullish divergence
            if df['close'].iloc[-1] < df['close'].iloc[-5] and current_rsi > df['RSI_14'].iloc[-5]:
                reasons.append("â­ Bullish RSI divergence detected")
            
            # Volume confirmation
            volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
            if volume_ratio > 1.5:
                reasons.append(f"Volume spike ({volume_ratio:.1f}x average)")
            
            return ScanResult(
                symbol=symbol,
                signal="RSI_OVERSOLD_BOUNCE",
                strength=min((self.config.RSI_OVERSOLD - current_rsi) / 20, 1.0),
                price=df['close'].iloc[-1],
                change_pct=df['close'].pct_change().iloc[-1] * 100,
                volume_ratio=volume_ratio,
                reasons=reasons
            )
        
        return None
    
    def scan_rsi_overbought(self, df: pd.DataFrame, symbol: str) -> Optional[ScanResult]:
        """Find RSI overbought stocks (potential pullback)"""
        if 'RSI_14' not in df.columns:
            return None
        
        current_rsi = df['RSI_14'].iloc[-1]
        prev_rsi = df['RSI_14'].iloc[-2]
        
        if current_rsi > self.config.RSI_OVERBOUGHT and current_rsi < prev_rsi:
            reasons = [
                f"RSI({current_rsi:.1f}) above {self.config.RSI_OVERBOUGHT}",
                "RSI showing reversal (turning down)",
            ]
            
            if df['close'].iloc[-1] > df['close'].iloc[-5] and current_rsi < df['RSI_14'].iloc[-5]:
                reasons.append("â­ Bearish RSI divergence detected")
            
            volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
            
            return ScanResult(
                symbol=symbol,
                signal="RSI_OVERBOUGHT_PULLBACK",
                strength=min((current_rsi - self.config.RSI_OVERBOUGHT) / 20, 1.0),
                price=df['close'].iloc[-1],
                change_pct=df['close'].pct_change().iloc[-1] * 100,
                volume_ratio=volume_ratio,
                reasons=reasons
            )
        
        return None
    
    def scan_macd_crossover(self, df: pd.DataFrame, symbol: str) -> Optional[ScanResult]:
        """Detect MACD bullish/bearish crossovers"""
        if 'MACD' not in df.columns or 'MACD_Signal' not in df.columns:
            return None
        
        macd = df['MACD']
        signal = df['MACD_Signal']
        
        # Bullish crossover
        if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]:
            reasons = [
                "MACD bullish crossover (MACD crossed above Signal)",
                f"MACD: {macd.iloc[-1]:.2f}, Signal: {signal.iloc[-1]:.2f}",
            ]
            
            if macd.iloc[-1] < 0:
                reasons.append("Crossover below zero line (early momentum)")
            
            volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
            
            return ScanResult(
                symbol=symbol,
                signal="MACD_BULLISH_CROSSOVER",
                strength=0.7,
                price=df['close'].iloc[-1],
                change_pct=df['close'].pct_change().iloc[-1] * 100,
                volume_ratio=volume_ratio,
                reasons=reasons
            )
        
        # Bearish crossover
        if macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]:
            reasons = [
                "MACD bearish crossover (MACD crossed below Signal)",
                f"MACD: {macd.iloc[-1]:.2f}, Signal: {signal.iloc[-1]:.2f}",
            ]
            
            volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
            
            return ScanResult(
                symbol=symbol,
                signal="MACD_BEARISH_CROSSOVER",
                strength=0.7,
                price=df['close'].iloc[-1],
                change_pct=df['close'].pct_change().iloc[-1] * 100,
                volume_ratio=volume_ratio,
                reasons=reasons
            )
        
        return None
    
    def scan_supertrend_signal(self, df: pd.DataFrame, symbol: str) -> Optional[ScanResult]:
        """Detect SuperTrend buy/sell signals"""
        if 'ST_Direction' not in df.columns:
            return None
        
        curr_dir = df['ST_Direction'].iloc[-1]
        prev_dir = df['ST_Direction'].iloc[-2]
        
        if curr_dir == 1 and prev_dir == -1:
            volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
            reasons = [
                "â­ SuperTrend BUY signal (trend turned bullish)",
                f"SuperTrend value: {df['SuperTrend'].iloc[-1]:.2f}",
            ]
            
            return ScanResult(
                symbol=symbol,
                signal="SUPERTREND_BUY",
                strength=0.8,
                price=df['close'].iloc[-1],
                change_pct=df['close'].pct_change().iloc[-1] * 100,
                volume_ratio=volume_ratio,
                reasons=reasons
            )
        
        if curr_dir == -1 and prev_dir == 1:
            volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
            reasons = [
                "âš ï¸ SuperTrend SELL signal (trend turned bearish)",
                f"SuperTrend value: {df['SuperTrend'].iloc[-1]:.2f}",
            ]
            
            return ScanResult(
                symbol=symbol,
                signal="SUPERTREND_SELL",
                strength=0.8,
                price=df['close'].iloc[-1],
                change_pct=df['close'].pct_change().iloc[-1] * 100,
                volume_ratio=volume_ratio,
                reasons=reasons
            )
        
        return None
    
    def scan_golden_cross(self, df: pd.DataFrame, symbol: str) -> Optional[ScanResult]:
        """Detect Golden Cross (50 SMA crosses above 200 SMA)"""
        if 'SMA_50' not in df.columns or 'SMA_200' not in df.columns:
            return None
        
        sma50 = df['SMA_50']
        sma200 = df['SMA_200']
        
        if sma50.iloc[-1] > sma200.iloc[-1] and sma50.iloc[-2] <= sma200.iloc[-2]:
            volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
            
            return ScanResult(
                symbol=symbol,
                signal="GOLDEN_CROSS",
                strength=0.9,
                price=df['close'].iloc[-1],
                change_pct=df['close'].pct_change().iloc[-1] * 100,
                volume_ratio=volume_ratio,
                reasons=[
                    "â­â­ GOLDEN CROSS: 50-day SMA crossed above 200-day SMA",
                    "Strong long-term bullish signal",
                    f"50-SMA: {sma50.iloc[-1]:.2f}, 200-SMA: {sma200.iloc[-1]:.2f}"
                ]
            )
        
        # Death Cross
        if sma50.iloc[-1] < sma200.iloc[-1] and sma50.iloc[-2] >= sma200.iloc[-2]:
            volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
            
            return ScanResult(
                symbol=symbol,
                signal="DEATH_CROSS",
                strength=0.9,
                price=df['close'].iloc[-1],
                change_pct=df['close'].pct_change().iloc[-1] * 100,
                volume_ratio=volume_ratio,
                reasons=[
                    "âš ï¸âš ï¸ DEATH CROSS: 50-day SMA crossed below 200-day SMA",
                    "Strong long-term bearish signal",
                    f"50-SMA: {sma50.iloc[-1]:.2f}, 200-SMA: {sma200.iloc[-1]:.2f}"
                ]
            )
        
        return None
    
    def scan_ema_crossover(self, df: pd.DataFrame, symbol: str) -> Optional[ScanResult]:
        """Detect EMA 9/20 crossover"""
        if 'EMA_9' not in df.columns or 'EMA_20' not in df.columns:
            return None
        
        ema9 = df['EMA_9']
        ema20 = df['EMA_20']
        
        if ema9.iloc[-1] > ema20.iloc[-1] and ema9.iloc[-2] <= ema20.iloc[-2]:
            volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
            
            return ScanResult(
                symbol=symbol,
                signal="EMA_9_20_BULLISH",
                strength=0.6,
                price=df['close'].iloc[-1],
                change_pct=df['close'].pct_change().iloc[-1] * 100,
                volume_ratio=volume_ratio,
                reasons=[
                    "EMA 9 crossed above EMA 20 (short-term bullish)",
                    f"EMA9: {ema9.iloc[-1]:.2f}, EMA20: {ema20.iloc[-1]:.2f}"
                ]
            )
        
        return None
    
    def scan_strong_trend(self, df: pd.DataFrame, symbol: str) -> Optional[ScanResult]:
        """Find stocks in strong uptrend"""
        if 'ADX' not in df.columns:
            return None
        
        adx = df['ADX'].iloc[-1]
        plus_di = df['Plus_DI'].iloc[-1] if 'Plus_DI' in df.columns else 0
        minus_di = df['Minus_DI'].iloc[-1] if 'Minus_DI' in df.columns else 0
        
        if adx > 25 and plus_di > minus_di:
            volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
            
            strength = min(adx / 50, 1.0)
            
            return ScanResult(
                symbol=symbol,
                signal="STRONG_UPTREND",
                strength=strength,
                price=df['close'].iloc[-1],
                change_pct=df['close'].pct_change().iloc[-1] * 100,
                volume_ratio=volume_ratio,
                reasons=[
                    f"Strong uptrend: ADX={adx:.1f}",
                    f"+DI={plus_di:.1f} > -DI={minus_di:.1f}",
                    "Trend is strong and bullish" if adx > 40 else "Trend is developing"
                ]
            )
        
        return None
    
    def run_all_scans(
        self,
        stock_data: Dict[str, pd.DataFrame]
    ) -> List[ScanResult]:
        """Run all momentum scans on a universe of stocks"""
        results = []
        
        scan_functions = [
            self.scan_rsi_oversold,
            self.scan_rsi_overbought,
            self.scan_macd_crossover,
            self.scan_supertrend_signal,
            self.scan_golden_cross,
            self.scan_ema_crossover,
            self.scan_strong_trend,
        ]
        
        for symbol, df in stock_data.items():
            for scan_fn in scan_functions:
                try:
                    result = scan_fn(df, symbol)
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error scanning {symbol} with {scan_fn.__name__}: {e}")
        
        # Sort by strength
        results.sort(key=lambda x: x.strength, reverse=True)
        
        return results
