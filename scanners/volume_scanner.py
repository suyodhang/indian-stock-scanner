"""
Volume-Based Scanner for Indian Stock Market

Scans for:
- Volume breakouts
- Volume dry-ups (pre-breakout)
- Accumulation patterns
- Distribution patterns
- Delivery volume signals (NSE specific)
- Smart money activity
- Volume-price divergences
- Unusual volume activity
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

from scanners.momentum_scanner import ScanResult

logger = logging.getLogger(__name__)


class VolumeScanner:
    """Scan for volume-based trading setups"""

    def __init__(self, config=None):
        from config.settings import scanner_config
        self.config = config or scanner_config

    def scan_volume_explosion(self, df: pd.DataFrame, symbol: str) -> Optional[ScanResult]:
        """Detect sudden volume explosion with price movement"""
        if len(df) < 25:
            return None
        
        vol_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
        price_change = df['close'].pct_change().iloc[-1] * 100
        
        if vol_ratio >= 3.0 and abs(price_change) > 1:
            direction = "BULLISH" if price_change > 0 else "BEARISH"
            
            return ScanResult(
                symbol=symbol,
                signal=f"VOLUME_EXPLOSION_{direction}",
                strength=min(vol_ratio / 5, 1.0),
                price=df['close'].iloc[-1],
                change_pct=price_change,
                volume_ratio=vol_ratio,
                reasons=[
                    f"ðŸ”¥ Volume Explosion: {vol_ratio:.1f}x average",
                    f"Price: {'+' if price_change > 0 else ''}{price_change:.2f}%",
                    f"Volume: {df['volume'].iloc[-1]:,.0f}",
                    f"20-day avg: {df['volume'].rolling(20).mean().iloc[-1]:,.0f}",
                    f"Direction: {direction}",
                ]
            )
        
        return None

    def scan_volume_dryup(self, df: pd.DataFrame, symbol: str) -> Optional[ScanResult]:
        """
        Detect volume dry-up (potential pre-breakout)
        
        Low volume + tight range = energy building up
        """
        if len(df) < 25:
            return None
        
        vol_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
        
        # Volume dry for multiple days
        dry_days = (df['volume'].tail(5) < df['volume'].rolling(20).mean().tail(5) * 0.5).sum()
        
        # Tight price range
        range_5d = (df['high'].tail(5).max() - df['low'].tail(5).min()) / df['close'].iloc[-1] * 100
        
        if dry_days >= 3 and range_5d < 3:
            return ScanResult(
                symbol=symbol,
                signal="VOLUME_DRYUP_BUILDUP",
                strength=0.65,
                price=df['close'].iloc[-1],
                change_pct=df['close'].pct_change().iloc[-1] * 100,
                volume_ratio=vol_ratio,
                reasons=[
                    "ðŸ“‰ Volume Dry-Up detected",
                    f"{dry_days}/5 days with below-average volume",
                    f"Tight range: {range_5d:.2f}% over 5 days",
                    "Energy building up - watch for breakout",
                    "Set alerts for volume spike",
                ]
            )
        
        return None

    def scan_accumulation(self, df: pd.DataFrame, symbol: str) -> Optional[ScanResult]:
        """Detect smart money accumulation"""
        if len(df) < 30:
            return None
        
        recent = df.tail(20)
        
        # Higher volume on up days
        up_days = recent[recent['close'] > recent['open']]
        down_days = recent[recent['close'] <= recent['open']]
        
        if len(up_days) == 0 or len(down_days) == 0:
            return None
        
        avg_up_vol = up_days['volume'].mean()
        avg_down_vol = down_days['volume'].mean()
        vol_ratio_ud = avg_up_vol / (avg_down_vol + 1)
        
        # Higher lows
        lows = recent['low']
        higher_lows = sum(lows.iloc[i] > lows.iloc[i-1] for i in range(1, len(lows))) / (len(lows) - 1)
        
        # OBV rising
        obv_rising = False
        if 'OBV' in df.columns:
            obv_slope = df['OBV'].iloc[-1] - df['OBV'].iloc[-20]
            obv_rising = obv_slope > 0
        
        volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
        
        conditions = []
        score = 0
        
        if vol_ratio_ud > 1.3:
            conditions.append(f"Up-day volume {vol_ratio_ud:.1f}x down-day volume")
            score += 0.3
        
        if higher_lows > 0.6:
            conditions.append(f"Higher lows pattern ({higher_lows:.0%} of days)")
            score += 0.3
        
        if obv_rising:
            conditions.append("OBV trending up (buying pressure)")
            score += 0.2
        
        # CMF positive
        if 'CMF_20' in df.columns and df['CMF_20'].iloc[-1] > 0.05:
            conditions.append(f"Positive money flow (CMF: {df['CMF_20'].iloc[-1]:.2f})")
            score += 0.2
        
        if score >= 0.5:
            return ScanResult(
                symbol=symbol,
                signal="ACCUMULATION",
                strength=min(score, 1.0),
                price=df['close'].iloc[-1],
                change_pct=df['close'].pct_change().iloc[-1] * 100,
                volume_ratio=volume_ratio,
                reasons=["ðŸŸ¢ Accumulation Pattern Detected"] + conditions
            )
        
        return None

    def scan_distribution(self, df: pd.DataFrame, symbol: str) -> Optional[ScanResult]:
        """Detect distribution (smart money selling)"""
        if len(df) < 30:
            return None
        
        recent = df.tail(20)
        
        up_days = recent[recent['close'] > recent['open']]
        down_days = recent[recent['close'] <= recent['open']]
        
        if len(up_days) == 0 or len(down_days) == 0:
            return None
        
        avg_up_vol = up_days['volume'].mean()
        avg_down_vol = down_days['volume'].mean()
        vol_ratio_du = avg_down_vol / (avg_up_vol + 1)
        
        # Lower highs
        highs = recent['high']
        lower_highs = sum(highs.iloc[i] < highs.iloc[i-1] for i in range(1, len(highs))) / (len(highs) - 1)
        
        volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
        
        conditions = []
        score = 0
        
        if vol_ratio_du > 1.3:
            conditions.append(f"Down-day volume {vol_ratio_du:.1f}x up-day volume")
            score += 0.3
        
        if lower_highs > 0.6:
            conditions.append(f"Lower highs pattern ({lower_highs:.0%} of days)")
            score += 0.3
        
        if 'CMF_20' in df.columns and df['CMF_20'].iloc[-1] < -0.05:
            conditions.append(f"Negative money flow (CMF: {df['CMF_20'].iloc[-1]:.2f})")
            score += 0.2
        
        if 'OBV' in df.columns:
            obv_slope = df['OBV'].iloc[-1] - df['OBV'].iloc[-20]
            if obv_slope < 0:
                conditions.append("OBV declining (selling pressure)")
                score += 0.2
        
        if score >= 0.5:
            return ScanResult(
                symbol=symbol,
                signal="DISTRIBUTION",
                strength=min(score, 1.0),
                price=df['close'].iloc[-1],
                change_pct=df['close'].pct_change().iloc[-1] * 100,
                volume_ratio=volume_ratio,
                reasons=["ðŸ”´ Distribution Pattern Detected"] + conditions
            )
        
        return None

    def scan_delivery_volume(self, df: pd.DataFrame, symbol: str) -> Optional[ScanResult]:
        """
        Scan delivery volume (NSE India specific)
        
        High delivery + price up = Genuine buying
        """
        if 'delivery_pct' not in df.columns or len(df) < 20:
            return None
        
        delivery = df['delivery_pct'].iloc[-1]
        avg_delivery = df['delivery_pct'].rolling(20).mean().iloc[-1]
        price_change = df['close'].pct_change().iloc[-1] * 100
        volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
        
        if delivery > 65 and delivery > avg_delivery * 1.3 and price_change > 1:
            return ScanResult(
                symbol=symbol,
                signal="HIGH_DELIVERY_BUYING",
                strength=min(delivery / 100, 1.0),
                price=df['close'].iloc[-1],
                change_pct=price_change,
                volume_ratio=volume_ratio,
                reasons=[
                    f"â­ High Delivery Volume: {delivery:.1f}%",
                    f"Average: {avg_delivery:.1f}%",
                    f"Price up {price_change:.2f}%",
                    "Genuine institutional buying interest",
                    f"Volume: {volume_ratio:.1f}x average",
                ]
            )
        
        return None

    def scan_price_volume_divergence(self, df: pd.DataFrame, symbol: str) -> Optional[ScanResult]:
        """Detect price-volume divergence"""
        if len(df) < 20:
            return None
        
        price_trend = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10]
        vol_trend = (df['volume'].iloc[-5:].mean() - df['volume'].iloc[-15:-5].mean()) / df['volume'].iloc[-15:-5].mean()
        
        volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
        
        # Price rising but volume falling (bearish divergence)
        if price_trend > 0.03 and vol_trend < -0.2:
            return ScanResult(
                symbol=symbol,
                signal="PRICE_VOL_BEARISH_DIVERGENCE",
                strength=0.65,
                price=df['close'].iloc[-1],
                change_pct=df['close'].pct_change().iloc[-1] * 100,
                volume_ratio=volume_ratio,
                reasons=[
                    "âš ï¸ Price-Volume Bearish Divergence",
                    f"Price rising ({price_trend*100:.1f}%) but volume declining",
                    "Rally may be losing steam",
                    "Watch for potential pullback",
                ]
            )
        
        # Price falling but volume falling (potential bottom)
        if price_trend < -0.03 and vol_trend < -0.3:
            return ScanResult(
                symbol=symbol,
                signal="PRICE_VOL_BULLISH_DIVERGENCE",
                strength=0.6,
                price=df['close'].iloc[-1],
                change_pct=df['close'].pct_change().iloc[-1] * 100,
                volume_ratio=volume_ratio,
                reasons=[
                    "ðŸŸ¢ Price declining but selling pressure reducing",
                    "Potential bottom forming",
                    "Volume drying up on down days",
                ]
            )
        
        return None

    def run_all_scans(
        self,
        stock_data: Dict[str, pd.DataFrame]
    ) -> List[ScanResult]:
        """Run all volume scans"""
        results = []
        
        scan_functions = [
            self.scan_volume_explosion,
            self.scan_volume_dryup,
            self.scan_accumulation,
            self.scan_distribution,
            self.scan_delivery_volume,
            self.scan_price_volume_divergence,
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

