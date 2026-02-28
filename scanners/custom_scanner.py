"""
Custom Multi-Condition Scanner with AI Integration
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
import logging

from scanners.momentum_scanner import ScanResult
from scanners.breakout_scanner import BreakoutScanner
from ai_models.trend_predictor import TrendPredictor
from ai_models.breakout_detector import BreakoutDetector

logger = logging.getLogger(__name__)


@dataclass
class ScanCondition:
    """Define a single scan condition"""
    name: str
    column: str
    operator: str  # 'gt', 'lt', 'eq', 'gte', 'lte', 'cross_above', 'cross_below'
    value: float = None
    column2: str = None  # For cross conditions
    weight: float = 1.0


class CustomScanner:
    """
    Highly customizable scanner that combines technical, 
    pattern, and AI signals
    """
    
    def __init__(self):
        self.trend_predictor = TrendPredictor()
        self.breakout_detector = BreakoutDetector()
    
    def evaluate_condition(
        self,
        df: pd.DataFrame,
        condition: ScanCondition
    ) -> bool:
        """Evaluate a single scan condition"""
        if condition.column not in df.columns:
            return False
        
        current_val = df[condition.column].iloc[-1]
        
        if condition.operator in ('cross_above', 'cross_below'):
            if condition.column2 not in df.columns:
                return False
            current_val2 = df[condition.column2].iloc[-1]
            prev_val = df[condition.column].iloc[-2]
            prev_val2 = df[condition.column2].iloc[-2]
            
            if condition.operator == 'cross_above':
                return current_val > current_val2 and prev_val <= prev_val2
            else:
                return current_val < current_val2 and prev_val >= prev_val2
        
        if condition.value is None:
            return False
        
        ops = {
            'gt': current_val > condition.value,
            'lt': current_val < condition.value,
            'eq': abs(current_val - condition.value) < 0.001,
            'gte': current_val >= condition.value,
            'lte': current_val <= condition.value,
        }
        
        return ops.get(condition.operator, False)
    
    def multi_condition_scan(
        self,
        stock_data: Dict[str, pd.DataFrame],
        conditions: List[ScanCondition],
        min_conditions_met: int = None
    ) -> List[ScanResult]:
        """
        Scan stocks against multiple conditions
        
        Args:
            stock_data: Dictionary of symbol -> DataFrame
            conditions: List of conditions to evaluate
            min_conditions_met: Minimum conditions that must be true (default: all)
        """
        if min_conditions_met is None:
            min_conditions_met = len(conditions)
        
        results = []
        
        for symbol, df in stock_data.items():
            try:
                met_conditions = []
                total_weight = 0
                met_weight = 0
                
                for condition in conditions:
                    total_weight += condition.weight
                    if self.evaluate_condition(df, condition):
                        met_conditions.append(condition.name)
                        met_weight += condition.weight
                
                if len(met_conditions) >= min_conditions_met:
                    strength = met_weight / total_weight
                    volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
                    
                    results.append(ScanResult(
                        symbol=symbol,
                        signal="CUSTOM_MULTI_CONDITION",
                        strength=strength,
                        price=df['close'].iloc[-1],
                        change_pct=df['close'].pct_change().iloc[-1] * 100,
                        volume_ratio=volume_ratio,
                        reasons=[f"âœ“ {cond}" for cond in met_conditions]
                    ))
            
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
        
        results.sort(key=lambda x: x.strength, reverse=True)
        return results
    
    def ai_enhanced_scan(
        self,
        stock_data: Dict[str, pd.DataFrame],
        confidence_threshold: float = 0.65
    ) -> List[ScanResult]:
        """
        Run AI-enhanced scan combining technical and ML predictions
        """
        results = []
        
        for symbol, df in stock_data.items():
            try:
                # Technical score
                tech_score = self._calculate_technical_score(df)
                
                # AI prediction
                ai_prediction = None
                if self.trend_predictor.is_trained:
                    X, _ = self.trend_predictor.prepare_features(df)
                    if not X.empty:
                        ai_prediction = self.trend_predictor.predict(X)
                
                # Breakout detection
                breakout_result = None
                if self.breakout_detector.is_trained:
                    breakout_result = self.breakout_detector.predict_breakout(df)
                
                # Combine scores
                combined_score = tech_score
                reasons = []
                
                if ai_prediction and ai_prediction['confidence'] > confidence_threshold:
                    ai_weight = ai_prediction['confidence']
                    if ai_prediction['prediction'] == 'BULLISH':
                        combined_score = combined_score * 0.5 + ai_weight * 0.5
                        reasons.append(
                            f"ðŸ¤– AI: BULLISH ({ai_prediction['confidence']:.0%} confidence, "
                            f"{ai_prediction['bullish_votes']}/{ai_prediction['total_models']} models agree)"
                        )
                    else:
                        combined_score = combined_score * 0.5 + (1 - ai_weight) * 0.5
                        reasons.append(
                            f"ðŸ¤– AI: BEARISH ({ai_prediction['confidence']:.0%} confidence)"
                        )
                
                if breakout_result and breakout_result['probability'] > 0.5:
                    reasons.append(
                        f"ðŸš€ Breakout probability: {breakout_result['probability']:.0%} ({breakout_result['signal']})"
                    )
                    combined_score = min(combined_score + 0.1, 1.0)
                
                # Add technical reasons
                tech_reasons = self._get_technical_reasons(df)
                reasons.extend(tech_reasons)
                
                if combined_score > 0.6:  # Only report significant signals
                    volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
                    
                    results.append(ScanResult(
                        symbol=symbol,
                        signal="AI_ENHANCED",
                        strength=combined_score,
                        price=df['close'].iloc[-1],
                        change_pct=df['close'].pct_change().iloc[-1] * 100,
                        volume_ratio=volume_ratio,
                        reasons=reasons,
                        ai_prediction=ai_prediction
                    ))
            
            except Exception as e:
                logger.error(f"Error in AI scan for {symbol}: {e}")
        
        results.sort(key=lambda x: x.strength, reverse=True)
        return results
    
    def _calculate_technical_score(self, df: pd.DataFrame) -> float:
        """Calculate composite technical score (0 to 1, where 1 = most bullish)"""
        score = 0.5  # Start neutral
        factors = 0
        
        # RSI
        if 'RSI_14' in df.columns:
            rsi = df['RSI_14'].iloc[-1]
            if rsi < 30:
                score += 0.1  # Oversold = bullish potential
            elif rsi > 70:
                score -= 0.1
            elif 40 < rsi < 60:
                score += 0.05  # Neutral zone
            factors += 1
        
        # MACD
        if 'MACD_Hist' in df.columns:
            hist = df['MACD_Hist'].iloc[-1]
            prev_hist = df['MACD_Hist'].iloc[-2]
            if hist > 0 and hist > prev_hist:
                score += 0.15
            elif hist < 0 and hist < prev_hist:
                score -= 0.15
            factors += 1
        
        # Moving average alignment
        if all(col in df.columns for col in ['EMA_9', 'EMA_20', 'SMA_50', 'SMA_200']):
            price = df['close'].iloc[-1]
            if price > df['EMA_9'].iloc[-1] > df['EMA_20'].iloc[-1] > df['SMA_50'].iloc[-1]:
                score += 0.2  # Strong bullish alignment
            elif price < df['EMA_9'].iloc[-1] < df['EMA_20'].iloc[-1] < df['SMA_50'].iloc[-1]:
                score -= 0.2
            factors += 1
        
        # SuperTrend
        if 'ST_Direction' in df.columns:
            if df['ST_Direction'].iloc[-1] == 1:
                score += 0.1
            else:
                score -= 0.1
            factors += 1
        
        # Volume
        volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
        price_change = df['close'].pct_change().iloc[-1]
        
        if volume_ratio > 1.5 and price_change > 0:
            score += 0.1  # Bullish volume
        elif volume_ratio > 1.5 and price_change < 0:
            score -= 0.1  # Bearish volume
        
        # ADX
        if 'ADX' in df.columns and 'Plus_DI' in df.columns:
            if df['ADX'].iloc[-1] > 25 and df['Plus_DI'].iloc[-1] > df['Minus_DI'].iloc[-1]:
                score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _get_technical_reasons(self, df: pd.DataFrame) -> List[str]:
        """Generate human-readable technical reasons"""
        reasons = []
        
        if 'RSI_14' in df.columns:
            rsi = df['RSI_14'].iloc[-1]
            reasons.append(f"RSI(14): {rsi:.1f}")
        
        if 'MACD_Hist' in df.columns:
            hist = df['MACD_Hist'].iloc[-1]
            reasons.append(f"MACD Histogram: {'Positive â†‘' if hist > 0 else 'Negative â†“'}")
        
        if 'ST_Direction' in df.columns:
            direction = "Bullish â†‘" if df['ST_Direction'].iloc[-1] == 1 else "Bearish â†“"
            reasons.append(f"SuperTrend: {direction}")
        
        if 'ADX' in df.columns:
            adx = df['ADX'].iloc[-1]
            trend_strength = "Strong" if adx > 25 else "Weak"
            reasons.append(f"ADX: {adx:.1f} ({trend_strength} trend)")
        
        return reasons


# =========================================
# Pre-built Scanner Presets
# =========================================

def get_bullish_scanner_conditions() -> List[ScanCondition]:
    """Pre-defined bullish scanner conditions"""
    return [
        ScanCondition("RSI < 40 (not overbought)", "RSI_14", "lt", 40, weight=1.0),
        ScanCondition("MACD Histogram positive", "MACD_Hist", "gt", 0, weight=1.5),
        ScanCondition("Price above 50 SMA", "Dist_SMA_50", "gt", 0, weight=1.0),
        ScanCondition("Price above 200 SMA", "Dist_SMA_200", "gt", 0, weight=1.0),
        ScanCondition("SuperTrend bullish", "ST_Direction", "eq", 1, weight=1.5),
        ScanCondition("ADX > 20 (trending)", "ADX", "gt", 20, weight=0.8),
        ScanCondition("Volume above average", "Volume_Ratio", "gt", 1.0, weight=0.5),
    ]


def get_bearish_scanner_conditions() -> List[ScanCondition]:
    """Pre-defined bearish scanner conditions"""
    return [
        ScanCondition("RSI > 60", "RSI_14", "gt", 60, weight=1.0),
        ScanCondition("MACD Histogram negative", "MACD_Hist", "lt", 0, weight=1.5),
        ScanCondition("Price below 50 SMA", "Dist_SMA_50", "lt", 0, weight=1.0),
        ScanCondition("SuperTrend bearish", "ST_Direction", "eq", -1, weight=1.5),
        ScanCondition("ADX > 20 (trending)", "ADX", "gt", 20, weight=0.8),
    ]


def get_swing_trade_conditions() -> List[ScanCondition]:
    """Conditions for swing trading setups"""
    return [
        ScanCondition("RSI between 40-60", "RSI_14", "gt", 40, weight=0.5),
        ScanCondition("RSI not overbought", "RSI_14", "lt", 60, weight=0.5),
        ScanCondition("Bollinger Band squeeze", "BB_Width", "lt", 0.05, weight=2.0),
        ScanCondition("Volume building up", "Volume_Ratio", "gt", 1.2, weight=1.0),
        ScanCondition("ADX rising (trend starting)", "ADX", "gt", 15, weight=1.0),
    ]
