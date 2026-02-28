"""
Candlestick & Chart Pattern Recognition
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class PatternResult:
    """Result of pattern detection"""
    name: str
    type: str  # 'bullish', 'bearish', 'neutral'
    confidence: float  # 0.0 to 1.0
    description: str
    date: str


class CandlestickPatterns:
    """Detect candlestick patterns"""
    
    @staticmethod
    def _body(df: pd.DataFrame) -> pd.Series:
        return df['close'] - df['open']
    
    @staticmethod
    def _body_abs(df: pd.DataFrame) -> pd.Series:
        return abs(df['close'] - df['open'])
    
    @staticmethod
    def _upper_shadow(df: pd.DataFrame) -> pd.Series:
        return df['high'] - df[['open', 'close']].max(axis=1)
    
    @staticmethod
    def _lower_shadow(df: pd.DataFrame) -> pd.Series:
        return df[['open', 'close']].min(axis=1) - df['low']
    
    @staticmethod
    def _is_bullish(df: pd.DataFrame) -> pd.Series:
        return df['close'] > df['open']
    
    @staticmethod
    def _is_bearish(df: pd.DataFrame) -> pd.Series:
        return df['close'] < df['open']
    
    @classmethod
    def detect_doji(cls, df: pd.DataFrame) -> pd.Series:
        """Detect Doji pattern"""
        body = cls._body_abs(df)
        total_range = df['high'] - df['low']
        return body / total_range < 0.1
    
    @classmethod
    def detect_hammer(cls, df: pd.DataFrame) -> pd.Series:
        """Detect Hammer pattern (bullish reversal)"""
        body = cls._body_abs(df)
        lower = cls._lower_shadow(df)
        upper = cls._upper_shadow(df)
        
        return (
            (lower >= 2 * body) &
            (upper <= body * 0.3) &
            (body > 0)
        )
    
    @classmethod
    def detect_inverted_hammer(cls, df: pd.DataFrame) -> pd.Series:
        """Detect Inverted Hammer pattern"""
        body = cls._body_abs(df)
        lower = cls._lower_shadow(df)
        upper = cls._upper_shadow(df)
        
        return (
            (upper >= 2 * body) &
            (lower <= body * 0.3) &
            (body > 0)
        )
    
    @classmethod
    def detect_engulfing_bullish(cls, df: pd.DataFrame) -> pd.Series:
        """Detect Bullish Engulfing pattern"""
        prev_bearish = cls._is_bearish(df).shift(1)
        curr_bullish = cls._is_bullish(df)
        
        return (
            prev_bearish &
            curr_bullish &
            (df['open'] <= df['close'].shift(1)) &
            (df['close'] >= df['open'].shift(1))
        )
    
    @classmethod
    def detect_engulfing_bearish(cls, df: pd.DataFrame) -> pd.Series:
        """Detect Bearish Engulfing pattern"""
        prev_bullish = cls._is_bullish(df).shift(1)
        curr_bearish = cls._is_bearish(df)
        
        return (
            prev_bullish &
            curr_bearish &
            (df['open'] >= df['close'].shift(1)) &
            (df['close'] <= df['open'].shift(1))
        )
    
    @classmethod
    def detect_morning_star(cls, df: pd.DataFrame) -> pd.Series:
        """Detect Morning Star pattern (3-candle bullish reversal)"""
        # Day 1: Large bearish candle
        day1_bearish = cls._is_bearish(df).shift(2)
        day1_large = cls._body_abs(df).shift(2) > cls._body_abs(df).shift(2).rolling(10).mean()
        
        # Day 2: Small body (gap down)
        day2_small = cls._body_abs(df).shift(1) < cls._body_abs(df).shift(1).rolling(10).mean() * 0.5
        
        # Day 3: Bullish candle closing above midpoint of day 1
        day3_bullish = cls._is_bullish(df)
        day3_closes_high = df['close'] > (df['open'].shift(2) + df['close'].shift(2)) / 2
        
        return day1_bearish & day1_large & day2_small & day3_bullish & day3_closes_high
    
    @classmethod
    def detect_evening_star(cls, df: pd.DataFrame) -> pd.Series:
        """Detect Evening Star pattern (3-candle bearish reversal)"""
        day1_bullish = cls._is_bullish(df).shift(2)
        day1_large = cls._body_abs(df).shift(2) > cls._body_abs(df).shift(2).rolling(10).mean()
        
        day2_small = cls._body_abs(df).shift(1) < cls._body_abs(df).shift(1).rolling(10).mean() * 0.5
        
        day3_bearish = cls._is_bearish(df)
        day3_closes_low = df['close'] < (df['open'].shift(2) + df['close'].shift(2)) / 2
        
        return day1_bullish & day1_large & day2_small & day3_bearish & day3_closes_low
    
    @classmethod
    def detect_three_white_soldiers(cls, df: pd.DataFrame) -> pd.Series:
        """Detect Three White Soldiers (strong bullish)"""
        return (
            cls._is_bullish(df) &
            cls._is_bullish(df).shift(1) &
            cls._is_bullish(df).shift(2) &
            (df['close'] > df['close'].shift(1)) &
            (df['close'].shift(1) > df['close'].shift(2)) &
            (df['open'] > df['open'].shift(1)) &
            (df['open'].shift(1) > df['open'].shift(2))
        )
    
    @classmethod
    def detect_three_black_crows(cls, df: pd.DataFrame) -> pd.Series:
        """Detect Three Black Crows (strong bearish)"""
        return (
            cls._is_bearish(df) &
            cls._is_bearish(df).shift(1) &
            cls._is_bearish(df).shift(2) &
            (df['close'] < df['close'].shift(1)) &
            (df['close'].shift(1) < df['close'].shift(2)) &
            (df['open'] < df['open'].shift(1)) &
            (df['open'].shift(1) < df['open'].shift(2))
        )
    
    @classmethod
    def detect_all_patterns(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Detect all candlestick patterns and return as DataFrame"""
        patterns = pd.DataFrame(index=df.index)
        
        patterns['Doji'] = cls.detect_doji(df)
        patterns['Hammer'] = cls.detect_hammer(df)
        patterns['Inverted_Hammer'] = cls.detect_inverted_hammer(df)
        patterns['Bullish_Engulfing'] = cls.detect_engulfing_bullish(df)
        patterns['Bearish_Engulfing'] = cls.detect_engulfing_bearish(df)
        patterns['Morning_Star'] = cls.detect_morning_star(df)
        patterns['Evening_Star'] = cls.detect_evening_star(df)
        patterns['Three_White_Soldiers'] = cls.detect_three_white_soldiers(df)
        patterns['Three_Black_Crows'] = cls.detect_three_black_crows(df)
        
        return patterns


class ChartPatterns:
    """Detect chart patterns using price action"""
    
    @staticmethod
    def find_support_resistance(
        df: pd.DataFrame,
        window: int = 20,
        num_levels: int = 5
    ) -> Dict:
        """Find key support and resistance levels"""
        pivots_high = df['high'].rolling(window=window, center=True).max()
        pivots_low = df['low'].rolling(window=window, center=True).min()
        
        # Cluster nearby levels
        from sklearn.cluster import KMeans
        
        highs = pivots_high.dropna().values.reshape(-1, 1)
        lows = pivots_low.dropna().values.reshape(-1, 1)
        
        all_levels = np.concatenate([highs, lows])
        
        if len(all_levels) < num_levels:
            return {'support': [], 'resistance': []}
        
        kmeans = KMeans(n_clusters=num_levels, random_state=42, n_init=10)
        kmeans.fit(all_levels)
        levels = sorted(kmeans.cluster_centers_.flatten())
        
        current_price = df['close'].iloc[-1]
        
        support = [l for l in levels if l < current_price]
        resistance = [l for l in levels if l > current_price]
        
        return {
            'support': support[-3:] if len(support) >= 3 else support,
            'resistance': resistance[:3] if len(resistance) >= 3 else resistance,
            'all_levels': levels
        }
    
    @staticmethod
    def detect_double_bottom(
        df: pd.DataFrame,
        window: int = 20,
        tolerance: float = 0.02
    ) -> bool:
        """Detect Double Bottom pattern"""
        lows = df['low'].rolling(window=window, center=True).min()
        
        # Find local minima
        local_mins = []
        for i in range(window, len(df) - window):
            if df['low'].iloc[i] == lows.iloc[i]:
                local_mins.append((i, df['low'].iloc[i]))
        
        if len(local_mins) < 2:
            return False
        
        # Check last two minima
        last_two = local_mins[-2:]
        price_diff = abs(last_two[0][1] - last_two[1][1]) / last_two[0][1]
        
        # Time between bottoms
        time_diff = last_two[1][0] - last_two[0][0]
        
        return price_diff < tolerance and time_diff > 10
    
    @staticmethod
    def detect_double_top(
        df: pd.DataFrame,
        window: int = 20,
        tolerance: float = 0.02
    ) -> bool:
        """Detect Double Top pattern"""
        highs = df['high'].rolling(window=window, center=True).max()
        
        local_maxs = []
        for i in range(window, len(df) - window):
            if df['high'].iloc[i] == highs.iloc[i]:
                local_maxs.append((i, df['high'].iloc[i]))
        
        if len(local_maxs) < 2:
            return False
        
        last_two = local_maxs[-2:]
        price_diff = abs(last_two[0][1] - last_two[1][1]) / last_two[0][1]
        time_diff = last_two[1][0] - last_two[0][0]
        
        return price_diff < tolerance and time_diff > 10
    
    @staticmethod
    def detect_trendline_breakout(
        df: pd.DataFrame,
        lookback: int = 50
    ) -> Dict:
        """Detect trendline breakouts"""
        from scipy.stats import linregress
        
        recent_data = df.tail(lookback)
        x = np.arange(len(recent_data))
        
        # Uptrend line (connecting lows)
        slope_low, intercept_low, _, _, _ = linregress(x, recent_data['low'].values)
        
        # Downtrend line (connecting highs)
        slope_high, intercept_high, _, _, _ = linregress(x, recent_data['high'].values)
        
        current_price = df['close'].iloc[-1]
        trendline_low_value = slope_low * (len(recent_data) - 1) + intercept_low
        trendline_high_value = slope_high * (len(recent_data) - 1) + intercept_high
        
        return {
            'uptrend_slope': slope_low,
            'downtrend_slope': slope_high,
            'price_above_uptrend': current_price > trendline_low_value,
            'price_below_downtrend': current_price < trendline_high_value,
            'uptrend_breakout': current_price > trendline_high_value and slope_high < 0,
            'downtrend_breakout': current_price < trendline_low_value and slope_low > 0,
        }
    
    @staticmethod
    def detect_consolidation(
        df: pd.DataFrame,
        lookback: int = 20,
        range_threshold: float = 0.05
    ) -> bool:
        """Detect if stock is in consolidation/range"""
        recent = df.tail(lookback)
        high = recent['high'].max()
        low = recent['low'].min()
        range_pct = (high - low) / low
        
        return range_pct < range_threshold
    
    @staticmethod
    def detect_head_and_shoulders(
        df: pd.DataFrame,
        window: int = 10
    ) -> Dict:
        """Simplified Head & Shoulders detection"""
        # Find local maxima
        highs = df['high'].rolling(window=window, center=True).max()
        
        local_maxs = []
        for i in range(window, len(df) - window):
            if df['high'].iloc[i] == highs.iloc[i]:
                local_maxs.append((i, df['high'].iloc[i]))
        
        if len(local_maxs) < 3:
            return {'detected': False}
        
        # Check last 3 peaks
        last_three = local_maxs[-3:]
        left_shoulder = last_three[0][1]
        head = last_three[1][1]
        right_shoulder = last_three[2][1]
        
        # Head should be highest, shoulders approximately equal
        is_h_and_s = (
            head > left_shoulder and
            head > right_shoulder and
            abs(left_shoulder - right_shoulder) / left_shoulder < 0.05
        )
        
        return {
            'detected': is_h_and_s,
            'left_shoulder': left_shoulder,
            'head': head,
            'right_shoulder': right_shoulder,
            'type': 'bearish' if is_h_and_s else None
        }
