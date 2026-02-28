"""
Advanced Volume Analysis for Indian Stock Market

Features:
- Volume Profile Analysis
- Volume Weighted Analysis (VWAP, VWMA)
- Volume Spread Analysis (VSA)
- Accumulation/Distribution detection
- Smart Money Flow detection
- Volume Breakout detection
- Delivery Volume Analysis (NSE specific)
- Volume Divergence detection
- Volume Climax detection
- OBV Analysis
- Chaikin Money Flow
- Force Index
- Ease of Movement
- Volume Rate of Change
- Relative Volume Analysis
- Institutional Activity detection
- Volume Pattern Recognition
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


# ============================================================
# DATA STRUCTURES
# ============================================================

class VolumeSignal(Enum):
    """Volume signal types"""
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    CLIMAX_TOP = "climax_top"
    CLIMAX_BOTTOM = "climax_bottom"
    BREAKOUT = "breakout"
    DRY_UP = "dry_up"
    SMART_MONEY_BUY = "smart_money_buy"
    SMART_MONEY_SELL = "smart_money_sell"
    DIVERGENCE_BULLISH = "divergence_bullish"
    DIVERGENCE_BEARISH = "divergence_bearish"
    NO_SIGNAL = "no_signal"


@dataclass
class VolumeAnalysisResult:
    """Result of volume analysis for a stock"""
    symbol: str
    date: str
    signal: VolumeSignal
    signal_strength: float  # 0.0 to 1.0
    volume: int
    avg_volume_20: float
    volume_ratio: float
    delivery_pct: float  # NSE specific
    price_change_pct: float
    description: str
    details: Dict = field(default_factory=dict)

    def __repr__(self):
        return (
            f"VolumeAnalysis({self.symbol} | {self.signal.value} | "
            f"Strength: {self.signal_strength:.0%} | Vol: {self.volume_ratio:.1f}x)"
        )


@dataclass
class VolumeProfile:
    """Volume Profile data for a price level"""
    price_level: float
    total_volume: int
    buy_volume: int
    sell_volume: int
    pct_of_total: float
    is_high_volume_node: bool
    is_low_volume_node: bool
    is_poc: bool  # Point of Control


# ============================================================
# VOLUME INDICATORS
# ============================================================

class VolumeIndicators:
    """Calculate all volume-based technical indicators"""

    # ----- BASIC VOLUME METRICS -----

    @staticmethod
    def volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
        """Volume Simple Moving Average"""
        return volume.rolling(window=period).mean()

    @staticmethod
    def volume_ema(volume: pd.Series, period: int = 20) -> pd.Series:
        """Volume Exponential Moving Average"""
        return volume.ewm(span=period, adjust=False).mean()

    @staticmethod
    def volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
        """
        Volume Ratio = Current Volume / Average Volume
        
        > 2.0 = High volume spike
        > 1.5 = Above average
        1.0 = Average
        < 0.5 = Below average (dry)
        """
        avg = volume.rolling(window=period).mean()
        return volume / avg

    @staticmethod
    def relative_volume(volume: pd.Series, period: int = 20) -> pd.Series:
        """
        Relative Volume (RVOL)
        How current volume compares to historical average
        """
        avg = volume.rolling(window=period).mean()
        std = volume.rolling(window=period).std()
        return (volume - avg) / std

    @staticmethod
    def volume_zscore(volume: pd.Series, period: int = 20) -> pd.Series:
        """Volume Z-Score for anomaly detection"""
        avg = volume.rolling(window=period).mean()
        std = volume.rolling(window=period).std()
        return (volume - avg) / (std + 1e-10)

    @staticmethod
    def volume_percentile(volume: pd.Series, period: int = 100) -> pd.Series:
        """Volume percentile rank over lookback period"""
        return volume.rolling(window=period).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )

    # ----- VWAP & VWMA -----

    @staticmethod
    def vwap(df: pd.DataFrame, anchor: str = "day") -> pd.Series:
        """
        Volume Weighted Average Price
        
        Args:
            df: DataFrame with high, low, close, volume
            anchor: 'day' (reset daily) or 'session' (cumulative)
        
        Widely used by institutional traders in India
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3

        if anchor == "session" or 'date' not in df.columns:
            # Cumulative VWAP (intraday)
            cum_volume = df['volume'].cumsum()
            cum_tp_vol = (typical_price * df['volume']).cumsum()
            return cum_tp_vol / cum_volume
        else:
            # Daily VWAP (reset each day)
            df_temp = df.copy()
            df_temp['tp_vol'] = typical_price * df['volume']

            if df_temp['date'].dt.tz is not None:
                df_temp['date'] = df_temp['date'].dt.tz_localize(None)

            df_temp['trade_date'] = df_temp['date'].dt.date

            vwap_series = pd.Series(index=df.index, dtype=float)

            for trade_date, group in df_temp.groupby('trade_date'):
                cum_vol = group['volume'].cumsum()
                cum_tp_vol = group['tp_vol'].cumsum()
                day_vwap = cum_tp_vol / cum_vol
                vwap_series.loc[group.index] = day_vwap

            return vwap_series

    @staticmethod
    def vwap_bands(
        df: pd.DataFrame,
        std_multiplier: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        VWAP with standard deviation bands
        
        Returns:
            vwap, upper_band, lower_band
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        cum_volume = df['volume'].cumsum()
        cum_tp_vol = (typical_price * df['volume']).cumsum()
        vwap_line = cum_tp_vol / cum_volume

        # Standard deviation around VWAP
        cum_tp_sq_vol = ((typical_price ** 2) * df['volume']).cumsum()
        variance = (cum_tp_sq_vol / cum_volume) - (vwap_line ** 2)
        std = np.sqrt(variance.clip(lower=0))

        upper = vwap_line + std_multiplier * std
        lower = vwap_line - std_multiplier * std

        return vwap_line, upper, lower

    @staticmethod
    def vwma(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Volume Weighted Moving Average"""
        return (
            (df['close'] * df['volume']).rolling(period).sum() /
            df['volume'].rolling(period).sum()
        )

    @staticmethod
    def anchored_vwap(
        df: pd.DataFrame,
        anchor_date: str
    ) -> pd.Series:
        """
        Anchored VWAP from a specific date
        
        Used to measure fair value from important events
        (earnings, IPO listing, breakout date, etc.)
        """
        anchor_dt = pd.to_datetime(anchor_date)
        mask = df['date'] >= anchor_dt

        typical_price = (df['high'] + df['low'] + df['close']) / 3
        tp_vol = typical_price * df['volume']

        result = pd.Series(np.nan, index=df.index)
        cum_vol = df.loc[mask, 'volume'].cumsum()
        cum_tp_vol = tp_vol[mask].cumsum()
        result[mask] = cum_tp_vol / cum_vol

        return result

    # ----- ON BALANCE VOLUME (OBV) -----

    @staticmethod
    def obv(df: pd.DataFrame) -> pd.Series:
        """On Balance Volume"""
        direction = np.sign(df['close'].diff())
        direction.iloc[0] = 0
        return (direction * df['volume']).cumsum()

    @staticmethod
    def obv_ema(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """OBV with EMA smoothing"""
        obv_val = VolumeIndicators.obv(df)
        return obv_val.ewm(span=period, adjust=False).mean()

    @staticmethod
    def obv_divergence(
        df: pd.DataFrame,
        lookback: int = 20
    ) -> pd.Series:
        """
        Detect OBV divergence with price
        
        +1 = Bullish divergence (price down, OBV up)
        -1 = Bearish divergence (price up, OBV down)
         0 = No divergence
        """
        obv_val = VolumeIndicators.obv(df)

        price_slope = df['close'].diff(lookback)
        obv_slope = obv_val.diff(lookback)

        divergence = pd.Series(0, index=df.index)

        # Bullish: Price falling, OBV rising
        divergence[(price_slope < 0) & (obv_slope > 0)] = 1

        # Bearish: Price rising, OBV falling
        divergence[(price_slope > 0) & (obv_slope < 0)] = -1

        return divergence

    # ----- ACCUMULATION / DISTRIBUTION -----

    @staticmethod
    def accumulation_distribution_line(df: pd.DataFrame) -> pd.Series:
        """
        Accumulation/Distribution Line (ADL)
        
        Measures the cumulative flow of money into/out of a stock
        """
        clv = (
            (2 * df['close'] - df['low'] - df['high']) /
            (df['high'] - df['low'] + 1e-10)
        )
        ad = (clv * df['volume']).cumsum()
        return ad

    @staticmethod
    def chaikin_money_flow(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Chaikin Money Flow (CMF)
        
        Range: -1 to +1
        > 0 = Buying pressure (accumulation)
        < 0 = Selling pressure (distribution)
        > 0.25 = Strong buying
        < -0.25 = Strong selling
        """
        clv = (
            (2 * df['close'] - df['low'] - df['high']) /
            (df['high'] - df['low'] + 1e-10)
        )
        mf_volume = clv * df['volume']

        cmf = (
            mf_volume.rolling(window=period).sum() /
            df['volume'].rolling(window=period).sum()
        )
        return cmf

    @staticmethod
    def chaikin_oscillator(
        df: pd.DataFrame,
        fast: int = 3,
        slow: int = 10
    ) -> pd.Series:
        """
        Chaikin Oscillator
        
        Difference between fast and slow EMA of A/D Line
        Crossover above 0 = Bullish
        Crossover below 0 = Bearish
        """
        ad = VolumeIndicators.accumulation_distribution_line(df)
        fast_ema = ad.ewm(span=fast, adjust=False).mean()
        slow_ema = ad.ewm(span=slow, adjust=False).mean()
        return fast_ema - slow_ema

    # ----- MONEY FLOW INDEX (MFI) -----

    @staticmethod
    def money_flow_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Money Flow Index (Volume-weighted RSI)
        
        Range: 0-100
        > 80 = Overbought
        < 20 = Oversold
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']

        positive_flow = pd.Series(0.0, index=df.index)
        negative_flow = pd.Series(0.0, index=df.index)

        for i in range(1, len(df)):
            if typical_price.iloc[i] > typical_price.iloc[i - 1]:
                positive_flow.iloc[i] = money_flow.iloc[i]
            else:
                negative_flow.iloc[i] = money_flow.iloc[i]

        positive_sum = positive_flow.rolling(window=period).sum()
        negative_sum = negative_flow.rolling(window=period).sum()

        mfi = 100 - (100 / (1 + positive_sum / (negative_sum + 1e-10)))
        return mfi

    # ----- FORCE INDEX -----

    @staticmethod
    def force_index(df: pd.DataFrame, period: int = 13) -> pd.Series:
        """
        Force Index = (Close - Previous Close) Ã— Volume
        
        Measures strength of bulls/bears
        EMA smoothed for signal generation
        """
        fi = df['close'].diff() * df['volume']
        return fi.ewm(span=period, adjust=False).mean()

    # ----- EASE OF MOVEMENT -----

    @staticmethod
    def ease_of_movement(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Ease of Movement (EOM)
        
        Measures price change relative to volume
        High positive = Price moving up easily on low volume
        High negative = Price moving down easily
        """
        dm = ((df['high'] + df['low']) / 2) - ((df['high'].shift(1) + df['low'].shift(1)) / 2)
        box_ratio = (df['volume'] / 1e6) / (df['high'] - df['low'] + 1e-10)

        eom = dm / box_ratio
        return eom.rolling(window=period).mean()

    # ----- VOLUME RATE OF CHANGE -----

    @staticmethod
    def volume_roc(volume: pd.Series, period: int = 14) -> pd.Series:
        """
        Volume Rate of Change
        
        Measures % change in volume over period
        Large positive = Volume increasing rapidly
        """
        return volume.pct_change(period) * 100

    # ----- KLINGER VOLUME OSCILLATOR -----

    @staticmethod
    def klinger_oscillator(
        df: pd.DataFrame,
        fast: int = 34,
        slow: int = 55,
        signal: int = 13
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Klinger Volume Oscillator
        
        Tracks volume flow to predict reversals
        
        Returns:
            kvo, signal_line
        """
        # Trend
        hlc = df['high'] + df['low'] + df['close']
        trend = pd.Series(0, index=df.index)
        trend[hlc > hlc.shift(1)] = 1
        trend[hlc <= hlc.shift(1)] = -1

        # dm
        dm = df['high'] - df['low']

        # cm (cumulative measurement)
        cm = pd.Series(0.0, index=df.index)
        cm.iloc[0] = dm.iloc[0]

        for i in range(1, len(df)):
            if trend.iloc[i] == trend.iloc[i - 1]:
                cm.iloc[i] = cm.iloc[i - 1] + dm.iloc[i]
            else:
                cm.iloc[i] = dm.iloc[i - 1] + dm.iloc[i]

        # Volume force
        vf = df['volume'] * abs(2 * dm / (cm + 1e-10) - 1) * trend * 100

        # KVO
        fast_ema = vf.ewm(span=fast, adjust=False).mean()
        slow_ema = vf.ewm(span=slow, adjust=False).mean()
        kvo = fast_ema - slow_ema
        signal_line = kvo.ewm(span=signal, adjust=False).mean()

        return kvo, signal_line

    # ----- NEGATIVE VOLUME INDEX (NVI) -----

    @staticmethod
    def negative_volume_index(df: pd.DataFrame) -> pd.Series:
        """
        Negative Volume Index
        
        Changes only on days when volume decreases
        Smart money tends to trade on low volume days
        Rising NVI = Smart money bullish
        """
        nvi = pd.Series(1000.0, index=df.index)

        for i in range(1, len(df)):
            if df['volume'].iloc[i] < df['volume'].iloc[i - 1]:
                pct_change = (df['close'].iloc[i] - df['close'].iloc[i - 1]) / df['close'].iloc[i - 1]
                nvi.iloc[i] = nvi.iloc[i - 1] * (1 + pct_change)
            else:
                nvi.iloc[i] = nvi.iloc[i - 1]

        return nvi

    @staticmethod
    def positive_volume_index(df: pd.DataFrame) -> pd.Series:
        """
        Positive Volume Index
        
        Changes only on days when volume increases
        General public tends to trade on high volume days
        """
        pvi = pd.Series(1000.0, index=df.index)

        for i in range(1, len(df)):
            if df['volume'].iloc[i] > df['volume'].iloc[i - 1]:
                pct_change = (df['close'].iloc[i] - df['close'].iloc[i - 1]) / df['close'].iloc[i - 1]
                pvi.iloc[i] = pvi.iloc[i - 1] * (1 + pct_change)
            else:
                pvi.iloc[i] = pvi.iloc[i - 1]

        return pvi

    # ----- CALCULATE ALL -----

    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all volume indicators at once"""
        result = df.copy()
        vi = VolumeIndicators

        # Basic
        result['Vol_SMA_10'] = vi.volume_sma(df['volume'], 10)
        result['Vol_SMA_20'] = vi.volume_sma(df['volume'], 20)
        result['Vol_SMA_50'] = vi.volume_sma(df['volume'], 50)
        result['Vol_EMA_20'] = vi.volume_ema(df['volume'], 20)
        result['Vol_Ratio'] = vi.volume_ratio(df['volume'], 20)
        result['Vol_ZScore'] = vi.volume_zscore(df['volume'], 20)
        result['RVOL'] = vi.relative_volume(df['volume'], 20)

        # VWAP & VWMA
        result['VWAP'] = vi.vwap(df, anchor="session")
        result['VWMA_20'] = vi.vwma(df, 20)
        result['Price_vs_VWAP'] = (df['close'] - result['VWAP']) / result['VWAP'] * 100

        # OBV
        result['OBV'] = vi.obv(df)
        result['OBV_EMA'] = vi.obv_ema(df, 20)
        result['OBV_Divergence'] = vi.obv_divergence(df, 20)

        # A/D Line & CMF
        result['AD_Line'] = vi.accumulation_distribution_line(df)
        result['CMF_20'] = vi.chaikin_money_flow(df, 20)
        result['Chaikin_Osc'] = vi.chaikin_oscillator(df)

        # MFI
        result['MFI_14'] = vi.money_flow_index(df, 14)

        # Force Index
        result['Force_Index_13'] = vi.force_index(df, 13)

        # EOM
        result['EOM_14'] = vi.ease_of_movement(df, 14)

        # Volume ROC
        result['Vol_ROC_14'] = vi.volume_roc(df['volume'], 14)

        # NVI / PVI
        result['NVI'] = vi.negative_volume_index(df)
        result['PVI'] = vi.positive_volume_index(df)

        # Klinger
        result['KVO'], result['KVO_Signal'] = vi.klinger_oscillator(df)

        return result


# ============================================================
# VOLUME SPREAD ANALYSIS (VSA)
# ============================================================

class VolumeSpreadAnalysis:
    """
    Volume Spread Analysis (VSA)
    
    Based on Wyckoff method - analyses relationship between
    price spread (high-low), closing position, and volume
    to detect smart money activity
    
    Widely used by Indian market traders for NIFTY & stock analysis
    """

    @staticmethod
    def classify_bar(
        row: pd.Series,
        avg_volume: float,
        avg_spread: float
    ) -> Dict:
        """
        Classify a single bar using VSA principles
        
        Returns:
            Dictionary with VSA classification
        """
        spread = row['high'] - row['low']
        close_position = (row['close'] - row['low']) / (spread + 1e-10)
        volume = row['volume']
        is_up = row['close'] > row['open']

        vol_class = (
            'ultra_high' if volume > avg_volume * 3 else
            'very_high' if volume > avg_volume * 2 else
            'high' if volume > avg_volume * 1.5 else
            'above_avg' if volume > avg_volume else
            'below_avg' if volume > avg_volume * 0.5 else
            'low' if volume > avg_volume * 0.25 else
            'ultra_low'
        )

        spread_class = (
            'wide' if spread > avg_spread * 1.5 else
            'average' if spread > avg_spread * 0.5 else
            'narrow'
        )

        close_class = (
            'high' if close_position > 0.7 else
            'middle' if close_position > 0.3 else
            'low'
        )

        return {
            'spread': spread,
            'close_position': close_position,
            'volume_class': vol_class,
            'spread_class': spread_class,
            'close_class': close_class,
            'is_up': is_up,
        }

    @classmethod
    def detect_signals(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect all VSA signals in the data
        
        Returns:
            DataFrame with VSA signal columns
        """
        result = df.copy()
        
        avg_volume = df['volume'].rolling(20).mean()
        avg_spread = (df['high'] - df['low']).rolling(20).mean()

        spread = df['high'] - df['low']
        close_pos = (df['close'] - df['low']) / (spread + 1e-10)
        vol_ratio = df['volume'] / avg_volume
        spread_ratio = spread / avg_spread
        is_up = df['close'] > df['open']

        # --- SUPPLY & DEMAND SIGNALS ---

        # Stopping Volume (Potential bottom)
        # High volume + down bar + narrow spread + close near high
        result['VSA_Stopping_Volume'] = (
            (vol_ratio > 1.5) &
            (~is_up) &
            (spread_ratio < 0.8) &
            (close_pos > 0.5)
        )

        # No Demand (Weak rallies)
        # Low volume + narrow spread up bar
        result['VSA_No_Demand'] = (
            (vol_ratio < 0.7) &
            is_up &
            (spread_ratio < 0.7)
        )

        # No Supply (Weak declines - bullish)
        # Low volume + narrow spread down bar
        result['VSA_No_Supply'] = (
            (vol_ratio < 0.7) &
            (~is_up) &
            (spread_ratio < 0.7)
        )

        # Selling Climax (Potential bottom)
        # Ultra high volume + wide spread down + close near low
        result['VSA_Selling_Climax'] = (
            (vol_ratio > 2.5) &
            (~is_up) &
            (spread_ratio > 1.5) &
            (close_pos < 0.3)
        )

        # Buying Climax (Potential top)
        # Ultra high volume + wide spread up + close near high
        result['VSA_Buying_Climax'] = (
            (vol_ratio > 2.5) &
            is_up &
            (spread_ratio > 1.5) &
            (close_pos > 0.7)
        )

        # Up Thrust (Bearish - weakness)
        # High volume + wide spread + close near low on up bar
        result['VSA_UpThrust'] = (
            (vol_ratio > 1.5) &
            (spread_ratio > 1.3) &
            (close_pos < 0.3) &
            (df['high'] > df['high'].shift(1))
        )

        # Test (Bullish - strength)
        # Low volume + narrow spread down bar after prior high volume
        result['VSA_Test'] = (
            (vol_ratio < 0.5) &
            (~is_up) &
            (spread_ratio < 0.6) &
            (vol_ratio.shift(1) > 1.5)  # Previous bar was high volume
        )

        # Effort vs Result
        # High volume but no price progress
        result['VSA_Effort_No_Result'] = (
            (vol_ratio > 2.0) &
            (spread_ratio < 0.5)
        )

        # Spring (Bullish - Wyckoff)
        # Price breaks below support then closes above
        recent_low = df['low'].rolling(20).min()
        result['VSA_Spring'] = (
            (df['low'] < recent_low.shift(1)) &
            (df['close'] > recent_low.shift(1)) &
            (vol_ratio > 1.0)
        )

        # Absorption Volume (Hidden buying/selling)
        # High volume but price doesn't fall much = buying absorption
        result['VSA_Absorption_Buy'] = (
            (vol_ratio > 2.0) &
            (~is_up) &
            (df['close'].pct_change().abs() < 0.005)  # Price barely moved
        )

        result['VSA_Absorption_Sell'] = (
            (vol_ratio > 2.0) &
            is_up &
            (df['close'].pct_change().abs() < 0.005)
        )

        # Composite VSA Score
        bullish_signals = (
            result['VSA_Stopping_Volume'].astype(int) * 2 +
            result['VSA_No_Supply'].astype(int) * 1 +
            result['VSA_Selling_Climax'].astype(int) * 3 +
            result['VSA_Test'].astype(int) * 2 +
            result['VSA_Spring'].astype(int) * 3 +
            result['VSA_Absorption_Buy'].astype(int) * 2
        )

        bearish_signals = (
            result['VSA_No_Demand'].astype(int) * 1 +
            result['VSA_Buying_Climax'].astype(int) * 3 +
            result['VSA_UpThrust'].astype(int) * 2 +
            result['VSA_Absorption_Sell'].astype(int) * 2
        )

        result['VSA_Bullish_Score'] = bullish_signals
        result['VSA_Bearish_Score'] = bearish_signals
        result['VSA_Net_Score'] = bullish_signals - bearish_signals

        return result


# ============================================================
# VOLUME PROFILE ANALYSIS
# ============================================================

class VolumeProfileAnalyzer:
    """
    Volume Profile Analysis
    
    Builds a volume profile showing how much volume was traded
    at each price level. Identifies:
    - Point of Control (POC) - most traded price
    - Value Area High/Low (VAH/VAL) 
    - High Volume Nodes (HVN)
    - Low Volume Nodes (LVN)
    """

    @staticmethod
    def build_volume_profile(
        df: pd.DataFrame,
        num_bins: int = 50,
        value_area_pct: float = 0.70
    ) -> Dict:
        """
        Build volume profile for given data
        
        Args:
            df: OHLCV DataFrame
            num_bins: Number of price bins
            value_area_pct: Percentage for value area (typically 70%)
        
        Returns:
            Dictionary with volume profile data
        """
        if df.empty:
            return {}

        price_min = df['low'].min()
        price_max = df['high'].max()

        # Create price bins
        bins = np.linspace(price_min, price_max, num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Distribute volume across price bins for each bar
        volume_profile = np.zeros(num_bins)

        for _, row in df.iterrows():
            # Find which bins this bar covers
            bar_low = row['low']
            bar_high = row['high']
            bar_volume = row['volume']

            low_bin = np.searchsorted(bins, bar_low, side='right') - 1
            high_bin = np.searchsorted(bins, bar_high, side='left')

            low_bin = max(0, low_bin)
            high_bin = min(num_bins - 1, high_bin)

            # Distribute volume equally across bins
            num_covered = high_bin - low_bin + 1
            if num_covered > 0:
                vol_per_bin = bar_volume / num_covered
                volume_profile[low_bin:high_bin + 1] += vol_per_bin

        total_volume = volume_profile.sum()

        # Point of Control (POC) - highest volume price level
        poc_idx = np.argmax(volume_profile)
        poc_price = bin_centers[poc_idx]

        # Value Area (70% of volume around POC)
        sorted_indices = np.argsort(volume_profile)[::-1]
        cum_volume = 0
        value_area_indices = []

        for idx in sorted_indices:
            cum_volume += volume_profile[idx]
            value_area_indices.append(idx)
            if cum_volume >= total_volume * value_area_pct:
                break

        val = bin_centers[min(value_area_indices)]
        vah = bin_centers[max(value_area_indices)]

        # High Volume Nodes (HVN) & Low Volume Nodes (LVN)
        avg_volume = volume_profile.mean()
        hvn_mask = volume_profile > avg_volume * 1.5
        lvn_mask = volume_profile < avg_volume * 0.5

        # Build profile list
        profile = []
        for i in range(num_bins):
            profile.append(VolumeProfile(
                price_level=bin_centers[i],
                total_volume=int(volume_profile[i]),
                buy_volume=0,
                sell_volume=0,
                pct_of_total=volume_profile[i] / total_volume * 100 if total_volume > 0 else 0,
                is_high_volume_node=bool(hvn_mask[i]),
                is_low_volume_node=bool(lvn_mask[i]),
                is_poc=i == poc_idx,
            ))

        return {
            'poc': poc_price,
            'vah': vah,
            'val': val,
            'total_volume': int(total_volume),
            'hvn_prices': bin_centers[hvn_mask].tolist(),
            'lvn_prices': bin_centers[lvn_mask].tolist(),
            'profile': profile,
            'bins': bins.tolist(),
            'volumes': volume_profile.tolist(),
            'bin_centers': bin_centers.tolist(),
        }

    @staticmethod
    def get_support_resistance_from_profile(
        profile: Dict,
        current_price: float
    ) -> Dict:
        """
        Extract support/resistance levels from volume profile
        
        HVN below price = Strong support
        HVN above price = Strong resistance
        LVN = Potential breakout zones (low friction)
        """
        support_levels = [
            p for p in profile.get('hvn_prices', [])
            if p < current_price
        ]
        resistance_levels = [
            p for p in profile.get('hvn_prices', [])
            if p > current_price
        ]
        breakout_zones = profile.get('lvn_prices', [])

        return {
            'poc': profile.get('poc', 0),
            'vah': profile.get('vah', 0),
            'val': profile.get('val', 0),
            'support_levels': sorted(support_levels, reverse=True)[:3],
            'resistance_levels': sorted(resistance_levels)[:3],
            'breakout_zones': breakout_zones,
            'price_vs_poc': 'ABOVE' if current_price > profile.get('poc', 0) else 'BELOW',
            'in_value_area': profile.get('val', 0) <= current_price <= profile.get('vah', 0),
        }


# ============================================================
# SMART MONEY DETECTOR
# ============================================================

class SmartMoneyDetector:
    """
    Detect institutional/smart money activity
    using volume analysis patterns
    """

    @staticmethod
    def detect_accumulation(
        df: pd.DataFrame,
        lookback: int = 20
    ) -> pd.Series:
        """
        Detect accumulation phases
        
        Characteristics:
        - Price range narrows
        - Volume decreases on down days
        - Volume increases on up days
        - Price makes higher lows
        """
        score = pd.Series(0.0, index=df.index)

        # Higher lows
        higher_lows = (df['low'] > df['low'].shift(1)).rolling(lookback).sum() / lookback
        score += higher_lows * 0.3

        # Volume pattern: higher on up days
        up_days = df['close'] > df['open']
        up_vol = df['volume'].where(up_days, 0).rolling(lookback).mean()
        down_vol = df['volume'].where(~up_days, 0).rolling(lookback).mean()
        vol_ratio = up_vol / (down_vol + 1)
        score += (vol_ratio > 1).astype(float) * 0.3

        # Range narrowing
        range_now = (df['high'] - df['low']).rolling(lookback // 2).mean()
        range_before = (df['high'] - df['low']).rolling(lookback).mean()
        score += (range_now < range_before).astype(float) * 0.2

        # CMF positive
        cmf = VolumeIndicators.chaikin_money_flow(df, lookback)
        score += (cmf > 0).astype(float) * 0.2

        return score

    @staticmethod
    def detect_distribution(
        df: pd.DataFrame,
        lookback: int = 20
    ) -> pd.Series:
        """
        Detect distribution phases
        
        Characteristics:
        - Price range stays wide
        - Volume increases on down days
        - Price makes lower highs
        """
        score = pd.Series(0.0, index=df.index)

        # Lower highs
        lower_highs = (df['high'] < df['high'].shift(1)).rolling(lookback).sum() / lookback
        score += lower_highs * 0.3

        # Volume pattern: higher on down days
        up_days = df['close'] > df['open']
        up_vol = df['volume'].where(up_days, 0).rolling(lookback).mean()
        down_vol = df['volume'].where(~up_days, 0).rolling(lookback).mean()
        vol_ratio = down_vol / (up_vol + 1)
        score += (vol_ratio > 1).astype(float) * 0.3

        # CMF negative
        cmf = VolumeIndicators.chaikin_money_flow(df, lookback)
        score += (cmf < 0).astype(float) * 0.2

        # High volume on down days
        avg_vol = df['volume'].rolling(20).mean()
        high_vol_down = (df['volume'] > avg_vol * 1.5) & (~up_days)
        score += high_vol_down.rolling(lookback).sum() / lookback * 0.2

        return score

    @staticmethod
    def detect_smart_money_divergence(
        df: pd.DataFrame,
        lookback: int = 10
    ) -> pd.DataFrame:
        """
        Detect price-volume divergences indicating smart money activity
        """
        result = pd.DataFrame(index=df.index)

        price_change = df['close'].pct_change(lookback)
        vol_change = df['volume'].pct_change(lookback)
        obv = VolumeIndicators.obv(df)
        obv_change = obv.pct_change(lookback)

        # Bullish divergence: price down, smart money buying (OBV up)
        result['bullish_divergence'] = (
            (price_change < -0.02) & (obv_change > 0)
        ).astype(int)

        # Bearish divergence: price up, smart money selling (OBV down)
        result['bearish_divergence'] = (
            (price_change > 0.02) & (obv_change < 0)
        ).astype(int)

        # Volume dry-up on pullback (bullish)
        avg_vol = df['volume'].rolling(20).mean()
        result['dry_pullback'] = (
            (price_change < 0) &
            (df['volume'] < avg_vol * 0.5)
        ).astype(int)

        # Volume surge on breakout
        result['volume_surge_breakout'] = (
            (df['close'] > df['high'].rolling(20).max().shift(1)) &
            (df['volume'] > avg_vol * 2)
        ).astype(int)

        return result


# ============================================================
# DELIVERY VOLUME ANALYZER (NSE SPECIFIC)
# ============================================================

class DeliveryVolumeAnalyzer:
    """
    Analyze delivery volume data (specific to Indian NSE/BSE)
    
    Delivery % indicates what percentage of traded volume
    was actually delivered (settled). High delivery = genuine buying
    
    Key thresholds for Indian market:
    - > 60% delivery = Strong genuine interest
    - > 50% delivery = Above average interest
    - < 30% delivery = Mostly speculative/intraday
    """

    @staticmethod
    def analyze_delivery(
        df: pd.DataFrame,
        delivery_col: str = 'delivery_pct'
    ) -> pd.DataFrame:
        """
        Analyze delivery volume patterns
        
        Args:
            df: DataFrame with delivery_pct column
            delivery_col: Name of delivery percentage column
        """
        if delivery_col not in df.columns:
            logger.warning(f"Column '{delivery_col}' not found in DataFrame")
            return df

        result = df.copy()

        # Delivery metrics
        result['del_sma_10'] = result[delivery_col].rolling(10).mean()
        result['del_sma_20'] = result[delivery_col].rolling(20).mean()

        # Delivery classification
        result['delivery_class'] = pd.cut(
            result[delivery_col],
            bins=[0, 20, 35, 50, 65, 100],
            labels=['Very_Low', 'Low', 'Average', 'High', 'Very_High']
        )

        # Delivery spike (unusual high delivery)
        result['delivery_spike'] = result[delivery_col] > result['del_sma_20'] * 1.5

        # High delivery + price up = Strong buying
        price_up = result['close'] > result['close'].shift(1)
        result['strong_buying'] = (
            (result[delivery_col] > 60) & price_up
        )

        # High delivery + price down = Strong selling
        result['strong_selling'] = (
            (result[delivery_col] > 60) & (~price_up)
        )

        # Low delivery = Speculative / Intraday
        result['speculative'] = result[delivery_col] < 30

        # Delivery increasing trend
        result['delivery_trend'] = (
            result[delivery_col].rolling(5).mean() -
            result[delivery_col].rolling(20).mean()
        )

        # Score: positive = accumulation, negative = distribution
        result['delivery_score'] = (
            result[delivery_col] / 100 *
            np.where(price_up, 1, -1) *
            (result['volume'] / result['volume'].rolling(20).mean())
        )

        return result


# ============================================================
# MAIN VOLUME ANALYZER
# ============================================================

class VolumeAnalyzer:
    """
    Main volume analysis class combining all volume tools
    
    Usage:
        analyzer = VolumeAnalyzer()
        
        # Full analysis
        result_df = analyzer.full_analysis(df)
        
        # Get current signal
        signal = analyzer.get_current_signal(df, "RELIANCE")
        
        # Volume profile
        profile = analyzer.get_volume_profile(df)
    """

    def __init__(self):
        self.indicators = VolumeIndicators()
        self.vsa = VolumeSpreadAnalysis()
        self.profile_analyzer = VolumeProfileAnalyzer()
        self.smart_money = SmartMoneyDetector()
        self.delivery_analyzer = DeliveryVolumeAnalyzer()

    def full_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run complete volume analysis"""
        result = self.indicators.calculate_all(df)
        result = self.vsa.detect_signals(result)

        # Smart money detection
        result['Accumulation_Score'] = self.smart_money.detect_accumulation(df)
        result['Distribution_Score'] = self.smart_money.detect_distribution(df)

        divergence = self.smart_money.detect_smart_money_divergence(df)
        for col in divergence.columns:
            result[f'SM_{col}'] = divergence[col]

        # Delivery analysis
        if 'delivery_pct' in df.columns:
            result = self.delivery_analyzer.analyze_delivery(result)

        return result

    def get_current_signal(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> VolumeAnalysisResult:
        """Get current volume signal for a stock"""
        analyzed = self.full_analysis(df)
        latest = analyzed.iloc[-1]

        vol_ratio = latest.get('Vol_Ratio', 1.0)
        delivery_pct = latest.get('delivery_pct', 50.0)
        price_change = df['close'].pct_change().iloc[-1] * 100

        # Determine signal
        signal = VolumeSignal.NO_SIGNAL
        strength = 0.0
        description = "No significant volume signal"

        vsa_bull = latest.get('VSA_Bullish_Score', 0)
        vsa_bear = latest.get('VSA_Bearish_Score', 0)
        accum = latest.get('Accumulation_Score', 0)
        distrib = latest.get('Distribution_Score', 0)

        if latest.get('VSA_Selling_Climax', False):
            signal = VolumeSignal.CLIMAX_BOTTOM
            strength = 0.9
            description = "Selling climax - potential bottom"
        elif latest.get('VSA_Buying_Climax', False):
            signal = VolumeSignal.CLIMAX_TOP
            strength = 0.9
            description = "Buying climax - potential top"
        elif vol_ratio > 2.5 and price_change > 2:
            signal = VolumeSignal.BREAKOUT
            strength = min(vol_ratio / 5, 1.0)
            description = f"Volume breakout: {vol_ratio:.1f}x average with {price_change:.1f}% gain"
        elif accum > 0.7:
            signal = VolumeSignal.ACCUMULATION
            strength = accum
            description = "Smart money accumulation detected"
        elif distrib > 0.7:
            signal = VolumeSignal.DISTRIBUTION
            strength = distrib
            description = "Distribution/selling pattern detected"
        elif latest.get('SM_bullish_divergence', 0):
            signal = VolumeSignal.DIVERGENCE_BULLISH
            strength = 0.7
            description = "Bullish volume divergence"
        elif latest.get('SM_bearish_divergence', 0):
            signal = VolumeSignal.DIVERGENCE_BEARISH
            strength = 0.7
            description = "Bearish volume divergence"
        elif vol_ratio < 0.3:
            signal = VolumeSignal.DRY_UP
            strength = 0.5
            description = "Volume dry-up - potential consolidation"

        return VolumeAnalysisResult(
            symbol=symbol,
            date=str(df['date'].iloc[-1]) if 'date' in df.columns else '',
            signal=signal,
            signal_strength=strength,
            volume=int(df['volume'].iloc[-1]),
            avg_volume_20=float(df['volume'].rolling(20).mean().iloc[-1]),
            volume_ratio=float(vol_ratio),
            delivery_pct=float(delivery_pct),
            price_change_pct=float(price_change),
            description=description,
            details={
                'cmf': float(latest.get('CMF_20', 0)),
                'mfi': float(latest.get('MFI_14', 0)),
                'obv_divergence': int(latest.get('OBV_Divergence', 0)),
                'vsa_net_score': int(vsa_bull - vsa_bear),
                'accumulation_score': float(accum),
                'distribution_score': float(distrib),
            }
        )

    def get_volume_profile(
        self,
        df: pd.DataFrame,
        num_bins: int = 50
    ) -> Dict:
        """Build volume profile"""
        return self.profile_analyzer.build_volume_profile(df, num_bins)

    def scan_volume_signals(
        self,
        stock_data: Dict[str, pd.DataFrame]
    ) -> List[VolumeAnalysisResult]:
        """Scan multiple stocks for volume signals"""
        results = []

        for symbol, df in stock_data.items():
            try:
                signal = self.get_current_signal(df, symbol)
                if signal.signal != VolumeSignal.NO_SIGNAL:
                    results.append(signal)
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")

        results.sort(key=lambda x: x.signal_strength, reverse=True)
        return results

