"""
Comprehensive Technical Indicators for Indian Stock Market
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


class TechnicalIndicators:
    """Calculate all technical indicators"""
    
    @staticmethod
    def SMA(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def EMA(data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def WMA(data: pd.Series, period: int) -> pd.Series:
        """Weighted Moving Average"""
        weights = np.arange(1, period + 1)
        return data.rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    @staticmethod
    def VWAP(df: pd.DataFrame) -> pd.Series:
        """Volume Weighted Average Price"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        return (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    
    @staticmethod
    def RSI(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def MACD(
        data: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def BollingerBands(
        data: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        middle = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower
    
    @staticmethod
    def ATR(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def SuperTrend(
        df: pd.DataFrame,
        period: int = 10,
        multiplier: float = 3.0
    ) -> Tuple[pd.Series, pd.Series]:
        """SuperTrend Indicator (very popular in Indian markets)"""
        hl2 = (df['high'] + df['low']) / 2
        atr = TechnicalIndicators.ATR(df, period)
        
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)
        
        supertrend.iloc[0] = upper_band.iloc[0]
        direction.iloc[0] = 1
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] > upper_band.iloc[i - 1]:
                direction.iloc[i] = 1
            elif df['close'].iloc[i] < lower_band.iloc[i - 1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i - 1]
                
                if direction.iloc[i] == 1 and lower_band.iloc[i] < lower_band.iloc[i - 1]:
                    lower_band.iloc[i] = lower_band.iloc[i - 1]
                if direction.iloc[i] == -1 and upper_band.iloc[i] > upper_band.iloc[i - 1]:
                    upper_band.iloc[i] = upper_band.iloc[i - 1]
            
            supertrend.iloc[i] = lower_band.iloc[i] if direction.iloc[i] == 1 else upper_band.iloc[i]
        
        return supertrend, direction
    
    @staticmethod
    def Stochastic(
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator"""
        lowest_low = df['low'].rolling(window=k_period).min()
        highest_high = df['high'].rolling(window=k_period).max()
        
        k_line = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
        d_line = k_line.rolling(window=d_period).mean()
        
        return k_line, d_line
    
    @staticmethod
    def ADX(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Average Directional Index"""
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # When both are positive, keep only the larger
        mask = plus_dm > minus_dm
        minus_dm[mask & (plus_dm > 0)] = 0
        plus_dm[~mask & (minus_dm > 0)] = 0
        
        atr = TechnicalIndicators.ATR(df, period)
        
        plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period).mean() / atr)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=1/period).mean()
        
        return adx, plus_di, minus_di
    
    @staticmethod
    def OBV(df: pd.DataFrame) -> pd.Series:
        """On Balance Volume"""
        obv = pd.Series(index=df.index, dtype=float)
        obv.iloc[0] = df['volume'].iloc[0]
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] - df['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i - 1]
        
        return obv
    
    @staticmethod
    def VWMA(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Volume Weighted Moving Average"""
        return (df['close'] * df['volume']).rolling(period).sum() / df['volume'].rolling(period).sum()
    
    @staticmethod
    def CCI(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )
        cci = (typical_price - sma) / (0.015 * mean_deviation)
        return cci
    
    @staticmethod
    def Williams_R(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Williams %R"""
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()
        
        williams_r = -100 * (highest_high - df['close']) / (highest_high - lowest_low)
        return williams_r
    
    @staticmethod
    def MFI(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Money Flow Index"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        positive_flow = pd.Series(0.0, index=df.index)
        negative_flow = pd.Series(0.0, index=df.index)
        
        for i in range(1, len(df)):
            if typical_price.iloc[i] > typical_price.iloc[i - 1]:
                positive_flow.iloc[i] = money_flow.iloc[i]
            else:
                negative_flow.iloc[i] = money_flow.iloc[i]
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi
    
    @staticmethod
    def Ichimoku(
        df: pd.DataFrame,
        tenkan: int = 9,
        kijun: int = 26,
        senkou_b: int = 52
    ) -> Dict:
        """Ichimoku Cloud"""
        tenkan_sen = (
            df['high'].rolling(window=tenkan).max() + 
            df['low'].rolling(window=tenkan).min()
        ) / 2
        
        kijun_sen = (
            df['high'].rolling(window=kijun).max() + 
            df['low'].rolling(window=kijun).min()
        ) / 2
        
        senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
        
        senkou_b_line = (
            (df['high'].rolling(window=senkou_b).max() + 
             df['low'].rolling(window=senkou_b).min()) / 2
        ).shift(kijun)
        
        chikou_span = df['close'].shift(-kijun)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_a': senkou_a,
            'senkou_b': senkou_b_line,
            'chikou_span': chikou_span
        }
    
    @staticmethod
    def PivotPoints(df: pd.DataFrame) -> Dict:
        """Calculate Pivot Points (Standard)"""
        pivot = (df['high'] + df['low'] + df['close']) / 3
        
        r1 = 2 * pivot - df['low']
        s1 = 2 * pivot - df['high']
        r2 = pivot + (df['high'] - df['low'])
        s2 = pivot - (df['high'] - df['low'])
        r3 = df['high'] + 2 * (pivot - df['low'])
        s3 = df['low'] - 2 * (df['high'] - pivot)
        
        return {
            'pivot': pivot,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3
        }
    
    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all indicators at once"""
        result = df.copy()
        ti = TechnicalIndicators
        
        # Moving Averages
        for period in [9, 20, 50, 100, 200]:
            result[f'SMA_{period}'] = ti.SMA(df['close'], period)
            result[f'EMA_{period}'] = ti.EMA(df['close'], period)
        
        # RSI
        result['RSI_14'] = ti.RSI(df['close'], 14)
        result['RSI_7'] = ti.RSI(df['close'], 7)
        
        # MACD
        result['MACD'], result['MACD_Signal'], result['MACD_Hist'] = ti.MACD(df['close'])
        
        # Bollinger Bands
        result['BB_Upper'], result['BB_Middle'], result['BB_Lower'] = ti.BollingerBands(df['close'])
        result['BB_Width'] = (result['BB_Upper'] - result['BB_Lower']) / result['BB_Middle']
        
        # ATR
        result['ATR_14'] = ti.ATR(df, 14)
        
        # SuperTrend
        result['SuperTrend'], result['ST_Direction'] = ti.SuperTrend(df)
        
        # Stochastic
        result['Stoch_K'], result['Stoch_D'] = ti.Stochastic(df)
        
        # ADX
        result['ADX'], result['Plus_DI'], result['Minus_DI'] = ti.ADX(df)
        
        # Volume indicators
        result['OBV'] = ti.OBV(df)
        result['VWMA_20'] = ti.VWMA(df, 20)
        result['MFI_14'] = ti.MFI(df, 14)
        
        # CCI
        result['CCI_20'] = ti.CCI(df, 20)
        
        # Williams %R
        result['Williams_R'] = ti.Williams_R(df)
        
        # Volume metrics
        result['Volume_SMA_20'] = ti.SMA(df['volume'], 20)
        result['Volume_Ratio'] = df['volume'] / result['Volume_SMA_20']
        
        # Price metrics
        result['Daily_Return'] = df['close'].pct_change()
        result['Log_Return'] = np.log(df['close'] / df['close'].shift(1))
        result['Volatility_20'] = result['Daily_Return'].rolling(20).std() * np.sqrt(252)
        
        # Distance from moving averages
        result['Dist_SMA_50'] = (df['close'] - result['SMA_50']) / result['SMA_50'] * 100
        result['Dist_SMA_200'] = (df['close'] - result['SMA_200']) / result['SMA_200'] * 100
        
        # 52-week high/low
        result['52W_High'] = df['high'].rolling(252).max()
        result['52W_Low'] = df['low'].rolling(252).min()
        result['Dist_52W_High'] = (df['close'] - result['52W_High']) / result['52W_High'] * 100
        result['Dist_52W_Low'] = (df['close'] - result['52W_Low']) / result['52W_Low'] * 100
        
        return result
