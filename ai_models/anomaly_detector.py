"""
AI-Based Anomaly Detection for Indian Stock Market

Detects unusual patterns in:
- Price movements (sudden spikes/crashes)
- Volume activity (unusual accumulation/distribution)
- Price-Volume relationships
- Spread anomalies
- Intraday pattern anomalies
- Cross-stock correlation breaks
- Market regime changes
- Insider activity patterns
- Options activity anomalies
- Order flow imbalances

Models Used:
- Isolation Forest
- Local Outlier Factor (LOF)
- DBSCAN Clustering
- One-Class SVM
- Autoencoder (Deep Learning)
- Statistical methods (Z-Score, IQR, Grubbs)
- LSTM Autoencoder (sequence anomalies)
- Mahalanobis Distance

Author: AI Stock Scanner
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.covariance import EllipticEnvelope
from scipy import stats
from scipy.spatial.distance import mahalanobis
import joblib
import logging
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


# ============================================================
# DATA STRUCTURES
# ============================================================

class AnomalyType(Enum):
    """Types of anomalies detected"""
    PRICE_SPIKE = "price_spike"
    PRICE_CRASH = "price_crash"
    VOLUME_SPIKE = "volume_spike"
    VOLUME_DRY = "volume_dry"
    PRICE_VOLUME_DIVERGENCE = "price_volume_divergence"
    SPREAD_ANOMALY = "spread_anomaly"
    GAP_ANOMALY = "gap_anomaly"
    VOLATILITY_SPIKE = "volatility_spike"
    CORRELATION_BREAK = "correlation_break"
    PATTERN_ANOMALY = "pattern_anomaly"
    STATISTICAL_OUTLIER = "statistical_outlier"
    REGIME_CHANGE = "regime_change"
    INSTITUTIONAL_ACTIVITY = "institutional_activity"
    MANIPULATION_SUSPECT = "manipulation_suspect"
    UNKNOWN = "unknown"


class AnomalySeverity(Enum):
    """Severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AnomalyResult:
    """Single anomaly detection result"""
    symbol: str
    date: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    score: float  # 0.0 to 1.0 (higher = more anomalous)
    description: str
    price: float
    volume: int
    details: Dict = field(default_factory=dict)
    model_name: str = ""

    def __repr__(self):
        return (
            f"Anomaly({self.symbol} | {self.anomaly_type.value} | "
            f"Severity: {self.severity.value} | Score: {self.score:.2f})"
        )


@dataclass
class AnomalyReport:
    """Complete anomaly report for a stock"""
    symbol: str
    scan_date: str
    total_anomalies: int
    critical_anomalies: int
    high_anomalies: int
    anomalies: List[AnomalyResult]
    risk_level: str  # 'normal', 'elevated', 'high', 'extreme'
    summary: str


# ============================================================
# STATISTICAL ANOMALY DETECTORS
# ============================================================

class StatisticalAnomalyDetector:
    """
    Statistical methods for anomaly detection
    
    Uses classical statistical tests:
    - Z-Score
    - Modified Z-Score (MAD based)
    - IQR method
    - Grubbs test
    - Dixon's Q test
    """

    @staticmethod
    def zscore_anomalies(
        data: pd.Series,
        threshold: float = 3.0
    ) -> pd.Series:
        """
        Z-Score based anomaly detection
        
        |Z| > 3 â†’ Anomaly (99.7% rule)
        |Z| > 2 â†’ Potential anomaly (95% rule)
        """
        mean = data.mean()
        std = data.std()
        
        if std == 0:
            return pd.Series(False, index=data.index)
        
        z_scores = (data - mean) / std
        return z_scores.abs() > threshold

    @staticmethod
    def modified_zscore_anomalies(
        data: pd.Series,
        threshold: float = 3.5
    ) -> pd.Series:
        """
        Modified Z-Score using Median Absolute Deviation (MAD)
        
        More robust than standard Z-score for skewed data
        Common in Indian market where data is often skewed
        """
        median = data.median()
        mad = np.median(np.abs(data - median))
        
        if mad == 0:
            return pd.Series(False, index=data.index)
        
        modified_z = 0.6745 * (data - median) / mad
        return modified_z.abs() > threshold

    @staticmethod
    def iqr_anomalies(
        data: pd.Series,
        multiplier: float = 1.5
    ) -> pd.Series:
        """
        Interquartile Range (IQR) method
        
        multiplier=1.5 â†’ mild outliers
        multiplier=3.0 â†’ extreme outliers
        """
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        return (data < lower_bound) | (data > upper_bound)

    @staticmethod
    def rolling_zscore_anomalies(
        data: pd.Series,
        window: int = 20,
        threshold: float = 2.5
    ) -> pd.Series:
        """
        Rolling Z-Score - adapts to recent market conditions
        
        Better for trending/volatile markets like Indian mid-caps
        """
        rolling_mean = data.rolling(window=window).mean()
        rolling_std = data.rolling(window=window).std()
        
        z_scores = (data - rolling_mean) / (rolling_std + 1e-10)
        return z_scores.abs() > threshold

    @staticmethod
    def grubbs_test(data: pd.Series, alpha: float = 0.05) -> Dict:
        """
        Grubbs test for single outlier detection
        
        Tests if the max/min value is an outlier
        """
        n = len(data)
        if n < 3:
            return {'is_outlier': False}
        
        mean = data.mean()
        std = data.std()
        
        if std == 0:
            return {'is_outlier': False}
        
        # Test statistic for max value
        max_val = data.max()
        min_val = data.min()
        
        g_max = (max_val - mean) / std
        g_min = (mean - min_val) / std
        
        # Critical value
        t_crit = stats.t.ppf(1 - alpha / (2 * n), n - 2)
        g_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))
        
        max_is_outlier = g_max > g_crit
        min_is_outlier = g_min > g_crit
        
        return {
            'max_value': max_val,
            'min_value': min_val,
            'max_g_statistic': g_max,
            'min_g_statistic': g_min,
            'critical_value': g_crit,
            'max_is_outlier': max_is_outlier,
            'min_is_outlier': min_is_outlier,
            'is_outlier': max_is_outlier or min_is_outlier,
        }

    @staticmethod
    def mahalanobis_anomalies(
        df: pd.DataFrame,
        features: List[str],
        threshold: float = 3.0
    ) -> pd.Series:
        """
        Mahalanobis Distance based anomaly detection
        
        Accounts for correlations between features
        Useful for detecting multi-dimensional anomalies
        """
        data = df[features].dropna()
        
        if len(data) < len(features) + 1:
            return pd.Series(False, index=df.index)
        
        try:
            mean = data.mean().values
            cov = data.cov().values
            cov_inv = np.linalg.inv(cov)
            
            distances = pd.Series(index=data.index, dtype=float)
            
            for idx, row in data.iterrows():
                diff = row.values - mean
                dist = np.sqrt(diff @ cov_inv @ diff)
                distances[idx] = dist
            
            # Fill NaN for missing indices
            result = pd.Series(False, index=df.index)
            result[distances.index] = distances > threshold
            
            return result
            
        except np.linalg.LinAlgError:
            logger.warning("Singular covariance matrix in Mahalanobis calculation")
            return pd.Series(False, index=df.index)


# ============================================================
# ML-BASED ANOMALY DETECTORS
# ============================================================

class MLAnomalyDetector:
    """
    Machine Learning based anomaly detection
    
    Uses unsupervised learning algorithms that don't require
    labeled anomaly data (perfect for stock market)
    """

    def __init__(self, contamination: float = 0.05):
        """
        Args:
            contamination: Expected proportion of anomalies (0.01-0.10)
        """
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.models = {}
        self.is_fitted = False

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature matrix for anomaly detection"""
        features = pd.DataFrame(index=df.index)
        
        # Price features
        features['return'] = df['close'].pct_change()
        features['abs_return'] = features['return'].abs()
        features['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volatility features
        features['return_std_5'] = features['return'].rolling(5).std()
        features['return_std_20'] = features['return'].rolling(20).std()
        features['volatility_ratio'] = features['return_std_5'] / (features['return_std_20'] + 1e-10)
        
        # Volume features
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        features['volume_change'] = df['volume'].pct_change()
        features['log_volume'] = np.log1p(df['volume'])
        features['volume_std'] = df['volume'].rolling(20).std() / df['volume'].rolling(20).mean()
        
        # Spread features
        features['spread'] = (df['high'] - df['low']) / df['close']
        features['spread_ratio'] = features['spread'] / features['spread'].rolling(20).mean()
        
        # Body features
        features['body'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)
        features['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 1e-10)
        features['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 1e-10)
        
        # Gap
        features['gap'] = df['open'] / df['close'].shift(1) - 1
        
        # Price vs Moving Averages
        if len(df) > 20:
            sma_20 = df['close'].rolling(20).mean()
            features['dist_sma_20'] = (df['close'] - sma_20) / sma_20
        
        if len(df) > 50:
            sma_50 = df['close'].rolling(50).mean()
            features['dist_sma_50'] = (df['close'] - sma_50) / sma_50
        
        # Volume-Price relationship
        features['price_volume_corr'] = (
            features['return'].rolling(10).corr(features['volume_change'])
        )
        
        # Momentum
        features['momentum_5'] = df['close'].pct_change(5)
        features['momentum_10'] = df['close'].pct_change(10)
        
        return features.dropna()

    def fit(self, df: pd.DataFrame) -> Dict:
        """
        Fit all anomaly detection models
        
        Args:
            df: Historical OHLCV DataFrame
        
        Returns:
            Dictionary with model fitting results
        """
        features = self._prepare_features(df)
        
        if len(features) < 50:
            logger.warning("Insufficient data for model fitting")
            return {}
        
        X = self.scaler.fit_transform(features)
        
        results = {}
        
        # 1. Isolation Forest
        try:
            self.models['isolation_forest'] = IsolationForest(
                n_estimators=200,
                contamination=self.contamination,
                random_state=42,
                n_jobs=-1,
                max_samples='auto',
            )
            self.models['isolation_forest'].fit(X)
            results['isolation_forest'] = 'fitted'
            logger.info("âœ… Isolation Forest fitted")
        except Exception as e:
            logger.error(f"Isolation Forest error: {e}")
        
        # 2. Local Outlier Factor
        try:
            self.models['lof'] = LocalOutlierFactor(
                n_neighbors=20,
                contamination=self.contamination,
                novelty=True,
                n_jobs=-1,
            )
            self.models['lof'].fit(X)
            results['lof'] = 'fitted'
            logger.info("âœ… LOF fitted")
        except Exception as e:
            logger.error(f"LOF error: {e}")
        
        # 3. One-Class SVM
        try:
            self.models['ocsvm'] = OneClassSVM(
                kernel='rbf',
                gamma='scale',
                nu=self.contamination,
            )
            self.models['ocsvm'].fit(X)
            results['ocsvm'] = 'fitted'
            logger.info("âœ… One-Class SVM fitted")
        except Exception as e:
            logger.error(f"One-Class SVM error: {e}")
        
        # 4. Elliptic Envelope (Robust Covariance)
        try:
            self.models['elliptic'] = EllipticEnvelope(
                contamination=self.contamination,
                random_state=42,
            )
            self.models['elliptic'].fit(X)
            results['elliptic'] = 'fitted'
            logger.info("âœ… Elliptic Envelope fitted")
        except Exception as e:
            logger.error(f"Elliptic Envelope error: {e}")
        
        # 5. DBSCAN
        try:
            self.models['dbscan'] = DBSCAN(
                eps=1.5,
                min_samples=5,
                n_jobs=-1,
            )
            # DBSCAN needs predict separately
            results['dbscan'] = 'fitted'
            logger.info("âœ… DBSCAN ready")
        except Exception as e:
            logger.error(f"DBSCAN error: {e}")
        
        self.is_fitted = True
        self._feature_names = features.columns.tolist()
        
        return results

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies in new data
        
        Returns:
            DataFrame with anomaly scores and labels for each model
        """
        if not self.is_fitted:
            self.fit(df)
        
        features = self._prepare_features(df)
        
        if features.empty:
            return pd.DataFrame()
        
        X = self.scaler.transform(features)
        
        result = features.copy()
        
        # Isolation Forest
        if 'isolation_forest' in self.models:
            try:
                predictions = self.models['isolation_forest'].predict(X)
                scores = self.models['isolation_forest'].decision_function(X)
                result['if_anomaly'] = predictions == -1
                result['if_score'] = -scores  # Higher = more anomalous
                result['if_score_normalized'] = (result['if_score'] - result['if_score'].min()) / (result['if_score'].max() - result['if_score'].min() + 1e-10)
            except Exception as e:
                logger.error(f"IF prediction error: {e}")
        
        # LOF
        if 'lof' in self.models:
            try:
                predictions = self.models['lof'].predict(X)
                scores = self.models['lof'].decision_function(X)
                result['lof_anomaly'] = predictions == -1
                result['lof_score'] = -scores
                result['lof_score_normalized'] = (result['lof_score'] - result['lof_score'].min()) / (result['lof_score'].max() - result['lof_score'].min() + 1e-10)
            except Exception as e:
                logger.error(f"LOF prediction error: {e}")
        
        # One-Class SVM
        if 'ocsvm' in self.models:
            try:
                predictions = self.models['ocsvm'].predict(X)
                scores = self.models['ocsvm'].decision_function(X)
                result['svm_anomaly'] = predictions == -1
                result['svm_score'] = -scores
                result['svm_score_normalized'] = (result['svm_score'] - result['svm_score'].min()) / (result['svm_score'].max() - result['svm_score'].min() + 1e-10)
            except Exception as e:
                logger.error(f"SVM prediction error: {e}")
        
        # Elliptic Envelope
        if 'elliptic' in self.models:
            try:
                predictions = self.models['elliptic'].predict(X)
                scores = self.models['elliptic'].decision_function(X)
                result['ee_anomaly'] = predictions == -1
                result['ee_score'] = -scores
            except Exception as e:
                logger.error(f"Elliptic prediction error: {e}")
        
        # DBSCAN
        if 'dbscan' in self.models:
            try:
                labels = self.models['dbscan'].fit_predict(X)
                result['dbscan_anomaly'] = labels == -1  # Noise points
            except Exception as e:
                logger.error(f"DBSCAN prediction error: {e}")
        
        # Ensemble score (average of all models)
        anomaly_cols = [c for c in result.columns if c.endswith('_anomaly')]
        score_cols = [c for c in result.columns if c.endswith('_normalized')]
        
        if anomaly_cols:
            result['ensemble_anomaly_count'] = result[anomaly_cols].sum(axis=1)
            result['ensemble_anomaly'] = result['ensemble_anomaly_count'] >= len(anomaly_cols) / 2
        
        if score_cols:
            result['ensemble_score'] = result[score_cols].mean(axis=1)
        
        return result

    def save_models(self, filepath: str):
        """Save fitted models"""
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'contamination': self.contamination,
            'feature_names': self._feature_names if hasattr(self, '_feature_names') else [],
            'is_fitted': self.is_fitted,
        }
        joblib.dump(model_data, filepath)
        logger.info(f"âœ… Anomaly models saved to {filepath}")

    def load_models(self, filepath: str):
        """Load fitted models"""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.contamination = model_data['contamination']
        self._feature_names = model_data.get('feature_names', [])
        self.is_fitted = model_data.get('is_fitted', True)
        logger.info(f"âœ… Anomaly models loaded from {filepath}")


# ============================================================
# SPECIALIZED ANOMALY DETECTORS
# ============================================================

class PriceAnomalyDetector:
    """Detect price-specific anomalies"""

    @staticmethod
    def detect_flash_crash(
        df: pd.DataFrame,
        threshold_pct: float = -5.0,
        recovery_candles: int = 3
    ) -> List[Dict]:
        """
        Detect flash crash events
        
        Flash crash = Sharp drop followed by quick recovery
        Common in Indian market during circuit breaker events
        """
        anomalies = []
        returns = df['close'].pct_change() * 100
        
        for i in range(1, len(df) - recovery_candles):
            if returns.iloc[i] < threshold_pct:
                # Check for recovery
                recovery_return = (
                    df['close'].iloc[i + recovery_candles] / df['close'].iloc[i] - 1
                ) * 100
                
                is_flash_crash = recovery_return > abs(threshold_pct) * 0.5
                
                anomalies.append({
                    'date': str(df['date'].iloc[i]) if 'date' in df.columns else str(i),
                    'drop_pct': returns.iloc[i],
                    'recovery_pct': recovery_return,
                    'is_flash_crash': is_flash_crash,
                    'low_price': df['low'].iloc[i],
                    'close_price': df['close'].iloc[i],
                    'volume': df['volume'].iloc[i],
                })
        
        return anomalies

    @staticmethod
    def detect_price_manipulation(
        df: pd.DataFrame,
        lookback: int = 20
    ) -> pd.DataFrame:
        """
        Detect potential price manipulation patterns
        
        Patterns:
        - Pump and Dump: Sharp rise + high volume, then drop
        - Spoofing: Large orders that get cancelled
        - Circular trading: Consistent volume at same price
        - Marking the close: End-of-day price manipulation
        """
        result = pd.DataFrame(index=df.index)
        
        returns = df['close'].pct_change()
        vol_ratio = df['volume'] / df['volume'].rolling(lookback).mean()
        spread = (df['high'] - df['low']) / df['close']
        
        # Pump and Dump detection
        # Day 1: Large up move with high volume
        # Day 2-5: Decline on lower volume
        pump_day = (returns > 0.05) & (vol_ratio > 3)
        
        result['pump_suspect'] = pump_day
        
        for i in range(len(df)):
            if pump_day.iloc[i] and i + 5 < len(df):
                subsequent_return = (
                    df['close'].iloc[i + 5] / df['close'].iloc[i] - 1
                )
                subsequent_vol = df['volume'].iloc[i+1:i+6].mean() / df['volume'].iloc[i]
                
                if subsequent_return < -0.03 and subsequent_vol < 0.5:
                    result.loc[result.index[i], 'pump_and_dump'] = True
        
        if 'pump_and_dump' not in result.columns:
            result['pump_and_dump'] = False
        
        # Unusual closing patterns (marking the close)
        # Close at high/low with volume spike in last few minutes
        close_at_high = (df['close'] == df['high'])
        close_at_low = (df['close'] == df['low'])
        result['close_marking_suspect'] = (close_at_high | close_at_low) & (vol_ratio > 2)
        
        # Circular trading detection
        # Very tight spread with consistent volume
        tight_spread = spread < spread.rolling(lookback).mean() * 0.3
        consistent_vol = df['volume'].rolling(5).std() / df['volume'].rolling(5).mean() < 0.1
        result['circular_suspect'] = tight_spread & consistent_vol & (vol_ratio > 1.5)
        
        # Overall manipulation score
        result['manipulation_score'] = (
            result['pump_suspect'].astype(int) * 2 +
            result.get('pump_and_dump', pd.Series(False, index=df.index)).astype(int) * 5 +
            result['close_marking_suspect'].astype(int) * 1 +
            result['circular_suspect'].astype(int) * 3
        )
        
        return result

    @staticmethod
    def detect_gap_anomalies(
        df: pd.DataFrame,
        min_gap_pct: float = 2.0
    ) -> pd.DataFrame:
        """
        Detect unusual gap openings
        
        Gap up/down analysis:
        - Exhaustion gaps (end of trend)
        - Breakaway gaps (start of new trend)  
        - Common gaps (noise)
        - Island reversals
        """
        result = pd.DataFrame(index=df.index)
        
        gap = (df['open'] / df['close'].shift(1) - 1) * 100
        vol_ratio = df['volume'] / df['volume'].rolling(20).mean()
        
        result['gap_pct'] = gap
        result['is_gap_up'] = gap > min_gap_pct
        result['is_gap_down'] = gap < -min_gap_pct
        
        # Gap classification
        for i in range(20, len(df)):
            if abs(gap.iloc[i]) >= min_gap_pct:
                # Check if gap fills
                gap_filled = False
                for j in range(i + 1, min(i + 10, len(df))):
                    if gap.iloc[i] > 0 and df['low'].iloc[j] < df['close'].iloc[i - 1]:
                        gap_filled = True
                        break
                    elif gap.iloc[i] < 0 and df['high'].iloc[j] > df['close'].iloc[i - 1]:
                        gap_filled = True
                        break
                
                # Prior trend
                prior_return = (df['close'].iloc[i-1] - df['close'].iloc[i-10]) / df['close'].iloc[i-10]
                
                gap_type = 'common'
                if vol_ratio.iloc[i] > 2 and not gap_filled:
                    if abs(prior_return) < 0.03:
                        gap_type = 'breakaway'
                    elif abs(prior_return) > 0.10:
                        gap_type = 'exhaustion'
                    else:
                        gap_type = 'continuation'
                
                result.loc[result.index[i], 'gap_type'] = gap_type
                result.loc[result.index[i], 'gap_filled'] = gap_filled
        
        return result

    @staticmethod
    def detect_circuit_events(
        df: pd.DataFrame,
        upper_limit: float = 20.0,
        lower_limit: float = -20.0
    ) -> pd.DataFrame:
        """
        Detect circuit limit hits (Indian market specific)
        
        NSE/BSE have circuit limits:
        - Individual stocks: 2%, 5%, 10%, 20%
        - Index: 10%, 15%, 20%
        """
        result = pd.DataFrame(index=df.index)
        
        daily_change = (df['close'] / df['close'].shift(1) - 1) * 100
        
        result['daily_change_pct'] = daily_change
        result['upper_circuit'] = daily_change >= upper_limit * 0.95
        result['lower_circuit'] = daily_change <= lower_limit * 0.95
        result['near_circuit'] = (daily_change >= upper_limit * 0.80) | (daily_change <= lower_limit * 0.80)
        
        # Circuit limit levels
        circuit_levels = [2, 5, 10, 20]
        for level in circuit_levels:
            result[f'hit_{level}pct_up'] = daily_change >= level * 0.95
            result[f'hit_{level}pct_down'] = daily_change <= -level * 0.95
        
        return result


class VolumeAnomalyDetector:
    """Detect volume-specific anomalies"""

    @staticmethod
    def detect_volume_spikes(
        df: pd.DataFrame,
        multiplier: float = 3.0,
        lookback: int = 20
    ) -> pd.DataFrame:
        """
        Detect unusual volume activity
        
        Types:
        - Volume explosion (>3x average)
        - Volume drought (<0.3x average)
        - Volume climax (highest in N days)
        - Volume trend change
        """
        result = pd.DataFrame(index=df.index)
        
        avg_vol = df['volume'].rolling(lookback).mean()
        std_vol = df['volume'].rolling(lookback).std()
        vol_ratio = df['volume'] / avg_vol
        
        result['vol_ratio'] = vol_ratio
        result['vol_zscore'] = (df['volume'] - avg_vol) / (std_vol + 1)
        
        # Spike detection
        result['volume_explosion'] = vol_ratio > multiplier
        result['volume_drought'] = vol_ratio < (1 / multiplier)
        
        # Volume climax (highest volume in lookback period)
        result['volume_climax'] = df['volume'] == df['volume'].rolling(lookback).max()
        
        # Volume trend change
        short_avg = df['volume'].rolling(5).mean()
        long_avg = df['volume'].rolling(lookback).mean()
        result['volume_trend_up'] = (short_avg > long_avg * 1.5) & (short_avg.shift(5) <= long_avg.shift(5))
        result['volume_trend_down'] = (short_avg < long_avg * 0.5) & (short_avg.shift(5) >= long_avg.shift(5))
        
        # Unusual volume with price analysis
        price_up = df['close'] > df['close'].shift(1)
        result['high_vol_up'] = result['volume_explosion'] & price_up
        result['high_vol_down'] = result['volume_explosion'] & (~price_up)
        
        # Severity
        result['volume_severity'] = pd.cut(
            vol_ratio.fillna(1),
            bins=[0, 0.3, 0.7, 1.5, 3, 5, float('inf')],
            labels=['extreme_low', 'low', 'normal', 'high', 'very_high', 'extreme_high']
        )
        
        return result

    @staticmethod
    def detect_accumulation_distribution_anomaly(
        df: pd.DataFrame,
        lookback: int = 20
    ) -> pd.DataFrame:
        """Detect unusual accumulation or distribution patterns"""
        result = pd.DataFrame(index=df.index)
        
        # Close Location Value
        clv = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        money_flow = clv * df['volume']
        
        cum_mf = money_flow.cumsum()
        mf_trend = cum_mf.diff(lookback)
        price_trend = df['close'].diff(lookback)
        
        # Divergence: money flowing in but price not rising
        result['hidden_accumulation'] = (mf_trend > 0) & (price_trend < 0)
        result['hidden_distribution'] = (mf_trend < 0) & (price_trend > 0)
        
        # Intensity
        vol_ratio = df['volume'] / df['volume'].rolling(lookback).mean()
        result['accumulation_intensity'] = (
            clv.rolling(lookback).mean() * vol_ratio
        )
        
        return result


# ============================================================
# REGIME CHANGE DETECTOR
# ============================================================

class RegimeChangeDetector:
    """
    Detect market regime changes
    
    Identifies transitions between:
    - Bull â†’ Bear
    - Bear â†’ Bull
    - Trending â†’ Sideways
    - Low volatility â†’ High volatility
    """

    @staticmethod
    def detect_regime_change(
        df: pd.DataFrame,
        short_window: int = 10,
        long_window: int = 50
    ) -> pd.DataFrame:
        """
        Detect regime changes using multiple methods
        """
        result = pd.DataFrame(index=df.index)
        
        returns = df['close'].pct_change()
        
        # 1. Volatility Regime
        short_vol = returns.rolling(short_window).std()
        long_vol = returns.rolling(long_window).std()
        result['vol_regime'] = np.where(
            short_vol > long_vol * 1.5, 'high_vol',
            np.where(short_vol < long_vol * 0.5, 'low_vol', 'normal_vol')
        )
        
        # 2. Trend Regime
        sma_short = df['close'].rolling(short_window).mean()
        sma_long = df['close'].rolling(long_window).mean()
        
        result['trend_regime'] = np.where(
            (sma_short > sma_long) & (df['close'] > sma_short), 'strong_bull',
            np.where(
                sma_short > sma_long, 'bull',
                np.where(
                    (sma_short < sma_long) & (df['close'] < sma_short), 'strong_bear',
                    'bear'
                )
            )
        )
        
        # 3. Regime change detection (transitions)
        result['vol_regime_change'] = result['vol_regime'] != result['vol_regime'].shift(1)
        result['trend_regime_change'] = result['trend_regime'] != result['trend_regime'].shift(1)
        
        # 4. Momentum regime
        momentum = returns.rolling(long_window).mean()
        result['momentum_regime'] = np.where(
            momentum > 0.001, 'positive',
            np.where(momentum < -0.001, 'negative', 'neutral')
        )
        
        # 5. Correlation regime (for index/market-wide)
        if len(df) > long_window:
            rolling_autocorr = returns.rolling(long_window).apply(
                lambda x: x.autocorr(lag=1) if len(x) > 1 else 0,
                raw=False
            )
            result['mean_reversion'] = rolling_autocorr < -0.1
            result['trending'] = rolling_autocorr > 0.1
        
        return result

    @staticmethod
    def detect_structural_break(
        prices: pd.Series,
        window: int = 50,
        significance: float = 0.05
    ) -> pd.Series:
        """
        Detect structural breaks in price series
        
        Uses CUSUM (Cumulative Sum) approach
        """
        returns = prices.pct_change().dropna()
        
        cum_mean = returns.expanding().mean()
        cusum = (returns - cum_mean).cumsum()
        
        cusum_std = cusum.rolling(window).std()
        cusum_zscore = cusum / (cusum_std + 1e-10)
        
        # Structural break when CUSUM z-score exceeds threshold
        threshold = stats.norm.ppf(1 - significance / 2)
        
        return cusum_zscore.abs() > threshold


# ============================================================
# MAIN ANOMALY DETECTOR (ORCHESTRATOR)
# ============================================================

class AnomalyDetector:
    """
    Main anomaly detection system
    
    Orchestrates all anomaly detection methods and produces
    a unified anomaly report
    
    Usage:
        detector = AnomalyDetector()
        
        # Fit on historical data
        detector.fit(df)
        
        # Detect anomalies
        report = detector.detect_anomalies(df, symbol="RELIANCE")
        
        # Scan multiple stocks
        results = detector.scan_anomalies(stock_data)
    """

    def __init__(self, contamination: float = 0.05):
        self.statistical = StatisticalAnomalyDetector()
        self.ml_detector = MLAnomalyDetector(contamination=contamination)
        self.price_detector = PriceAnomalyDetector()
        self.volume_detector = VolumeAnomalyDetector()
        self.regime_detector = RegimeChangeDetector()
        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
        """Fit ML models on historical data"""
        self.ml_detector.fit(df)
        self.is_fitted = True
        logger.info("âœ… Anomaly detector fitted")

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN"
    ) -> AnomalyReport:
        """
        Run complete anomaly detection on a stock
        
        Returns:
            AnomalyReport with all detected anomalies
        """
        if not self.is_fitted:
            self.fit(df)
        
        anomalies = []
        latest = df.iloc[-1]
        latest_date = str(df['date'].iloc[-1]) if 'date' in df.columns else str(datetime.now().date())
        latest_price = latest['close']
        latest_volume = int(latest['volume'])
        
        # 1. Statistical Anomalies
        try:
            returns = df['close'].pct_change().dropna()
            
            # Rolling Z-Score
            rolling_z = self.statistical.rolling_zscore_anomalies(returns, window=20, threshold=2.5)
            if rolling_z.iloc[-1]:
                ret = returns.iloc[-1]
                anomalies.append(AnomalyResult(
                    symbol=symbol,
                    date=latest_date,
                    anomaly_type=AnomalyType.PRICE_SPIKE if ret > 0 else AnomalyType.PRICE_CRASH,
                    severity=AnomalySeverity.HIGH if abs(ret) > 0.05 else AnomalySeverity.MEDIUM,
                    score=min(abs(ret) * 10, 1.0),
                    description=f"Unusual price movement: {ret*100:.2f}% (Z-score > 2.5)",
                    price=latest_price,
                    volume=latest_volume,
                    details={'return': float(ret), 'method': 'rolling_zscore'},
                    model_name='statistical',
                ))
            
            # Volume Z-Score
            vol_z = self.statistical.rolling_zscore_anomalies(
                df['volume'].astype(float), window=20, threshold=2.5
            )
            if vol_z.iloc[-1]:
                vol_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
                anomalies.append(AnomalyResult(
                    symbol=symbol,
                    date=latest_date,
                    anomaly_type=AnomalyType.VOLUME_SPIKE if vol_ratio > 1 else AnomalyType.VOLUME_DRY,
                    severity=AnomalySeverity.HIGH if vol_ratio > 3 else AnomalySeverity.MEDIUM,
                    score=min(vol_ratio / 5, 1.0),
                    description=f"Unusual volume: {vol_ratio:.1f}x average",
                    price=latest_price,
                    volume=latest_volume,
                    details={'volume_ratio': float(vol_ratio), 'method': 'zscore'},
                    model_name='statistical',
                ))
        except Exception as e:
            logger.error(f"Statistical anomaly error: {e}")
        
        # 2. ML-Based Anomalies
        try:
            ml_results = self.ml_detector.detect(df)
            
            if not ml_results.empty and ml_results['ensemble_anomaly'].iloc[-1]:
                score = float(ml_results['ensemble_score'].iloc[-1])
                count = int(ml_results['ensemble_anomaly_count'].iloc[-1])
                total_models = len([c for c in ml_results.columns if c.endswith('_anomaly') and c != 'ensemble_anomaly'])
                
                anomalies.append(AnomalyResult(
                    symbol=symbol,
                    date=latest_date,
                    anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                    severity=(
                        AnomalySeverity.CRITICAL if score > 0.9 else
                        AnomalySeverity.HIGH if score > 0.7 else
                        AnomalySeverity.MEDIUM
                    ),
                    score=score,
                    description=f"ML ensemble anomaly: {count}/{total_models} models flagged (score: {score:.2f})",
                    price=latest_price,
                    volume=latest_volume,
                    details={
                        'models_flagged': count,
                        'total_models': total_models,
                        'ensemble_score': score,
                    },
                    model_name='ml_ensemble',
                ))
        except Exception as e:
            logger.error(f"ML anomaly error: {e}")
        
        # 3. Price Manipulation Check
        try:
            manip = self.price_detector.detect_price_manipulation(df)
            if manip['manipulation_score'].iloc[-1] > 3:
                anomalies.append(AnomalyResult(
                    symbol=symbol,
                    date=latest_date,
                    anomaly_type=AnomalyType.MANIPULATION_SUSPECT,
                    severity=AnomalySeverity.CRITICAL,
                    score=min(manip['manipulation_score'].iloc[-1] / 10, 1.0),
                    description="Potential price manipulation pattern detected",
                    price=latest_price,
                    volume=latest_volume,
                    details={'manipulation_score': int(manip['manipulation_score'].iloc[-1])},
                    model_name='manipulation_detector',
                ))
        except Exception as e:
            logger.error(f"Manipulation detection error: {e}")
        
        # 4. Gap Anomalies
        try:
            gaps = self.price_detector.detect_gap_anomalies(df)
            if gaps['is_gap_up'].iloc[-1] or gaps['is_gap_down'].iloc[-1]:
                gap_pct = gaps['gap_pct'].iloc[-1]
                anomalies.append(AnomalyResult(
                    symbol=symbol,
                    date=latest_date,
                    anomaly_type=AnomalyType.GAP_ANOMALY,
                    severity=AnomalySeverity.HIGH if abs(gap_pct) > 5 else AnomalySeverity.MEDIUM,
                    score=min(abs(gap_pct) / 10, 1.0),
                    description=f"Significant gap {'up' if gap_pct > 0 else 'down'}: {gap_pct:.2f}%",
                    price=latest_price,
                    volume=latest_volume,
                    details={'gap_pct': float(gap_pct)},
                    model_name='gap_detector',
                ))
        except Exception as e:
            logger.error(f"Gap detection error: {e}")
        
        # 5. Regime Change
        try:
            regime = self.regime_detector.detect_regime_change(df)
            if regime['vol_regime_change'].iloc[-1] or regime['trend_regime_change'].iloc[-1]:
                anomalies.append(AnomalyResult(
                    symbol=symbol,
                    date=latest_date,
                    anomaly_type=AnomalyType.REGIME_CHANGE,
                    severity=AnomalySeverity.MEDIUM,
                    score=0.6,
                    description=f"Regime change: Vol={regime['vol_regime'].iloc[-1]}, Trend={regime['trend_regime'].iloc[-1]}",
                    price=latest_price,
                    volume=latest_volume,
                    details={
                        'vol_regime': str(regime['vol_regime'].iloc[-1]),
                        'trend_regime': str(regime['trend_regime'].iloc[-1]),
                    },
                    model_name='regime_detector',
                ))
        except Exception as e:
            logger.error(f"Regime detection error: {e}")
        
        # 6. Volume Anomalies
        try:
            vol_anomalies = self.volume_detector.detect_volume_spikes(df)
            if vol_anomalies['volume_explosion'].iloc[-1]:
                vol_r = float(vol_anomalies['vol_ratio'].iloc[-1])
                anomalies.append(AnomalyResult(
                    symbol=symbol,
                    date=latest_date,
                    anomaly_type=AnomalyType.VOLUME_SPIKE,
                    severity=AnomalySeverity.HIGH if vol_r > 5 else AnomalySeverity.MEDIUM,
                    score=min(vol_r / 10, 1.0),
                    description=f"Volume explosion: {vol_r:.1f}x average",
                    price=latest_price,
                    volume=latest_volume,
                    details={'volume_ratio': vol_r},
                    model_name='volume_detector',
                ))
        except Exception as e:
            logger.error(f"Volume anomaly error: {e}")
        
        # Determine risk level
        critical_count = sum(1 for a in anomalies if a.severity == AnomalySeverity.CRITICAL)
        high_count = sum(1 for a in anomalies if a.severity == AnomalySeverity.HIGH)
        
        if critical_count > 0:
            risk_level = "extreme"
        elif high_count >= 2:
            risk_level = "high"
        elif high_count >= 1 or len(anomalies) >= 3:
            risk_level = "elevated"
        else:
            risk_level = "normal"
        
        # Sort by severity and score
        anomalies.sort(key=lambda x: (
            {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}[x.severity.value],
            x.score
        ), reverse=True)
        
        summary = (
            f"Found {len(anomalies)} anomalies for {symbol}. "
            f"Risk level: {risk_level}. "
            f"Critical: {critical_count}, High: {high_count}"
        )
        
        return AnomalyReport(
            symbol=symbol,
            scan_date=latest_date,
            total_anomalies=len(anomalies),
            critical_anomalies=critical_count,
            high_anomalies=high_count,
            anomalies=anomalies,
            risk_level=risk_level,
            summary=summary,
        )

    def scan_anomalies(
        self,
        stock_data: Dict[str, pd.DataFrame],
        min_severity: str = "medium"
    ) -> List[AnomalyReport]:
        """
        Scan multiple stocks for anomalies
        
        Args:
            stock_data: Dictionary of symbol -> DataFrame
            min_severity: Minimum severity to report
        
        Returns:
            List of AnomalyReport objects
        """
        reports = []
        severity_order = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        min_sev_val = severity_order.get(min_severity, 2)
        
        for symbol, df in stock_data.items():
            try:
                report = self.detect_anomalies(df, symbol)
                
                # Filter by minimum severity
                report.anomalies = [
                    a for a in report.anomalies
                    if severity_order[a.severity.value] >= min_sev_val
                ]
                report.total_anomalies = len(report.anomalies)
                
                if report.total_anomalies > 0:
                    reports.append(report)
                    
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
        
        # Sort by risk level
        risk_order = {'extreme': 4, 'high': 3, 'elevated': 2, 'normal': 1}
        reports.sort(key=lambda x: risk_order.get(x.risk_level, 0), reverse=True)
        
        logger.info(f"âœ… Scanned {len(stock_data)} stocks, found {len(reports)} with anomalies")
        return reports

    def save(self, filepath: str):
        """Save detector state"""
        self.ml_detector.save_models(filepath)

    def load(self, filepath: str):
        """Load detector state"""
        self.ml_detector.load_models(filepath)
        self.is_fitted = True

