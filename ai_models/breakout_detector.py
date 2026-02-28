"""
AI-based Breakout Detection System
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import logging

logger = logging.getLogger(__name__)


class BreakoutDetector:
    """Detect potential breakout stocks using ML"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.is_trained = False
    
    def create_breakout_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features specific to breakout detection"""
        features = pd.DataFrame(index=df.index)
        
        # Consolidation features
        for period in [10, 20, 30]:
            rolling_high = df['high'].rolling(period).max()
            rolling_low = df['low'].rolling(period).min()
            features[f'range_{period}d'] = (rolling_high - rolling_low) / rolling_low * 100
            features[f'range_narrowing_{period}d'] = features[f'range_{period}d'].diff(5)
        
        # Volume accumulation
        features['volume_trend_5'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()
        features['volume_trend_10'] = df['volume'].rolling(10).mean() / df['volume'].rolling(50).mean()
        features['volume_spike'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Distance from resistance
        features['dist_from_20d_high'] = (df['close'] - df['high'].rolling(20).max()) / df['close'] * 100
        features['dist_from_50d_high'] = (df['close'] - df['high'].rolling(50).max()) / df['close'] * 100
        features['dist_from_52w_high'] = (df['close'] - df['high'].rolling(252).max()) / df['close'] * 100
        
        # Bollinger Band squeeze (volatility contraction)
        if 'BB_Width' in df.columns:
            features['bb_width'] = df['BB_Width']
            features['bb_width_percentile'] = df['BB_Width'].rolling(100).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1]
            )
        
        # ATR contraction
        if 'ATR_14' in df.columns:
            features['atr_ratio'] = df['ATR_14'] / df['close'] * 100
            features['atr_contraction'] = features['atr_ratio'].rolling(10).mean() / features['atr_ratio'].rolling(50).mean()
        
        # ADX (trend strength building)
        if 'ADX' in df.columns:
            features['adx'] = df['ADX']
            features['adx_rising'] = df['ADX'].diff(5)
        
        # Price pattern features
        features['higher_lows'] = (df['low'] > df['low'].shift(1)).rolling(5).sum()
        features['higher_highs'] = (df['high'] > df['high'].shift(1)).rolling(5).sum()
        features['inside_bars'] = (
            (df['high'] < df['high'].shift(1)) & 
            (df['low'] > df['low'].shift(1))
        ).rolling(5).sum()
        
        # Relative strength
        features['return_5d'] = df['close'].pct_change(5)
        features['return_20d'] = df['close'].pct_change(20)
        
        # RSI momentum
        if 'RSI_14' in df.columns:
            features['rsi'] = df['RSI_14']
            features['rsi_slope'] = df['RSI_14'].diff(5)
        
        # OBV trend
        if 'OBV' in df.columns:
            features['obv_slope'] = df['OBV'].diff(10)
            features['obv_ma_ratio'] = df['OBV'] / df['OBV'].rolling(20).mean()
        
        return features.dropna()
    
    def create_labels(
        self,
        df: pd.DataFrame,
        horizon: int = 5,
        threshold: float = 0.05
    ) -> pd.Series:
        """
        Create breakout labels
        A breakout is defined as > threshold% move in horizon days
        """
        future_max = df['high'].rolling(horizon).max().shift(-horizon)
        future_return = (future_max - df['close']) / df['close']
        
        labels = (future_return > threshold).astype(int)
        return labels
    
    def train(self, df: pd.DataFrame) -> Dict:
        """Train breakout detection model"""
        features = self.create_breakout_features(df)
        labels = self.create_labels(df)
        
        # Align
        common_idx = features.index.intersection(labels.dropna().index)
        X = features.loc[common_idx]
        y = labels.loc[common_idx]
        
        # Time-series split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1),
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        y_pred = self.model.predict(X_test_scaled)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
        }
        
        self.is_trained = True
        logger.info(f"Breakout model trained: {metrics}")
        
        return metrics
    
    def predict_breakout(self, df: pd.DataFrame) -> Dict:
        """Predict breakout probability for current state"""
        if not self.is_trained:
            raise ValueError("Model not trained!")
        
        features = self.create_breakout_features(df)
        if features.empty:
            return {'probability': 0, 'signal': 'NO_DATA'}
        
        X = self.scaler.transform(features.iloc[[-1]])
        
        prob = self.model.predict_proba(X)[0]
        prediction = self.model.predict(X)[0]
        
        breakout_prob = prob[1] if len(prob) > 1 else 0
        
        signal = 'STRONG_BREAKOUT' if breakout_prob > 0.7 else \
                 'MODERATE_BREAKOUT' if breakout_prob > 0.5 else \
                 'WEAK_BREAKOUT' if breakout_prob > 0.3 else 'NO_BREAKOUT'
        
        return {
            'probability': float(breakout_prob),
            'signal': signal,
            'prediction': int(prediction),
            'features': features.iloc[-1].to_dict()
        }


class AnomalyDetector:
    """Detect anomalous price/volume activity"""
    
    def __init__(self, contamination: float = 0.05):
        self.model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42
        )
        self.scaler = StandardScaler()
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalous trading activity
        
        Returns:
            DataFrame with anomaly scores
        """
        features = pd.DataFrame({
            'return': df['close'].pct_change(),
            'volume_ratio': df['volume'] / df['volume'].rolling(20).mean(),
            'range_ratio': (df['high'] - df['low']) / df['close'],
            'gap': df['open'] / df['close'].shift(1) - 1,
            'body_ratio': abs(df['close'] - df['open']) / (df['high'] - df['low'] + 0.001),
        }).dropna()
        
        X_scaled = self.scaler.fit_transform(features)
        
        # -1 for anomaly, 1 for normal
        predictions = self.model.fit_predict(X_scaled)
        scores = self.model.decision_function(X_scaled)
        
        result = features.copy()
        result['anomaly'] = predictions
        result['anomaly_score'] = scores
        result['is_anomaly'] = predictions == -1
        
        return result
    
    def detect_volume_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect unusual volume patterns"""
        volume_features = pd.DataFrame({
            'volume': df['volume'],
            'volume_ma5': df['volume'].rolling(5).mean(),
            'volume_ma20': df['volume'].rolling(20).mean(),
            'volume_ratio': df['volume'] / df['volume'].rolling(20).mean(),
            'volume_std_ratio': df['volume'] / df['volume'].rolling(20).std(),
            'price_change': df['close'].pct_change(),
        }).dropna()
        
        # Volume spike detection using z-score
        volume_features['volume_zscore'] = (
            volume_features['volume'] - volume_features['volume_ma20']
        ) / df['volume'].rolling(20).std()
        
        volume_features['is_volume_spike'] = volume_features['volume_zscore'] > 2.0
        volume_features['is_volume_dry'] = volume_features['volume_zscore'] < -1.5
        
        return volume_features
