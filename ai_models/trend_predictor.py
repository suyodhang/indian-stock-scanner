"""
AI-based Trend Prediction using Multiple Models
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, VotingClassifier
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)
try:
    import xgboost as xgb
except Exception:
    xgb = None
try:
    import lightgbm as lgb
except Exception:
    lgb = None
from sklearn.neural_network import MLPClassifier
import joblib
import logging
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class TrendPredictor:
    """
    Predicts stock trend (Up/Down/Sideways) using ensemble ML models
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.ensemble_model = None
        self.feature_importance = None
        self.feature_columns: List[str] = []
        self.is_trained = False
        
    def prepare_features(
        self,
        df: pd.DataFrame,
        prediction_horizon: int = 5
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare feature matrix from OHLCV data with technical indicators
        
        Args:
            df: DataFrame with technical indicators already calculated
            prediction_horizon: Number of days ahead to predict
        
        Returns:
            X: Feature DataFrame
            y: Target Series (1=Up, 0=Down)
        """
        features = df.copy()
        
        # ---- Price-based features ----
        for lag in [1, 2, 3, 5, 10]:
            features[f'return_{lag}d'] = features['close'].pct_change(lag)
            features[f'high_low_ratio_{lag}d'] = (
                features['high'].rolling(lag).max() / 
                features['low'].rolling(lag).min()
            )
        
        # Price momentum
        features['momentum_5'] = features['close'] / features['close'].shift(5) - 1
        features['momentum_10'] = features['close'] / features['close'].shift(10) - 1
        features['momentum_20'] = features['close'] / features['close'].shift(20) - 1
        
        # Gap analysis
        features['gap'] = features['open'] / features['close'].shift(1) - 1
        
        # Candle features
        features['body_ratio'] = (features['close'] - features['open']) / (features['high'] - features['low'] + 0.001)
        features['upper_shadow_ratio'] = (features['high'] - features[['open', 'close']].max(axis=1)) / (features['high'] - features['low'] + 0.001)
        features['lower_shadow_ratio'] = (features[['open', 'close']].min(axis=1) - features['low']) / (features['high'] - features['low'] + 0.001)
        
        # ---- Volume features ----
        features['volume_change'] = features['volume'].pct_change()
        features['volume_ma_ratio'] = features['volume'] / features['volume'].rolling(20).mean()
        features['volume_trend'] = features['volume'].rolling(5).mean() / features['volume'].rolling(20).mean()
        
        # Price-Volume relationship
        features['price_volume_corr'] = features['close'].rolling(10).corr(features['volume'])
        
        # ---- Indicator features (ensure they exist) ----
        indicator_cols = [
            'RSI_14', 'RSI_7', 'MACD', 'MACD_Signal', 'MACD_Hist',
            'BB_Width', 'ATR_14', 'ADX', 'Plus_DI', 'Minus_DI',
            'Stoch_K', 'Stoch_D', 'CCI_20', 'MFI_14', 'Williams_R',
            'Dist_SMA_50', 'Dist_SMA_200', 'Volatility_20',
            'ST_Direction'
        ]
        
        for col in indicator_cols:
            if col not in features.columns:
                features[col] = np.nan
        
        # ---- Indicator-derived features ----
        features['RSI_slope'] = features['RSI_14'].diff(3)
        features['MACD_slope'] = features['MACD'].diff(3)
        features['ADX_slope'] = features['ADX'].diff(3)
        
        # MA crossover signals
        if 'EMA_9' in features.columns and 'EMA_20' in features.columns:
            features['EMA_9_20_cross'] = (features['EMA_9'] > features['EMA_20']).astype(int)
        if 'SMA_50' in features.columns and 'SMA_200' in features.columns:
            features['Golden_Cross'] = (features['SMA_50'] > features['SMA_200']).astype(int)
        
        # ---- Day/Week features ----
        if 'date' in features.columns:
            features['day_of_week'] = pd.to_datetime(features['date']).dt.dayofweek
            features['month'] = pd.to_datetime(features['date']).dt.month
            features['is_month_end'] = pd.to_datetime(features['date']).dt.is_month_end.astype(int)
            features['is_month_start'] = pd.to_datetime(features['date']).dt.is_month_start.astype(int)
        
        # ---- Target Variable ----
        future_return = features['close'].shift(-prediction_horizon) / features['close'] - 1
        y = (future_return > 0).astype(int)  # 1 = Up, 0 = Down
        
        # Remove non-feature columns
        drop_cols = ['date', 'Symbol', 'symbol', 'ticker', 'open', 'high', 'low', 'close', 'volume',
                     'dividends', 'stock_splits', 'ltp', 'value', 'no_of_trades']
        feature_cols = [c for c in features.columns if c not in drop_cols]
        
        X = features[feature_cols]
        # Keep only numeric features to avoid scaler failures (e.g., ticker strings).
        X = X.select_dtypes(include=[np.number])
        # Replace infinities from ratio/division features before NaN filtering.
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Drop rows with NaN
        valid_idx = X.dropna().index.intersection(y.dropna().index)
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
        
        return X, y
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        quick_mode: bool = False
    ) -> Dict:
        """
        Train ensemble model
        
        Returns:
            Dictionary with training metrics
        """
        if X is None or y is None or len(X) == 0 or len(y) == 0:
            raise ValueError("Training data is empty after feature preparation.")
        if len(X) < 50:
            raise ValueError(f"Not enough training samples: {len(X)} (need at least 50).")
        # Final finite-value guard for sklearn scaler/model stability.
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        y = y.loc[X.index]
        if len(X) < 50:
            raise ValueError(f"Not enough valid finite samples after cleaning: {len(X)}.")

        # Time-series split (no shuffling!)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self.feature_columns = list(X.columns)
        
        # Define models
        rf_estimators = 80 if quick_mode else 200
        gb_estimators = 70 if quick_mode else 150
        xgb_estimators = 80 if quick_mode else 200
        lgb_estimators = 80 if quick_mode else 200

        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=rf_estimators,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=gb_estimators,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
        }
        
        if xgb is not None:
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=xgb_estimators,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss',
                verbosity=0
            )
        else:
            logger.warning("xgboost not installed; skipping xgboost model.")

        if lgb is not None:
            self.models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=lgb_estimators,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )
        else:
            logger.warning("lightgbm not installed; skipping lightgbm model.")
        
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            if name in ['random_forest', 'gradient_boosting']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            logger.info(f"  {name}: Acc={accuracy:.4f}, F1={f1:.4f}")
        
        # Create ensemble (Voting Classifier)
        self.ensemble_model = VotingClassifier(
            estimators=[(name, model) for name, model in self.models.items()],
            voting='soft'
        )
        self.ensemble_model.fit(X_train_scaled, y_train)
        
        y_pred_ensemble = self.ensemble_model.predict(X_test_scaled)
        ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
        ensemble_f1 = f1_score(y_test, y_pred_ensemble, zero_division=0)
        
        results['ensemble'] = {
            'accuracy': ensemble_accuracy,
            'precision': precision_score(y_test, y_pred_ensemble, zero_division=0),
            'recall': recall_score(y_test, y_pred_ensemble, zero_division=0),
            'f1': ensemble_f1
        }
        
        logger.info(f"  Ensemble: Acc={ensemble_accuracy:.4f}, F1={ensemble_f1:.4f}")
        
        # Feature importance from best available tree model.
        if 'xgboost' in self.models and hasattr(self.models['xgboost'], 'feature_importances_'):
            importance_values = self.models['xgboost'].feature_importances_
        elif 'random_forest' in self.models and hasattr(self.models['random_forest'], 'feature_importances_'):
            importance_values = self.models['random_forest'].feature_importances_
        else:
            importance_values = np.zeros(len(X.columns))

        self.feature_importance = pd.Series(
            importance_values,
            index=X.columns
        ).sort_values(ascending=False)
        
        self.is_trained = True
        
        return results
    
    def predict(self, X: pd.DataFrame) -> Dict:
        """
        Make predictions with confidence scores
        
        Returns:
            Dictionary with predictions and confidence
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        X_in = X.copy()
        # Ensure prediction uses same numeric features as training.
        if self.feature_columns:
            for col in self.feature_columns:
                if col not in X_in.columns:
                    X_in[col] = 0.0
            X_in = X_in[self.feature_columns]
        X_in = X_in.select_dtypes(include=[np.number])
        X_scaled = self.scaler.transform(X_in)
        
        # Individual model predictions
        predictions = {}
        for name, model in self.models.items():
            pred = model.predict(X_scaled)
            prob = model.predict_proba(X_scaled)
            predictions[name] = {
                'prediction': pred[-1],
                'confidence': max(prob[-1])
            }
        
        # Ensemble prediction
        ensemble_pred = self.ensemble_model.predict(X_scaled)
        ensemble_prob = self.ensemble_model.predict_proba(X_scaled)
        
        avg_confidence = np.mean([p['confidence'] for p in predictions.values()])
        bullish_votes = sum(1 for p in predictions.values() if p['prediction'] == 1)
        total_models = len(predictions)
        
        return {
            'prediction': 'BULLISH' if ensemble_pred[-1] == 1 else 'BEARISH',
            'confidence': float(max(ensemble_prob[-1])),
            'avg_confidence': float(avg_confidence),
            'bullish_votes': bullish_votes,
            'bearish_votes': total_models - bullish_votes,
            'total_models': total_models,
            'individual_predictions': predictions,
            'top_features': self.feature_importance.head(10).to_dict()
        }
    
    def save_model(self, filepath: str):
        """Save trained model"""
        model_data = {
            'scaler': self.scaler,
            'models': self.models,
            'ensemble_model': self.ensemble_model,
            'feature_importance': self.feature_importance,
            'feature_columns': self.feature_columns,
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        model_data = joblib.load(filepath)
        self.scaler = model_data['scaler']
        self.models = model_data['models']
        self.ensemble_model = model_data['ensemble_model']
        self.feature_importance = model_data['feature_importance']
        self.feature_columns = model_data.get('feature_columns', [])
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")
