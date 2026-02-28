"""
AI Model Training Pipeline for Stock Scanner

Handles:
- Data preparation & feature engineering
- Model training with cross-validation
- Hyperparameter tuning
- Model evaluation & comparison
- Model persistence (save/load)
- Automated retraining
- Performance tracking
- Walk-forward optimization
- Feature selection
- Ensemble model creation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import joblib
import json
import os
from pathlib import Path
import warnings
import time

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor,
    VotingClassifier, StackingClassifier,
    AdaBoostClassifier, BaggingClassifier
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    # Data
    prediction_horizon: int = 5  # Days ahead to predict
    test_size: float = 0.2
    validation_size: float = 0.1
    min_training_samples: int = 200
    
    # Target
    target_type: str = "classification"  # 'classification' or 'regression'
    classification_threshold: float = 0.0  # Return threshold for up/down
    
    # Features
    feature_selection: bool = True
    max_features: int = 50
    feature_selection_method: str = "mutual_info"  # 'f_classif', 'mutual_info'
    
    # Scaling
    scaler_type: str = "standard"  # 'standard', 'minmax', 'robust'
    
    # Cross-validation
    n_splits: int = 5
    
    # Hyperparameter tuning
    tune_hyperparams: bool = False
    n_iter_search: int = 50
    
    # Ensemble
    use_ensemble: bool = True
    ensemble_method: str = "voting"  # 'voting', 'stacking'
    
    # Save
    model_dir: str = "models"
    auto_save: bool = True


@dataclass
class TrainingResult:
    """Result of model training"""
    model_name: str
    train_date: str
    
    # Metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    roc_auc: float = 0.0
    
    # Data info
    training_samples: int = 0
    test_samples: int = 0
    feature_count: int = 0
    
    # Feature importance
    top_features: Dict = field(default_factory=dict)
    
    # Confusion matrix
    confusion_matrix: List = field(default_factory=list)
    
    # Time
    training_time_sec: float = 0.0
    
    # Hyperparameters
    best_params: Dict = field(default_factory=dict)


@dataclass
class ModelInfo:
    """Stored model information"""
    model_name: str
    model_type: str
    version: str
    train_date: str
    accuracy: float
    f1: float
    feature_names: List[str]
    filepath: str
    config: Dict = field(default_factory=dict)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

class FeatureEngineer:
    """
    Create features from OHLCV data for ML models
    
    Features categories:
    1. Price-based (returns, momentum, gaps)
    2. Volume-based (ratios, trends)
    3. Technical indicators (RSI, MACD, etc.)
    4. Statistical (volatility, skew, kurtosis)
    5. Calendar (day of week, month effects)
    6. Relative (vs index, vs sector)
    """

    @staticmethod
    def create_features(
        df: pd.DataFrame,
        include_technical: bool = True
    ) -> pd.DataFrame:
        """Create complete feature set from OHLCV data"""
        features = pd.DataFrame(index=df.index)
        
        # ---- 1. PRICE RETURNS ----
        for lag in [1, 2, 3, 5, 10, 20, 60]:
            features[f'return_{lag}d'] = df['close'].pct_change(lag)
        
        features['log_return_1d'] = np.log(df['close'] / df['close'].shift(1))
        
        # ---- 2. MOMENTUM ----
        for period in [5, 10, 20, 50]:
            features[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
        
        # Rate of Change
        for period in [5, 10, 20]:
            features[f'roc_{period}'] = df['close'].pct_change(period) * 100
        
        # ---- 3. VOLATILITY ----
        for window in [5, 10, 20, 50]:
            features[f'volatility_{window}d'] = df['close'].pct_change().rolling(window).std()
        
        # Volatility ratio
        features['vol_ratio_5_20'] = features['volatility_5d'] / (features['volatility_20d'] + 1e-10)
        
        # Parkinson volatility
        features['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * (np.log(df['high'] / df['low'])) ** 2
        ).rolling(20).mean()
        
        # ---- 4. VOLUME FEATURES ----
        features['volume_change'] = df['volume'].pct_change()
        features['volume_sma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        features['volume_trend'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()
        
        for period in [5, 10, 20]:
            features[f'volume_sma_{period}'] = df['volume'].rolling(period).mean()
        
        features['volume_std_20'] = df['volume'].rolling(20).std() / df['volume'].rolling(20).mean()
        
        # Price-Volume correlation
        features['price_vol_corr_10'] = (
            df['close'].pct_change().rolling(10).corr(df['volume'].pct_change())
        )
        
        # ---- 5. CANDLE FEATURES ----
        range_hl = df['high'] - df['low']
        features['body_ratio'] = (df['close'] - df['open']) / (range_hl + 1e-10)
        features['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (range_hl + 1e-10)
        features['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (range_hl + 1e-10)
        features['range_pct'] = range_hl / df['close']
        features['body_pct'] = abs(df['close'] - df['open']) / df['close']
        
        # ---- 6. GAP FEATURES ----
        features['gap'] = df['open'] / df['close'].shift(1) - 1
        features['gap_filled'] = (
            ((df['open'] > df['close'].shift(1)) & (df['low'] <= df['close'].shift(1))) |
            ((df['open'] < df['close'].shift(1)) & (df['high'] >= df['close'].shift(1)))
        ).astype(int)
        
        # ---- 7. MOVING AVERAGE FEATURES ----
        for period in [5, 10, 20, 50, 100, 200]:
            if len(df) > period:
                sma = df['close'].rolling(period).mean()
                ema = df['close'].ewm(span=period, adjust=False).mean()
                features[f'dist_sma_{period}'] = (df['close'] - sma) / sma * 100
                features[f'dist_ema_{period}'] = (df['close'] - ema) / ema * 100
                features[f'sma_{period}_slope'] = sma.pct_change(5) * 100
        
        # MA Crossovers
        if len(df) > 50:
            features['sma_20_50_cross'] = (
                df['close'].rolling(20).mean() > df['close'].rolling(50).mean()
            ).astype(int)
        
        if len(df) > 200:
            features['golden_cross'] = (
                df['close'].rolling(50).mean() > df['close'].rolling(200).mean()
            ).astype(int)
        
        # ---- 8. TECHNICAL INDICATORS ----
        if include_technical:
            features = FeatureEngineer._add_technical_features(df, features)
        
        # ---- 9. STATISTICAL FEATURES ----
        for window in [10, 20, 50]:
            ret = df['close'].pct_change()
            features[f'skew_{window}'] = ret.rolling(window).skew()
            features[f'kurtosis_{window}'] = ret.rolling(window).kurt()
            features[f'max_return_{window}'] = ret.rolling(window).max()
            features[f'min_return_{window}'] = ret.rolling(window).min()
        
        # ---- 10. CALENDAR FEATURES ----
        if 'date' in df.columns:
            dt = pd.to_datetime(df['date'])
            features['day_of_week'] = dt.dt.dayofweek
            features['month'] = dt.dt.month
            features['quarter'] = dt.dt.quarter
            features['is_month_start'] = dt.dt.is_month_start.astype(int)
            features['is_month_end'] = dt.dt.is_month_end.astype(int)
            features['is_quarter_end'] = dt.dt.is_quarter_end.astype(int)
            
            # Expiry week effect (Indian market)
            features['is_last_week'] = (dt.dt.day > 20).astype(int)
            
            # Monday/Friday effect
            features['is_monday'] = (dt.dt.dayofweek == 0).astype(int)
            features['is_friday'] = (dt.dt.dayofweek == 4).astype(int)
        
        # ---- 11. DISTANCE FROM HIGHS/LOWS ----
        for period in [20, 50, 252]:
            if len(df) > period:
                features[f'dist_{period}d_high'] = (
                    df['close'] / df['high'].rolling(period).max() - 1
                ) * 100
                features[f'dist_{period}d_low'] = (
                    df['close'] / df['low'].rolling(period).min() - 1
                ) * 100
        
        # ---- 12. CONSECUTIVE FEATURES ----
        up_days = (df['close'] > df['close'].shift(1)).astype(int)
        features['consecutive_up'] = up_days.groupby(
            (up_days != up_days.shift()).cumsum()
        ).cumsum() * up_days
        
        down_days = (df['close'] < df['close'].shift(1)).astype(int)
        features['consecutive_down'] = down_days.groupby(
            (down_days != down_days.shift()).cumsum()
        ).cumsum() * down_days
        
        return features

    @staticmethod
    def _add_technical_features(df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicator features"""
        try:
            from analysis.technical_indicators import TechnicalIndicators as TI
            
            # RSI
            features['rsi_14'] = TI.RSI(df['close'], 14)
            features['rsi_7'] = TI.RSI(df['close'], 7)
            features['rsi_slope'] = features['rsi_14'].diff(3)
            
            # MACD
            macd, signal, hist = TI.MACD(df['close'])
            features['macd'] = macd
            features['macd_signal'] = signal
            features['macd_hist'] = hist
            features['macd_hist_slope'] = hist.diff(3)
            
            # Bollinger Bands
            upper, middle, lower = TI.BollingerBands(df['close'])
            features['bb_width'] = (upper - lower) / middle
            features['bb_position'] = (df['close'] - lower) / (upper - lower + 1e-10)
            
            # ATR
            features['atr_14'] = TI.ATR(df, 14)
            features['atr_ratio'] = features['atr_14'] / df['close']
            
            # ADX
            adx, plus_di, minus_di = TI.ADX(df)
            features['adx'] = adx
            features['di_diff'] = plus_di - minus_di
            
            # Stochastic
            k, d = TI.Stochastic(df)
            features['stoch_k'] = k
            features['stoch_d'] = d
            
            # SuperTrend
            _, direction = TI.SuperTrend(df)
            features['supertrend_dir'] = direction
            
            # OBV
            features['obv'] = TI.OBV(df)
            features['obv_slope'] = features['obv'].pct_change(10)
            
            # CCI
            features['cci_20'] = TI.CCI(df, 20)
            
            # MFI
            features['mfi_14'] = TI.MFI(df, 14)
            
            # Williams %R
            features['williams_r'] = TI.Williams_R(df)
            
        except ImportError:
            logger.warning("TechnicalIndicators not available, skipping technical features")
        
        return features

    @staticmethod
    def create_target(
        df: pd.DataFrame,
        horizon: int = 5,
        target_type: str = "classification",
        threshold: float = 0.0
    ) -> pd.Series:
        """
        Create target variable
        
        Args:
            df: OHLCV DataFrame
            horizon: Prediction horizon in days
            target_type: 'classification' or 'regression'
            threshold: Return threshold for classification
        
        Returns:
            Target Series
        """
        future_return = df['close'].shift(-horizon) / df['close'] - 1
        
        if target_type == "classification":
            # Binary: 1 = Up, 0 = Down
            return (future_return > threshold).astype(int)
        else:
            # Regression: actual return
            return future_return


# ============================================================
# MODEL TRAINER
# ============================================================

class ModelTrainer:
    """
    Main model training class
    
    Handles the complete ML pipeline:
    1. Feature creation
    2. Feature selection
    3. Train/test split (time-series aware)
    4. Model training
    5. Hyperparameter tuning
    6. Evaluation
    7. Ensemble creation
    8. Model persistence
    """

    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.feature_engineer = FeatureEngineer()
        self.scaler = None
        self.models = {}
        self.best_model = None
        self.feature_names = []
        self.selected_features = []
        self.training_results = {}
        
        # Create model directory
        Path(self.config.model_dir).mkdir(parents=True, exist_ok=True)

    def _get_scaler(self) -> object:
        """Get scaler based on config"""
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler(),
        }
        return scalers.get(self.config.scaler_type, StandardScaler())

    def _get_models(self) -> Dict:
        """Get model dictionary based on task type"""
        if self.config.target_type == "classification":
            return {
                'random_forest': RandomForestClassifier(
                    n_estimators=200, max_depth=10,
                    min_samples_split=10, min_samples_leaf=5,
                    random_state=42, n_jobs=-1
                ),
                'xgboost': xgb.XGBClassifier(
                    n_estimators=200, max_depth=6,
                    learning_rate=0.1, subsample=0.8,
                    colsample_bytree=0.8, random_state=42,
                    eval_metric='logloss', verbosity=0
                ),
                'lightgbm': lgb.LGBMClassifier(
                    n_estimators=200, max_depth=6,
                    learning_rate=0.1, subsample=0.8,
                    colsample_bytree=0.8, random_state=42,
                    verbose=-1
                ),
                'gradient_boosting': GradientBoostingClassifier(
                    n_estimators=150, max_depth=5,
                    learning_rate=0.1, random_state=42
                ),
                'adaboost': AdaBoostClassifier(
                    n_estimators=100, learning_rate=0.1,
                    random_state=42
                ),
                'logistic': LogisticRegression(
                    max_iter=1000, random_state=42,
                    C=1.0, solver='lbfgs'
                ),
            }
        else:
            return {
                'random_forest': RandomForestRegressor(
                    n_estimators=200, max_depth=10,
                    random_state=42, n_jobs=-1
                ),
                'xgboost': xgb.XGBRegressor(
                    n_estimators=200, max_depth=6,
                    learning_rate=0.1, random_state=42,
                    verbosity=0
                ),
                'lightgbm': lgb.LGBMRegressor(
                    n_estimators=200, max_depth=6,
                    learning_rate=0.1, random_state=42,
                    verbose=-1
                ),
            }

    def prepare_data(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target from raw OHLCV data
        
        Returns:
            X (features DataFrame), y (target Series)
        """
        logger.info("ðŸ“Š Preparing features...")
        
        # Create features
        X = self.feature_engineer.create_features(df)
        
        # Create target
        y = self.feature_engineer.create_target(
            df,
            horizon=self.config.prediction_horizon,
            target_type=self.config.target_type,
            threshold=self.config.classification_threshold
        )
        
        # Align indices
        common_idx = X.dropna().index.intersection(y.dropna().index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Drop columns with too many NaN
        nan_threshold = 0.3  # Max 30% NaN
        valid_cols = X.columns[X.isna().mean() < nan_threshold]
        X = X[valid_cols]
        
        # Fill remaining NaN
        X = X.fillna(method='ffill').fillna(0)
        
        # Remove constant columns
        non_constant = X.columns[X.std() > 1e-10]
        X = X[non_constant]
        
        self.feature_names = X.columns.tolist()
        
        logger.info(f"  Features: {len(self.feature_names)}, Samples: {len(X)}")
        
        return X, y

    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> pd.DataFrame:
        """Select most important features"""
        if not self.config.feature_selection:
            self.selected_features = X.columns.tolist()
            return X
        
        max_features = min(self.config.max_features, len(X.columns))
        
        method = self.config.feature_selection_method
        
        if method == "mutual_info":
            selector = SelectKBest(
                score_func=mutual_info_classif,
                k=max_features
            )
        else:
            selector = SelectKBest(
                score_func=f_classif,
                k=max_features
            )
        
        try:
            X_selected = selector.fit_transform(X, y)
            selected_mask = selector.get_support()
            self.selected_features = X.columns[selected_mask].tolist()
            
            logger.info(f"  Selected {len(self.selected_features)}/{len(X.columns)} features")
            return X[self.selected_features]
            
        except Exception as e:
            logger.warning(f"Feature selection failed: {e}. Using all features.")
            self.selected_features = X.columns.tolist()
            return X

    def train(
        self,
        df: pd.DataFrame,
        symbol: str = "GENERAL"
    ) -> Dict[str, TrainingResult]:
        """
        Complete training pipeline
        
        Args:
            df: Raw OHLCV DataFrame
            symbol: Stock symbol (for labeling)
        
        Returns:
            Dictionary of model_name -> TrainingResult
        """
        logger.info(f"ðŸ§  Starting training pipeline for {symbol}...")
        start_time = time.time()
        
        # 1. Prepare data
        X, y = self.prepare_data(df)
        
        if len(X) < self.config.min_training_samples:
            logger.error(f"Insufficient data: {len(X)} < {self.config.min_training_samples}")
            return {}
        
        # 2. Feature selection
        X = self.select_features(X, y)
        
        # 3. Time-series split
        split_idx = int(len(X) * (1 - self.config.test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        logger.info(f"  Train: {len(X_train)}, Test: {len(X_test)}")
        
        # 4. Scale features
        self.scaler = self._get_scaler()
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # 5. Train models
        models = self._get_models()
        results = {}
        
        for name, model in models.items():
            try:
                model_start = time.time()
                logger.info(f"  Training {name}...")
                
                model.fit(X_train_scaled.values, y_train.values)
                
                y_pred = model.predict(X_test_scaled.values)
                
                if self.config.target_type == "classification":
                    try:
                        y_prob = model.predict_proba(X_test_scaled.values)[:, 1]
                        auc = roc_auc_score(y_test, y_prob)
                    except:
                        auc = 0
                    
                    result = TrainingResult(
                        model_name=name,
                        train_date=datetime.now().strftime('%Y-%m-%d %H:%M'),
                        accuracy=accuracy_score(y_test, y_pred),
                        precision=precision_score(y_test, y_pred, zero_division=0),
                        recall=recall_score(y_test, y_pred, zero_division=0),
                        f1=f1_score(y_test, y_pred, zero_division=0),
                        roc_auc=auc,
                        training_samples=len(X_train),
                        test_samples=len(X_test),
                        feature_count=len(self.selected_features),
                        confusion_matrix=confusion_matrix(y_test, y_pred).tolist(),
                        training_time_sec=time.time() - model_start,
                    )
                else:
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    result = TrainingResult(
                        model_name=name,
                        train_date=datetime.now().strftime('%Y-%m-%d %H:%M'),
                        accuracy=1 - mae,  # Approximate
                        training_samples=len(X_train),
                        test_samples=len(X_test),
                        feature_count=len(self.selected_features),
                        training_time_sec=time.time() - model_start,
                    )
                
                # Feature importance
                try:
                    if hasattr(model, 'feature_importances_'):
                        fi = pd.Series(
                            model.feature_importances_,
                            index=self.selected_features
                        ).sort_values(ascending=False)
                        result.top_features = fi.head(15).to_dict()
                except:
                    pass
                
                self.models[name] = model
                results[name] = result
                
                logger.info(
                    f"    âœ… {name}: Acc={result.accuracy:.4f}, "
                    f"F1={result.f1:.4f}, AUC={result.roc_auc:.4f}"
                )
                
            except Exception as e:
                logger.error(f"    âŒ {name} failed: {e}")
        
        # 6. Create ensemble
        if self.config.use_ensemble and len(self.models) >= 2:
            try:
                ensemble_result = self._create_ensemble(
                    X_train_scaled, y_train,
                    X_test_scaled, y_test
                )
                if ensemble_result:
                    results['ensemble'] = ensemble_result
            except Exception as e:
                logger.error(f"Ensemble creation failed: {e}")
        
        # 7. Find best model
        if results:
            best_name = max(results, key=lambda k: results[k].f1)
            self.best_model = self.models.get(best_name) or self.models.get('ensemble')
            logger.info(f"\nðŸ† Best model: {best_name} (F1: {results[best_name].f1:.4f})")
        
        # 8. Auto-save
        if self.config.auto_save and results:
            self.save_all(symbol)
        
        total_time = time.time() - start_time
        logger.info(f"â±ï¸ Total training time: {total_time:.1f}s")
        
        self.training_results = results
        return results

    def _create_ensemble(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Optional[TrainingResult]:
        """Create ensemble model"""
        logger.info("  Creating ensemble model...")
        
        estimators = [
            (name, model) for name, model in self.models.items()
            if name not in ['logistic']
        ]
        
        if len(estimators) < 2:
            return None
        
        if self.config.ensemble_method == "stacking":
            ensemble = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(max_iter=1000),
                cv=3, n_jobs=-1
            )
        else:
            ensemble = VotingClassifier(
                estimators=estimators,
                voting='soft',
                n_jobs=-1
            )
        
        ensemble.fit(X_train.values, y_train.values)
        y_pred = ensemble.predict(X_test.values)
        
        try:
            y_prob = ensemble.predict_proba(X_test.values)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        except:
            auc = 0
        
        self.models['ensemble'] = ensemble
        
        return TrainingResult(
            model_name='ensemble',
            train_date=datetime.now().strftime('%Y-%m-%d %H:%M'),
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred, zero_division=0),
            recall=recall_score(y_test, y_pred, zero_division=0),
            f1=f1_score(y_test, y_pred, zero_division=0),
            roc_auc=auc,
            training_samples=len(X_train),
            test_samples=len(X_test),
            feature_count=len(self.selected_features),
        )

    def predict(
        self,
        df: pd.DataFrame,
        model_name: str = None
    ) -> Dict:
        """
        Make prediction on new data
        
        Returns:
            Dictionary with prediction details
        """
        model = self.models.get(model_name or 'ensemble') or self.best_model
        
        if model is None:
            raise ValueError("No trained model available!")
        
        X = self.feature_engineer.create_features(df)
        
        # Use same features
        available = [f for f in self.selected_features if f in X.columns]
        X = X[available].fillna(method='ffill').fillna(0)
        
        X_scaled = self.scaler.transform(X.iloc[[-1]])
        
        prediction = model.predict(X_scaled)[0]
        
        try:
            probabilities = model.predict_proba(X_scaled)[0]
            confidence = max(probabilities)
        except:
            probabilities = [0, 0]
            confidence = 0.5
        
        return {
            'prediction': 'BULLISH' if prediction == 1 else 'BEARISH',
            'confidence': float(confidence),
            'bullish_prob': float(probabilities[1]) if len(probabilities) > 1 else 0,
            'bearish_prob': float(probabilities[0]) if len(probabilities) > 0 else 0,
            'model_used': model_name or 'best_model',
            'horizon': self.config.prediction_horizon,
        }

    def walk_forward_validation(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        train_size: int = 252,
        test_size: int = 63
    ) -> Dict:
        """
        Walk-forward optimization (most realistic backtest)
        
        Trains on rolling window, tests on next period
        """
        logger.info("ðŸ”„ Running walk-forward validation...")
        
        X, y = self.prepare_data(df)
        
        all_results = []
        
        for i in range(n_splits):
            train_start = i * test_size
            train_end = train_start + train_size
            test_end = train_end + test_size
            
            if test_end > len(X):
                break
            
            X_train = X.iloc[train_start:train_end]
            y_train = y.iloc[train_start:train_end]
            X_test = X.iloc[train_end:test_end]
            y_test = y.iloc[train_end:test_end]
            
            # Scale
            scaler = self._get_scaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            
            # Train best model type
            model = xgb.XGBClassifier(
                n_estimators=100, max_depth=5,
                learning_rate=0.1, random_state=42,
                verbosity=0
            )
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            all_results.append({
                'split': i,
                'train_start': train_start,
                'train_end': train_end,
                'test_end': test_end,
                'accuracy': acc,
                'f1': f1,
            })
            
            logger.info(f"  Split {i}: Acc={acc:.4f}, F1={f1:.4f}")
        
        avg_acc = np.mean([r['accuracy'] for r in all_results])
        avg_f1 = np.mean([r['f1'] for r in all_results])
        
        logger.info(f"ðŸ“Š Walk-Forward Results: Avg Acc={avg_acc:.4f}, Avg F1={avg_f1:.4f}")
        
        return {
            'splits': all_results,
            'avg_accuracy': avg_acc,
            'avg_f1': avg_f1,
            'std_accuracy': np.std([r['accuracy'] for r in all_results]),
            'std_f1': np.std([r['f1'] for r in all_results]),
        }

    # =========================================================
    # SAVE / LOAD
    # =========================================================

    def save_all(self, symbol: str = "general"):
        """Save all models, scaler, and metadata"""
        base_path = os.path.join(self.config.model_dir, symbol.lower())
        Path(base_path).mkdir(parents=True, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            filepath = os.path.join(base_path, f"{name}.pkl")
            joblib.dump(model, filepath)
        
        # Save scaler
        if self.scaler:
            joblib.dump(self.scaler, os.path.join(base_path, "scaler.pkl"))
        
        # Save metadata
        metadata = {
            'symbol': symbol,
            'feature_names': self.feature_names,
            'selected_features': self.selected_features,
            'config': self.config.__dict__,
            'train_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'results': {
                name: {
                    'accuracy': r.accuracy,
                    'f1': r.f1,
                    'precision': r.precision,
                    'recall': r.recall,
                }
                for name, r in self.training_results.items()
            }
        }
        
        with open(os.path.join(base_path, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"âœ… Models saved to {base_path}")

    def load_all(self, symbol: str = "general"):
        """Load all models from disk"""
        base_path = os.path.join(self.config.model_dir, symbol.lower())
        
        if not os.path.exists(base_path):
            logger.error(f"Model path not found: {base_path}")
            return False
        
        # Load metadata
        meta_path = os.path.join(base_path, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            self.feature_names = metadata.get('feature_names', [])
            self.selected_features = metadata.get('selected_features', [])
        
        # Load scaler
        scaler_path = os.path.join(base_path, "scaler.pkl")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        
        # Load models
        for filepath in Path(base_path).glob("*.pkl"):
            if filepath.stem != "scaler":
                self.models[filepath.stem] = joblib.load(filepath)
        
        if self.models:
            self.best_model = self.models.get('ensemble') or list(self.models.values())[0]
        
        logger.info(f"âœ… Loaded {len(self.models)} models from {base_path}")
        return True

    def get_training_summary(self) -> pd.DataFrame:
        """Get summary of all training results"""
        if not self.training_results:
            return pd.DataFrame()
        
        data = []
        for name, result in self.training_results.items():
            data.append({
                'model': name,
                'accuracy': result.accuracy,
                'precision': result.precision,
                'recall': result.recall,
                'f1': result.f1,
                'roc_auc': result.roc_auc,
                'train_time': result.training_time_sec,
            })
        
        df = pd.DataFrame(data)
        df.sort_values('f1', ascending=False, inplace=True)
        return df

