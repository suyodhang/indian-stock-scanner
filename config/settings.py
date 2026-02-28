import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

load_dotenv()

@dataclass
class AppConfig:
    """Main application configuration"""
    APP_NAME: str = "AI Indian Stock Scanner"
    DEBUG: bool = True
    
    # Database
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", 5432))
    DB_NAME: str = os.getenv("DB_NAME", "stock_scanner")
    DB_USER: str = os.getenv("DB_USER", "postgres")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "password")
    
    # Redis
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    
    # API Keys
    ZERODHA_API_KEY: str = os.getenv("ZERODHA_API_KEY", "")
    ZERODHA_API_SECRET: str = os.getenv("ZERODHA_API_SECRET", "")
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
    
    # Scanner Settings
    SCAN_INTERVAL_MINUTES: int = 5
    MARKET_OPEN_TIME: str = "09:15"
    MARKET_CLOSE_TIME: str = "15:30"
    
    # AI Model Settings
    MODEL_RETRAIN_DAYS: int = 30
    PREDICTION_CONFIDENCE_THRESHOLD: float = 0.7
    LOOKBACK_PERIOD: int = 252  # 1 year trading days
    
    @property
    def DATABASE_URL(self):
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"


@dataclass
class ScannerConfig:
    """Scanner-specific configuration"""
    # Momentum Scanner
    RSI_OVERSOLD: float = 30.0
    RSI_OVERBOUGHT: float = 70.0
    MACD_SIGNAL_THRESHOLD: float = 0.0
    
    # Volume Scanner
    VOLUME_SPIKE_MULTIPLIER: float = 2.0
    MIN_VOLUME: int = 100000
    
    # Breakout Scanner
    BREAKOUT_LOOKBACK: int = 20
    BREAKOUT_VOLUME_FACTOR: float = 1.5
    CONSOLIDATION_RANGE_PCT: float = 5.0
    
    # Price Filters
    MIN_PRICE: float = 10.0
    MAX_PRICE: float = 50000.0
    MIN_MARKET_CAP_CR: float = 500.0  # 500 Crore minimum


config = AppConfig()
scanner_config = ScannerConfig()
