"""
Database Models for AI Stock Scanner

Tables:
- stocks: Stock master data
- stock_prices: Historical OHLCV data
- technical_signals: Scanner signals
- ai_predictions: AI model predictions
- scan_results: Scanner run results
- alerts: Alert history
- watchlists: User watchlists
- portfolio: Portfolio tracking
- model_metrics: AI model performance
- market_data: Market-level data (indices, breadth)
- corporate_actions: Dividends, splits, bonus
- fundamental_data: Quarterly fundamentals
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column, Integer, BigInteger, String, Float, Boolean,
    DateTime, Date, Text, JSON, ForeignKey, Index,
    UniqueConstraint, Enum as SQLEnum, create_engine
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
import enum

Base = declarative_base()


# ============================================================
# ENUMS
# ============================================================

class ExchangeEnum(enum.Enum):
    NSE = "NSE"
    BSE = "BSE"


class SignalTypeEnum(enum.Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class AlertStatusEnum(enum.Enum):
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    EXPIRED = "expired"


class PredictionEnum(enum.Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


# ============================================================
# MODELS
# ============================================================

class Stock(Base):
    """Stock master table"""
    __tablename__ = 'stocks'

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(50), nullable=False, index=True)
    company_name = Column(String(200))
    exchange = Column(SQLEnum(ExchangeEnum), default=ExchangeEnum.NSE)
    isin = Column(String(20), unique=True)
    bse_code = Column(String(10))
    nse_symbol = Column(String(50))
    sector = Column(String(100))
    industry = Column(String(100))
    market_cap = Column(BigInteger, default=0)
    face_value = Column(Float, default=10.0)
    listing_date = Column(Date)
    is_active = Column(Boolean, default=True)
    group_name = Column(String(10))  # A, B, T for BSE
    lot_size = Column(Integer, default=1)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    prices = relationship("StockPrice", back_populates="stock", cascade="all, delete-orphan")
    signals = relationship("TechnicalSignal", back_populates="stock")
    predictions = relationship("AIPrediction", back_populates="stock")
    fundamentals = relationship("FundamentalData", back_populates="stock")

    __table_args__ = (
        UniqueConstraint('symbol', 'exchange', name='uq_symbol_exchange'),
        Index('idx_sector', 'sector'),
        Index('idx_industry', 'industry'),
    )

    def __repr__(self):
        return f"<Stock({self.symbol} | {self.exchange.value})>"


class StockPrice(Base):
    """Historical OHLCV price data"""
    __tablename__ = 'stock_prices'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    stock_id = Column(Integer, ForeignKey('stocks.id'), nullable=False)
    date = Column(Date, nullable=False)
    timeframe = Column(String(10), default='1d')  # 1d, 1h, 5m, etc.

    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(BigInteger, default=0)
    value = Column(Float, default=0)  # Turnover
    no_of_trades = Column(Integer, default=0)

    # Delivery data (NSE specific)
    delivery_qty = Column(BigInteger, default=0)
    delivery_pct = Column(Float, default=0)

    # Adjusted prices
    adj_close = Column(Float)

    # Pre-calculated metrics
    daily_return = Column(Float)
    gap_pct = Column(Float)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    stock = relationship("Stock", back_populates="prices")

    __table_args__ = (
        UniqueConstraint('stock_id', 'date', 'timeframe', name='uq_stock_date_tf'),
        Index('idx_date', 'date'),
        Index('idx_stock_date', 'stock_id', 'date'),
    )


class TechnicalSignal(Base):
    """Scanner-generated technical signals"""
    __tablename__ = 'technical_signals'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    stock_id = Column(Integer, ForeignKey('stocks.id'), nullable=False)
    date = Column(Date, nullable=False)

    signal_name = Column(String(100), nullable=False)
    signal_type = Column(SQLEnum(SignalTypeEnum), nullable=False)
    signal_strength = Column(Float, default=0.0)  # 0.0 to 1.0

    scanner_name = Column(String(100))
    price_at_signal = Column(Float)
    volume_at_signal = Column(BigInteger)
    volume_ratio = Column(Float)

    # Indicator values at signal time
    rsi = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    adx = Column(Float)
    supertrend_dir = Column(Integer)  # 1 or -1
    bb_width = Column(Float)

    reasons = Column(JSON)  # List of reason strings
    metadata_json = Column(JSON)  # Additional data

    # Outcome tracking
    price_after_1d = Column(Float)
    price_after_5d = Column(Float)
    price_after_10d = Column(Float)
    price_after_20d = Column(Float)
    signal_success = Column(Boolean)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    stock = relationship("Stock", back_populates="signals")

    __table_args__ = (
        Index('idx_signal_date', 'date'),
        Index('idx_signal_type', 'signal_type'),
        Index('idx_signal_name', 'signal_name'),
        Index('idx_stock_signal_date', 'stock_id', 'date'),
    )


class AIPrediction(Base):
    """AI model predictions"""
    __tablename__ = 'ai_predictions'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    stock_id = Column(Integer, ForeignKey('stocks.id'), nullable=False)
    date = Column(Date, nullable=False)

    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50))

    prediction = Column(SQLEnum(PredictionEnum), nullable=False)
    confidence = Column(Float, nullable=False)
    prediction_horizon = Column(Integer, default=5)  # days

    # Detailed predictions
    bullish_probability = Column(Float)
    bearish_probability = Column(Float)
    predicted_return = Column(Float)
    predicted_price_target = Column(Float)

    # Model agreement
    bullish_votes = Column(Integer)
    bearish_votes = Column(Integer)
    total_models = Column(Integer)

    # Feature importance
    top_features = Column(JSON)

    # Price at prediction
    price_at_prediction = Column(Float)

    # Outcome
    actual_return = Column(Float)
    actual_price = Column(Float)
    was_correct = Column(Boolean)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    stock = relationship("Stock", back_populates="predictions")

    __table_args__ = (
        Index('idx_pred_date', 'date'),
        Index('idx_pred_model', 'model_name'),
        Index('idx_pred_stock_date', 'stock_id', 'date'),
    )


class ScanResult(Base):
    """Scanner run results (aggregated)"""
    __tablename__ = 'scan_results'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    scan_date = Column(DateTime, nullable=False)
    scanner_name = Column(String(100), nullable=False)

    total_stocks_scanned = Column(Integer, default=0)
    bullish_signals = Column(Integer, default=0)
    bearish_signals = Column(Integer, default=0)
    neutral_signals = Column(Integer, default=0)

    top_bullish = Column(JSON)  # List of top bullish picks
    top_bearish = Column(JSON)

    market_breadth = Column(JSON)  # Advances/Declines
    execution_time_sec = Column(Float)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_scan_date', 'scan_date'),
    )


class Alert(Base):
    """Alert history"""
    __tablename__ = 'alerts'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    stock_id = Column(Integer, ForeignKey('stocks.id'))
    alert_type = Column(String(100), nullable=False)
    channel = Column(String(50))  # telegram, email, webhook

    title = Column(String(200))
    message = Column(Text)
    status = Column(SQLEnum(AlertStatusEnum), default=AlertStatusEnum.PENDING)

    signal_id = Column(BigInteger, ForeignKey('technical_signals.id'))
    prediction_id = Column(BigInteger, ForeignKey('ai_predictions.id'))

    sent_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_alert_status', 'status'),
        Index('idx_alert_date', 'created_at'),
    )


class Watchlist(Base):
    """User watchlists"""
    __tablename__ = 'watchlists'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)

    symbols = Column(JSON)  # List of symbols
    is_default = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Portfolio(Base):
    """Portfolio tracking"""
    __tablename__ = 'portfolio'

    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_id = Column(Integer, ForeignKey('stocks.id'), nullable=False)

    buy_date = Column(Date, nullable=False)
    buy_price = Column(Float, nullable=False)
    quantity = Column(Integer, nullable=False)

    sell_date = Column(Date)
    sell_price = Column(Float)

    # Calculated
    investment = Column(Float)
    current_value = Column(Float)
    pnl = Column(Float)
    pnl_pct = Column(Float)

    stop_loss = Column(Float)
    target_price = Column(Float)

    notes = Column(Text)
    is_open = Column(Boolean, default=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index('idx_portfolio_open', 'is_open'),
    )


class ModelMetrics(Base):
    """AI model performance tracking"""
    __tablename__ = 'model_metrics'

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50))
    train_date = Column(DateTime, nullable=False)

    accuracy = Column(Float)
    precision_val = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)

    total_predictions = Column(Integer, default=0)
    correct_predictions = Column(Integer, default=0)
    win_rate = Column(Float)

    avg_return = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)

    training_samples = Column(Integer)
    feature_count = Column(Integer)
    top_features = Column(JSON)

    parameters = Column(JSON)
    notes = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_model_name', 'model_name'),
        Index('idx_model_date', 'train_date'),
    )


class MarketData(Base):
    """Market-level data (indices, breadth, FII/DII)"""
    __tablename__ = 'market_data'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False)
    data_type = Column(String(50), nullable=False)  # 'index', 'breadth', 'fii_dii'

    # Index data
    index_name = Column(String(50))
    index_open = Column(Float)
    index_high = Column(Float)
    index_low = Column(Float)
    index_close = Column(Float)
    index_change = Column(Float)
    index_change_pct = Column(Float)

    # Market breadth
    advances = Column(Integer)
    declines = Column(Integer)
    unchanged = Column(Integer)
    ad_ratio = Column(Float)

    # FII/DII
    fii_buy = Column(Float)
    fii_sell = Column(Float)
    fii_net = Column(Float)
    dii_buy = Column(Float)
    dii_sell = Column(Float)
    dii_net = Column(Float)

    # VIX
    india_vix = Column(Float)

    metadata_json = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_market_date', 'date'),
        Index('idx_market_type', 'data_type'),
        UniqueConstraint('date', 'data_type', 'index_name', name='uq_market_data'),
    )


class CorporateAction(Base):
    """Corporate actions - dividends, splits, bonus"""
    __tablename__ = 'corporate_actions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_id = Column(Integer, ForeignKey('stocks.id'), nullable=False)

    action_type = Column(String(50), nullable=False)  # dividend, split, bonus, rights
    ex_date = Column(Date)
    record_date = Column(Date)
    payment_date = Column(Date)

    # Dividend specific
    dividend_amount = Column(Float)
    dividend_pct = Column(Float)

    # Split/Bonus specific
    ratio_from = Column(Integer)
    ratio_to = Column(Integer)

    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_corp_action_date', 'ex_date'),
        Index('idx_corp_action_type', 'action_type'),
    )


class FundamentalData(Base):
    """Quarterly/Annual fundamental data"""
    __tablename__ = 'fundamental_data'

    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_id = Column(Integer, ForeignKey('stocks.id'), nullable=False)
    period = Column(String(20), nullable=False)  # 'Q1FY24', 'FY24', etc.
    period_type = Column(String(10), nullable=False)  # 'quarterly', 'annual'
    date = Column(Date)

    # Income Statement
    revenue = Column(Float)
    operating_profit = Column(Float)
    net_profit = Column(Float)
    eps = Column(Float)

    # Balance Sheet
    total_assets = Column(Float)
    total_liabilities = Column(Float)
    total_equity = Column(Float)
    book_value = Column(Float)

    # Ratios
    pe_ratio = Column(Float)
    pb_ratio = Column(Float)
    roe = Column(Float)
    roa = Column(Float)
    debt_to_equity = Column(Float)
    current_ratio = Column(Float)
    operating_margin = Column(Float)
    net_margin = Column(Float)

    # Cash Flow
    operating_cash_flow = Column(Float)
    free_cash_flow = Column(Float)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    stock = relationship("Stock", back_populates="fundamentals")

    __table_args__ = (
        UniqueConstraint('stock_id', 'period', name='uq_stock_period'),
        Index('idx_fund_date', 'date'),
    )


# ============================================================
# DATABASE CREATION UTILITY
# ============================================================

def create_tables(database_url: str):
    """Create all tables in the database"""
    engine = create_engine(database_url, echo=False)
    Base.metadata.create_all(engine)
    logger.info(f"âœ… All tables created successfully")
    return engine


def drop_tables(database_url: str):
    """Drop all tables (DANGEROUS!)"""
    engine = create_engine(database_url, echo=False)
    Base.metadata.drop_all(engine)
    logger.info("âš ï¸ All tables dropped")
    return engine


if __name__ == "__main__":
    # Test with SQLite
    engine = create_tables("sqlite:///test_scanner.db")
    print("âœ… Database created successfully!")
    print(f"Tables: {list(Base.metadata.tables.keys())}")

