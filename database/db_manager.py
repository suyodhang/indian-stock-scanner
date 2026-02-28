"""
Database Manager for AI Stock Scanner

Handles all database operations:
- CRUD operations for all models
- Bulk insert/update
- Query helpers
- Data migration
- Backup/Restore
- Performance optimization
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from sqlalchemy import create_engine, func, and_, or_, desc, asc, text
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.dialects.postgresql import insert as pg_insert
from contextlib import contextmanager
import logging
import json

from database.models import (
    Base, Stock, StockPrice, TechnicalSignal, AIPrediction,
    ScanResult, Alert, Watchlist, Portfolio, ModelMetrics,
    MarketData, CorporateAction, FundamentalData,
    ExchangeEnum, SignalTypeEnum, AlertStatusEnum, PredictionEnum,
    create_tables
)

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Central database manager for all CRUD operations
    
    Usage:
        db = DatabaseManager("postgresql://user:pass@localhost/scanner")
        
        # Add stock
        db.add_stock("RELIANCE", "Reliance Industries", "NSE")
        
        # Save prices
        db.save_price_data("RELIANCE", df)
        
        # Query
        prices = db.get_price_data("RELIANCE", period="1y")
        signals = db.get_signals(date="2024-01-15")
    """

    def __init__(self, database_url: str = None, echo: bool = False):
        """
        Args:
            database_url: SQLAlchemy database URL
            echo: Whether to log SQL statements
        """
        if database_url is None:
            database_url = "sqlite:///stock_scanner.db"

        self.engine = create_engine(
            database_url,
            echo=echo,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
        )

        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)

        self.SessionFactory = sessionmaker(bind=self.engine)
        self.ScopedSession = scoped_session(self.SessionFactory)

        logger.info(f"âœ… Database connected: {database_url.split('@')[-1] if '@' in database_url else database_url}")

    @contextmanager
    def get_session(self) -> Session:
        """Get a database session with automatic commit/rollback"""
        session = self.SessionFactory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()

    # =========================================================
    # STOCK OPERATIONS
    # =========================================================

    def add_stock(
        self,
        symbol: str,
        company_name: str = "",
        exchange: str = "NSE",
        **kwargs
    ) -> Optional[Stock]:
        """Add a new stock to the database"""
        with self.get_session() as session:
            existing = session.query(Stock).filter_by(
                symbol=symbol,
                exchange=ExchangeEnum[exchange]
            ).first()

            if existing:
                # Update existing
                for key, value in kwargs.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                existing.company_name = company_name or existing.company_name
                return existing

            stock = Stock(
                symbol=symbol,
                company_name=company_name,
                exchange=ExchangeEnum[exchange],
                **{k: v for k, v in kwargs.items() if hasattr(Stock, k)}
            )
            session.add(stock)
            session.flush()
            logger.info(f"âœ… Added stock: {symbol} ({exchange})")
            return stock

    def add_bulk_stocks(self, stocks: List[Dict]) -> int:
        """Add multiple stocks at once"""
        count = 0
        with self.get_session() as session:
            for stock_data in stocks:
                try:
                    existing = session.query(Stock).filter_by(
                        symbol=stock_data['symbol'],
                        exchange=ExchangeEnum[stock_data.get('exchange', 'NSE')]
                    ).first()

                    if not existing:
                        stock = Stock(
                            symbol=stock_data['symbol'],
                            company_name=stock_data.get('company_name', ''),
                            exchange=ExchangeEnum[stock_data.get('exchange', 'NSE')],
                            sector=stock_data.get('sector', ''),
                            industry=stock_data.get('industry', ''),
                            isin=stock_data.get('isin', ''),
                            bse_code=stock_data.get('bse_code', ''),
                        )
                        session.add(stock)
                        count += 1
                except IntegrityError:
                    session.rollback()
                    continue

        logger.info(f"âœ… Added {count} new stocks")
        return count

    def get_stock(self, symbol: str, exchange: str = "NSE") -> Optional[Stock]:
        """Get stock by symbol"""
        with self.get_session() as session:
            return session.query(Stock).filter_by(
                symbol=symbol,
                exchange=ExchangeEnum[exchange]
            ).first()

    def get_stock_id(self, symbol: str, exchange: str = "NSE") -> Optional[int]:
        """Get stock ID by symbol"""
        with self.get_session() as session:
            stock = session.query(Stock.id).filter_by(
                symbol=symbol,
                exchange=ExchangeEnum[exchange]
            ).first()
            return stock.id if stock else None

    def get_all_stocks(self, exchange: str = None, sector: str = None) -> List[Stock]:
        """Get all stocks with optional filters"""
        with self.get_session() as session:
            query = session.query(Stock).filter_by(is_active=True)

            if exchange:
                query = query.filter_by(exchange=ExchangeEnum[exchange])
            if sector:
                query = query.filter_by(sector=sector)

            return query.order_by(Stock.symbol).all()

    def get_all_symbols(self, exchange: str = "NSE") -> List[str]:
        """Get all active stock symbols"""
        with self.get_session() as session:
            results = session.query(Stock.symbol).filter_by(
                exchange=ExchangeEnum[exchange],
                is_active=True
            ).all()
            return [r[0] for r in results]

    def search_stocks(self, query: str) -> List[Stock]:
        """Search stocks by symbol or company name"""
        with self.get_session() as session:
            return session.query(Stock).filter(
                or_(
                    Stock.symbol.ilike(f"%{query}%"),
                    Stock.company_name.ilike(f"%{query}%")
                )
            ).all()

    # =========================================================
    # PRICE DATA OPERATIONS
    # =========================================================

    def save_price_data(
        self,
        symbol: str,
        df: pd.DataFrame,
        exchange: str = "NSE",
        timeframe: str = "1d"
    ) -> int:
        """
        Save OHLCV price data to database
        
        Args:
            symbol: Stock symbol
            df: DataFrame with date, open, high, low, close, volume
            exchange: Exchange
            timeframe: Data timeframe
        
        Returns:
            Number of rows saved
        """
        stock_id = self.get_stock_id(symbol, exchange)
        if not stock_id:
            self.add_stock(symbol, exchange=exchange)
            stock_id = self.get_stock_id(symbol, exchange)

        if stock_id is None:
            logger.error(f"Cannot find/create stock: {symbol}")
            return 0

        count = 0
        with self.get_session() as session:
            for _, row in df.iterrows():
                try:
                    price_date = pd.to_datetime(row.get('date', row.name)).date()

                    existing = session.query(StockPrice).filter_by(
                        stock_id=stock_id,
                        date=price_date,
                        timeframe=timeframe
                    ).first()

                    if existing:
                        existing.open = float(row.get('open', 0))
                        existing.high = float(row.get('high', 0))
                        existing.low = float(row.get('low', 0))
                        existing.close = float(row.get('close', 0))
                        existing.volume = int(row.get('volume', 0))
                        existing.delivery_pct = float(row.get('delivery_pct', 0))
                        existing.daily_return = float(row.get('daily_return', 0)) if pd.notna(row.get('daily_return')) else None
                    else:
                        price = StockPrice(
                            stock_id=stock_id,
                            date=price_date,
                            timeframe=timeframe,
                            open=float(row.get('open', 0)),
                            high=float(row.get('high', 0)),
                            low=float(row.get('low', 0)),
                            close=float(row.get('close', 0)),
                            volume=int(row.get('volume', 0)),
                            value=float(row.get('value', 0)),
                            no_of_trades=int(row.get('no_of_trades', 0)),
                            delivery_qty=int(row.get('delivery_qty', 0)),
                            delivery_pct=float(row.get('delivery_pct', 0)),
                            daily_return=float(row.get('daily_return', 0)) if pd.notna(row.get('daily_return')) else None,
                        )
                        session.add(price)
                        count += 1

                except Exception as e:
                    logger.debug(f"Error saving price for {symbol} on {row.get('date')}: {e}")
                    continue

        logger.info(f"âœ… Saved {count} new price rows for {symbol}")
        return count

    def save_bulk_price_data(
        self,
        stock_data: Dict[str, pd.DataFrame],
        exchange: str = "NSE"
    ) -> Dict[str, int]:
        """Save price data for multiple stocks"""
        results = {}
        for symbol, df in stock_data.items():
            count = self.save_price_data(symbol, df, exchange)
            results[symbol] = count

        total = sum(results.values())
        logger.info(f"âœ… Bulk save: {total} rows for {len(results)} stocks")
        return results

    def get_price_data(
        self,
        symbol: str,
        exchange: str = "NSE",
        start_date: Union[str, date] = None,
        end_date: Union[str, date] = None,
        period: str = None,
        timeframe: str = "1d",
        limit: int = None
    ) -> pd.DataFrame:
        """
        Get price data from database
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            period: Period shorthand ('1mo', '3mo', '6mo', '1y')
            timeframe: Data timeframe
            limit: Max number of rows
        
        Returns:
            DataFrame with OHLCV data
        """
        stock_id = self.get_stock_id(symbol, exchange)
        if not stock_id:
            return pd.DataFrame()

        with self.get_session() as session:
            query = session.query(StockPrice).filter_by(
                stock_id=stock_id,
                timeframe=timeframe
            )

            # Date filters
            if period and not start_date:
                period_map = {
                    '1mo': 30, '3mo': 90, '6mo': 180,
                    '1y': 365, '2y': 730, '5y': 1825,
                }
                days = period_map.get(period, 365)
                start_date = date.today() - timedelta(days=days)

            if start_date:
                if isinstance(start_date, str):
                    start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
                query = query.filter(StockPrice.date >= start_date)

            if end_date:
                if isinstance(end_date, str):
                    end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
                query = query.filter(StockPrice.date <= end_date)

            query = query.order_by(StockPrice.date)

            if limit:
                query = query.limit(limit)

            results = query.all()

            if not results:
                return pd.DataFrame()

            data = [{
                'date': r.date,
                'open': r.open,
                'high': r.high,
                'low': r.low,
                'close': r.close,
                'volume': r.volume,
                'delivery_pct': r.delivery_pct,
                'daily_return': r.daily_return,
            } for r in results]

            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df['symbol'] = symbol
            return df

    def get_latest_price(self, symbol: str, exchange: str = "NSE") -> Optional[Dict]:
        """Get latest price for a stock"""
        stock_id = self.get_stock_id(symbol, exchange)
        if not stock_id:
            return None

        with self.get_session() as session:
            price = session.query(StockPrice).filter_by(
                stock_id=stock_id, timeframe='1d'
            ).order_by(desc(StockPrice.date)).first()

            if price:
                return {
                    'date': price.date,
                    'open': price.open,
                    'high': price.high,
                    'low': price.low,
                    'close': price.close,
                    'volume': price.volume,
                }
            return None

    # =========================================================
    # SIGNAL OPERATIONS
    # =========================================================

    def save_signal(
        self,
        symbol: str,
        signal_name: str,
        signal_type: str,
        signal_strength: float,
        price: float,
        volume: int = 0,
        reasons: List[str] = None,
        **kwargs
    ) -> Optional[int]:
        """Save a technical signal"""
        stock_id = self.get_stock_id(symbol)
        if not stock_id:
            return None

        with self.get_session() as session:
            signal = TechnicalSignal(
                stock_id=stock_id,
                date=date.today(),
                signal_name=signal_name,
                signal_type=SignalTypeEnum[signal_type.upper()],
                signal_strength=signal_strength,
                price_at_signal=price,
                volume_at_signal=volume,
                reasons=reasons or [],
                **{k: v for k, v in kwargs.items() if hasattr(TechnicalSignal, k)}
            )
            session.add(signal)
            session.flush()
            return signal.id

    def save_bulk_signals(self, signals: List[Dict]) -> int:
        """Save multiple signals"""
        count = 0
        for signal_data in signals:
            try:
                sid = self.save_signal(**signal_data)
                if sid:
                    count += 1
            except Exception as e:
                logger.error(f"Error saving signal: {e}")

        logger.info(f"âœ… Saved {count} signals")
        return count

    def get_signals(
        self,
        signal_date: Union[str, date] = None,
        signal_type: str = None,
        symbol: str = None,
        min_strength: float = 0.0,
        limit: int = 100
    ) -> pd.DataFrame:
        """Query signals with filters"""
        with self.get_session() as session:
            query = session.query(
                TechnicalSignal, Stock.symbol
            ).join(Stock)

            if signal_date:
                if isinstance(signal_date, str):
                    signal_date = datetime.strptime(signal_date, '%Y-%m-%d').date()
                query = query.filter(TechnicalSignal.date == signal_date)

            if signal_type:
                query = query.filter(
                    TechnicalSignal.signal_type == SignalTypeEnum[signal_type.upper()]
                )

            if symbol:
                query = query.filter(Stock.symbol == symbol)

            if min_strength > 0:
                query = query.filter(TechnicalSignal.signal_strength >= min_strength)

            query = query.order_by(desc(TechnicalSignal.signal_strength))

            if limit:
                query = query.limit(limit)

            results = query.all()

            data = [{
                'symbol': r[1],
                'signal_name': r[0].signal_name,
                'signal_type': r[0].signal_type.value,
                'strength': r[0].signal_strength,
                'price': r[0].price_at_signal,
                'date': r[0].date,
                'reasons': r[0].reasons,
            } for r in results]

            return pd.DataFrame(data)

    def get_today_signals(self, min_strength: float = 0.5) -> pd.DataFrame:
        """Get today's signals"""
        return self.get_signals(
            signal_date=date.today(),
            min_strength=min_strength
        )

    # =========================================================
    # AI PREDICTION OPERATIONS
    # =========================================================

    def save_prediction(
        self,
        symbol: str,
        model_name: str,
        prediction: str,
        confidence: float,
        price: float,
        **kwargs
    ) -> Optional[int]:
        """Save an AI prediction"""
        stock_id = self.get_stock_id(symbol)
        if not stock_id:
            return None

        with self.get_session() as session:
            pred = AIPrediction(
                stock_id=stock_id,
                date=date.today(),
                model_name=model_name,
                prediction=PredictionEnum[prediction.upper()],
                confidence=confidence,
                price_at_prediction=price,
                **{k: v for k, v in kwargs.items() if hasattr(AIPrediction, k)}
            )
            session.add(pred)
            session.flush()
            return pred.id

    def get_predictions(
        self,
        symbol: str = None,
        pred_date: date = None,
        model_name: str = None,
        limit: int = 50
    ) -> pd.DataFrame:
        """Get AI predictions"""
        with self.get_session() as session:
            query = session.query(
                AIPrediction, Stock.symbol
            ).join(Stock)

            if symbol:
                query = query.filter(Stock.symbol == symbol)
            if pred_date:
                query = query.filter(AIPrediction.date == pred_date)
            if model_name:
                query = query.filter(AIPrediction.model_name == model_name)

            query = query.order_by(desc(AIPrediction.confidence)).limit(limit)
            results = query.all()

            data = [{
                'symbol': r[1],
                'prediction': r[0].prediction.value,
                'confidence': r[0].confidence,
                'price': r[0].price_at_prediction,
                'model': r[0].model_name,
                'date': r[0].date,
            } for r in results]

            return pd.DataFrame(data)

    # =========================================================
    # ALERT OPERATIONS
    # =========================================================

    def save_alert(
        self,
        alert_type: str,
        title: str,
        message: str,
        channel: str = "telegram",
        stock_id: int = None,
    ) -> int:
        """Save an alert"""
        with self.get_session() as session:
            alert = Alert(
                stock_id=stock_id,
                alert_type=alert_type,
                title=title,
                message=message,
                channel=channel,
                status=AlertStatusEnum.PENDING,
            )
            session.add(alert)
            session.flush()
            return alert.id

    def mark_alert_sent(self, alert_id: int):
        """Mark alert as sent"""
        with self.get_session() as session:
            alert = session.query(Alert).get(alert_id)
            if alert:
                alert.status = AlertStatusEnum.SENT
                alert.sent_at = datetime.utcnow()

    def get_pending_alerts(self) -> List[Alert]:
        """Get all pending alerts"""
        with self.get_session() as session:
            return session.query(Alert).filter_by(
                status=AlertStatusEnum.PENDING
            ).all()

    # =========================================================
    # WATCHLIST OPERATIONS
    # =========================================================

    def create_watchlist(self, name: str, symbols: List[str], description: str = "") -> int:
        """Create a watchlist"""
        with self.get_session() as session:
            wl = Watchlist(name=name, symbols=symbols, description=description)
            session.add(wl)
            session.flush()
            return wl.id

    def get_watchlist(self, name: str) -> Optional[Dict]:
        """Get watchlist by name"""
        with self.get_session() as session:
            wl = session.query(Watchlist).filter_by(name=name).first()
            if wl:
                return {
                    'id': wl.id,
                    'name': wl.name,
                    'symbols': wl.symbols,
                    'description': wl.description,
                }
            return None

    def get_all_watchlists(self) -> List[Dict]:
        """Get all watchlists"""
        with self.get_session() as session:
            watchlists = session.query(Watchlist).all()
            return [{
                'id': wl.id,
                'name': wl.name,
                'symbols': wl.symbols,
                'count': len(wl.symbols) if wl.symbols else 0,
            } for wl in watchlists]

    def add_to_watchlist(self, name: str, symbol: str):
        """Add symbol to watchlist"""
        with self.get_session() as session:
            wl = session.query(Watchlist).filter_by(name=name).first()
            if wl:
                symbols = wl.symbols or []
                if symbol not in symbols:
                    symbols.append(symbol)
                    wl.symbols = symbols

    def remove_from_watchlist(self, name: str, symbol: str):
        """Remove symbol from watchlist"""
        with self.get_session() as session:
            wl = session.query(Watchlist).filter_by(name=name).first()
            if wl and wl.symbols:
                wl.symbols = [s for s in wl.symbols if s != symbol]

    # =========================================================
    # PORTFOLIO OPERATIONS
    # =========================================================

    def add_portfolio_entry(
        self,
        symbol: str,
        buy_date: date,
        buy_price: float,
        quantity: int,
        stop_loss: float = None,
        target_price: float = None,
        notes: str = ""
    ) -> int:
        """Add a portfolio entry"""
        stock_id = self.get_stock_id(symbol)
        if not stock_id:
            self.add_stock(symbol)
            stock_id = self.get_stock_id(symbol)

        with self.get_session() as session:
            entry = Portfolio(
                stock_id=stock_id,
                buy_date=buy_date,
                buy_price=buy_price,
                quantity=quantity,
                investment=buy_price * quantity,
                stop_loss=stop_loss,
                target_price=target_price,
                notes=notes,
            )
            session.add(entry)
            session.flush()
            return entry.id

    def close_portfolio_entry(self, entry_id: int, sell_price: float, sell_date: date = None):
        """Close a portfolio entry"""
        with self.get_session() as session:
            entry = session.query(Portfolio).get(entry_id)
            if entry:
                entry.sell_date = sell_date or date.today()
                entry.sell_price = sell_price
                entry.current_value = sell_price * entry.quantity
                entry.pnl = (sell_price - entry.buy_price) * entry.quantity
                entry.pnl_pct = (sell_price / entry.buy_price - 1) * 100
                entry.is_open = False

    def get_open_positions(self) -> pd.DataFrame:
        """Get all open portfolio positions"""
        with self.get_session() as session:
            results = session.query(
                Portfolio, Stock.symbol
            ).join(Stock).filter(
                Portfolio.is_open == True
            ).all()

            data = [{
                'symbol': r[1],
                'buy_date': r[0].buy_date,
                'buy_price': r[0].buy_price,
                'quantity': r[0].quantity,
                'investment': r[0].investment,
                'stop_loss': r[0].stop_loss,
                'target': r[0].target_price,
            } for r in results]

            return pd.DataFrame(data)

    def get_portfolio_summary(self) -> Dict:
        """Get portfolio performance summary"""
        with self.get_session() as session:
            # Open positions
            open_positions = session.query(Portfolio).filter_by(is_open=True).all()
            total_investment = sum(p.investment or 0 for p in open_positions)

            # Closed positions
            closed = session.query(Portfolio).filter_by(is_open=False).all()
            total_pnl = sum(p.pnl or 0 for p in closed)
            winning_trades = sum(1 for p in closed if (p.pnl or 0) > 0)
            total_closed = len(closed)

            return {
                'open_positions': len(open_positions),
                'total_investment': total_investment,
                'closed_trades': total_closed,
                'total_pnl': total_pnl,
                'win_rate': winning_trades / max(total_closed, 1) * 100,
                'winning_trades': winning_trades,
                'losing_trades': total_closed - winning_trades,
            }

    # =========================================================
    # MODEL METRICS
    # =========================================================

    def save_model_metrics(self, metrics: Dict) -> int:
        """Save AI model training metrics"""
        with self.get_session() as session:
            mm = ModelMetrics(
                model_name=metrics.get('model_name', ''),
                model_version=metrics.get('model_version', '1.0'),
                train_date=datetime.utcnow(),
                accuracy=metrics.get('accuracy'),
                precision_val=metrics.get('precision'),
                recall=metrics.get('recall'),
                f1_score=metrics.get('f1'),
                training_samples=metrics.get('training_samples'),
                feature_count=metrics.get('feature_count'),
                top_features=metrics.get('top_features'),
                parameters=metrics.get('parameters'),
            )
            session.add(mm)
            session.flush()
            return mm.id

    def get_model_performance(self, model_name: str) -> pd.DataFrame:
        """Get historical model performance"""
        with self.get_session() as session:
            results = session.query(ModelMetrics).filter_by(
                model_name=model_name
            ).order_by(desc(ModelMetrics.train_date)).all()

            data = [{
                'date': r.train_date,
                'accuracy': r.accuracy,
                'precision': r.precision_val,
                'recall': r.recall,
                'f1': r.f1_score,
                'samples': r.training_samples,
            } for r in results]

            return pd.DataFrame(data)

    # =========================================================
    # MARKET DATA
    # =========================================================

    def save_market_data(self, data: Dict):
        """Save market-level data (index, breadth, FII/DII)"""
        with self.get_session() as session:
            md = MarketData(
                date=data.get('date', date.today()),
                data_type=data.get('data_type', 'index'),
                index_name=data.get('index_name'),
                index_close=data.get('close'),
                index_change=data.get('change'),
                index_change_pct=data.get('change_pct'),
                advances=data.get('advances'),
                declines=data.get('declines'),
                ad_ratio=data.get('ad_ratio'),
                fii_net=data.get('fii_net'),
                dii_net=data.get('dii_net'),
                india_vix=data.get('india_vix'),
                metadata_json=data.get('metadata'),
            )
            session.add(md)

    # =========================================================
    # STATISTICS & ANALYTICS
    # =========================================================

    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        with self.get_session() as session:
            stats = {
                'stocks': session.query(func.count(Stock.id)).scalar(),
                'price_rows': session.query(func.count(StockPrice.id)).scalar(),
                'signals': session.query(func.count(TechnicalSignal.id)).scalar(),
                'predictions': session.query(func.count(AIPrediction.id)).scalar(),
                'alerts': session.query(func.count(Alert.id)).scalar(),
                'watchlists': session.query(func.count(Watchlist.id)).scalar(),
                'portfolio_entries': session.query(func.count(Portfolio.id)).scalar(),
            }

            # Date range
            first_date = session.query(func.min(StockPrice.date)).scalar()
            last_date = session.query(func.max(StockPrice.date)).scalar()
            stats['first_date'] = str(first_date) if first_date else 'N/A'
            stats['last_date'] = str(last_date) if last_date else 'N/A'

            return stats

    def get_signal_performance(
        self,
        signal_name: str = None,
        days: int = 30
    ) -> Dict:
        """Analyze signal performance (backtest signals)"""
        with self.get_session() as session:
            query = session.query(TechnicalSignal).filter(
                TechnicalSignal.signal_success.isnot(None)
            )

            if signal_name:
                query = query.filter(TechnicalSignal.signal_name == signal_name)

            if days:
                cutoff = date.today() - timedelta(days=days)
                query = query.filter(TechnicalSignal.date >= cutoff)

            signals = query.all()

            if not signals:
                return {}

            total = len(signals)
            successful = sum(1 for s in signals if s.signal_success)

            return {
                'total_signals': total,
                'successful': successful,
                'failed': total - successful,
                'success_rate': successful / total * 100 if total > 0 else 0,
                'avg_strength': np.mean([s.signal_strength for s in signals]),
            }

    # =========================================================
    # CLEANUP & MAINTENANCE
    # =========================================================

    def cleanup_old_data(self, days: int = 365 * 5):
        """Remove old price data"""
        cutoff = date.today() - timedelta(days=days)
        with self.get_session() as session:
            deleted = session.query(StockPrice).filter(
                StockPrice.date < cutoff
            ).delete()
            logger.info(f"ðŸ§¹ Cleaned up {deleted} old price rows")
            return deleted

    def vacuum_database(self):
        """Optimize database (SQLite specific)"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("VACUUM"))
            logger.info("âœ… Database vacuumed")
        except:
            pass

    def backup_to_csv(self, output_dir: str = "backups"):
        """Export all data to CSV files for backup"""
        from pathlib import Path
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        tables = {
            'stocks': Stock,
            'watchlists': Watchlist,
        }

        with self.get_session() as session:
            for name, model in tables.items():
                try:
                    results = session.query(model).all()
                    if results:
                        data = [
                            {c.name: getattr(r, c.name) for c in r.__table__.columns}
                            for r in results
                        ]
                        df = pd.DataFrame(data)
                        filepath = f"{output_dir}/{name}_{date.today()}.csv"
                        df.to_csv(filepath, index=False)
                        logger.info(f"âœ… Backed up {name}: {len(data)} rows")
                except Exception as e:
                    logger.error(f"Error backing up {name}: {e}")

    def close(self):
        """Close database connection"""
        self.engine.dispose()
        logger.info("Database connection closed")


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

_default_db = None


def get_db(database_url: str = None) -> DatabaseManager:
    """Get or create default database manager"""
    global _default_db
    if _default_db is None:
        _default_db = DatabaseManager(database_url)
    return _default_db


if __name__ == "__main__":
    """Test database operations"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    print("=" * 60)
    print("ðŸ—„ï¸ Database Manager Test")
    print("=" * 60)

    db = DatabaseManager("sqlite:///test_scanner.db")

    # Add stocks
    db.add_stock("RELIANCE", "Reliance Industries Ltd", "NSE", sector="Energy")
    db.add_stock("TCS", "Tata Consultancy Services", "NSE", sector="IT")
    db.add_stock("INFY", "Infosys Limited", "NSE", sector="IT")

    # Create watchlist
    db.create_watchlist("My Watchlist", ["RELIANCE", "TCS", "INFY"])

    # Add portfolio entry
    db.add_portfolio_entry("RELIANCE", date.today(), 2500.0, 10, stop_loss=2400.0, target_price=2800.0)

    # Get stats
    stats = db.get_database_stats()
    print(f"\nðŸ“Š Database Stats:")
    for key, val in stats.items():
        print(f"  {key}: {val}")

    # Search
    results = db.search_stocks("Reliance")
    print(f"\nðŸ” Search results: {[r.symbol for r in results]}")

    # Watchlists
    wls = db.get_all_watchlists()
    print(f"\nðŸ“‹ Watchlists: {wls}")

    # Portfolio
    portfolio = db.get_portfolio_summary()
    print(f"\nðŸ’¼ Portfolio: {portfolio}")

    print("\nâœ… Database tests completed!")
    db.close()

