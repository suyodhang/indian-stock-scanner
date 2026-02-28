"""
Yahoo Finance Data Fetcher - Complete Implementation
For Indian Stock Market (NSE/BSE)

Features:
- Historical OHLCV data (daily, weekly, monthly, intraday)
- Real-time / delayed quotes
- Fundamental data (financials, balance sheet, cash flow)
- Options chain data
- Analyst recommendations
- Institutional holders
- Mutual fund holders
- Earnings & revenue estimates
- Dividend history
- Stock splits history
- Bulk data download (concurrent)
- Technical data ready (pre-processed for indicators)
- Sector & industry comparison
- Currency & commodity data
- Index data (NIFTY, SENSEX, etc.)
- News & calendar events
- ESG scores
- Error handling & retry logic
- Caching mechanism
- Rate limiting

Author: AI Stock Scanner
License: MIT
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import json
import time
import logging
import hashlib
from datetime import datetime, timedelta, date
from typing import (
    Dict, List, Optional, Tuple, Union, Any, Callable
)
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache, wraps
from pathlib import Path
import threading
import warnings
import pickle
import os

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


# ============================================================
# CONSTANTS & MAPPINGS
# ============================================================

# NSE suffix for Yahoo Finance
NSE_SUFFIX = ".NS"
BSE_SUFFIX = ".BO"

# Indian Indices on Yahoo Finance
INDIAN_INDICES = {
    "NIFTY50": "^NSEI",
    "SENSEX": "^BSESN",
    "BANKNIFTY": "^NSEBANK",
    "NIFTYIT": "^CNXIT",
    "NIFTYPHARMA": "^CNXPHARMA",
    "NIFTYAUTO": "^CNXAUTO",
    "NIFTYFMCG": "^CNXFMCG",
    "NIFTYMETAL": "^CNXMETAL",
    "NIFTYREALTY": "^CNXREALTY",
    "NIFTYENERGY": "^CNXENERGY",
    "NIFTYINFRA": "^CNXINFRA",
    "NIFTYPSE": "^CNXPSE",
    "NIFTYMIDCAP50": "^NSEMDCP50",
    "NIFTYSMLCAP50": "NIFTYSMLCAP50.NS",
    "NIFTYMIDCAP100": "NIFTY_MID_SELECT.NS",
    "NIFTY100": "^CNX100",
    "NIFTY200": "^CNX200",
    "NIFTY500": "^CNX500",
    "NIFTYNEXT50": "^NSMIDCP",
    "NIFTYFINANCE": "^CNXFIN",
    "NIFTYPVTBANK": "NIFTYPVTBANK.NS",
    "NIFTYCONSUMER": "^CNXCONSUME",
    "INDIAVIX": "^INDIAVIX",
}

# Common Indian stocks symbol mapping
# Yahoo Finance uses .NS for NSE and .BO for BSE
NIFTY_50_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
    "LT", "HCLTECH", "AXISBANK", "ASIANPAINT", "MARUTI",
    "SUNPHARMA", "TITAN", "BAJFINANCE", "WIPRO", "ULTRACEMCO",
    "NESTLEIND", "NTPC", "TATAMOTORS", "M&M", "POWERGRID",
    "TECHM", "ADANIENT", "ADANIPORTS", "BAJAJFINSV", "COALINDIA",
    "GRASIM", "CIPLA", "JSWSTEEL", "ONGC", "TATASTEEL",
    "HINDALCO", "DRREDDY", "EICHERMOT", "DIVISLAB", "BPCL",
    "BRITANNIA", "APOLLOHOSP", "INDUSINDBK", "TATACONSUM", "SBILIFE",
    "HDFCLIFE", "HEROMOTOCO", "BAJAJ-AUTO", "UPL", "LTIM"
]

# Known Yahoo ticker fallbacks for unstable mappings.
SYMBOL_FALLBACKS = {
    "TATAMOTORS": ["TATAMOTORS.BO", "TATAMTRDVR.NS"],
}

# Valid periods and intervals for yfinance
VALID_PERIODS = [
    "1d", "5d", "1mo", "3mo", "6mo",
    "1y", "2y", "5y", "10y", "ytd", "max"
]

VALID_INTERVALS = [
    "1m", "2m", "5m", "15m", "30m", "60m", "90m",
    "1h", "1d", "5d", "1wk", "1mo", "3mo"
]

# Interval constraints (Yahoo Finance limits)
INTERVAL_MAX_PERIOD = {
    "1m": "7d",       # Max 7 days for 1-minute data
    "2m": "60d",      # Max 60 days
    "5m": "60d",
    "15m": "60d",
    "30m": "60d",
    "60m": "730d",    # ~2 years
    "90m": "60d",
    "1h": "730d",
    "1d": "max",
    "5d": "max",
    "1wk": "max",
    "1mo": "max",
    "3mo": "max",
}


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class StockQuote:
    """Real-time stock quote data"""
    symbol: str
    exchange: str  # 'NSE' or 'BSE'
    company_name: str
    current_price: float
    previous_close: float
    open_price: float
    day_high: float
    day_low: float
    volume: int
    avg_volume: int
    market_cap: float
    pe_ratio: float
    eps: float
    week_52_high: float
    week_52_low: float
    change: float
    change_pct: float
    bid: float = 0.0
    ask: float = 0.0
    bid_size: int = 0
    ask_size: int = 0
    currency: str = "INR"
    timestamp: str = ""

    def __repr__(self):
        return (
            f"StockQuote({self.symbol} | â‚¹{self.current_price:.2f} | "
            f"{self.change_pct:+.2f}%)"
        )


@dataclass
class FundamentalData:
    """Comprehensive fundamental data"""
    symbol: str
    company_name: str
    sector: str
    industry: str
    market_cap: float
    enterprise_value: float

    # Valuation
    pe_ratio: float
    forward_pe: float
    peg_ratio: float
    pb_ratio: float
    ps_ratio: float
    ev_to_ebitda: float
    ev_to_revenue: float

    # Profitability
    profit_margin: float
    operating_margin: float
    gross_margin: float
    roe: float
    roa: float
    roic: float

    # Growth
    revenue_growth: float
    earnings_growth: float
    revenue_per_share: float
    earnings_quarterly_growth: float

    # Financial Health
    total_debt: float
    total_cash: float
    debt_to_equity: float
    current_ratio: float
    quick_ratio: float
    free_cash_flow: float
    operating_cash_flow: float

    # Per Share
    eps: float
    book_value: float
    dividend_rate: float
    dividend_yield: float
    payout_ratio: float

    # Trading
    beta: float
    avg_volume: int
    avg_volume_10d: int
    shares_outstanding: int
    float_shares: int
    shares_short: int
    short_ratio: float

    # Price
    week_52_high: float
    week_52_low: float
    day_50_ma: float
    day_200_ma: float
    target_mean_price: float
    target_high_price: float
    target_low_price: float

    # Misc
    full_time_employees: int = 0
    website: str = ""
    description: str = ""


@dataclass
class EarningsData:
    """Earnings information"""
    symbol: str
    earnings_dates: List[Dict] = field(default_factory=list)
    quarterly_earnings: List[Dict] = field(default_factory=list)
    yearly_earnings: List[Dict] = field(default_factory=list)
    earnings_estimate: Dict = field(default_factory=dict)
    revenue_estimate: Dict = field(default_factory=dict)


@dataclass
class DividendData:
    """Dividend history"""
    symbol: str
    dividend_rate: float
    dividend_yield: float
    ex_dividend_date: str
    payout_ratio: float
    five_year_avg_yield: float
    history: pd.DataFrame = field(default_factory=pd.DataFrame)


# ============================================================
# CACHING SYSTEM
# ============================================================

class DataCache:
    """
    Simple file-based cache for API data
    Reduces API calls and speeds up repeated requests
    """

    def __init__(self, cache_dir: str = ".cache/yahoo", default_ttl: int = 300):
        """
        Args:
            cache_dir: Directory to store cache files
            default_ttl: Default time-to-live in seconds (5 min)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl
        self.memory_cache = {}
        self._lock = threading.Lock()
        logger.debug(f"Cache initialized at {self.cache_dir}")

    def _get_cache_key(self, key: str) -> str:
        """Generate hash-based cache key"""
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get cached data if not expired"""
        cache_key = self._get_cache_key(key)

        # Check memory cache first
        with self._lock:
            if cache_key in self.memory_cache:
                data, expiry = self.memory_cache[cache_key]
                if datetime.now().timestamp() < expiry:
                    logger.debug(f"Cache HIT (memory): {key[:50]}")
                    return data
                else:
                    del self.memory_cache[cache_key]

        # Check file cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached = pickle.load(f)
                    if datetime.now().timestamp() < cached['expiry']:
                        logger.debug(f"Cache HIT (file): {key[:50]}")
                        # Store in memory for faster access
                        with self._lock:
                            self.memory_cache[cache_key] = (cached['data'], cached['expiry'])
                        return cached['data']
                    else:
                        cache_file.unlink()  # Remove expired
            except Exception as e:
                logger.debug(f"Cache read error: {e}")
                cache_file.unlink(missing_ok=True)

        logger.debug(f"Cache MISS: {key[:50]}")
        return None

    def set(self, key: str, data: Any, ttl: int = None):
        """Store data in cache"""
        if ttl is None:
            ttl = self.default_ttl

        cache_key = self._get_cache_key(key)
        expiry = datetime.now().timestamp() + ttl

        # Memory cache
        with self._lock:
            self.memory_cache[cache_key] = (data, expiry)

        # File cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({'data': data, 'expiry': expiry}, f)
        except Exception as e:
            logger.debug(f"Cache write error: {e}")

    def clear(self):
        """Clear all cache"""
        with self._lock:
            self.memory_cache.clear()

        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink(missing_ok=True)

        logger.info("Cache cleared")

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        file_count = len(list(self.cache_dir.glob("*.pkl")))
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.pkl"))

        return {
            'memory_entries': len(self.memory_cache),
            'file_entries': file_count,
            'total_size_mb': total_size / (1024 * 1024),
        }


# ============================================================
# RATE LIMITER
# ============================================================

class RateLimiter:
    """Thread-safe rate limiter for API calls"""

    def __init__(self, max_calls: int = 5, period: float = 1.0):
        """
        Args:
            max_calls: Maximum calls per period
            period: Time period in seconds
        """
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self._lock = threading.Lock()

    def wait(self):
        """Wait if rate limit is exceeded"""
        with self._lock:
            now = time.time()
            # Remove old calls
            self.calls = [t for t in self.calls if now - t < self.period]

            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (now - self.calls[0])
                if sleep_time > 0:
                    logger.debug(f"Rate limit: sleeping {sleep_time:.2f}s")
                    time.sleep(sleep_time)

            self.calls.append(time.time())


# ============================================================
# MAIN YAHOO FINANCE FETCHER
# ============================================================

class YahooFinanceFetcher:
    """
    Comprehensive Yahoo Finance data fetcher for Indian stocks

    Usage:
        fetcher = YahooFinanceFetcher()

        # Get historical data
        df = fetcher.get_historical_data("RELIANCE", period="1y")

        # Get real-time quote
        quote = fetcher.get_quote("RELIANCE")

        # Get fundamentals
        fundamentals = fetcher.get_fundamentals("TCS")

        # Bulk download
        data = fetcher.get_bulk_historical_data(["RELIANCE", "TCS", "INFY"])

        # Get with indicators
        df = fetcher.get_data_with_indicators("RELIANCE")
    """

    def __init__(
        self,
        exchange: str = "NSE",
        use_cache: bool = True,
        cache_ttl: int = 300,
        max_workers: int = 10,
        rate_limit: int = 5
    ):
        """
        Args:
            exchange: Default exchange ('NSE' or 'BSE')
            use_cache: Enable data caching
            cache_ttl: Cache time-to-live in seconds
            max_workers: Max concurrent threads for bulk operations
            rate_limit: Max API calls per second
        """
        self.exchange = exchange.upper()
        self.suffix = NSE_SUFFIX if self.exchange == "NSE" else BSE_SUFFIX
        self.max_workers = max_workers

        # Cache & Rate Limiting
        self.cache = DataCache(default_ttl=cache_ttl) if use_cache else None
        self.rate_limiter = RateLimiter(max_calls=rate_limit, period=1.0)

        # Ticker cache (yfinance Ticker objects)
        self._ticker_cache: Dict[str, yf.Ticker] = {}

        logger.info(
            f"âœ… YahooFinanceFetcher initialized | Exchange: {self.exchange} | "
            f"Cache: {'ON' if use_cache else 'OFF'} | Workers: {max_workers}"
        )

    # =========================================================
    # TICKER MANAGEMENT
    # =========================================================

    def _get_ticker_symbol(
        self,
        symbol: str,
        exchange: str = None,
        is_index: bool = False
    ) -> str:
        """
        Convert symbol to Yahoo Finance format

        Args:
            symbol: Stock symbol (e.g., 'RELIANCE')
            exchange: Override exchange ('NSE' or 'BSE')
            is_index: Whether it's an index symbol

        Returns:
            Yahoo Finance ticker (e.g., 'RELIANCE.NS')
        """
        # Check if it's an index
        if is_index or symbol.upper() in INDIAN_INDICES:
            return INDIAN_INDICES.get(symbol.upper(), symbol)

        # Already has suffix
        if symbol.endswith(('.NS', '.BO')):
            return symbol

        # If it looks like a Yahoo symbol already (starts with ^)
        if symbol.startswith('^'):
            return symbol

        # Add exchange suffix
        suffix = (
            NSE_SUFFIX if (exchange or self.exchange).upper() == "NSE"
            else BSE_SUFFIX
        )

        return f"{symbol}{suffix}"

    def _get_ticker_candidates(
        self,
        symbol: str,
        exchange: str = None,
        is_index: bool = False
    ) -> List[str]:
        """Return primary Yahoo ticker followed by optional fallback candidates."""
        primary = self._get_ticker_symbol(symbol, exchange=exchange, is_index=is_index)
        clean = self._clean_symbol(symbol).upper()
        candidates = [primary]
        for alt in SYMBOL_FALLBACKS.get(clean, []):
            if alt not in candidates:
                candidates.append(alt)
        return candidates

    def _get_ticker(self, symbol: str, **kwargs) -> yf.Ticker:
        """Get or create yfinance Ticker object with caching"""
        ticker_symbol = self._get_ticker_symbol(symbol, **kwargs)

        if ticker_symbol not in self._ticker_cache:
            self._ticker_cache[ticker_symbol] = yf.Ticker(ticker_symbol)

        return self._ticker_cache[ticker_symbol]

    def _clean_symbol(self, symbol: str) -> str:
        """Remove exchange suffix from symbol"""
        for suffix in ['.NS', '.BO', '.NSE', '.BSE']:
            symbol = symbol.replace(suffix, '')
        return symbol

    # =========================================================
    # HISTORICAL DATA
    # =========================================================

    def get_historical_data(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        start_date: Union[str, datetime] = None,
        end_date: Union[str, datetime] = None,
        exchange: str = None,
        auto_adjust: bool = True,
        include_actions: bool = True,
        add_symbol_column: bool = True,
        prepost: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data

        Args:
            symbol: Stock symbol (e.g., 'RELIANCE', 'TCS')
            period: Data period ('1d','5d','1mo','3mo','6mo','1y','2y','5y','10y','ytd','max')
            interval: Candle interval ('1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo')
            start_date: Start date (overrides period if provided)
            end_date: End date
            exchange: Override exchange ('NSE' or 'BSE')
            auto_adjust: Auto-adjust for splits/dividends
            include_actions: Include dividend & split data
            add_symbol_column: Add symbol column to DataFrame
            prepost: Include pre/post market data

        Returns:
            DataFrame with columns: date, open, high, low, close, volume
            Plus optional: dividends, stock_splits, symbol
        """
        # Validate inputs
        if period not in VALID_PERIODS and start_date is None:
            logger.warning(f"Invalid period '{period}', using '1y'")
            period = "1y"

        if interval not in VALID_INTERVALS:
            logger.warning(f"Invalid interval '{interval}', using '1d'")
            interval = "1d"

        # Check cache
        clean_sym = self._clean_symbol(symbol)
        cache_key = f"hist_{clean_sym}_{period}_{interval}_{start_date}_{end_date}"

        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        # Rate limiting
        self.rate_limiter.wait()

        # Get ticker candidates (primary + fallbacks)
        ticker_candidates = self._get_ticker_candidates(symbol, exchange=exchange)
        ticker_symbol = ticker_candidates[0]

        try:
            # Date normalization once
            if start_date is not None and isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            if start_date is not None and end_date is None:
                end_date = datetime.now()

            # Try primary first, then fallbacks.
            df = pd.DataFrame()
            used_ticker = ticker_symbol
            for candidate in ticker_candidates:
                ticker = yf.Ticker(candidate)
                if start_date is not None:
                    df = ticker.history(
                        start=start_date,
                        end=end_date,
                        interval=interval,
                        auto_adjust=auto_adjust,
                        actions=include_actions,
                        prepost=prepost,
                    )
                else:
                    df = ticker.history(
                        period=period,
                        interval=interval,
                        auto_adjust=auto_adjust,
                        actions=include_actions,
                        prepost=prepost,
                    )
                if not df.empty:
                    used_ticker = candidate
                    break

            if df.empty:
                logger.info(f"No data returned for {ticker_symbol}")
                return pd.DataFrame()

            # Reset index
            df.reset_index(inplace=True)

            # Standardize column names
            column_map = {
                'Date': 'date',
                'Datetime': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Adj Close': 'adj_close',
                'Volume': 'volume',
                'Dividends': 'dividends',
                'Stock Splits': 'stock_splits',
                'Capital Gains': 'capital_gains',
            }
            df.rename(columns=column_map, inplace=True)

            # Ensure date column exists
            if 'date' not in df.columns:
                if df.index.name in ['Date', 'Datetime', 'date']:
                    df.reset_index(inplace=True)
                    df.rename(columns={df.columns[0]: 'date'}, inplace=True)

            # Convert date to datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                # Remove timezone if present
                if df['date'].dt.tz is not None:
                    df['date'] = df['date'].dt.tz_localize(None)

            # Add symbol column
            if add_symbol_column:
                df['symbol'] = clean_sym
                df['ticker'] = used_ticker

            # Remove zero volume rows (holidays/non-trading)
            if 'volume' in df.columns:
                df = df[df['volume'] > 0].copy()

            # Sort by date
            if 'date' in df.columns:
                df.sort_values('date', inplace=True)
                df.reset_index(drop=True, inplace=True)

            # Calculate basic metrics
            if 'close' in df.columns and len(df) > 1:
                df['daily_return'] = df['close'].pct_change()
                df['log_return'] = np.log(df['close'] / df['close'].shift(1))
                df['cum_return'] = (1 + df['daily_return']).cumprod() - 1
                df['high_low_range'] = df['high'] - df['low']
                df['body'] = df['close'] - df['open']
                df['body_pct'] = df['body'] / df['open'] * 100
                df['gap'] = df['open'] / df['close'].shift(1) - 1

            logger.info(
                f"âœ… {clean_sym}: {len(df)} rows | "
                f"{df['date'].iloc[0].strftime('%Y-%m-%d') if 'date' in df.columns and len(df) > 0 else 'N/A'} to "
                f"{df['date'].iloc[-1].strftime('%Y-%m-%d') if 'date' in df.columns and len(df) > 0 else 'N/A'}"
            )

            # Cache the result
            if self.cache:
                # Longer TTL for daily data, shorter for intraday
                ttl = 3600 if interval in ['1d', '5d', '1wk', '1mo'] else 60
                self.cache.set(cache_key, df, ttl=ttl)

            return df

        except Exception as e:
            logger.error(f"âŒ Error fetching {ticker_symbol}: {e}")
            return pd.DataFrame()

    def get_intraday_data(
        self,
        symbol: str,
        interval: str = "5m",
        days: int = 5,
        exchange: str = None
    ) -> pd.DataFrame:
        """
        Get intraday data

        Args:
            symbol: Stock symbol
            interval: Time interval ('1m','2m','5m','15m','30m','60m')
            days: Number of days (max depends on interval)
            exchange: Override exchange

        Returns:
            DataFrame with intraday OHLCV data
        """
        # Validate interval constraints
        max_days_map = {
            '1m': 7, '2m': 60, '5m': 60,
            '15m': 60, '30m': 60, '60m': 730, '1h': 730,
        }

        max_allowed = max_days_map.get(interval, 60)
        if days > max_allowed:
            logger.warning(
                f"Reducing days from {days} to {max_allowed} "
                f"(max for {interval} interval)"
            )
            days = max_allowed

        return self.get_historical_data(
            symbol=symbol,
            period=f"{days}d",
            interval=interval,
            exchange=exchange,
        )

    def get_daily_data(
        self,
        symbol: str,
        years: int = 1,
        exchange: str = None
    ) -> pd.DataFrame:
        """Convenience method for daily OHLCV data"""
        period = f"{years}y" if years <= 10 else "max"
        return self.get_historical_data(
            symbol=symbol,
            period=period,
            interval="1d",
            exchange=exchange,
        )

    def get_weekly_data(
        self,
        symbol: str,
        years: int = 5,
        exchange: str = None
    ) -> pd.DataFrame:
        """Convenience method for weekly OHLCV data"""
        period = f"{years}y" if years <= 10 else "max"
        return self.get_historical_data(
            symbol=symbol,
            period=period,
            interval="1wk",
            exchange=exchange,
        )

    def get_monthly_data(
        self,
        symbol: str,
        years: int = 10,
        exchange: str = None
    ) -> pd.DataFrame:
        """Convenience method for monthly OHLCV data"""
        period = f"{years}y" if years <= 10 else "max"
        return self.get_historical_data(
            symbol=symbol,
            period=period,
            interval="1mo",
            exchange=exchange,
        )

    # =========================================================
    # BULK DATA DOWNLOAD
    # =========================================================

    def get_bulk_historical_data(
        self,
        symbols: List[str],
        period: str = "1y",
        interval: str = "1d",
        exchange: str = None,
        max_workers: int = None,
        progress_callback: Callable = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple stocks concurrently

        Args:
            symbols: List of stock symbols
            period: Data period
            interval: Data interval
            exchange: Override exchange
            max_workers: Override max concurrent threads
            progress_callback: Optional callback fn(symbol, status, current, total)

        Returns:
            Dictionary of symbol -> DataFrame
        """
        workers = max_workers or self.max_workers
        results = {}
        errors = []
        total = len(symbols)
        completed = 0

        logger.info(f"ðŸ“¡ Fetching data for {total} stocks with {workers} workers...")
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_symbol = {
                executor.submit(
                    self.get_historical_data,
                    symbol, period, interval, exchange=exchange
                ): symbol
                for symbol in symbols
            }

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                completed += 1

                try:
                    data = future.result()
                    if data is not None and not data.empty:
                        results[self._clean_symbol(symbol)] = data
                        status = "success"
                    else:
                        errors.append(symbol)
                        status = "no_data"
                except Exception as e:
                    logger.error(f"âŒ Error for {symbol}: {e}")
                    errors.append(symbol)
                    status = "error"

                if progress_callback:
                    progress_callback(symbol, status, completed, total)

                # Log progress
                if completed % 10 == 0 or completed == total:
                    elapsed = time.time() - start_time
                    logger.info(
                        f"  Progress: {completed}/{total} "
                        f"({completed/total*100:.0f}%) | "
                        f"Time: {elapsed:.1f}s"
                    )

        elapsed = time.time() - start_time
        logger.info(
            f"âœ… Bulk download complete: {len(results)}/{total} successful | "
            f"Time: {elapsed:.1f}s | "
            f"Errors: {len(errors)}"
        )

        if errors:
            logger.info(f"Failed symbols: {errors[:20]}{'...' if len(errors) > 20 else ''}")

        return results

    def download_multiple(
        self,
        symbols: List[str],
        period: str = "1y",
        interval: str = "1d",
        exchange: str = None,
        group_by: str = "ticker"
    ) -> pd.DataFrame:
        """
        Download multiple stocks using yfinance bulk download

        Args:
            symbols: List of symbols
            period: Data period
            interval: Data interval
            exchange: Override exchange
            group_by: 'ticker' or 'column'

        Returns:
            Multi-level DataFrame or dict of DataFrames
        """
        # Prepare tickers
        tickers = []
        for symbol in symbols:
            tickers.append(self._get_ticker_symbol(symbol, exchange=exchange))

        ticker_string = " ".join(tickers)

        self.rate_limiter.wait()

        try:
            data = yf.download(
                tickers=ticker_string,
                period=period,
                interval=interval,
                group_by=group_by,
                auto_adjust=True,
                threads=True,
                progress=False,
            )

            if data.empty:
                logger.warning("No data returned from bulk download")
                return pd.DataFrame()

            logger.info(f"âœ… Bulk download: {len(symbols)} tickers, shape={data.shape}")
            return data

        except Exception as e:
            logger.error(f"âŒ Bulk download error: {e}")
            return pd.DataFrame()

    # =========================================================
    # REAL-TIME QUOTES
    # =========================================================

    def get_quote(
        self,
        symbol: str,
        exchange: str = None
    ) -> Optional[StockQuote]:
        """
        Get real-time / last available quote

        Args:
            symbol: Stock symbol
            exchange: Override exchange

        Returns:
            StockQuote object
        """
        cache_key = f"quote_{symbol}_{exchange or self.exchange}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        self.rate_limiter.wait()

        try:
            ticker = self._get_ticker(symbol, exchange=exchange)
            info = ticker.info

            if not info or 'regularMarketPrice' not in info:
                # Try fast_info
                try:
                    fast = ticker.fast_info
                    current_price = fast.get('lastPrice', fast.get('last_price', 0))
                    prev_close = fast.get('previousClose', fast.get('previous_close', 0))
                except:
                    logger.warning(f"No quote data for {symbol}")
                    return None
            else:
                current_price = info.get('regularMarketPrice', info.get('currentPrice', 0))
                prev_close = info.get('regularMarketPreviousClose', info.get('previousClose', 0))

            change = current_price - prev_close if prev_close else 0
            change_pct = (change / prev_close * 100) if prev_close else 0

            quote = StockQuote(
                symbol=self._clean_symbol(symbol),
                exchange=exchange or self.exchange,
                company_name=info.get('longName', info.get('shortName', '')),
                current_price=current_price,
                previous_close=prev_close,
                open_price=info.get('regularMarketOpen', info.get('open', 0)),
                day_high=info.get('regularMarketDayHigh', info.get('dayHigh', 0)),
                day_low=info.get('regularMarketDayLow', info.get('dayLow', 0)),
                volume=info.get('regularMarketVolume', info.get('volume', 0)),
                avg_volume=info.get('averageVolume', 0),
                market_cap=info.get('marketCap', 0),
                pe_ratio=info.get('trailingPE', 0) or 0,
                eps=info.get('trailingEps', 0) or 0,
                week_52_high=info.get('fiftyTwoWeekHigh', 0),
                week_52_low=info.get('fiftyTwoWeekLow', 0),
                change=change,
                change_pct=change_pct,
                bid=info.get('bid', 0),
                ask=info.get('ask', 0),
                bid_size=info.get('bidSize', 0),
                ask_size=info.get('askSize', 0),
                currency=info.get('currency', 'INR'),
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            )

            if self.cache:
                self.cache.set(cache_key, quote, ttl=30)  # 30 sec cache for quotes

            return quote

        except Exception as e:
            logger.error(f"âŒ Error fetching quote for {symbol}: {e}")
            return None

    def get_multiple_quotes(
        self,
        symbols: List[str],
        exchange: str = None
    ) -> Dict[str, StockQuote]:
        """Get quotes for multiple stocks"""
        results = {}

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(symbols))) as executor:
            future_to_symbol = {
                executor.submit(self.get_quote, s, exchange): s
                for s in symbols
            }

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    quote = future.result()
                    if quote:
                        results[self._clean_symbol(symbol)] = quote
                except Exception as e:
                    logger.error(f"Error getting quote for {symbol}: {e}")

        return results

    def get_ltp(
        self,
        symbols: Union[str, List[str]],
        exchange: str = None
    ) -> Dict[str, float]:
        """
        Get Last Traded Price for one or more symbols

        Args:
            symbols: Single symbol or list
            exchange: Override exchange

        Returns:
            Dictionary of symbol -> price
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        results = {}
        for symbol in symbols:
            try:
                ticker = self._get_ticker(symbol, exchange=exchange)
                fast = ticker.fast_info

                price = 0
                for attr in ['last_price', 'lastPrice', 'regularMarketPrice']:
                    try:
                        price = getattr(fast, attr, 0)
                        if price and price > 0:
                            break
                    except:
                        continue

                if price and price > 0:
                    results[self._clean_symbol(symbol)] = float(price)

            except Exception as e:
                logger.debug(f"LTP error for {symbol}: {e}")

        return results

    # =========================================================
    # FUNDAMENTAL DATA
    # =========================================================

    def get_fundamentals(
        self,
        symbol: str,
        exchange: str = None
    ) -> Optional[FundamentalData]:
        """
        Get comprehensive fundamental data

        Args:
            symbol: Stock symbol
            exchange: Override exchange

        Returns:
            FundamentalData object with all fundamentals
        """
        cache_key = f"fundamentals_{symbol}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        self.rate_limiter.wait()

        try:
            ticker = self._get_ticker(symbol, exchange=exchange)
            info = ticker.info

            if not info:
                return None

            def safe_get(key, default=0):
                val = info.get(key, default)
                return val if val is not None else default

            fundamentals = FundamentalData(
                symbol=self._clean_symbol(symbol),
                company_name=safe_get('longName', safe_get('shortName', '')),
                sector=safe_get('sector', ''),
                industry=safe_get('industry', ''),
                market_cap=safe_get('marketCap', 0),
                enterprise_value=safe_get('enterpriseValue', 0),

                # Valuation
                pe_ratio=safe_get('trailingPE', 0),
                forward_pe=safe_get('forwardPE', 0),
                peg_ratio=safe_get('pegRatio', 0),
                pb_ratio=safe_get('priceToBook', 0),
                ps_ratio=safe_get('priceToSalesTrailing12Months', 0),
                ev_to_ebitda=safe_get('enterpriseToEbitda', 0),
                ev_to_revenue=safe_get('enterpriseToRevenue', 0),

                # Profitability
                profit_margin=safe_get('profitMargins', 0),
                operating_margin=safe_get('operatingMargins', 0),
                gross_margin=safe_get('grossMargins', 0),
                roe=safe_get('returnOnEquity', 0),
                roa=safe_get('returnOnAssets', 0),
                roic=0,  # Not available via yfinance

                # Growth
                revenue_growth=safe_get('revenueGrowth', 0),
                earnings_growth=safe_get('earningsGrowth', 0),
                revenue_per_share=safe_get('revenuePerShare', 0),
                earnings_quarterly_growth=safe_get('earningsQuarterlyGrowth', 0),

                # Financial Health
                total_debt=safe_get('totalDebt', 0),
                total_cash=safe_get('totalCash', 0),
                debt_to_equity=safe_get('debtToEquity', 0),
                current_ratio=safe_get('currentRatio', 0),
                quick_ratio=safe_get('quickRatio', 0),
                free_cash_flow=safe_get('freeCashflow', 0),
                operating_cash_flow=safe_get('operatingCashflow', 0),

                # Per Share
                eps=safe_get('trailingEps', 0),
                book_value=safe_get('bookValue', 0),
                dividend_rate=safe_get('dividendRate', 0),
                dividend_yield=safe_get('dividendYield', 0),
                payout_ratio=safe_get('payoutRatio', 0),

                # Trading
                beta=safe_get('beta', 0),
                avg_volume=safe_get('averageVolume', 0),
                avg_volume_10d=safe_get('averageVolume10days', 0),
                shares_outstanding=safe_get('sharesOutstanding', 0),
                float_shares=safe_get('floatShares', 0),
                shares_short=safe_get('sharesShort', 0),
                short_ratio=safe_get('shortRatio', 0),

                # Price
                week_52_high=safe_get('fiftyTwoWeekHigh', 0),
                week_52_low=safe_get('fiftyTwoWeekLow', 0),
                day_50_ma=safe_get('fiftyDayAverage', 0),
                day_200_ma=safe_get('twoHundredDayAverage', 0),
                target_mean_price=safe_get('targetMeanPrice', 0),
                target_high_price=safe_get('targetHighPrice', 0),
                target_low_price=safe_get('targetLowPrice', 0),

                # Misc
                full_time_employees=safe_get('fullTimeEmployees', 0),
                website=safe_get('website', ''),
                description=safe_get('longBusinessSummary', ''),
            )

            if self.cache:
                self.cache.set(cache_key, fundamentals, ttl=3600)  # 1 hour

            return fundamentals

        except Exception as e:
            logger.error(f"âŒ Error fetching fundamentals for {symbol}: {e}")
            return None

    def get_fundamentals_dict(
        self,
        symbol: str,
        exchange: str = None
    ) -> Dict:
        """
        Get fundamentals as a simple dictionary

        Returns:
            Dictionary with all fundamental data
        """
        fund = self.get_fundamentals(symbol, exchange)
        if fund:
            return fund.__dict__
        return {}

    def get_bulk_fundamentals(
        self,
        symbols: List[str],
        exchange: str = None
    ) -> pd.DataFrame:
        """
        Get fundamentals for multiple stocks as DataFrame

        Returns:
            DataFrame with fundamental data for all stocks
        """
        results = []

        with ThreadPoolExecutor(max_workers=min(5, len(symbols))) as executor:
            future_to_symbol = {
                executor.submit(self.get_fundamentals_dict, s, exchange): s
                for s in symbols
            }

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    if data:
                        results.append(data)
                except Exception as e:
                    logger.error(f"Error for {symbol}: {e}")

        if results:
            df = pd.DataFrame(results)
            # Sort by market cap
            if 'market_cap' in df.columns:
                df.sort_values('market_cap', ascending=False, inplace=True)
            df.reset_index(drop=True, inplace=True)
            return df

        return pd.DataFrame()

    # =========================================================
    # FINANCIAL STATEMENTS
    # =========================================================

    def get_income_statement(
        self,
        symbol: str,
        quarterly: bool = False,
        exchange: str = None
    ) -> pd.DataFrame:
        """
        Get income statement (P&L)

        Args:
            symbol: Stock symbol
            quarterly: True for quarterly, False for annual
            exchange: Override exchange

        Returns:
            DataFrame with income statement data
        """
        self.rate_limiter.wait()

        try:
            ticker = self._get_ticker(symbol, exchange=exchange)

            if quarterly:
                df = ticker.quarterly_income_stmt
            else:
                df = ticker.income_stmt

            if df is not None and not df.empty:
                df.columns = [col.strftime('%Y-%m-%d') if isinstance(col, datetime) else str(col) for col in df.columns]
                return df

        except Exception as e:
            logger.error(f"Error fetching income statement for {symbol}: {e}")

        return pd.DataFrame()

    def get_balance_sheet(
        self,
        symbol: str,
        quarterly: bool = False,
        exchange: str = None
    ) -> pd.DataFrame:
        """Get balance sheet"""
        self.rate_limiter.wait()

        try:
            ticker = self._get_ticker(symbol, exchange=exchange)

            if quarterly:
                df = ticker.quarterly_balance_sheet
            else:
                df = ticker.balance_sheet

            if df is not None and not df.empty:
                df.columns = [col.strftime('%Y-%m-%d') if isinstance(col, datetime) else str(col) for col in df.columns]
                return df

        except Exception as e:
            logger.error(f"Error fetching balance sheet for {symbol}: {e}")

        return pd.DataFrame()

    def get_cash_flow(
        self,
        symbol: str,
        quarterly: bool = False,
        exchange: str = None
    ) -> pd.DataFrame:
        """Get cash flow statement"""
        self.rate_limiter.wait()

        try:
            ticker = self._get_ticker(symbol, exchange=exchange)

            if quarterly:
                df = ticker.quarterly_cashflow
            else:
                df = ticker.cashflow

            if df is not None and not df.empty:
                df.columns = [col.strftime('%Y-%m-%d') if isinstance(col, datetime) else str(col) for col in df.columns]
                return df

        except Exception as e:
            logger.error(f"Error fetching cash flow for {symbol}: {e}")

        return pd.DataFrame()

    def get_all_financials(
        self,
        symbol: str,
        quarterly: bool = False,
        exchange: str = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Get all financial statements at once

        Returns:
            Dictionary with 'income_statement', 'balance_sheet', 'cash_flow'
        """
        return {
            'income_statement': self.get_income_statement(symbol, quarterly, exchange),
            'balance_sheet': self.get_balance_sheet(symbol, quarterly, exchange),
            'cash_flow': self.get_cash_flow(symbol, quarterly, exchange),
        }

    # =========================================================
    # DIVIDENDS & SPLITS
    # =========================================================

    def get_dividends(
        self,
        symbol: str,
        exchange: str = None
    ) -> DividendData:
        """
        Get dividend history and details

        Returns:
            DividendData object with full dividend history
        """
        self.rate_limiter.wait()

        try:
            ticker = self._get_ticker(symbol, exchange=exchange)
            info = ticker.info

            # Dividend history
            div_history = ticker.dividends
            if div_history is not None and not div_history.empty:
                div_df = div_history.reset_index()
                div_df.columns = ['date', 'dividend']
                div_df['date'] = pd.to_datetime(div_df['date'])
                if div_df['date'].dt.tz is not None:
                    div_df['date'] = div_df['date'].dt.tz_localize(None)
            else:
                div_df = pd.DataFrame()

            return DividendData(
                symbol=self._clean_symbol(symbol),
                dividend_rate=info.get('dividendRate', 0) or 0,
                dividend_yield=info.get('dividendYield', 0) or 0,
                ex_dividend_date=str(info.get('exDividendDate', '')),
                payout_ratio=info.get('payoutRatio', 0) or 0,
                five_year_avg_yield=info.get('fiveYearAvgDividendYield', 0) or 0,
                history=div_df,
            )

        except Exception as e:
            logger.error(f"Error fetching dividends for {symbol}: {e}")
            return DividendData(symbol=self._clean_symbol(symbol),
                              dividend_rate=0, dividend_yield=0,
                              ex_dividend_date='', payout_ratio=0,
                              five_year_avg_yield=0)

    def get_splits(
        self,
        symbol: str,
        exchange: str = None
    ) -> pd.DataFrame:
        """Get stock split history"""
        self.rate_limiter.wait()

        try:
            ticker = self._get_ticker(symbol, exchange=exchange)
            splits = ticker.splits

            if splits is not None and not splits.empty:
                df = splits.reset_index()
                df.columns = ['date', 'split_ratio']
                df['date'] = pd.to_datetime(df['date'])
                if df['date'].dt.tz is not None:
                    df['date'] = df['date'].dt.tz_localize(None)
                return df

        except Exception as e:
            logger.error(f"Error fetching splits for {symbol}: {e}")

        return pd.DataFrame()

    def get_actions(
        self,
        symbol: str,
        exchange: str = None
    ) -> pd.DataFrame:
        """Get all corporate actions (dividends + splits)"""
        self.rate_limiter.wait()

        try:
            ticker = self._get_ticker(symbol, exchange=exchange)
            actions = ticker.actions

            if actions is not None and not actions.empty:
                actions.reset_index(inplace=True)
                return actions

        except Exception as e:
            logger.error(f"Error fetching actions for {symbol}: {e}")

        return pd.DataFrame()

    # =========================================================
    # ANALYST DATA
    # =========================================================

    def get_analyst_recommendations(
        self,
        symbol: str,
        exchange: str = None
    ) -> pd.DataFrame:
        """Get analyst recommendations (Buy/Hold/Sell)"""
        self.rate_limiter.wait()

        try:
            ticker = self._get_ticker(symbol, exchange=exchange)
            recs = ticker.recommendations

            if recs is not None and not recs.empty:
                recs.reset_index(inplace=True)
                return recs

        except Exception as e:
            logger.error(f"Error fetching recommendations for {symbol}: {e}")

        return pd.DataFrame()

    def get_analyst_price_targets(
        self,
        symbol: str,
        exchange: str = None
    ) -> Dict:
        """Get analyst price targets"""
        self.rate_limiter.wait()

        try:
            ticker = self._get_ticker(symbol, exchange=exchange)
            info = ticker.info

            return {
                'symbol': self._clean_symbol(symbol),
                'current_price': info.get('currentPrice', 0),
                'target_mean': info.get('targetMeanPrice', 0),
                'target_high': info.get('targetHighPrice', 0),
                'target_low': info.get('targetLowPrice', 0),
                'target_median': info.get('targetMedianPrice', 0),
                'recommendation': info.get('recommendationKey', ''),
                'num_analysts': info.get('numberOfAnalystOpinions', 0),
                'upside_pct': (
                    (info.get('targetMeanPrice', 0) / info.get('currentPrice', 1) - 1) * 100
                    if info.get('currentPrice', 0) > 0 else 0
                ),
            }

        except Exception as e:
            logger.error(f"Error fetching price targets for {symbol}: {e}")
            return {}

    def get_upgrades_downgrades(
        self,
        symbol: str,
        exchange: str = None
    ) -> pd.DataFrame:
        """Get analyst upgrades and downgrades"""
        self.rate_limiter.wait()

        try:
            ticker = self._get_ticker(symbol, exchange=exchange)
            upgrades = ticker.upgrades_downgrades

            if upgrades is not None and not upgrades.empty:
                upgrades.reset_index(inplace=True)
                return upgrades

        except Exception as e:
            logger.error(f"Error fetching upgrades/downgrades for {symbol}: {e}")

        return pd.DataFrame()

    # =========================================================
    # INSTITUTIONAL DATA
    # =========================================================

    def get_institutional_holders(
        self,
        symbol: str,
        exchange: str = None
    ) -> pd.DataFrame:
        """Get institutional holders"""
        self.rate_limiter.wait()

        try:
            ticker = self._get_ticker(symbol, exchange=exchange)
            holders = ticker.institutional_holders

            if holders is not None and not holders.empty:
                return holders

        except Exception as e:
            logger.error(f"Error fetching institutional holders for {symbol}: {e}")

        return pd.DataFrame()

    def get_major_holders(
        self,
        symbol: str,
        exchange: str = None
    ) -> pd.DataFrame:
        """Get major holders breakdown"""
        self.rate_limiter.wait()

        try:
            ticker = self._get_ticker(symbol, exchange=exchange)
            holders = ticker.major_holders

            if holders is not None and not holders.empty:
                return holders

        except Exception as e:
            logger.error(f"Error fetching major holders for {symbol}: {e}")

        return pd.DataFrame()

    def get_mutual_fund_holders(
        self,
        symbol: str,
        exchange: str = None
    ) -> pd.DataFrame:
        """Get mutual fund holders"""
        self.rate_limiter.wait()

        try:
            ticker = self._get_ticker(symbol, exchange=exchange)
            holders = ticker.mutualfund_holders

            if holders is not None and not holders.empty:
                return holders

        except Exception as e:
            logger.error(f"Error fetching MF holders for {symbol}: {e}")

        return pd.DataFrame()

    def get_insider_transactions(
        self,
        symbol: str,
        exchange: str = None
    ) -> pd.DataFrame:
        """Get insider transactions"""
        self.rate_limiter.wait()

        try:
            ticker = self._get_ticker(symbol, exchange=exchange)
            insiders = ticker.insider_transactions

            if insiders is not None and not insiders.empty:
                return insiders

        except Exception as e:
            logger.error(f"Error fetching insider transactions for {symbol}: {e}")

        return pd.DataFrame()

    def get_insider_purchases(
        self,
        symbol: str,
        exchange: str = None
    ) -> pd.DataFrame:
        """Get insider purchases"""
        self.rate_limiter.wait()

        try:
            ticker = self._get_ticker(symbol, exchange=exchange)
            purchases = ticker.insider_purchases

            if purchases is not None and not purchases.empty:
                return purchases

        except Exception as e:
            logger.error(f"Error: {e}")

        return pd.DataFrame()

    # =========================================================
    # EARNINGS DATA
    # =========================================================

    def get_earnings(
        self,
        symbol: str,
        exchange: str = None
    ) -> EarningsData:
        """Get comprehensive earnings data"""
        self.rate_limiter.wait()

        try:
            ticker = self._get_ticker(symbol, exchange=exchange)

            # Earnings dates
            earnings_dates = []
            try:
                ed = ticker.earnings_dates
                if ed is not None and not ed.empty:
                    ed.reset_index(inplace=True)
                    earnings_dates = ed.to_dict('records')
            except:
                pass

            # Quarterly earnings
            quarterly_earnings = []
            try:
                qe = ticker.quarterly_earnings
                if qe is not None and not qe.empty:
                    qe.reset_index(inplace=True)
                    quarterly_earnings = qe.to_dict('records')
            except:
                pass

            # Yearly earnings
            yearly_earnings = []
            try:
                ye = ticker.earnings
                if ye is not None and not ye.empty:
                    ye.reset_index(inplace=True)
                    yearly_earnings = ye.to_dict('records')
            except:
                pass

            # Estimates
            earnings_estimate = {}
            revenue_estimate = {}
            try:
                ee = ticker.earnings_estimate
                if ee is not None and not ee.empty:
                    earnings_estimate = ee.to_dict()
            except:
                pass

            try:
                re = ticker.revenue_estimate
                if re is not None and not re.empty:
                    revenue_estimate = re.to_dict()
            except:
                pass

            return EarningsData(
                symbol=self._clean_symbol(symbol),
                earnings_dates=earnings_dates,
                quarterly_earnings=quarterly_earnings,
                yearly_earnings=yearly_earnings,
                earnings_estimate=earnings_estimate,
                revenue_estimate=revenue_estimate,
            )

        except Exception as e:
            logger.error(f"Error fetching earnings for {symbol}: {e}")
            return EarningsData(symbol=self._clean_symbol(symbol))

    def get_earnings_calendar(
        self,
        symbols: List[str] = None,
        exchange: str = None
    ) -> pd.DataFrame:
        """
        Get upcoming earnings dates for stocks

        Args:
            symbols: List of symbols (default: NIFTY 50)
        """
        if symbols is None:
            symbols = NIFTY_50_SYMBOLS[:20]

        results = []

        for symbol in symbols:
            try:
                ticker = self._get_ticker(symbol, exchange=exchange)
                ed = ticker.earnings_dates

                if ed is not None and not ed.empty:
                    ed.reset_index(inplace=True)
                    # Get next earnings date
                    future_dates = ed[ed.iloc[:, 0] > datetime.now()]
                    if not future_dates.empty:
                        next_date = future_dates.iloc[0]
                        results.append({
                            'symbol': self._clean_symbol(symbol),
                            'earnings_date': str(next_date.iloc[0]),
                            'eps_estimate': next_date.get('EPS Estimate', None),
                        })
            except:
                pass

            time.sleep(0.2)

        return pd.DataFrame(results) if results else pd.DataFrame()

    # =========================================================
    # OPTIONS DATA
    # =========================================================

    def get_options_chain(
        self,
        symbol: str,
        expiry_date: str = None,
        exchange: str = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Get options chain data

        Args:
            symbol: Stock symbol
            expiry_date: Specific expiry date (YYYY-MM-DD)
            exchange: Override exchange

        Returns:
            Dictionary with 'calls' and 'puts' DataFrames
        """
        self.rate_limiter.wait()

        try:
            ticker = self._get_ticker(symbol, exchange=exchange)

            # Get available expiry dates
            expirations = ticker.options

            if not expirations:
                logger.warning(f"No options data for {symbol}")
                return {'calls': pd.DataFrame(), 'puts': pd.DataFrame(), 'expiries': []}

            # Select expiry
            if expiry_date:
                expiry = expiry_date
            else:
                expiry = expirations[0]  # Nearest expiry

            # Get chain
            chain = ticker.option_chain(expiry)

            calls = chain.calls if hasattr(chain, 'calls') else pd.DataFrame()
            puts = chain.puts if hasattr(chain, 'puts') else pd.DataFrame()

            return {
                'calls': calls,
                'puts': puts,
                'expiry': expiry,
                'all_expiries': list(expirations),
            }

        except Exception as e:
            logger.error(f"Error fetching options for {symbol}: {e}")
            return {'calls': pd.DataFrame(), 'puts': pd.DataFrame(), 'expiries': []}

    def get_all_options_expiries(
        self,
        symbol: str,
        exchange: str = None
    ) -> List[str]:
        """Get all available options expiry dates"""
        self.rate_limiter.wait()

        try:
            ticker = self._get_ticker(symbol, exchange=exchange)
            return list(ticker.options)
        except:
            return []

    # =========================================================
    # INDEX DATA
    # =========================================================

    def get_index_data(
        self,
        index_name: str = "NIFTY50",
        period: str = "1y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get index historical data

        Args:
            index_name: Index name ('NIFTY50', 'SENSEX', 'BANKNIFTY', etc.)
            period: Data period
            interval: Data interval

        Returns:
            DataFrame with index OHLCV data
        """
        # Map to Yahoo Finance ticker
        ticker_symbol = INDIAN_INDICES.get(
            index_name.upper().replace(" ", ""),
            index_name
        )

        return self.get_historical_data(
            symbol=ticker_symbol,
            period=period,
            interval=interval,
            exchange=None,  # Indices don't need exchange suffix
        )

    def get_india_vix(self, period: str = "1y") -> pd.DataFrame:
        """Get India VIX (volatility index) data"""
        return self.get_index_data("INDIAVIX", period)

    def get_all_indices_data(
        self,
        period: str = "1y",
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """Get historical data for all major Indian indices"""
        results = {}

        for name, ticker in INDIAN_INDICES.items():
            try:
                df = self.get_historical_data(
                    symbol=ticker,
                    period=period,
                    interval=interval,
                )
                if not df.empty:
                    results[name] = df
            except Exception as e:
                logger.debug(f"Error fetching {name}: {e}")

        return results

    def get_indices_snapshot(self) -> pd.DataFrame:
        """Get current snapshot of all Indian indices"""
        results = []

        for name, ticker in INDIAN_INDICES.items():
            try:
                self.rate_limiter.wait()
                t = yf.Ticker(ticker)

                try:
                    fast = t.fast_info
                    price = getattr(fast, 'last_price', 0) or getattr(fast, 'lastPrice', 0)
                    prev = getattr(fast, 'previous_close', 0) or getattr(fast, 'previousClose', 0)
                except:
                    hist = t.history(period="2d")
                    if len(hist) >= 2:
                        price = hist['Close'].iloc[-1]
                        prev = hist['Close'].iloc[-2]
                    else:
                        continue

                if price and prev:
                    results.append({
                        'index': name,
                        'ticker': ticker,
                        'value': float(price),
                        'prev_close': float(prev),
                        'change': float(price - prev),
                        'change_pct': float((price - prev) / prev * 100),
                    })

            except Exception as e:
                logger.debug(f"Error for {name}: {e}")

        df = pd.DataFrame(results)
        if not df.empty:
            df.sort_values('change_pct', ascending=False, inplace=True)
            df.reset_index(drop=True, inplace=True)

        return df

    # =========================================================
    # NEWS & EVENTS
    # =========================================================

    def get_news(
        self,
        symbol: str,
        exchange: str = None
    ) -> List[Dict]:
        """
        Get recent news for a stock

        Returns:
            List of news articles with title, link, publisher, date
        """
        self.rate_limiter.wait()

        try:
            ticker = self._get_ticker(symbol, exchange=exchange)
            news = ticker.news

            if news:
                articles = []
                for article in news:
                    articles.append({
                        'title': article.get('title', ''),
                        'publisher': article.get('publisher', ''),
                        'link': article.get('link', ''),
                        'published': datetime.fromtimestamp(
                            article.get('providerPublishTime', 0)
                        ).strftime('%Y-%m-%d %H:%M'),
                        'type': article.get('type', ''),
                        'thumbnail': article.get('thumbnail', {}).get('resolutions', [{}])[0].get('url', '') if article.get('thumbnail') else '',
                    })
                return articles

        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")

        return []

    def get_calendar(
        self,
        symbol: str,
        exchange: str = None
    ) -> Dict:
        """Get upcoming events calendar (earnings, dividends)"""
        self.rate_limiter.wait()

        try:
            ticker = self._get_ticker(symbol, exchange=exchange)
            calendar = ticker.calendar

            if calendar is not None:
                if isinstance(calendar, pd.DataFrame):
                    return calendar.to_dict()
                elif isinstance(calendar, dict):
                    return calendar

        except Exception as e:
            logger.error(f"Error fetching calendar for {symbol}: {e}")

        return {}

    # =========================================================
    # ESG (Environmental, Social, Governance)
    # =========================================================

    def get_sustainability(
        self,
        symbol: str,
        exchange: str = None
    ) -> Dict:
        """Get ESG / Sustainability scores"""
        self.rate_limiter.wait()

        try:
            ticker = self._get_ticker(symbol, exchange=exchange)
            sustainability = ticker.sustainability

            if sustainability is not None and not sustainability.empty:
                return sustainability.to_dict()

        except Exception as e:
            logger.debug(f"ESG data not available for {symbol}: {e}")

        return {}

    # =========================================================
    # CURRENCY & COMMODITY DATA
    # =========================================================

    def get_currency_data(
        self,
        pair: str = "USDINR",
        period: str = "1y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get currency pair data

        Args:
            pair: Currency pair (e.g., 'USDINR', 'EURINR', 'GBPINR')
            period: Data period
            interval: Data interval
        """
        # Yahoo Finance format: USDINR=X
        ticker = f"{pair}=X"

        return self.get_historical_data(
            symbol=ticker,
            period=period,
            interval=interval,
        )

    def get_commodity_data(
        self,
        commodity: str = "GOLD",
        period: str = "1y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get commodity data

        Args:
            commodity: 'GOLD', 'SILVER', 'CRUDE', 'NATURALGAS'
        """
        commodity_map = {
            'GOLD': 'GC=F',
            'SILVER': 'SI=F',
            'CRUDE': 'CL=F',
            'BRENT': 'BZ=F',
            'NATURALGAS': 'NG=F',
            'COPPER': 'HG=F',
            'PLATINUM': 'PL=F',
            'PALLADIUM': 'PA=F',
        }

        ticker = commodity_map.get(commodity.upper(), commodity)

        return self.get_historical_data(
            symbol=ticker,
            period=period,
            interval=interval,
        )

    def get_mcx_gold_inr(self, period: str = "1y") -> pd.DataFrame:
        """Get Gold price in INR"""
        gold_usd = self.get_commodity_data("GOLD", period)
        usdinr = self.get_currency_data("USDINR", period)

        if gold_usd.empty or usdinr.empty:
            return pd.DataFrame()

        # Merge on date
        merged = gold_usd[['date', 'close']].merge(
            usdinr[['date', 'close']],
            on='date',
            suffixes=('_gold_usd', '_usdinr')
        )

        # Convert to INR per 10 grams (1 troy oz = 31.1035 grams)
        merged['gold_inr_per_10g'] = (
            merged['close_gold_usd'] * merged['close_usdinr'] / 31.1035 * 10
        )

        return merged

    # =========================================================
    # COMPARISON & SCREENING
    # =========================================================

    def compare_stocks(
        self,
        symbols: List[str],
        metrics: List[str] = None,
        exchange: str = None
    ) -> pd.DataFrame:
        """
        Compare multiple stocks on key metrics

        Args:
            symbols: List of symbols to compare
            metrics: List of metrics to compare (default: comprehensive)
            exchange: Override exchange

        Returns:
            DataFrame with comparison data
        """
        if metrics is None:
            metrics = [
                'market_cap', 'pe_ratio', 'pb_ratio', 'dividend_yield',
                'roe', 'profit_margin', 'debt_to_equity', 'revenue_growth',
                'eps', 'beta', 'week_52_high', 'week_52_low',
            ]

        results = []

        for symbol in symbols:
            fund = self.get_fundamentals(symbol, exchange)
            if fund:
                row = {'symbol': fund.symbol}
                for metric in metrics:
                    row[metric] = getattr(fund, metric, None)
                results.append(row)

        return pd.DataFrame(results) if results else pd.DataFrame()

    def screen_stocks(
        self,
        symbols: List[str] = None,
        filters: Dict = None,
        exchange: str = None
    ) -> pd.DataFrame:
        """
        Screen stocks based on fundamental criteria

        Args:
            symbols: List of symbols (default: NIFTY 50)
            filters: Dictionary of {metric: (min, max)} filters
            exchange: Override exchange

        Example:
            filters = {
                'pe_ratio': (0, 25),
                'roe': (0.15, None),  # ROE > 15%
                'debt_to_equity': (None, 1.0),  # D/E < 1
                'dividend_yield': (0.02, None),  # Yield > 2%
            }
        """
        if symbols is None:
            symbols = NIFTY_50_SYMBOLS

        if filters is None:
            filters = {
                'pe_ratio': (0, 30),
                'roe': (0.10, None),
            }

        # Get all fundamentals
        all_funds = self.get_bulk_fundamentals(symbols, exchange)

        if all_funds.empty:
            return pd.DataFrame()

        # Apply filters
        mask = pd.Series(True, index=all_funds.index)

        for metric, (min_val, max_val) in filters.items():
            if metric in all_funds.columns:
                if min_val is not None:
                    mask &= all_funds[metric] >= min_val
                if max_val is not None:
                    mask &= all_funds[metric] <= max_val

        filtered = all_funds[mask].copy()
        filtered.reset_index(drop=True, inplace=True)

        return filtered

    def get_sector_comparison(
        self,
        sector: str = None,
        symbols: List[str] = None,
        exchange: str = None
    ) -> pd.DataFrame:
        """
        Compare stocks within a sector

        Args:
            sector: Sector name (if provided, auto-selects stocks)
            symbols: Manual list of symbols
        """
        if symbols:
            return self.compare_stocks(symbols, exchange=exchange)

        # If sector provided, get stocks in that sector
        if sector:
            sector_symbols = []
            for sym in NIFTY_50_SYMBOLS[:30]:
                fund = self.get_fundamentals(sym, exchange)
                if fund and fund.sector.lower() == sector.lower():
                    sector_symbols.append(sym)

            if sector_symbols:
                return self.compare_stocks(sector_symbols, exchange=exchange)

        return pd.DataFrame()

    # =========================================================
    # DATA WITH TECHNICAL INDICATORS
    # =========================================================

    def get_data_with_indicators(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        exchange: str = None
    ) -> pd.DataFrame:
        """
        Get historical data with pre-calculated technical indicators

        This integrates with the TechnicalIndicators class

        Returns:
            DataFrame with OHLCV + all technical indicators
        """
        # Get raw data
        df = self.get_historical_data(
            symbol=symbol,
            period=period,
            interval=interval,
            exchange=exchange,
        )

        if df.empty:
            return df

        # Calculate indicators
        try:
            from analysis.technical_indicators import TechnicalIndicators
            ti = TechnicalIndicators()
            df = ti.calculate_all(df)
            logger.info(f"âœ… Indicators calculated for {symbol}: {len(df.columns)} columns")
        except ImportError:
            logger.warning(
                "TechnicalIndicators not available. "
                "Returning raw OHLCV data. "
                "Import from analysis.technical_indicators to get indicators."
            )

        return df

    def get_bulk_data_with_indicators(
        self,
        symbols: List[str],
        period: str = "1y",
        interval: str = "1d",
        exchange: str = None,
        max_workers: int = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Get data with indicators for multiple stocks

        Returns:
            Dictionary of symbol -> DataFrame (with indicators)
        """
        # First get raw data
        raw_data = self.get_bulk_historical_data(
            symbols=symbols,
            period=period,
            interval=interval,
            exchange=exchange,
            max_workers=max_workers,
        )

        # Calculate indicators
        try:
            from analysis.technical_indicators import TechnicalIndicators
            ti = TechnicalIndicators()

            for symbol, df in raw_data.items():
                raw_data[symbol] = ti.calculate_all(df)

            logger.info(f"âœ… Indicators calculated for {len(raw_data)} stocks")
        except ImportError:
            logger.warning("TechnicalIndicators module not available")

        return raw_data

    # =========================================================
    # PERFORMANCE ANALYTICS
    # =========================================================

    def get_stock_returns(
        self,
        symbol: str,
        periods: List[str] = None,
        exchange: str = None
    ) -> Dict[str, float]:
        """
        Calculate returns over various periods

        Returns:
            Dictionary of period -> return percentage
        """
        if periods is None:
            periods = ['1d', '1w', '1m', '3m', '6m', '1y', '2y', '3y', '5y', 'ytd']

        df = self.get_historical_data(symbol, period="5y", exchange=exchange)

        if df.empty:
            return {}

        results = {}
        current_price = df['close'].iloc[-1]
        today = df['date'].iloc[-1]

        period_map = {
            '1d': 1, '1w': 5, '2w': 10,
            '1m': 21, '3m': 63, '6m': 126,
            '1y': 252, '2y': 504, '3y': 756, '5y': 1260,
        }

        for period in periods:
            try:
                if period == 'ytd':
                    year_start = df[df['date'].dt.year == today.year].iloc[0]
                    start_price = year_start['close']
                elif period in period_map:
                    days = period_map[period]
                    idx = max(0, len(df) - days - 1)
                    start_price = df['close'].iloc[idx]
                else:
                    continue

                ret = (current_price - start_price) / start_price * 100
                results[period] = round(ret, 2)

            except:
                results[period] = None

        return results

    def get_volatility_metrics(
        self,
        symbol: str,
        period: str = "1y",
        exchange: str = None
    ) -> Dict:
        """
        Calculate volatility metrics

        Returns:
            Dictionary with various volatility measures
        """
        df = self.get_historical_data(symbol, period=period, exchange=exchange)

        if df.empty or len(df) < 30:
            return {}

        returns = df['close'].pct_change().dropna()

        return {
            'symbol': self._clean_symbol(symbol),
            'daily_volatility': float(returns.std()),
            'annualized_volatility': float(returns.std() * np.sqrt(252)),
            'max_daily_gain': float(returns.max() * 100),
            'max_daily_loss': float(returns.min() * 100),
            'avg_daily_return': float(returns.mean() * 100),
            'sharpe_ratio': float(
                (returns.mean() * 252) / (returns.std() * np.sqrt(252))
                if returns.std() > 0 else 0
            ),
            'sortino_ratio': float(
                (returns.mean() * 252) / (returns[returns < 0].std() * np.sqrt(252))
                if len(returns[returns < 0]) > 0 and returns[returns < 0].std() > 0 else 0
            ),
            'max_drawdown': float(self._calculate_max_drawdown(df['close'])),
            'calmar_ratio': float(
                (returns.mean() * 252) / abs(self._calculate_max_drawdown(df['close']))
                if abs(self._calculate_max_drawdown(df['close'])) > 0 else 0
            ),
            'positive_days_pct': float(
                len(returns[returns > 0]) / len(returns) * 100
            ),
            'skewness': float(returns.skew()),
            'kurtosis': float(returns.kurtosis()),
            'var_95': float(np.percentile(returns, 5) * 100),  # 95% VaR
            'cvar_95': float(returns[returns <= np.percentile(returns, 5)].mean() * 100),
        }

    @staticmethod
    def _calculate_max_drawdown(prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return float(drawdown.min())

    def get_correlation_matrix(
        self,
        symbols: List[str],
        period: str = "1y",
        exchange: str = None
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix between stocks

        Returns:
            DataFrame with pairwise correlations
        """
        data = self.download_multiple(symbols, period=period, exchange=exchange)

        if data.empty:
            return pd.DataFrame()

        # Extract close prices
        try:
            if isinstance(data.columns, pd.MultiIndex):
                closes = data.xs('Close', axis=1, level=0)
            else:
                closes = data[['Close']]
        except:
            closes = data

        # Calculate returns
        returns = closes.pct_change().dropna()

        # Clean column names
        returns.columns = [self._clean_symbol(str(c)) for c in returns.columns]

        return returns.corr()

    # =========================================================
    # UTILITY METHODS
    # =========================================================

    def validate_symbol(self, symbol: str, exchange: str = None) -> bool:
        """Check if a symbol is valid on Yahoo Finance"""
        try:
            ticker_symbol = self._get_ticker_symbol(symbol, exchange=exchange)
            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(period="5d")
            return not hist.empty
        except:
            return False

    def get_trading_days(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime] = None
    ) -> pd.DatetimeIndex:
        """Get list of trading days (NSE) between dates"""
        if end_date is None:
            end_date = datetime.now()

        df = self.get_historical_data(
            "^NSEI",
            start_date=start_date,
            end_date=end_date,
        )

        if not df.empty and 'date' in df.columns:
            return pd.DatetimeIndex(df['date'])

        return pd.DatetimeIndex([])

    def is_market_open(self) -> bool:
        """Check if Indian market is currently open"""
        now = datetime.now()

        # Weekend check
        if now.weekday() >= 5:
            return False

        # Market hours: 9:15 AM to 3:30 PM IST
        market_open = now.replace(hour=9, minute=15, second=0)
        market_close = now.replace(hour=15, minute=30, second=0)

        return market_open <= now <= market_close

    def clear_cache(self):
        """Clear all cached data"""
        if self.cache:
            self.cache.clear()
        self._ticker_cache.clear()
        logger.info("All caches cleared")

    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        stats = {
            'ticker_cache_size': len(self._ticker_cache),
        }
        if self.cache:
            stats.update(self.cache.get_stats())
        return stats

    # =========================================================
    # EXPORT METHODS
    # =========================================================

    def export_to_csv(
        self,
        symbol: str,
        filepath: str = None,
        period: str = "1y",
        exchange: str = None
    ) -> str:
        """Export stock data to CSV file"""
        df = self.get_historical_data(symbol, period=period, exchange=exchange)

        if df.empty:
            logger.warning(f"No data to export for {symbol}")
            return ""

        if filepath is None:
            filepath = f"{self._clean_symbol(symbol)}_{period}_{datetime.now().strftime('%Y%m%d')}.csv"

        df.to_csv(filepath, index=False)
        logger.info(f"âœ… Exported {symbol} to {filepath}")
        return filepath

    def export_bulk_to_csv(
        self,
        symbols: List[str],
        output_dir: str = "data/exports",
        period: str = "1y",
        exchange: str = None
    ) -> List[str]:
        """Export multiple stocks to CSV files"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        data = self.get_bulk_historical_data(symbols, period=period, exchange=exchange)
        files = []

        for symbol, df in data.items():
            filepath = os.path.join(
                output_dir,
                f"{symbol}_{period}_{datetime.now().strftime('%Y%m%d')}.csv"
            )
            df.to_csv(filepath, index=False)
            files.append(filepath)

        logger.info(f"âœ… Exported {len(files)} files to {output_dir}")
        return files

    def export_to_excel(
        self,
        symbol: str,
        filepath: str = None,
        period: str = "1y",
        include_fundamentals: bool = True,
        include_financials: bool = True,
        exchange: str = None
    ) -> str:
        """
        Export comprehensive stock data to Excel

        Creates multiple sheets:
        - Price History
        - Fundamentals
        - Income Statement
        - Balance Sheet
        - Cash Flow
        - Dividends
        """
        if filepath is None:
            filepath = f"{self._clean_symbol(symbol)}_analysis_{datetime.now().strftime('%Y%m%d')}.xlsx"

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Price History
            df = self.get_historical_data(symbol, period=period, exchange=exchange)
            if not df.empty:
                df.to_excel(writer, sheet_name='Price History', index=False)

            # Fundamentals
            if include_fundamentals:
                fund = self.get_fundamentals_dict(symbol, exchange)
                if fund:
                    fund_df = pd.DataFrame([fund])
                    fund_df.to_excel(writer, sheet_name='Fundamentals', index=False)

            # Financials
            if include_financials:
                for name, method in [
                    ('Income Statement', self.get_income_statement),
                    ('Balance Sheet', self.get_balance_sheet),
                    ('Cash Flow', self.get_cash_flow),
                ]:
                    try:
                        fin_df = method(symbol, exchange=exchange)
                        if not fin_df.empty:
                            fin_df.to_excel(writer, sheet_name=name)
                    except:
                        pass

            # Dividends
            div_data = self.get_dividends(symbol, exchange)
            if not div_data.history.empty:
                div_data.history.to_excel(writer, sheet_name='Dividends', index=False)

        logger.info(f"âœ… Excel exported: {filepath}")
        return filepath


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

# Global fetcher instance
_default_fetcher = None


def _get_fetcher() -> YahooFinanceFetcher:
    """Get or create default fetcher"""
    global _default_fetcher
    if _default_fetcher is None:
        _default_fetcher = YahooFinanceFetcher()
    return _default_fetcher


def get_stock_data(
    symbol: str,
    period: str = "1y",
    interval: str = "1d"
) -> pd.DataFrame:
    """Quick function to get stock data"""
    return _get_fetcher().get_historical_data(symbol, period, interval)


def get_stock_quote(symbol: str) -> Optional[StockQuote]:
    """Quick function to get stock quote"""
    return _get_fetcher().get_quote(symbol)


def get_stock_fundamentals(symbol: str) -> Optional[FundamentalData]:
    """Quick function to get fundamentals"""
    return _get_fetcher().get_fundamentals(symbol)


def get_nifty50_data(period: str = "1y") -> Dict[str, pd.DataFrame]:
    """Quick function to get all NIFTY 50 stock data"""
    return _get_fetcher().get_bulk_historical_data(NIFTY_50_SYMBOLS, period=period)


def get_index(index_name: str = "NIFTY50", period: str = "1y") -> pd.DataFrame:
    """Quick function to get index data"""
    return _get_fetcher().get_index_data(index_name, period)


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    """Test Yahoo Finance Fetcher"""

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    print("=" * 70)
    print("ðŸ“Š Yahoo Finance Fetcher - Test Suite (Indian Market)")
    print("=" * 70)

    fetcher = YahooFinanceFetcher(exchange="NSE", use_cache=True)

    # Test 1: Historical Data
    print("\nðŸ“Œ Test 1: Historical Data (RELIANCE - 6 months)")
    df = fetcher.get_historical_data("RELIANCE", period="6mo")
    if not df.empty:
        print(f"  Rows: {len(df)} | Columns: {len(df.columns)}")
        print(f"  Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
        print(f"  Latest close: â‚¹{df['close'].iloc[-1]:.2f}")
        print(f"  Columns: {list(df.columns[:10])}...")

    # Test 2: Intraday Data
    print("\nðŸ“Œ Test 2: Intraday Data (TCS - 5min)")
    intra = fetcher.get_intraday_data("TCS", interval="5m", days=2)
    if not intra.empty:
        print(f"  Rows: {len(intra)}")
        print(f"  Latest: â‚¹{intra['close'].iloc[-1]:.2f}")

    # Test 3: Stock Quote
    print("\nðŸ“Œ Test 3: Real-time Quote (INFY)")
    quote = fetcher.get_quote("INFY")
    if quote:
        print(f"  {quote}")
        print(f"  Market Cap: â‚¹{quote.market_cap/1e7:,.0f} Cr")
        print(f"  P/E: {quote.pe_ratio:.2f}")
        print(f"  52W High: â‚¹{quote.week_52_high:.2f}")
        print(f"  52W Low: â‚¹{quote.week_52_low:.2f}")

    # Test 4: Fundamentals
    print("\nðŸ“Œ Test 4: Fundamentals (HDFCBANK)")
    fund = fetcher.get_fundamentals("HDFCBANK")
    if fund:
        print(f"  Company: {fund.company_name}")
        print(f"  Sector: {fund.sector} | Industry: {fund.industry}")
        print(f"  Market Cap: â‚¹{fund.market_cap/1e7:,.0f} Cr")
        print(f"  P/E: {fund.pe_ratio:.2f} | P/B: {fund.pb_ratio:.2f}")
        print(f"  ROE: {fund.roe*100:.1f}% | Debt/Equity: {fund.debt_to_equity:.2f}")
        print(f"  Dividend Yield: {fund.dividend_yield*100:.2f}%")

    # Test 5: Bulk Download
    print("\nðŸ“Œ Test 5: Bulk Download (5 stocks)")
    bulk = fetcher.get_bulk_historical_data(
        ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"],
        period="3mo"
    )
    for sym, data in bulk.items():
        print(f"  {sym}: {len(data)} rows | Latest: â‚¹{data['close'].iloc[-1]:.2f}")

    # Test 6: Financial Statements
    print("\nðŸ“Œ Test 6: Income Statement (TCS)")
    income = fetcher.get_income_statement("TCS")
    if not income.empty:
        print(f"  Rows: {len(income)} | Periods: {list(income.columns[:4])}")

    # Test 7: Dividends
    print("\nðŸ“Œ Test 7: Dividends (ITC)")
    div = fetcher.get_dividends("ITC")
    print(f"  Dividend Rate: â‚¹{div.dividend_rate}")
    print(f"  Yield: {div.dividend_yield*100:.2f}%")
    if not div.history.empty:
        print(f"  History entries: {len(div.history)}")

    # Test 8: Analyst Recommendations
    print("\nðŸ“Œ Test 8: Analyst Targets (RELIANCE)")
    targets = fetcher.get_analyst_price_targets("RELIANCE")
    if targets:
        print(f"  Target Mean: â‚¹{targets.get('target_mean', 0):.2f}")
        print(f"  Upside: {targets.get('upside_pct', 0):.1f}%")
        print(f"  Analysts: {targets.get('num_analysts', 0)}")

    # Test 9: Index Data
    print("\nðŸ“Œ Test 9: Index Snapshot")
    indices = fetcher.get_indices_snapshot()
    if not indices.empty:
        for _, row in indices.head(5).iterrows():
            print(f"  {row['index']}: {row['value']:,.2f} ({row['change_pct']:+.2f}%)")

    # Test 10: Returns
    print("\nðŸ“Œ Test 10: Stock Returns (BAJFINANCE)")
    returns = fetcher.get_stock_returns("BAJFINANCE")
    if returns:
        for period, ret in returns.items():
            if ret is not None:
                print(f"  {period}: {ret:+.2f}%")

    # Test 11: Volatility
    print("\nðŸ“Œ Test 11: Volatility Metrics (TATAMOTORS)")
    vol = fetcher.get_volatility_metrics("TATAMOTORS")
    if vol:
        print(f"  Annual Volatility: {vol['annualized_volatility']*100:.1f}%")
        print(f"  Max Drawdown: {vol['max_drawdown']*100:.1f}%")
        print(f"  Sharpe Ratio: {vol['sharpe_ratio']:.2f}")

    # Test 12: News
    print("\nðŸ“Œ Test 12: News (RELIANCE)")
    news = fetcher.get_news("RELIANCE")
    for article in news[:3]:
        print(f"  ðŸ“° {article['title'][:80]}")
        print(f"     {article['publisher']} | {article['published']}")

    # Test 13: Currency
    print("\nðŸ“Œ Test 13: USD/INR")
    usdinr = fetcher.get_currency_data("USDINR", period="1mo")
    if not usdinr.empty:
        print(f"  Current: â‚¹{usdinr['close'].iloc[-1]:.2f}")

    # Test 14: Correlation
    print("\nðŸ“Œ Test 14: Correlation (IT stocks)")
    corr = fetcher.get_correlation_matrix(
        ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM"],
        period="6mo"
    )
    if not corr.empty:
        print(corr.to_string())

    # Test 15: Stock Screening
    print("\nðŸ“Œ Test 15: Stock Screener")
    screened = fetcher.screen_stocks(
        symbols=NIFTY_50_SYMBOLS[:20],
        filters={
            'pe_ratio': (0, 25),
            'roe': (0.15, None),
            'dividend_yield': (0.01, None),
        }
    )
    if not screened.empty:
        print(f"  Stocks passing filters: {len(screened)}")
        for _, row in screened.head(5).iterrows():
            print(f"    {row['symbol']} | PE: {row.get('pe_ratio', 'N/A'):.1f} | ROE: {row.get('roe', 0)*100:.1f}%")

    # Test 16: Cache Stats
    print("\nðŸ“Œ Test 16: Cache Stats")
    stats = fetcher.get_cache_stats()
    print(f"  {stats}")

    # Test 17: Options
    print("\nðŸ“Œ Test 17: Options Chain (RELIANCE)")
    options = fetcher.get_options_chain("RELIANCE")
    if options.get('all_expiries'):
        print(f"  Available expiries: {len(options['all_expiries'])}")
        print(f"  Nearest expiry: {options.get('expiry', 'N/A')}")
        if not options['calls'].empty:
            print(f"  Call options: {len(options['calls'])}")
        if not options['puts'].empty:
            print(f"  Put options: {len(options['puts'])}")

    # Test 18: Market Status
    print("\nðŸ“Œ Test 18: Market Status")
    print(f"  Market open: {fetcher.is_market_open()}")

    print("\n" + "=" * 70)
    print("âœ… Yahoo Finance Fetcher - All tests completed!")
    print("=" * 70)


