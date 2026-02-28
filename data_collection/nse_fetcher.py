"""
Fetch data from NSE India using multiple sources
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import requests
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


class NSEFetcher:
    """Fetch stock data from NSE India"""
    
    BASE_URL = "https://www.nseindia.com"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
        })
        self._init_session()
    
    def _init_session(self):
        """Initialize session with NSE cookies"""
        try:
            self.session.get(self.BASE_URL, timeout=10)
            time.sleep(1)
        except Exception as e:
            logger.error(f"Failed to initialize NSE session: {e}")
    
    def get_stock_quote(self, symbol: str) -> Dict:
        """Get real-time quote for a stock"""
        url = f"{self.BASE_URL}/api/quote-equity?symbol={symbol}"
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
        return {}
    
    def get_stock_info(self, symbol: str) -> Dict:
        """Get detailed stock information"""
        url = f"{self.BASE_URL}/api/quote-equity?symbol={symbol}&section=trade_info"
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {e}")
        return {}
    
    def get_market_status(self) -> Dict:
        """Get current market status"""
        url = f"{self.BASE_URL}/api/marketStatus"
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Error fetching market status: {e}")
        return {}
    
    def get_nifty50_stocks(self) -> pd.DataFrame:
        """Get all NIFTY 50 stock data"""
        url = f"{self.BASE_URL}/api/equity-stockIndices?index=NIFTY%2050"
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return pd.DataFrame(data.get('data', []))
        except Exception as e:
            logger.error(f"Error fetching NIFTY 50 data: {e}")
        return pd.DataFrame()
    
    def get_all_stocks_data(self, index: str = "NIFTY 500") -> pd.DataFrame:
        """Get data for all stocks in an index"""
        index_encoded = index.replace(" ", "%20")
        url = f"{self.BASE_URL}/api/equity-stockIndices?index={index_encoded}"
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data.get('data', []))
                return df
        except Exception as e:
            logger.error(f"Error fetching {index} data: {e}")
        return pd.DataFrame()
    
    def get_advances_declines(self) -> Dict:
        """Get market breadth data"""
        url = f"{self.BASE_URL}/api/market-data-pre-open?key=ALL"
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Error fetching advances/declines: {e}")
        return {}
    
    def get_option_chain(self, symbol: str = "NIFTY") -> Dict:
        """Get option chain data for derivatives analysis"""
        url = f"{self.BASE_URL}/api/option-chain-indices?symbol={symbol}"
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Error fetching option chain for {symbol}: {e}")
        return {}
    
    def get_bulk_deals(self) -> pd.DataFrame:
        """Get bulk deal data"""
        url = f"{self.BASE_URL}/api/snapshot-capital-market-largedeal"
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return pd.DataFrame(data.get('data', []))
        except Exception as e:
            logger.error(f"Error fetching bulk deals: {e}")
        return pd.DataFrame()


class YahooFinanceFetcher:
    """Fetch historical data from Yahoo Finance for Indian stocks"""
    
    def __init__(self):
        pass  # yfinance handles sessions internally
    
    def get_historical_data(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        add_suffix: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE')
            period: Data period ('1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
            interval: Data interval ('1m', '5m', '15m', '1h', '1d', '1wk', '1mo')
            add_suffix: Whether to add '.NS' suffix for NSE stocks
        """
        import yfinance as yf
        
        ticker = f"{symbol}.NS" if add_suffix else symbol
        fallback_tickers = {
            "TATAMOTORS": ["TATAMTRDVR.NS", "TATAMOTORS.BO"],
            "M&M": ["M&M.NS", "M_M.NS"],
            "BAJAJ-AUTO": ["BAJAJ-AUTO.NS", "BAJAJAUTO.NS"],
        }
        tickers_to_try = [ticker]
        if add_suffix and symbol.upper() in fallback_tickers:
            tickers_to_try.extend(fallback_tickers[symbol.upper()])
        
        try:
            df = pd.DataFrame()
            used_ticker = ticker
            for candidate in tickers_to_try:
                stock = yf.Ticker(candidate)
                df = stock.history(period=period, interval=interval)
                if not df.empty:
                    used_ticker = candidate
                    break

            if df.empty:
                logger.warning(f"No data found for {ticker}")
                return pd.DataFrame()
            
            df.reset_index(inplace=True)
            df['Symbol'] = symbol
            df['ticker'] = used_ticker
            
            # Standardize column names
            df.rename(columns={
                'Date': 'date',
                'Datetime': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Dividends': 'dividends',
                'Stock Splits': 'stock_splits'
            }, inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_bulk_historical_data(
        self,
        symbols: List[str],
        period: str = "1y",
        interval: str = "1d",
        max_workers: int = 10
    ) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for multiple stocks concurrently"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(
                    self.get_historical_data, symbol, period, interval
                ): symbol
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    if not data.empty:
                        results[symbol] = data
                        logger.info(f"âœ… Fetched data for {symbol}: {len(data)} rows")
                    else:
                        logger.warning(f"âš ï¸ No data for {symbol}")
                except Exception as e:
                    logger.error(f"âŒ Error for {symbol}: {e}")
        
        return results
    
    def get_intraday_data(self, symbol: str, interval: str = "5m") -> pd.DataFrame:
        """Fetch intraday data (last 60 days max for 5m intervals)"""
        return self.get_historical_data(
            symbol, period="60d", interval=interval
        )
    
    def get_fundamentals(self, symbol: str) -> Dict:
        """Fetch fundamental data"""
        import yfinance as yf
        
        ticker = f"{symbol}.NS"
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                'symbol': symbol,
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'pb_ratio': info.get('priceToBook', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'eps': info.get('trailingEps', 0),
                'roe': info.get('returnOnEquity', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'profit_margin': info.get('profitMargins', 0),
                'operating_margin': info.get('operatingMargins', 0),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0),
                'avg_volume': info.get('averageVolume', 0),
                'beta': info.get('beta', 0),
                'book_value': info.get('bookValue', 0),
            }
        except Exception as e:
            logger.error(f"Error fetching fundamentals for {symbol}: {e}")
            return {}


class JugaadDataFetcher:
    """Fetch data using jugaad-data library (free NSE data)"""
    
    def get_stock_history(
        self,
        symbol: str,
        from_date: datetime,
        to_date: datetime
    ) -> pd.DataFrame:
        """Fetch historical data from NSE using jugaad-data"""
        try:
            from jugaad_data.nse import stock_df
            
            df = stock_df(
                symbol=symbol,
                from_date=from_date.date(),
                to_date=to_date.date(),
                series="EQ"
            )
            
            df.rename(columns={
                'DATE': 'date',
                'OPEN': 'open',
                'HIGH': 'high',
                'LOW': 'low',
                'CLOSE': 'close',
                'LTP': 'ltp',
                'VOLUME': 'volume',
                'VALUE': 'value',
                'NO OF TRADES': 'no_of_trades',
            }, inplace=True)
            
            df['Symbol'] = symbol
            return df
            
        except Exception as e:
            logger.error(f"Error fetching jugaad data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_index_history(
        self,
        symbol: str = "NIFTY 50",
        from_date: datetime = None,
        to_date: datetime = None
    ) -> pd.DataFrame:
        """Fetch index historical data"""
        try:
            from jugaad_data.nse import index_df
            
            if from_date is None:
                from_date = datetime.now() - timedelta(days=365)
            if to_date is None:
                to_date = datetime.now()
            
            df = index_df(
                symbol=symbol,
                from_date=from_date.date(),
                to_date=to_date.date()
            )
            return df
            
        except Exception as e:
            logger.error(f"Error fetching index data for {symbol}: {e}")
            return pd.DataFrame()
