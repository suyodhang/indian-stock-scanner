"""
Live data feed using Zerodha Kite or other brokers
"""

import logging
from typing import Callable, List, Dict
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class ZerodhaLiveFeed:
    """Live market data using Zerodha Kite Connect"""
    
    def __init__(self, api_key: str, access_token: str):
        self.api_key = api_key
        self.access_token = access_token
        self.kite = None
        self.kws = None
        self.callbacks = []
        self._setup_connection()
    
    def _setup_connection(self):
        """Setup Kite Connect connection"""
        try:
            from kiteconnect import KiteConnect, KiteTicker
            
            self.kite = KiteConnect(api_key=self.api_key)
            self.kite.set_access_token(self.access_token)
            
            self.kws = KiteTicker(self.api_key, self.access_token)
            self.kws.on_ticks = self._on_ticks
            self.kws.on_connect = self._on_connect
            self.kws.on_close = self._on_close
            self.kws.on_error = self._on_error
            
            logger.info("âœ… Zerodha connection established")
            
        except ImportError:
            logger.warning("kiteconnect not installed. Install with: pip install kiteconnect")
        except Exception as e:
            logger.error(f"Error setting up Zerodha connection: {e}")
    
    def get_instrument_tokens(self, symbols: List[str], exchange: str = "NSE") -> Dict[str, int]:
        """Get instrument tokens for given symbols"""
        instruments = self.kite.instruments(exchange)
        token_map = {}
        
        for inst in instruments:
            if inst['tradingsymbol'] in symbols:
                token_map[inst['tradingsymbol']] = inst['instrument_token']
        
        return token_map
    
    def subscribe(self, tokens: List[int], mode: str = "full"):
        """Subscribe to live ticks"""
        self.kws.subscribe(tokens)
        self.kws.set_mode(self.kws.MODE_FULL if mode == "full" else self.kws.MODE_LTP, tokens)
    
    def register_callback(self, callback: Callable):
        """Register a callback function for tick data"""
        self.callbacks.append(callback)
    
    def _on_ticks(self, ws, ticks):
        """Handle incoming tick data"""
        for callback in self.callbacks:
            try:
                callback(ticks)
            except Exception as e:
                logger.error(f"Error in tick callback: {e}")
    
    def _on_connect(self, ws, response):
        """Handle connection event"""
        logger.info("âœ… WebSocket connected")
    
    def _on_close(self, ws, code, reason):
        """Handle connection close"""
        logger.warning(f"WebSocket closed: {code} - {reason}")
    
    def _on_error(self, ws, code, reason):
        """Handle error"""
        logger.error(f"WebSocket error: {code} - {reason}")
    
    def start(self):
        """Start the live feed"""
        if self.kws:
            self.kws.connect(threaded=True)
    
    def stop(self):
        """Stop the live feed"""
        if self.kws:
            self.kws.close()
    
    def get_ltp(self, symbols: List[str], exchange: str = "NSE") -> Dict:
        """Get Last Traded Price for symbols"""
        instruments = [f"{exchange}:{symbol}" for symbol in symbols]
        return self.kite.ltp(instruments)
    
    def get_ohlc(self, symbols: List[str], exchange: str = "NSE") -> Dict:
        """Get OHLC data for symbols"""
        instruments = [f"{exchange}:{symbol}" for symbol in symbols]
        return self.kite.ohlc(instruments)


class SimulatedLiveFeed:
    """Simulated live feed for testing (uses delayed Yahoo Finance data)"""
    
    def __init__(self):
        import yfinance as yf
        self.yf = yf
    
    def get_current_data(self, symbols: List[str]) -> pd.DataFrame:
        """Get near real-time data"""
        import yfinance as yf
        
        tickers = [f"{s}.NS" for s in symbols]
        data = yf.download(
            tickers=" ".join(tickers),
            period="1d",
            interval="1m",
            progress=False
        )
        return data
