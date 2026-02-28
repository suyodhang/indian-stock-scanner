"""
Main Application - Stock Scanner Orchestrator
"""

import schedule
import time
import logging
import sys
from datetime import datetime
from typing import Dict, List

from config.settings import config, scanner_config
from config.stock_universe import NIFTY_50, NIFTY_NEXT_50
from data_collection.nse_fetcher import YahooFinanceFetcher
from analysis.technical_indicators import TechnicalIndicators
from scanners.momentum_scanner import MomentumScanner
from scanners.breakout_scanner import BreakoutScanner
from scanners.custom_scanner import CustomScanner, get_bullish_scanner_conditions
from alerts.telegram_bot import TelegramBot
from ai_models.trend_predictor import TrendPredictor

# Setup logging
# Avoid Windows console encoding errors for non-ASCII log messages.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('scanner.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class StockScannerApp:
    """Main application orchestrator"""
    
    def __init__(self):
        self.fetcher = YahooFinanceFetcher()
        self.indicators = TechnicalIndicators()
        self.momentum_scanner = MomentumScanner()
        self.breakout_scanner = BreakoutScanner()
        self.custom_scanner = CustomScanner()
        
        # AI Models
        self.trend_predictor = TrendPredictor()
        
        # Alerts
        self.telegram_bot = None
        if config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID:
            self.telegram_bot = TelegramBot(
                config.TELEGRAM_BOT_TOKEN,
                config.TELEGRAM_CHAT_ID
            )
        
        # Stock universe
        self.symbols = NIFTY_50
        
        # Cache
        self.stock_data: Dict = {}
        self.last_fetch_time = None
    
    def fetch_data(self):
        """Fetch latest data for all stocks"""
        logger.info(f"ðŸ“¡ Fetching data for {len(self.symbols)} stocks...")
        
        raw_data = self.fetcher.get_bulk_historical_data(
            self.symbols, period="1y", interval="1d"
        )
        
        # Calculate indicators
        for symbol, df in raw_data.items():
            self.stock_data[symbol] = self.indicators.calculate_all(df)
        
        self.last_fetch_time = datetime.now()
        logger.info(f"âœ… Data fetched for {len(self.stock_data)} stocks")
    
    def run_scanners(self):
        """Run all scanners"""
        logger.info("ðŸ” Running scanners...")
        
        all_results = []
        
        # Momentum Scanner
        momentum_results = self.momentum_scanner.run_all_scans(self.stock_data)
        all_results.extend(momentum_results)
        logger.info(f"  Momentum: {len(momentum_results)} signals")
        
        # Breakout Scanner
        breakout_results = self.breakout_scanner.run_all_scans(self.stock_data)
        all_results.extend(breakout_results)
        logger.info(f"  Breakout: {len(breakout_results)} signals")
        
        # Custom Bullish Scanner
        bullish_conditions = get_bullish_scanner_conditions()
        custom_results = self.custom_scanner.multi_condition_scan(
            self.stock_data, bullish_conditions, min_conditions_met=5
        )
        all_results.extend(custom_results)
        logger.info(f"  Custom Bullish: {len(custom_results)} signals")
        
        # AI-Enhanced Scanner
        if self.trend_predictor.is_trained:
            ai_results = self.custom_scanner.ai_enhanced_scan(self.stock_data)
            all_results.extend(ai_results)
            logger.info(f"  AI Enhanced: {len(ai_results)} signals")
        
        # Sort by strength
        all_results.sort(key=lambda x: x.strength, reverse=True)
        
        logger.info(f"ðŸ“Š Total signals: {len(all_results)}")
        
        # Send alerts
        if self.telegram_bot and all_results:
            top_results = all_results[:10]
            self.telegram_bot.send_scan_results(top_results, "Scanner Results")
        
        return all_results
    
    def train_ai_models(self):
        """Train/retrain AI models"""
        logger.info("ðŸ§  Training AI models...")
        
        for symbol in ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK']:
            if symbol in self.stock_data:
                df = self.stock_data[symbol]
                X, y = self.trend_predictor.prepare_features(df)
                
                if len(X) > 200:
                    results = self.trend_predictor.train(X, y)
                    logger.info(f"  {symbol} - Ensemble Accuracy: {results['ensemble']['accuracy']:.4f}")
                    break
        
        # Save model
        self.trend_predictor.save_model('models/trend_predictor.pkl')
        logger.info("âœ… AI models trained and saved")
    
    def run_scheduled(self):
        """Run on a schedule during market hours"""
        logger.info("ðŸš€ Starting Stock Scanner...")
        
        # Initial fetch and scan
        self.fetch_data()
        self.train_ai_models()
        self.run_scanners()
        
        # Schedule periodic scans
        schedule.every(config.SCAN_INTERVAL_MINUTES).minutes.do(self.fetch_data)
        schedule.every(config.SCAN_INTERVAL_MINUTES).minutes.do(self.run_scanners)
        
        # Daily model retrain
        schedule.every().day.at("08:00").do(self.train_ai_models)
        
        # End of day summary
        schedule.every().day.at("15:35").do(self._send_eod_summary)
        
        while True:
            # Only run during market hours
            now = datetime.now()
            market_open = now.replace(
                hour=int(config.MARKET_OPEN_TIME.split(":")[0]),
                minute=int(config.MARKET_OPEN_TIME.split(":")[1])
            )
            market_close = now.replace(
                hour=int(config.MARKET_CLOSE_TIME.split(":")[0]),
                minute=int(config.MARKET_CLOSE_TIME.split(":")[1])
            )
            
            if market_open <= now <= market_close and now.weekday() < 5:
                schedule.run_pending()
            
            time.sleep(30)
    
    def run_once(self):
        """Run scanner once (for testing/manual runs)"""
        logger.info("ðŸš€ Running scanner (one-time)...")
        self.fetch_data()
        results = self.run_scanners()
        
        # Print results
        print("\n" + "="*80)
        print("ðŸ“Š SCAN RESULTS")
        print("="*80)
        
        for result in results[:20]:
            print(f"\n{'='*60}")
            print(f"ðŸ“Œ {result.symbol} | {result.signal}")
            print(f"   Price: â‚¹{result.price:.2f} | Change: {result.change_pct:+.2f}%")
            print(f"   Volume: {result.volume_ratio:.1f}x | Strength: {result.strength:.0%}")
            print(f"   Reasons:")
            for reason in result.reasons:
                print(f"     â€¢ {reason}")
        
        return results
    
    def _send_eod_summary(self):
        """Send end of day summary"""
        if self.telegram_bot:
            results = self.run_scanners()
            bullish = [r for r in results if 'BUY' in r.signal or 'BULLISH' in r.signal or 'BREAKOUT' in r.signal]
            bearish = [r for r in results if 'SELL' in r.signal or 'BEARISH' in r.signal]
            self.telegram_bot.send_daily_summary(bullish, bearish)


def main():
    """Entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Indian Stock Scanner")
    parser.add_argument(
        '--mode',
        choices=['once', 'scheduled', 'dashboard'],
        default='once',
        help='Run mode: once (single scan), scheduled (continuous), dashboard (web UI)'
    )
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=None,
        help='Custom symbols to scan'
    )
    
    args = parser.parse_args()
    
    app = StockScannerApp()
    
    if args.symbols:
        app.symbols = [s.upper() for s in args.symbols]
    
    if args.mode == 'once':
        app.run_once()
    elif args.mode == 'scheduled':
        app.run_scheduled()
    elif args.mode == 'dashboard':
        import subprocess
        subprocess.run([
            'streamlit', 'run', 'dashboard/app.py',
            '--server.port', '8501',
            '--theme.base', 'dark'
        ])


if __name__ == "__main__":
    main()
