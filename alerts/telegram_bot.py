"""
Telegram Alert System for Stock Scanner
"""

import requests
import logging
from typing import List, Dict, Optional
from datetime import datetime
from scanners.momentum_scanner import ScanResult

logger = logging.getLogger(__name__)


class TelegramBot:
    """Send scan alerts via Telegram"""
    
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
    
    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a text message"""
        url = f"{self.base_url}/sendMessage"
        data = {
            'chat_id': self.chat_id,
            'text': text,
            'parse_mode': parse_mode,
            'disable_web_page_preview': True
        }
        
        try:
            response = requests.post(url, data=data, timeout=10)
            if response.status_code == 200:
                return True
            else:
                logger.error(f"Telegram error: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            return False
    
    def format_scan_result(self, result: ScanResult) -> str:
        """Format a ScanResult into a Telegram message"""
        # Signal emoji
        signal_emojis = {
            'RSI_OVERSOLD_BOUNCE': 'ðŸŸ¢',
            'RSI_OVERBOUGHT_PULLBACK': 'ðŸ”´',
            'MACD_BULLISH_CROSSOVER': 'ðŸŸ¢',
            'MACD_BEARISH_CROSSOVER': 'ðŸ”´',
            'SUPERTREND_BUY': 'ðŸŸ¢',
            'SUPERTREND_SELL': 'ðŸ”´',
            'GOLDEN_CROSS': 'ðŸŸ¢â­',
            'DEATH_CROSS': 'ðŸ”´âš ï¸',
            '52_WEEK_HIGH_BREAKOUT': 'ðŸš€',
            'RESISTANCE_BREAKOUT': 'ðŸŸ¢',
            'BB_SQUEEZE_BULLISH': 'ðŸŸ¢',
            'BB_SQUEEZE_BEARISH': 'ðŸ”´',
            'VOLUME_BREAKOUT': 'ðŸ”¥',
            'CONSOLIDATION_BREAKOUT': 'ðŸŸ¢',
            'GAP_UP': 'ðŸ“ˆ',
            'AI_ENHANCED': 'ðŸ¤–',
            'STRONG_UPTREND': 'ðŸ’ª',
        }
        
        emoji = signal_emojis.get(result.signal, 'ðŸ“Š')
        
        strength_bar = 'â–ˆ' * int(result.strength * 10) + 'â–‘' * (10 - int(result.strength * 10))
        
        message = f"""
{emoji} <b>{result.symbol}</b> - {result.signal.replace('_', ' ')}

ðŸ’° Price: â‚¹{result.price:.2f} ({'+' if result.change_pct > 0 else ''}{result.change_pct:.2f}%)
ðŸ“Š Volume: {result.volume_ratio:.1f}x average
ðŸ’ª Signal Strength: [{strength_bar}] {result.strength:.0%}

ðŸ“ <b>Reasons:</b>
{chr(10).join('  â€¢ ' + r for r in result.reasons)}
"""
        
        if result.ai_prediction:
            ai = result.ai_prediction
            message += f"""
ðŸ¤– <b>AI Analysis:</b>
  Prediction: {ai['prediction']} ({ai['confidence']:.0%})
  Model Votes: {ai['bullish_votes']} Bullish / {ai['bearish_votes']} Bearish
"""
        
        message += f"\nâ° {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return message
    
    def send_scan_results(self, results: List[ScanResult], title: str = "Scanner Alert"):
        """Send multiple scan results"""
        if not results:
            return
        
        # Header message
        header = f"""
ðŸ“¡ <b>{title}</b>
â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
Found {len(results)} signals
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
"""
        self.send_message(header)
        
        # Send top results (limit to avoid spam)
        for result in results[:10]:
            message = self.format_scan_result(result)
            self.send_message(message)
    
    def send_daily_summary(
        self,
        bullish_signals: List[ScanResult],
        bearish_signals: List[ScanResult],
        market_data: Dict = None
    ):
        """Send end-of-day summary"""
        message = f"""
ðŸ“Š <b>Daily Market Scanner Summary</b>
ðŸ“… {datetime.now().strftime('%Y-%m-%d')}
â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”

ðŸŸ¢ <b>Bullish Signals: {len(bullish_signals)}</b>
{chr(10).join(f'  â€¢ {r.symbol} ({r.signal.replace("_", " ")}) - â‚¹{r.price:.2f}' for r in bullish_signals[:10])}

ðŸ”´ <b>Bearish Signals: {len(bearish_signals)}</b>
{chr(10).join(f'  â€¢ {r.symbol} ({r.signal.replace("_", " ")}) - â‚¹{r.price:.2f}' for r in bearish_signals[:10])}
"""
        
        if market_data:
            message += f"""
ðŸ“ˆ <b>Market Overview:</b>
  NIFTY 50: {market_data.get('nifty50', 'N/A')}
  BANK NIFTY: {market_data.get('banknifty', 'N/A')}
  Advances: {market_data.get('advances', 'N/A')}
  Declines: {market_data.get('declines', 'N/A')}
"""
        
        self.send_message(message)
    
    def send_ai_alert(
        self,
        symbol: str,
        prediction: Dict,
        current_price: float
    ):
        """Send AI-specific alert"""
        message = f"""
ðŸ¤– <b>AI Trading Signal</b>
â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”

ðŸ“ˆ <b>{symbol}</b>
ðŸ’° Current Price: â‚¹{current_price:.2f}

ðŸ”® <b>Prediction: {prediction['prediction']}</b>
ðŸ“Š Confidence: {prediction['confidence']:.0%}
ðŸ—³ï¸ Model Votes: {prediction['bullish_votes']}/{prediction['total_models']} Bullish

<b>Top Influential Features:</b>
{chr(10).join(f'  â€¢ {k}: {v:.4f}' for k, v in list(prediction.get('top_features', {}).items())[:5])}

â° {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.send_message(message)
