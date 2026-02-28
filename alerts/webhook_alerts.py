"""
Webhook Alert System for Stock Scanner

Supports:
- Custom webhook URLs
- Slack webhooks
- Discord webhooks
- Microsoft Teams webhooks
- Custom HTTP endpoints
- JSON/Form data formats
- Retry logic
- Rate limiting
"""

import requests
import json
import time
import logging
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass
import os

from scanners.momentum_scanner import ScanResult

logger = logging.getLogger(__name__)


@dataclass
class WebhookConfig:
    """Webhook configuration"""
    name: str
    url: str
    platform: str = "custom"  # 'custom', 'slack', 'discord', 'teams'
    headers: Dict = None
    auth_token: str = ""
    is_active: bool = True
    max_retries: int = 3
    timeout: int = 10


class WebhookAlerts:
    """Send alerts via webhooks to various platforms"""

    def __init__(self, webhooks: List[WebhookConfig] = None):
        self.webhooks = webhooks or []
        
        # Auto-configure from environment
        self._auto_configure()
        
        logger.info(f"âœ… Webhook alerts configured: {len(self.webhooks)} endpoints")

    def _auto_configure(self):
        """Auto-configure from environment variables"""
        # Slack
        slack_url = os.getenv("SLACK_WEBHOOK_URL")
        if slack_url:
            self.webhooks.append(WebhookConfig(
                name="Slack", url=slack_url, platform="slack"
            ))
        
        # Discord
        discord_url = os.getenv("DISCORD_WEBHOOK_URL")
        if discord_url:
            self.webhooks.append(WebhookConfig(
                name="Discord", url=discord_url, platform="discord"
            ))
        
        # Teams
        teams_url = os.getenv("TEAMS_WEBHOOK_URL")
        if teams_url:
            self.webhooks.append(WebhookConfig(
                name="Teams", url=teams_url, platform="teams"
            ))
        
        # Custom
        custom_url = os.getenv("CUSTOM_WEBHOOK_URL")
        if custom_url:
            self.webhooks.append(WebhookConfig(
                name="Custom", url=custom_url, platform="custom"
            ))

    def add_webhook(self, config: WebhookConfig):
        """Add a webhook endpoint"""
        self.webhooks.append(config)

    def _send_request(self, webhook: WebhookConfig, payload: Dict) -> bool:
        """Send HTTP request with retries"""
        headers = webhook.headers or {'Content-Type': 'application/json'}
        
        if webhook.auth_token:
            headers['Authorization'] = f'Bearer {webhook.auth_token}'
        
        for attempt in range(webhook.max_retries):
            try:
                response = requests.post(
                    webhook.url,
                    json=payload,
                    headers=headers,
                    timeout=webhook.timeout
                )
                
                if response.status_code in [200, 201, 204]:
                    return True
                elif response.status_code == 429:
                    wait = int(response.headers.get('Retry-After', 5))
                    time.sleep(wait)
                else:
                    logger.warning(f"Webhook {webhook.name}: HTTP {response.status_code}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Webhook {webhook.name}: timeout (attempt {attempt+1})")
            except Exception as e:
                logger.error(f"Webhook {webhook.name}: {e}")
            
            time.sleep(2 ** attempt)
        
        return False

    # =========================================================
    # PLATFORM-SPECIFIC FORMATTERS
    # =========================================================

    def _format_slack(self, title: str, text: str, results: List[ScanResult] = None) -> Dict:
        """Format message for Slack"""
        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"ðŸ“Š {title}"}
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": text}
            },
        ]
        
        if results:
            for r in results[:10]:
                emoji = "ðŸŸ¢" if "BUY" in r.signal or "BULLISH" in r.signal else "ðŸ”´"
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            f"{emoji} *{r.symbol}* | {r.signal.replace('_', ' ')}\n"
                            f"Price: â‚¹{r.price:.2f} ({r.change_pct:+.2f}%) | "
                            f"Volume: {r.volume_ratio:.1f}x | Strength: {r.strength:.0%}"
                        )
                    }
                })
        
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f"â° {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"}]
        })
        
        return {"blocks": blocks}

    def _format_discord(self, title: str, text: str, results: List[ScanResult] = None) -> Dict:
        """Format message for Discord"""
        embeds = [{
            "title": f"ðŸ“Š {title}",
            "description": text,
            "color": 3447003,  # Blue
            "timestamp": datetime.now().isoformat(),
            "footer": {"text": "AI Stock Scanner"}
        }]
        
        if results:
            fields = []
            for r in results[:10]:
                emoji = "ðŸŸ¢" if "BUY" in r.signal or "BULLISH" in r.signal else "ðŸ”´"
                fields.append({
                    "name": f"{emoji} {r.symbol}",
                    "value": (
                        f"Signal: {r.signal.replace('_', ' ')}\n"
                        f"â‚¹{r.price:.2f} ({r.change_pct:+.2f}%)\n"
                        f"Strength: {r.strength:.0%}"
                    ),
                    "inline": True
                })
            embeds[0]["fields"] = fields
        
        return {"embeds": embeds}

    def _format_teams(self, title: str, text: str, results: List[ScanResult] = None) -> Dict:
        """Format message for Microsoft Teams"""
        facts = []
        if results:
            for r in results[:10]:
                facts.append({
                    "name": r.symbol,
                    "value": f"{r.signal.replace('_', ' ')} | â‚¹{r.price:.2f} | {r.strength:.0%}"
                })
        
        return {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": "0076D7",
            "summary": title,
            "sections": [{
                "activityTitle": f"ðŸ“Š {title}",
                "activitySubtitle": datetime.now().strftime('%Y-%m-%d %H:%M'),
                "facts": facts,
                "text": text,
                "markdown": True
            }]
        }

    def _format_custom(self, title: str, text: str, results: List[ScanResult] = None) -> Dict:
        """Format message for custom webhook"""
        signals = []
        if results:
            for r in results:
                signals.append({
                    'symbol': r.symbol,
                    'signal': r.signal,
                    'price': r.price,
                    'change_pct': r.change_pct,
                    'volume_ratio': r.volume_ratio,
                    'strength': r.strength,
                    'reasons': r.reasons,
                })
        
        return {
            'title': title,
            'message': text,
            'timestamp': datetime.now().isoformat(),
            'signals': signals,
            'signal_count': len(signals),
        }

    def _format_payload(self, webhook: WebhookConfig, title: str, text: str, results: List[ScanResult] = None) -> Dict:
        """Format payload based on platform"""
        formatters = {
            'slack': self._format_slack,
            'discord': self._format_discord,
            'teams': self._format_teams,
            'custom': self._format_custom,
        }
        formatter = formatters.get(webhook.platform, self._format_custom)
        return formatter(title, text, results)

    # =========================================================
    # SEND METHODS
    # =========================================================

    def send_alert(
        self,
        title: str,
        message: str,
        results: List[ScanResult] = None,
        webhook_names: List[str] = None
    ):
        """Send alert to all or specific webhooks"""
        for webhook in self.webhooks:
            if not webhook.is_active:
                continue
            if webhook_names and webhook.name not in webhook_names:
                continue
            
            payload = self._format_payload(webhook, title, message, results)
            success = self._send_request(webhook, payload)
            
            if success:
                logger.info(f"âœ… Webhook sent: {webhook.name}")
            else:
                logger.error(f"âŒ Webhook failed: {webhook.name}")

    def send_scan_results(self, results: List[ScanResult], title: str = "Scanner Alert"):
        """Send scan results to all webhooks"""
        if not results:
            return
        
        text = f"Found {len(results)} signals at {datetime.now().strftime('%H:%M:%S')}"
        self.send_alert(title, text, results)

    def send_daily_summary(
        self,
        bullish: List[ScanResult],
        bearish: List[ScanResult]
    ):
        """Send daily summary"""
        text = (
            f"Bullish: {len(bullish)} signals | "
            f"Bearish: {len(bearish)} signals"
        )
        all_results = bullish[:5] + bearish[:5]
        self.send_alert("Daily Summary", text, all_results)

    def send_ai_prediction(self, symbol: str, prediction: Dict, price: float):
        """Send AI prediction alert"""
        text = (
            f"ðŸ¤– {symbol}: {prediction['prediction']} "
            f"(Confidence: {prediction['confidence']:.0%})"
        )
        self.send_alert(f"AI Signal: {symbol}", text)

    def send_anomaly_alert(self, symbol: str, anomaly_type: str, details: str):
        """Send anomaly detection alert"""
        text = f"âš ï¸ {symbol}: {anomaly_type} - {details}"
        self.send_alert(f"Anomaly: {symbol}", text)

    def send_custom_data(self, data: Dict):
        """Send arbitrary data to all webhooks"""
        for webhook in self.webhooks:
            if webhook.is_active:
                self._send_request(webhook, data)

    def test_webhooks(self) -> Dict[str, bool]:
        """Test all webhook connections"""
        results = {}
        for webhook in self.webhooks:
            payload = self._format_payload(
                webhook,
                "ðŸ§ª Test Alert",
                "This is a test message from AI Stock Scanner"
            )
            results[webhook.name] = self._send_request(webhook, payload)
        return results



