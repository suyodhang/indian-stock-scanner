"""
Email Alert System for Stock Scanner

Features:
- HTML formatted alerts
- Scan result summaries
- Daily/Weekly reports
- Portfolio alerts
- AI prediction alerts
- Attachment support (charts, CSV)
- Multiple recipient support
- Template system
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import List, Dict, Optional
from datetime import datetime
import logging
import os

from scanners.momentum_scanner import ScanResult

logger = logging.getLogger(__name__)


class EmailAlerts:
    """Email alert system"""

    def __init__(
        self,
        smtp_server: str = None,
        smtp_port: int = None,
        sender_email: str = None,
        sender_password: str = None,
        recipients: List[str] = None
    ):
        self.smtp_server = smtp_server or os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = smtp_port or int(os.getenv("SMTP_PORT", "587"))
        self.sender_email = sender_email or os.getenv("SENDER_EMAIL", "")
        self.sender_password = sender_password or os.getenv("SENDER_PASSWORD", "")
        self.recipients = recipients or os.getenv("ALERT_RECIPIENTS", "").split(",")
        
        self.is_configured = bool(self.sender_email and self.sender_password)
        
        if self.is_configured:
            logger.info("âœ… Email alerts configured")
        else:
            logger.warning("âš ï¸ Email alerts not configured")

    def _create_connection(self):
        """Create SMTP connection"""
        context = ssl.create_default_context()
        server = smtplib.SMTP(self.smtp_server, self.smtp_port)
        server.ehlo()
        server.starttls(context=context)
        server.login(self.sender_email, self.sender_password)
        return server

    def send_email(
        self,
        subject: str,
        body_html: str,
        recipients: List[str] = None,
        attachments: List[str] = None
    ) -> bool:
        """Send email with HTML body and optional attachments"""
        if not self.is_configured:
            logger.warning("Email not configured")
            return False
        
        to_list = recipients or self.recipients
        
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = self.sender_email
        msg['To'] = ", ".join(to_list)
        
        # HTML body
        html_part = MIMEText(body_html, 'html')
        msg.attach(html_part)
        
        # Attachments
        if attachments:
            for filepath in attachments:
                try:
                    with open(filepath, 'rb') as f:
                        part = MIMEBase('application', 'octet-stream')
                        part.set_payload(f.read())
                    encoders.encode_base64(part)
                    filename = os.path.basename(filepath)
                    part.add_header('Content-Disposition', f'attachment; filename="{filename}"')
                    msg.attach(part)
                except Exception as e:
                    logger.error(f"Error attaching {filepath}: {e}")
        
        try:
            server = self._create_connection()
            server.sendmail(self.sender_email, to_list, msg.as_string())
            server.quit()
            logger.info(f"âœ… Email sent: {subject}")
            return True
        except Exception as e:
            logger.error(f"âŒ Email failed: {e}")
            return False

    def format_scan_results_html(self, results: List[ScanResult], title: str = "Scanner Alert") -> str:
        """Format scan results as HTML email"""
        rows = ""
        for r in results[:20]:
            color = "#00cc00" if "BUY" in r.signal or "BULLISH" in r.signal else "#cc0000"
            strength_bar = f'<div style="background:#333;border-radius:4px;width:100px;height:12px;"><div style="background:{color};border-radius:4px;width:{int(r.strength*100)}px;height:12px;"></div></div>'
            
            rows += f"""
            <tr>
                <td style="padding:8px;border-bottom:1px solid #333;font-weight:bold;">{r.symbol}</td>
                <td style="padding:8px;border-bottom:1px solid #333;"><span style="color:{color}">{r.signal.replace('_',' ')}</span></td>
                <td style="padding:8px;border-bottom:1px solid #333;">â‚¹{r.price:.2f}</td>
                <td style="padding:8px;border-bottom:1px solid #333;color:{'#00cc00' if r.change_pct > 0 else '#cc0000'}">{r.change_pct:+.2f}%</td>
                <td style="padding:8px;border-bottom:1px solid #333;">{r.volume_ratio:.1f}x</td>
                <td style="padding:8px;border-bottom:1px solid #333;">{strength_bar} {r.strength:.0%}</td>
            </tr>
            """
        
        html = f"""
        <html>
        <body style="background:#1a1a1a;color:#e0e0e0;font-family:Arial,sans-serif;padding:20px;">
            <div style="max-width:800px;margin:0 auto;background:#2a2a2a;border-radius:10px;padding:20px;">
                <h1 style="color:#00bfff;border-bottom:2px solid #00bfff;padding-bottom:10px;">
                    ðŸ“Š {title}
                </h1>
                <p style="color:#888;">
                    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
                    {len(results)} signals found
                </p>
                
                <table style="width:100%;border-collapse:collapse;margin-top:15px;">
                    <tr style="background:#333;">
                        <th style="padding:10px;text-align:left;">Symbol</th>
                        <th style="padding:10px;text-align:left;">Signal</th>
                        <th style="padding:10px;text-align:left;">Price</th>
                        <th style="padding:10px;text-align:left;">Change</th>
                        <th style="padding:10px;text-align:left;">Volume</th>
                        <th style="padding:10px;text-align:left;">Strength</th>
                    </tr>
                    {rows}
                </table>
                
                <div style="margin-top:20px;padding:15px;background:#1a1a1a;border-radius:8px;">
                    <h3 style="color:#ffd700;">ðŸ“‹ Signal Details</h3>
                    {''.join(f'<div style="margin:10px 0;padding:10px;border-left:3px solid #00bfff;"><b>{r.symbol}</b>: {"; ".join(r.reasons[:3])}</div>' for r in results[:10])}
                </div>
                
                <p style="color:#666;font-size:12px;margin-top:20px;text-align:center;">
                    AI Stock Scanner | Auto-generated alert | 
                    <a href="#" style="color:#00bfff;">Manage Preferences</a>
                </p>
            </div>
        </body>
        </html>
        """
        return html

    def send_scan_alert(self, results: List[ScanResult], title: str = "Scanner Alert"):
        """Send scan results email"""
        if not results:
            return
        
        html = self.format_scan_results_html(results, title)
        subject = f"ðŸ“Š {title} | {len(results)} Signals | {datetime.now().strftime('%Y-%m-%d')}"
        self.send_email(subject, html)

    def send_daily_summary(
        self,
        bullish: List[ScanResult],
        bearish: List[ScanResult],
        market_data: Dict = None
    ):
        """Send end-of-day summary email"""
        market_section = ""
        if market_data:
            market_section = f"""
            <div style="margin:15px 0;padding:15px;background:#1a1a1a;border-radius:8px;">
                <h3>ðŸ“ˆ Market Overview</h3>
                <p>NIFTY 50: {market_data.get('nifty50', 'N/A')}</p>
                <p>SENSEX: {market_data.get('sensex', 'N/A')}</p>
            </div>
            """
        
        html = f"""
        <html>
        <body style="background:#1a1a1a;color:#e0e0e0;font-family:Arial,sans-serif;padding:20px;">
            <div style="max-width:800px;margin:0 auto;background:#2a2a2a;border-radius:10px;padding:20px;">
                <h1 style="color:#00bfff;">ðŸ“Š Daily Market Summary</h1>
                <p>{datetime.now().strftime('%A, %B %d, %Y')}</p>
                
                {market_section}
                
                <h2 style="color:#00cc00;">ðŸŸ¢ Bullish Signals ({len(bullish)})</h2>
                {''.join(f'<p>â€¢ <b>{r.symbol}</b> - {r.signal.replace("_"," ")} (â‚¹{r.price:.2f}, {r.strength:.0%})</p>' for r in bullish[:10])}
                
                <h2 style="color:#cc0000;">ðŸ”´ Bearish Signals ({len(bearish)})</h2>
                {''.join(f'<p>â€¢ <b>{r.symbol}</b> - {r.signal.replace("_"," ")} (â‚¹{r.price:.2f}, {r.strength:.0%})</p>' for r in bearish[:10])}
                
                <p style="color:#666;font-size:12px;margin-top:20px;text-align:center;">
                    AI Stock Scanner | Daily Summary
                </p>
            </div>
        </body>
        </html>
        """
        
        subject = f"ðŸ“Š Daily Summary | {len(bullish)} Bullish, {len(bearish)} Bearish | {datetime.now().strftime('%Y-%m-%d')}"
        self.send_email(subject, html)

    def send_ai_alert(self, symbol: str, prediction: Dict, price: float):
        """Send AI prediction alert"""
        color = "#00cc00" if prediction['prediction'] == 'BULLISH' else "#cc0000"
        
        html = f"""
        <html>
        <body style="background:#1a1a1a;color:#e0e0e0;font-family:Arial,sans-serif;padding:20px;">
            <div style="max-width:600px;margin:0 auto;background:#2a2a2a;border-radius:10px;padding:20px;">
                <h1 style="color:#00bfff;">ðŸ¤– AI Trading Signal</h1>
                <h2>{symbol} - â‚¹{price:.2f}</h2>
                <div style="background:{color};color:white;padding:15px;border-radius:8px;text-align:center;font-size:24px;">
                    {prediction['prediction']}
                </div>
                <p>Confidence: {prediction['confidence']:.0%}</p>
                <p>Model Votes: {prediction.get('bullish_votes', 'N/A')}/{prediction.get('total_models', 'N/A')}</p>
                <p style="color:#666;font-size:12px;">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </body>
        </html>
        """
        
        subject = f"ðŸ¤– AI Signal: {symbol} - {prediction['prediction']} ({prediction['confidence']:.0%})"
        self.send_email(subject, html)

