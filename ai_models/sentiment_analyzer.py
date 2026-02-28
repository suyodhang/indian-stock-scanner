"""
News & Social Media Sentiment Analysis for Indian Stocks
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
import re
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Analyze sentiment from news and social media"""
    
    def __init__(self, news_api_key: str = None):
        self.news_api_key = news_api_key
        self.sentiment_model = None
        self._load_model()
        
        # Source reliability multipliers (heuristic).
        self.source_weights = {
            'reuters': 1.15,
            'bloomberg': 1.15,
            'cnbc': 1.10,
            'moneycontrol': 1.10,
            'economic times': 1.10,
            'livemint': 1.05,
            'business standard': 1.05,
            'ndtv profit': 1.00,
            'yahoo finance': 1.00,
            'google news': 0.95,
        }

        # Event impact lexicon (positive and negative drivers).
        self.event_impact_weights = {
            # Positive events
            'beat estimates': 0.70,
            'strong results': 0.65,
            'record profit': 0.70,
            'profit jumps': 0.60,
            'upgraded': 0.55,
            'buy rating': 0.55,
            'order win': 0.65,
            'new contract': 0.55,
            'merger': 0.45,
            'acquisition': 0.45,
            'dividend': 0.40,
            'bonus issue': 0.45,
            'share buyback': 0.60,
            'guidance raised': 0.70,
            'expansion': 0.40,
            'ipo oversubscribed': 0.60,
            # Negative events
            'miss estimates': -0.70,
            'weak results': -0.65,
            'profit falls': -0.60,
            'downgraded': -0.55,
            'sell rating': -0.55,
            'fraud': -0.95,
            'scam': -0.95,
            'penalty': -0.60,
            'investigation': -0.70,
            'lawsuit': -0.65,
            'default': -0.90,
            'debt stress': -0.70,
            'guidance cut': -0.75,
            'stake sale': -0.40,
            'pledged shares': -0.50,
            'ban': -0.70,
            'data breach': -0.70,
            'resignation': -0.45,
            'geopolitical tension': -0.45,
        }
    
    def _load_model(self):
        """Load pre-trained sentiment model"""
        try:
            from transformers import pipeline
            
            # Use FinBERT for financial sentiment
            self.sentiment_model = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                top_k=None
            )
            logger.info("âœ… FinBERT model loaded for sentiment analysis")
        except ImportError:
            logger.warning("transformers not installed. Using basic sentiment analysis.")
            self.sentiment_model = None
        except Exception as e:
            logger.warning(f"Could not load FinBERT: {e}. Using basic analysis.")
            self.sentiment_model = None
    
    def analyze_text(self, text: str) -> Dict:
        """Analyze sentiment of a single text"""
        if self.sentiment_model:
            result = self.sentiment_model(text[:512])  # FinBERT max length
            
            # FinBERT returns: positive, negative, neutral
            scores = {item['label']: item['score'] for item in result[0]}
            
            return {
                'text': text[:200],
                'positive': scores.get('positive', 0),
                'negative': scores.get('negative', 0),
                'neutral': scores.get('neutral', 0),
                'sentiment': max(scores, key=scores.get),
                'confidence': max(scores.values())
            }
        else:
            return self._basic_sentiment(text)

    def _parse_published_at(self, published_at: str) -> Optional[datetime]:
        """Parse published timestamp from mixed RSS/API formats."""
        if not published_at:
            return None
        try:
            return parsedate_to_datetime(published_at)
        except Exception:
            pass
        for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(published_at, fmt)
            except Exception:
                continue
        return None

    def _source_weight(self, source: str) -> float:
        """Get source reliability multiplier."""
        src = str(source or "").lower()
        for name, weight in self.source_weights.items():
            if name in src:
                return weight
        return 1.0

    def _recency_weight(self, published_at: str) -> float:
        """Recent news has stronger impact."""
        dt = self._parse_published_at(published_at)
        if dt is None:
            return 1.0
        if dt.tzinfo is not None:
            now = datetime.now(dt.tzinfo)
        else:
            now = datetime.now()
        hours = max((now - dt).total_seconds() / 3600.0, 0.0)
        if hours <= 6:
            return 1.20
        if hours <= 24:
            return 1.10
        if hours <= 72:
            return 1.00
        if hours <= 168:
            return 0.85
        return 0.70

    def estimate_news_impact(
        self,
        title: str,
        description: str = "",
        source: str = "",
        published_at: str = "",
        sentiment: Optional[Dict] = None
    ) -> Dict:
        """
        Estimate directional impact of a news item.
        Returns normalized impact score in [-100, +100].
        """
        text = f"{title}. {description}".lower()
        if sentiment is None:
            sentiment = self.analyze_text(text[:512])

        # Sentiment component (-1 to +1).
        sentiment_component = float(sentiment.get('positive', 0.0) - sentiment.get('negative', 0.0))

        # Event component from financial trigger keywords.
        event_component = 0.0
        matched = []
        for phrase, weight in self.event_impact_weights.items():
            if phrase in text:
                event_component += weight
                matched.append(phrase)

        # Cap cumulative event component.
        event_component = float(np.clip(event_component, -1.5, 1.5))

        # Blend components and apply context multipliers.
        raw = (0.55 * sentiment_component) + (0.45 * (event_component / 1.5))
        weighted = raw * self._source_weight(source) * self._recency_weight(published_at)
        impact_score = float(np.clip(weighted * 100.0, -100.0, 100.0))

        if impact_score >= 35:
            impact_label = "HIGH_POSITIVE"
        elif impact_score >= 15:
            impact_label = "POSITIVE"
        elif impact_score <= -35:
            impact_label = "HIGH_NEGATIVE"
        elif impact_score <= -15:
            impact_label = "NEGATIVE"
        else:
            impact_label = "NEUTRAL"

        return {
            'impact_score': impact_score,
            'impact_label': impact_label,
            'event_component': float(event_component),
            'sentiment_component': float(sentiment_component),
            'matched_events': matched[:5],
            'source_weight': self._source_weight(source),
            'recency_weight': self._recency_weight(published_at),
        }
    
    def _basic_sentiment(self, text: str) -> Dict:
        """Basic rule-based sentiment analysis"""
        text_lower = text.lower()
        
        positive_words = [
            'bullish', 'surge', 'rally', 'gain', 'profit', 'growth', 
            'upgrade', 'outperform', 'buy', 'strong', 'positive',
            'record high', 'breakout', 'momentum', 'upbeat', 'boom',
            'dividend', 'bonus', 'robust', 'expansion'
        ]
        
        negative_words = [
            'bearish', 'crash', 'fall', 'loss', 'decline', 'downgrade',
            'sell', 'weak', 'negative', 'slump', 'correction', 'risk',
            'debt', 'default', 'scam', 'fraud', 'penalty', 'ban',
            'recession', 'inflation', 'crisis'
        ]
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        total = pos_count + neg_count + 1
        
        pos_score = pos_count / total
        neg_score = neg_count / total
        neu_score = 1 - pos_score - neg_score
        
        if pos_score > neg_score:
            sentiment = 'positive'
        elif neg_score > pos_score:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'text': text[:200],
            'positive': pos_score,
            'negative': neg_score,
            'neutral': neu_score,
            'sentiment': sentiment,
            'confidence': max(pos_score, neg_score, neu_score)
        }
    
    def fetch_news(self, symbol: str, days: int = 7) -> List[Dict]:
        """Fetch news articles for a stock"""
        articles = []
        
        # Method 1: NewsAPI
        if self.news_api_key:
            try:
                from newsapi import NewsApiClient
                newsapi = NewsApiClient(api_key=self.news_api_key)
                
                from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
                
                response = newsapi.get_everything(
                    q=f"{symbol} stock NSE BSE India",
                    from_param=from_date,
                    language='en',
                    sort_by='relevancy',
                    page_size=20
                )
                
                for article in response.get('articles', []):
                    articles.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'url': article.get('url', ''),
                        'published_at': article.get('publishedAt', ''),
                    })
            except Exception as e:
                logger.error(f"Error fetching news from NewsAPI: {e}")
        
        # Method 2: Google News RSS (free, no API key needed)
        try:
            import feedparser
            
            query = f"{symbol}+stock+india"
            url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
            
            feed = feedparser.parse(url)
            
            for entry in feed.entries[:15]:
                articles.append({
                    'title': entry.get('title', ''),
                    'description': entry.get('summary', ''),
                    'source': entry.get('source', {}).get('title', 'Google News'),
                    'url': entry.get('link', ''),
                    'published_at': entry.get('published', ''),
                })
        except Exception as e:
            logger.error(f"Error fetching Google News: {e}")
        
        return articles
    
    def get_stock_sentiment(self, symbol: str, days: int = 7) -> Dict:
        """
        Get overall sentiment for a stock
        
        Returns:
            Aggregated sentiment scores and individual article sentiments
        """
        articles = self.fetch_news(symbol, days)
        
        if not articles:
            return {
                'symbol': symbol,
                'overall_sentiment': 'neutral',
                'sentiment_score': 0.0,
                'impact_score': 0.0,
                'overall_impact': 'NEUTRAL',
                'article_count': 0,
                'articles': []
            }
        
        sentiments = []
        impacts = []
        for article in articles:
            text = f"{article['title']}. {article.get('description', '')}"
            sentiment = self.analyze_text(text)
            impact = self.estimate_news_impact(
                title=article.get('title', ''),
                description=article.get('description', ''),
                source=article.get('source', ''),
                published_at=article.get('published_at', ''),
                sentiment=sentiment,
            )
            article['sentiment'] = sentiment
            article['impact'] = impact
            sentiments.append(sentiment)
            impacts.append(impact['impact_score'])
        
        # Aggregate sentiments
        avg_positive = np.mean([s['positive'] for s in sentiments])
        avg_negative = np.mean([s['negative'] for s in sentiments])
        avg_neutral = np.mean([s['neutral'] for s in sentiments])
        
        # Composite score: -1 (very negative) to +1 (very positive)
        sentiment_score = avg_positive - avg_negative
        
        if sentiment_score > 0.1:
            overall = 'positive'
        elif sentiment_score < -0.1:
            overall = 'negative'
        else:
            overall = 'neutral'
        
        positive_count = sum(1 for s in sentiments if s['sentiment'] == 'positive')
        negative_count = sum(1 for s in sentiments if s['sentiment'] == 'negative')
        neutral_count = sum(1 for s in sentiments if s['sentiment'] == 'neutral')
        impact_score = float(np.mean(impacts)) if impacts else 0.0
        if impact_score >= 20:
            overall_impact = 'POSITIVE'
        elif impact_score <= -20:
            overall_impact = 'NEGATIVE'
        else:
            overall_impact = 'NEUTRAL'
        high_impact_count = int(sum(1 for v in impacts if abs(v) >= 35))
        
        return {
            'symbol': symbol,
            'overall_sentiment': overall,
            'sentiment_score': float(sentiment_score),
            'impact_score': impact_score,
            'overall_impact': overall_impact,
            'high_impact_count': high_impact_count,
            'avg_positive': float(avg_positive),
            'avg_negative': float(avg_negative),
            'avg_neutral': float(avg_neutral),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'article_count': len(articles),
            'articles': articles
        }
    
    def batch_sentiment_analysis(self, symbols: List[str]) -> pd.DataFrame:
        """Analyze sentiment for multiple stocks"""
        results = []
        
        for symbol in symbols:
            logger.info(f"Analyzing sentiment for {symbol}...")
            sentiment = self.get_stock_sentiment(symbol)
            results.append({
                'symbol': symbol,
                'sentiment': sentiment['overall_sentiment'],
                'score': sentiment['sentiment_score'],
                'impact': sentiment.get('impact_score', 0.0),
                'positive': sentiment['avg_positive'],
                'negative': sentiment['avg_negative'],
                'articles': sentiment['article_count']
            })
        
        return pd.DataFrame(results)
